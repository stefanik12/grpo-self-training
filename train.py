import os
import time
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Any, Dict

import hydra
import numpy as np
import torch
import torch.distributed as dist
import wandb
import yaml
from torch.utils.data import DataLoader

from data_types import Split
from objects import Task, GenerationStrategy, Objective, Evaluator
from optimizer import MemoryEfficientAdamW
from tokenizer import Tokenizer


def evaluate(model, tokenizer: Tokenizer, task: Task, evaluators: List[Evaluator],
             generation_strategy: GenerationStrategy, device: torch.device, dtype: torch.dtype,
             config: Dict[str, Any], user_gen_kwargs: Dict[str, Any]) -> None:
    test_dataset = task.get_dataset(Split.test)
    generator = torch.Generator(device=device)
    # We reduce the batch size by half as we want to
    # generate twice as long trajectories.
    dataloader = DataLoader(
        test_dataset,
        shuffle=False,
        collate_fn=test_dataset.collator_function,
        generator=generator,
        batch_size=config["training"]["batch_size"] // 2,
        drop_last=False,
    )
    inputs, preds, expected, success, episodes = [], [], [], [], []
    metrics = {e.evaluator_id: [] for e in evaluators}
    for input_batch in dataloader:
        responses_str, responses_tokens = generation_strategy.generate(
                model=model,
                tokenizer=tokenizer,
                batch=input_batch,
                num_responses=1,
                dtype=dtype,
                extra_generate_kwargs={**task.gen_kwargs, **user_gen_kwargs}
        )
        batch_episodes = task.reward_responses(input_batch, responses_str, responses_tokens)
        inputs_str_b = input_batch.input_strs
        preds_str_b = [resp[0] for resp in responses_str]
        extra_reward_info_b = [str({k: v[i] for k, v in input_batch.extra_reward_info.items()})
                          for i in range(len(input_batch.input_strs))]
        expected_v = input_batch.extra_reward_info.get("dataset_target", [""]*len(preds_str_b))
        inputs.extend(inputs_str_b)
        preds.extend(preds_str_b)
        expected.extend(extra_reward_info_b)
        success.extend([str(episode.reward_info) for episode in batch_episodes])
        episodes.extend(batch_episodes)

        metrics = {e.evaluator_id: metrics[e.evaluator_id] + e.evaluate_batch(inputs_str_b, expected_v, preds_str_b)
                   for e in evaluators}

    metrics_keys = [e.evaluator_id for e in evaluators]
    metrics_vals = [sum(metrics[k]) / len(metrics[k]) for k in metrics]

    table = wandb.Table(columns=["Input", "Predicted", "Expected", "Rewards", *metrics_keys],
                        data=list(zip(inputs, preds, expected, success, *[metrics[k] for k in metrics])))
    wandb.log({f"eval/Predictions": table})
    [wandb.log({"eval/"+ k: v for k, v in zip(metrics_keys, metrics_vals)})]

    mean_reward = np.mean([episode.reward for episode in episodes])
    print(f"\rEval mean_reward: {mean_reward:.2f}" + " " * 100)
    wandb.log({"eval/reward": mean_reward})


def main(config_path: str, local_rank: int):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device(config["model"]["device"])
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(config["model"]["dtype"], torch.bfloat16)
    torch.random.manual_seed(config["training"]["random_seed"])
    torch.set_default_device(device)

    pretrained_model_path = Path(config["model"]["pretrained_model_path"])
    batch_size = config["training"]["batch_size"]
    num_prompts_per_batch = config["training"]["num_prompts_per_batch"]
    num_responses_per_prompt = batch_size // num_prompts_per_batch

    # TODO: resolve logging in multi-gpu training so only one process logs
    wandb_logger = wandb.init(project="selftraining", entity="transformersclub", config={})

    tokenizer = hydra.utils.instantiate(config["tokenizer"])

    task = hydra.utils.instantiate(config["task"])

    evaluators = []
    for evaluator_id, evaluator_cfg in config.get("evaluators", {}).items():
        EvalCls = hydra.utils.get_class(evaluator_cfg["class"])
        evaluators.append(EvalCls(**evaluator_cfg["init_kwargs"]))

    # train data resolution
    train_dataset = task.get_dataset(Split.train)
    generator = torch.Generator(device=device)
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=train_dataset.collator_function,
        generator=generator,
        batch_size=num_prompts_per_batch,
    )

    # model resolution
    ModelCls = hydra.utils.get_class(config["model"]["class"])
    model = ModelCls.from_pretrained(pretrained_model_path).to(device).train()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    generation_strategy: GenerationStrategy = hydra.utils.instantiate(config["generation_strategy"])
    user_gen_kwargs = config.get("generation_kwargs", {})

    optimizer = MemoryEfficientAdamW(model.parameters(), **config["optimizer"])

    objective: Objective = hydra.utils.instantiate(config["objective"])

    start_time = time.time()
    ckpt_dir = Path(config["training"]["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # main training iteration
    for step, input_batch in enumerate(train_dataloader, start=1):
        # start with evaluation
        if step % config["training"]["eval_interval"] == 0:
            evaluate(model, tokenizer, task, evaluators, generation_strategy, device, dtype, config, user_gen_kwargs)

        responses_str, responses_tokens = generation_strategy.generate(
                model=model,
                tokenizer=tokenizer,
                batch=input_batch,
                num_responses=num_responses_per_prompt,
                dtype=dtype,
                extra_generate_kwargs={**task.gen_kwargs, **user_gen_kwargs}
        )
        episodes = task.reward_responses(input_batch, responses_str, responses_tokens)

        if config["training"]["skip_unfinished_episodes"]:
            episodes = [episode for episode in episodes if episode.is_finished]
        results = objective.update_model(
                model=model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                episodes=episodes,
                dtype=dtype
        )
        torch.cuda.synchronize()
        end_time = time.time()
        duration = end_time - start_time
        start_time = end_time

        # train metrics logging
        reward = [episode.reward for episode in episodes]
        reward_infos = {k: np.mean([e.reward_info[k] for e in episodes]) for k, vals in episodes[0].reward_info.items()}

        mean_reward = np.mean(reward)
        std_reward = np.std(reward)

        grad_norm = results["grad_norm"]
        entropy = results["entropy"]
        lr = optimizer.param_groups[0]["lr"]
        loss = results["loss"]
        mean_response_len = np.mean(
            [len(episode.generated_token_ids) for episode in episodes]
        )
        print(
            f"\rStep {step}, "
            f"mean_reward: {mean_reward:.2f}, "
            f"loss: {loss:.2f}, "
            f"grad_norm: {grad_norm:.2f}, "
            f"duration: {duration:.2f}, "
            f"mean_response_len: {mean_response_len:.2f}, "
            f"entropy: {entropy:.2f}"
        )

        wandb_logger.log({"loss": loss})
        wandb_logger.log({"mean_reward": mean_reward})
        wandb_logger.log({"std_reward": std_reward})

        [wandb_logger.log({k: v}) for k, v in reward_infos.items()]
        wandb_logger.log({"grad_norm": grad_norm})
        wandb_logger.log({"duration": duration})
        wandb_logger.log({"learning_rate": lr})
        wandb_logger.log({"mean_response_len": mean_response_len})
        wandb_logger.log({"entropy": entropy})

        # save checkpoint
        if step % config["training"]["ckpt_save_interval"] == 0:
            output_file = ckpt_dir / f"ckpt_{step:06d}.pt"
            torch.save(model.state_dict(), output_file)
            print(f"Saved checkpoint to {output_file}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))  # set by torchrun
    torch.cuda.set_device(local_rank)
    torch.set_default_device("cuda")

    if not dist.is_initialized():  # only one worker does the group init
        if "RANK" not in os.environ:
            # local/single-process defaults
            os.environ["RANK"] = os.environ.get("RANK", "0")
            os.environ["WORLD_SIZE"] = os.environ.get("WORLD_SIZE", "1")
            os.environ["LOCAL_RANK"] = os.environ.get("LOCAL_RANK", "0")
            os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
            os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")

    main(args.config, local_rank)

    dist.destroy_process_group()
