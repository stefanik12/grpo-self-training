import os
import time
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.distributed as dist
import wandb
import yaml
from torch.utils.data import DataLoader

from data_types import Split
from objects import Task, GenerationStrategy, Objective
from optimizer import MemoryEfficientAdamW
from tokenizer import Tokenizer


def evaluate(model, tokenizer: Tokenizer, task: Task, generation_strategy: GenerationStrategy,
             device, dtype, config, user_gen_kwargs):
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
    inputs, preds, expected, success = [], [], [], []
    for input_batch in dataloader:
        responses_str, responses_tokens = generation_strategy.generate(
                model=model,
                tokenizer=tokenizer,
                batch=input_batch,
                num_responses=1,
                dtype=dtype,
                extra_generate_kwargs={**task.gen_kwargs, **user_gen_kwargs}
        )
        episodes = task.reward_responses(input_batch, responses_str, responses_tokens)

        inputs.extend(input_batch.input_strs)
        preds.extend([resp[0] for resp in responses_str])
        expected.extend([str({k: v[i] for k, v in input_batch.extra_reward_info.items()})
                         for i in range(len(input_batch.input_strs))])
        success.extend([str(episode.reward_info) for episode in episodes])

    table = wandb.Table(columns=["Input", "Predicted", "Expected", "Rewards"],
                        data=list(zip(inputs, preds, expected, success)))
    wandb.log({f"Predictions": table})

    return np.mean([episode.reward for episode in episodes])


def main(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    pretrained_model_path = Path(config["model"]["pretrained_model_path"])
    device = torch.device(config["model"]["device"])
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(config["model"]["dtype"], torch.bfloat16)
    torch.set_default_device(device)
    torch.random.manual_seed(config["training"]["random_seed"])
    BATCH_SIZE = config["training"]["batch_size"]
    NUM_QUESTIONS_PER_BATCH = config["training"]["num_questions_per_batch"]
    NUM_ANSWERS_PER_QUESTION = BATCH_SIZE // NUM_QUESTIONS_PER_BATCH

    current_time = datetime.now().strftime(r"%Y%m%d-%H%M%S")
    # tb_writer = SummaryWriter(log_dir=f"{config['training']['log_dir']}/{current_time}")
    # TODO: resolve singular logging in multi-gpu training so only one process logs
    wandb_logger = wandb.init(project="selftraining", entity="transformersclub", config={})

    tokenizer = hydra.utils.instantiate(config["tokenizer"])

    task = hydra.utils.instantiate(config["task"])

    # data resolution
    train_dataset = task.get_dataset(Split.train)
    generator = torch.Generator(device=device)
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=train_dataset.collator_function,
        generator=generator,
        batch_size=NUM_QUESTIONS_PER_BATCH,
    )
    local_rank = int(os.environ.get("LOCAL_RANK", 0))  # set by torchrun
    torch.cuda.set_device(local_rank)

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
        responses_str, responses_tokens = generation_strategy.generate(
                model=model,
                tokenizer=tokenizer,
                batch=input_batch,
                num_responses=NUM_ANSWERS_PER_QUESTION,
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

        # compute and log important metrics

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
        if step % config["training"]["eval_interval"] == 0:
            eval_reward = evaluate(model, tokenizer, task, generation_strategy, device, dtype, config, user_gen_kwargs)
            print(f"\rEval mean_reward: {eval_reward:.2f}" + " " * 100)
            wandb_logger.log({"mean_reward/eval": eval_reward})

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

    if not dist.is_initialized():  # make sure only one worker does the group init
        if "RANK" not in os.environ:
            # Local single-process fallback
            os.environ["RANK"] = os.environ.get("RANK", "0")
            os.environ["WORLD_SIZE"] = os.environ.get("WORLD_SIZE", "1")
            os.environ["LOCAL_RANK"] = os.environ.get("LOCAL_RANK", "0")
            os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
            os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")

        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")

    main(args.config)
