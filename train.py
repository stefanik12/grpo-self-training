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
from grpo import rollout, update_policy
from objects import Task
from optimizer import MemoryEfficientAdamW
from qwen2_model import Transformer
from tokenizer import Tokenizer


def evaluate(model, task: Task, device, dtype, config):
    test_dataset = task.get_dataset(Split.test)
    generator = torch.Generator(device=device)
    # We reduce the batch size by half as we want to
    # generate twice as long trajectories.
    dataloader = DataLoader(
        test_dataset,
        shuffle=False,
        collate_fn=task.collator_function(),
        generator=generator,
        batch_size=config["training"]["batch_size"] // 2,
        drop_last=False,
    )
    success = []
    for batch in dataloader:
        episodes = rollout(
            model=model,
            tokenizer=task.tokenizer,
            batch=batch,
            max_gen_len=config["training"]["max_gen_len"] * 2,
            num_answer_per_question=1,
            reward_function=task.reward_function,
            device=device,
            dtype=dtype,
        )
        success.extend([episode.reward_info["answer_reward"] for episode in episodes])
    return np.mean(success)


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
    wandb_logger = wandb.init(project="selftraining", entity="transformersclub", group="distillation", config={})

    tokenizer = Tokenizer(str(pretrained_model_path / "tokenizer.json"))

    task = hydra.utils.instantiate(config["task"])

    train_dataset = task.get_dataset(Split.train)
    generator = torch.Generator(device=device)
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=task.collator_function(),
        generator=generator,
        batch_size=NUM_QUESTIONS_PER_BATCH,
    )
    local_rank = int(os.environ.get("LOCAL_RANK", 0))  # set by torchrun
    torch.cuda.set_device(local_rank)

    model = Transformer.from_pretrained(pretrained_model_path, device=device).train()
    model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank
    )

    optimizer = MemoryEfficientAdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        betas=config["training"]["betas"],
        enabled=config["training"]["memory_efficient_adamw"],
    )

    start_time = time.time()
    ckpt_dir = Path(config["training"]["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for step, batch in enumerate(train_dataloader, start=1):
        episodes = rollout(  # TODO: rollout will be probably parametrized by a pipeline
            model=model,     #  but we need to check that pipe
            tokenizer=tokenizer,
            batch=batch,
            max_gen_len=config["training"]["max_gen_len"],
            num_answer_per_question=NUM_ANSWERS_PER_QUESTION,
            reward_function=task.reward_function,
            device=device,
            dtype=dtype,
        )
        if config["training"]["skip_unfinished_episodes"]:
            episodes = [episode for episode in episodes if episode.is_finished]
        results = update_policy(
            model=model,
            optimizer=optimizer,
            episodes=episodes,
            micro_batch_size=config["training"]["micro_batch_size"],
            pad_token_id=tokenizer.pad_token_id,
            max_grad_norm=config["training"]["max_grad_norm"],
            device=device,
            dtype=dtype,
        )
        torch.cuda.synchronize()
        end_time = time.time()
        duration = end_time - start_time
        start_time = end_time

        # compute and log important metrics
        reward = [episode.reward for episode in episodes]
        formatted_reward = [
            episode.reward_info["format_reward"] for episode in episodes
        ]
        answer_reward = [episode.reward_info["answer_reward"] for episode in episodes]
        num_finished_episodes = sum(episode.is_finished for episode in episodes)
        mean_reward = np.mean(reward)
        std_reward = np.std(reward)
        success_rate = np.mean(answer_reward)
        format_reward = np.mean(formatted_reward)
        grad_norm = results["grad_norm"]
        entropy = results["entropy"]
        lr = optimizer.param_groups[0]["lr"]
        loss = results["loss"]
        mean_response_len = np.mean(
            [len(episode.generated_token_ids) for episode in episodes]
        )
        print(
            f"\rStep {step}, mean_reward: {mean_reward:.2f}, "
            f"train success_rate: {success_rate:.2f}, "
            f"grad_norm: {grad_norm:.2f}, duration: {duration:.2f}, "
            f"num_finished_episodes: {num_finished_episodes}, "
            f"mean_response_len: {mean_response_len:.2f}, "
            f"entropy: {entropy:.2f}"
        )
        if step % config["training"]["eval_interval"] == 0:
            eval_success_rate = evaluate(model, task, device, dtype, config)
            print(f"\rEval success rate: {eval_success_rate:.2f}" + " " * 100)
            wandb_logger.log({"success_rate/eval": eval_success_rate})

        wandb_logger.log({"loss": loss})
        wandb_logger.log({"mean_reward": mean_reward})
        wandb_logger.log({"std_reward": std_reward})
        wandb_logger.log({"success_rate/train": success_rate})
        wandb_logger.log({"format_reward": format_reward})
        wandb_logger.log({"grad_norm": grad_norm})
        wandb_logger.log({"duration": duration})
        wandb_logger.log({"num_finished_episodes": num_finished_episodes})
        wandb_logger.log({"learning_rate": lr})
        wandb_logger.log({"mean_response_len": mean_response_len})
        wandb_logger.log({"entropy": entropy})
        # TODO: log output texts
        # for i, episode in enumerate(episodes):
        #     # TensorBoard treats text as markdown.
        #     text = html.escape(episode.text)
        #     wandb_logger.log({f"text_{i}": f"<pre>{text}</pre>"})

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
            os.environ["RANK"] = "0"
            os.environ["WORLD_SIZE"] = "1"
            os.environ["LOCAL_RANK"] = "0"
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = "29500"  # can be any free port

        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")

    main(args.config)

    # TODO future plans:
    # 1. Abstractize task, retype and integrate it into grpo
    # 2. Implement it with current task
    # 3. Implement it with MT

