import dataclasses
from collections import defaultdict
from typing import Union, List, Dict

import math
import numpy as np
import torch
from transformers import PreTrainedTokenizer

from data_types import Episode
from objects import Objective
from tokenizer import Tokenizer


class GRPO(Objective):

    @staticmethod
    def normalize_rewards_per_group(episodes: List[Episode]) -> List[Episode]:
        """Normalize rewards per group. A group is defined by the prefix."""
        groups = defaultdict(list)
        for episode in episodes:
            groups[tuple(episode.prefix)].append(episode)
        output = []
        for group in groups.values():
            group_rewards = [item.reward for item in group]
            mean_reward = np.mean(group_rewards)
            std_reward = np.std(group_rewards)
            for episode in group:
                normalized_reward = (episode.reward - mean_reward) / (std_reward + 1e-4)
                episode = dataclasses.replace(episode, reward=normalized_reward)
                output.append(episode)
        return output

    @staticmethod
    def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
        probs = torch.nn.functional.softmax(logits, dim=-1)
        entropy = torch.logsumexp(logits, dim=-1) - torch.sum(probs * logits, dim=-1)
        return entropy

    def update_model(self,
                     model : torch.nn.Module,
                     tokenizer: Union[Tokenizer, PreTrainedTokenizer],
                     optimizer: torch.optim.Optimizer,
                     episodes: List[Episode],
                     dtype: torch.dtype) -> Dict[str, float]:
        """Update the policy using the GRPO algorithm."""
        device = model.device

        episodes = self.normalize_rewards_per_group(episodes)
        # sort episodes by token length for efficient (micro-)batching
        episodes.sort(key=lambda x: len(x.prefix_token_ids) + len(x.generated_token_ids))

        num_target_tokens = sum(len(episode.generated_token_ids) for episode in episodes)
        entropy = 0.0

        for i in range(0, len(episodes), self.micro_batch_size):
            print(f"\r* Computing policy gradient: {i:>2d}/{len(episodes):>2d}", flush=True, end="")

            j = min(i + self.micro_batch_size, len(episodes))
            batch_episodes = episodes[i:j]
            batch_lengths = [
                len(episode.prefix_token_ids) + len(episode.generated_token_ids)
                for episode in batch_episodes
            ]
            batch_max_length = max(batch_lengths)
            batch_token_ids = [
                episode.prefix_token_ids
                + episode.generated_token_ids
                + [tokenizer.pad_token_id] * (batch_max_length - batch_lengths[i])
                for i, episode in enumerate(batch_episodes)
            ]
            batch_masks = [
                [0] * len(episode.prefix_token_ids)
                + [1] * len(episode.generated_token_ids)
                + [0] * (batch_max_length - batch_lengths[i])
                for i, episode in enumerate(batch_episodes)
            ]
            batch_advantages = [episode.reward for episode in batch_episodes]
            batch_token_ids = torch.tensor(batch_token_ids, device=device, dtype=torch.long)
            batch_masks = torch.tensor(batch_masks, device=device, dtype=torch.bool)
            batch_advantages = torch.tensor(batch_advantages, device=device, dtype=torch.float32)

            with torch.autocast(device_type=device.type, dtype=dtype):
                input_token_ids = batch_token_ids[:, :-1]
                target_token_ids = batch_token_ids[:, 1:]
                target_masks = batch_masks[:, 1:]
                logits = model(input_token_ids).float()

            log_probs = -torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    target_token_ids.reshape(-1),
                    ignore_index=tokenizer.pad_token_id,
                    reduction="none",
            ).reshape(input_token_ids.shape[0], -1)

            with torch.no_grad():
                token_entropy = self.compute_entropy(logits)
                entropy = entropy + (token_entropy * target_masks).sum() / num_target_tokens

            obj = log_probs * batch_advantages[:, None]
            # per-token objective
            obj = (obj * target_masks).sum() / num_target_tokens
            loss = -obj
            loss.backward()

        # update the policy
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.max_grad_norm)
        optimizer.step()

        optimizer.zero_grad(set_to_none=True)
        return {
            "loss": loss.item(),
            "grad_norm": grad_norm.item(),
            "entropy": entropy.item(),
        }
