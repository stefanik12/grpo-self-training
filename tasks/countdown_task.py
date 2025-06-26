import json
import re
from pathlib import Path
from typing import Any, Optional
from typing import Dict, List

import pandas as pd
import torch
from jinja2 import Environment
from tokenizers import Encoding
from tokenizers import Tokenizer as TokenizerBase

from data_types import MiniBatch, Split, Episode
from objects import Task, CollatedDataset

SYSTEM_MESSAGE = (
    "You are a helpful assistant. You first think about the reasoning process "
    "in your mind and then provide the user with the answer."
)
USER_TEMPLATE = (
    "Using the numbers {numbers}, create an equation that equals {target}. "
    "You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. "
    "Show your work in <think> </think> tags. "
    "And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>."
)
RESPONSE_PROMPT = "Let me solve this step by step.\n<think>"


class Tokenizer:
    """Tokenizer with chat template supported using jinja2 engine"""

    def __init__(self, tokenizer_path: str):
        super().__init__()
        tokenizer_config_path = Path(tokenizer_path) / "tokenizer_config.json"
        self.tokenizer_config = json.load(open(tokenizer_config_path))
        self.tokenizer = TokenizerBase.from_file(str(Path(tokenizer_path) / "tokenizer.json"))

        self.chat_template = Environment().from_string(self.tokenizer_config["chat_template"])

        self.eos_token = self.tokenizer_config["eos_token"]
        self.eos_token_id = self.tokenizer.token_to_id(self.eos_token)
        self.pad_token = self.tokenizer_config["pad_token"]
        self.pad_token_id = self.tokenizer.token_to_id(self.pad_token)

    def encode_chat(self, messages: List[Dict[str, str]]) -> str:
        return self.chat_template.render(messages=messages, add_generation_prompt=True)

    def encode_chat_with_response_prompt(
        self, messages: List[Dict[str, str]], prompt: str
    ) -> str:
        return self.encode_chat(messages) + prompt

    def tokenize(self, text: str) -> Encoding:
        return self.tokenizer.encode(text)

    def detokenize(self, token_ids: List[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)


class CountdownTasksDataset(CollatedDataset):
    """Prepare Countdown Tasks for training"""

    def __init__(
        self,
        tokenizer: Tokenizer,
        data_path: str,
        split: Split,
        test_size: int = 100,
    ):
        super().__init__()
        data = pd.read_parquet(Path(data_path) / "data")
        # use the last `test_size` examples for testing
        self.data = (
            data.iloc[:-test_size] if split == Split.train else data.iloc[-test_size:]
        )
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # this should produce exclusive indices for each worker:
        worker_idx = idx * self.num_workers + self.worker_rank
        item = self.data.iloc[worker_idx].to_dict()
        item.update(self.encode_prefix(item["nums"], item["target"]))
        return item

    def encode_prefix(self, numbers: List[int], target: int):
        """Prefix is the *actual* input to the model."""
        user_message = USER_TEMPLATE.format(numbers=numbers, target=target)
        prefix = self.tokenizer.encode_chat_with_response_prompt(
            [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": user_message},
            ],
            RESPONSE_PROMPT,
        )
        tokens = self.tokenizer.tokenize(prefix)
        return {
            "prefix": prefix,
            "prefix_tokens": tokens.tokens,
            "prefix_token_ids": tokens.ids,
        }

    @staticmethod
    def collator_function(batch: List[Dict[str, Any]]) -> MiniBatch:
        """Collate examples into a batch."""
        numbers = [item["nums"] for item in batch]
        target = [item["target"] for item in batch]
        prefix = [item["prefix"] for item in batch]
        prefix_token_ids = [item["prefix_token_ids"] for item in batch]
        return MiniBatch(
                input_strs=prefix,
                input_token_ids=prefix_token_ids,
                extra_reward_info={"numbers": numbers, "target_number": target},
        )


class CountdownTask(Task):

    think_regex = r"<think>.*?<\/think>"
    answer_regex = r"<answer>(.*?)<\/answer>"
    full_format_regex = r"^<think>.*?<\/think>\n<answer>.*?<\/answer>$"

    def __init__(self, data_path: str, test_size: int, tokenizer: str):
        self.data_path = data_path
        self.test_size = test_size
        self.tokenizer = Tokenizer(tokenizer_path=tokenizer)

    def get_dataset(self, split: Split) -> torch.utils.data.Dataset:
        return CountdownTasksDataset(self.tokenizer, self.data_path, split, test_size=self.test_size)

    def reward_responses(self,
                         in_batch: MiniBatch,
                         generated_strs: List[List[str]],
                         generated_encodings: List[List[torch.Tensor]]) -> List[Episode]:
        """Reward function for Countdown Tasks.

        Total reward = 0.1 * format_reward + answer_reward
        """
        out_episodes = []

        nums, tgt = in_batch.extra_reward_info["numbers"], in_batch.extra_reward_info["target_number"]

        batch_iter_args = zip(in_batch.input_strs, in_batch.input_token_ids, nums, tgt)
        for batch_i, (batch_input_str, batch_input_ids, input_nums, tgt_num) in enumerate(batch_iter_args):

            sample_iter_args = zip(generated_strs[batch_i], generated_encodings[batch_i])
            for generated_str, generated_ids in sample_iter_args:

                format_reward = self._format_reward_function("<think>" + generated_str, self.tokenizer.eos_token)
                answer_reward = self._answer_reward_function(generated_str, input_nums, tgt_num)
                reward = format_reward * 0.1 + answer_reward

                new_episode = Episode(input_str=batch_input_str,
                                      input_ids=batch_input_ids,
                                      generated_str=generated_str,
                                      generated_token_ids=generated_ids.tolist(),
                                      reward=reward,
                                      reward_info={"format_reward": format_reward, "answer_reward": answer_reward})
                out_episodes.append(new_episode)

        return out_episodes

    def _format_reward_function(self, response: str, end_token: Optional[str] = None) -> float:
        """
        Checks if the response follows the format <think>...</think><answer>...</answer>
        """
        # Strip end token if present
        if end_token and response.endswith(end_token):
            response = response[: -len(end_token)]

        think_match = re.search(self.think_regex, response, re.DOTALL)
        answer_match = re.search(self.answer_regex, response, re.DOTALL)
        full_format_match = re.match(self.full_format_regex, response, re.DOTALL)

        if full_format_match:
            return 1.0

        reward = 0.0

        if think_match:
            reward += 0.1

        if answer_match:
            reward += 0.5

        return reward

    def _answer_reward_function(self, response: str, numbers: List[int] = None, target: int = None) -> float:
        """
        Checks if the answer uses all numbers exactly once and evaluates to the target
        """
        answer_match = re.search(self.answer_regex, response, re.DOTALL)
        if not answer_match:
            return 0.0

        answer_content = answer_match.group(1)
        if not answer_content:
            return 0.0

        allowed_chars = r"^[0-9+\-*/() ]+$"
        if not re.match(allowed_chars, answer_content):
            return 0.0

        # Check if the answer uses all numbers exactly once
        used_numbers = [int(n) for n in re.findall(r"\d+", answer_content)]
        if sorted(used_numbers) != sorted(numbers):
            return 0.0

        # Check if the answer evaluates to the target
        try:
            result = eval(answer_content, {"__builtins__": None}, {})
            if abs(float(result) - float(target)) < 1e-5:
                return 1.0
        except:
            pass

        return 0.0
