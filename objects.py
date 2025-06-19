import abc
from typing import Dict, Any, Optional, Callable, List, Union, Tuple

import torch
from transformers import PreTrainedTokenizer

from data_types import MiniBatch, Split, Episode
from tokenizer import Tokenizer


class CollatedDataset(abc.ABC, torch.utils.data.Dataset):

    @staticmethod
    @abc.abstractmethod
    def collator_function(batch: List[Dict[str, Any]]) -> Callable[[List[Dict[str, Any]]], MiniBatch]:
        pass


class Task(abc.ABC):

    @abc.abstractmethod
    def get_dataset(self, split: Split) -> CollatedDataset:
        pass

    @abc.abstractmethod
    def reward_responses(self,
                         input_batch: MiniBatch,
                         generated_strs: List[str],
                         generated_encodings: List[torch.Tensor]) -> List[Episode]:
        pass


class GenerationStrategy(abc.ABC):

    def __init__(self, max_gen_len: int):
        self.max_gen_len = max_gen_len

    @abc.abstractmethod
    @torch.no_grad()
    def generate(self,
                 model: torch.nn.Module,
                 tokenizer: Union[Tokenizer, PreTrainedTokenizer],
                 batch: MiniBatch,
                 num_responses: int,
                 dtype: torch.dtype) -> Tuple[List[str], List[torch.Tensor]]:
        pass


class Objective(abc.ABC):

    def __init__(self, micro_batch_size: int, max_grad_norm: float):
        self.micro_batch_size = micro_batch_size
        self.max_grad_norm = max_grad_norm

    @abc.abstractmethod
    def update_model(self,
                     model : torch.nn.Module,
                     tokenizer: Union[Tokenizer, PreTrainedTokenizer],
                     optimizer: torch.optim.Optimizer,
                     episodes: List[Episode],
                     dtype: torch.dtype) -> Dict[str, float]:
        pass
