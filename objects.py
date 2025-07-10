import abc
from typing import Dict, Any, Optional, Callable, List, Union, Tuple

import torch
from transformers import PreTrainedTokenizer

from data_types import MiniBatch, Split, Episode
from tokenizer import Tokenizer


class CollatedDataset(abc.ABC, torch.utils.data.Dataset):

    def __init__(self):
        # distributed support
        worker_info = torch.utils.data.get_worker_info()
        self.worker_rank = worker_info.id if worker_info is not None else 0
        self.num_workers = worker_info.num_workers if worker_info is not None else 1

    @abc.abstractmethod
    def __len__(self):
        pass

    @staticmethod
    @abc.abstractmethod
    def collator_function(batch: List[Dict[str, Any]]) -> Callable[[List[Dict[str, Any]]], MiniBatch]:
        pass


class Evaluator(abc.ABC):

    evaluator_id: str

    def __init__(self, evaluator_id: str):
        self.evaluator_id = evaluator_id

    @abc.abstractmethod
    def evaluate_batch(self, inputs: List[str], expected: List[str], actual: List[str]) -> List[float]:
        pass


class Task(abc.ABC):

    @abc.abstractmethod
    def get_dataset(self, split: Split) -> CollatedDataset:
        pass

    @property
    def gen_kwargs(self) -> Dict[str, Any]:
        return {}

    @abc.abstractmethod
    def reward_responses(self,
                         input_batch: MiniBatch,
                         generated_strs: List[List[str]],
                         generated_encodings: List[List[torch.Tensor]]) -> List[Episode]:
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
                 dtype: torch.dtype,
                 extra_generate_kwargs: Optional[Dict[str, Any]] = None) -> Tuple[List[str], List[torch.Tensor]]:
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
