import abc
from typing import Dict, Any, Optional, Callable, List

import torch

from data_types import MiniBatch, Split


class Task(abc.ABC):

    def get_dataset(self, split: Split) -> torch.utils.data.Dataset:
        pass

    def collator_function(self) -> Callable[[List[Dict[str, Any]]], MiniBatch]:
        pass

    def reward_function(self, response: str, label: Optional[str] = None) -> Dict[str, Any]:
        pass
