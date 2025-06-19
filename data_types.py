from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any


@dataclass
class Episode:
    """Store all relevant information of an episode."""

    input_str: str
    input_ids: List[int]
    generated_str: str
    generated_token_ids: List[int]
    reward: float
    reward_info: Dict[str, float]


@dataclass
class MiniBatch:
    """Batch of data for each training step."""

    input_strs: List[str]
    input_token_ids: List[List[int]]
    extra_reward_info: Optional[Dict[str, Any]] = None


class Split(Enum):
    train = 1
    test = 2
