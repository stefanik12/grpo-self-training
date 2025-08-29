from typing import Union, List, Tuple

import torch
from transformers import PreTrainedTokenizer

from data_types import MiniBatch
from objects import GenerationStrategy
from tokenizer import Tokenizer


class BeamSearch(GenerationStrategy):
    """
    Support for any generation strategy implemented in HF Transformers generate(),
    `extra_generate_kwargs` are passed to generate() when sampling predictions.
    """

    def __init__(self, max_gen_len: int, num_beams: int, batch_size: int = 0):
        super().__init__(max_gen_len)
        self.num_beams = num_beams
        self.batch_size = batch_size

    def generate(self, model: torch.nn.Module,
                 tokenizer: Union[Tokenizer, PreTrainedTokenizer],
                 batch: MiniBatch,
                 num_responses: int,
                 dtype: torch.dtype,
                 extra_generate_kwargs=None) -> Tuple[List[str], List[torch.Tensor]]:
        if extra_generate_kwargs is None:
            extra_generate_kwargs = {}

        bs = self.batch_size if self.batch_size else len(batch.input_strs)
        out_strs_per_input = []
        out_ids_all = []

        for batch_i in range(0, len(batch.input_strs), bs):
            batch_str = batch.input_strs[batch_i: batch_i + bs]
            with torch.autocast(device_type=model.module.device.type, dtype=dtype):
                batch_inputs = tokenizer.tokenizer(batch_str, return_tensors="pt", padding=True)
                out = model.module.generate(**batch_inputs,
                                            max_new_tokens=self.max_gen_len,
                                            num_return_sequences=num_responses,
                                            num_beams=self.num_beams,
                                            **extra_generate_kwargs)
            out_ids = out.reshape(-1, num_responses, out.shape[-1])
            out_ids_all.extend(list(out_ids))
            out_strs_flat = tokenizer.tokenizer.batch_decode(out, skip_special_tokens=True)
            out_strs_per_input.extend([out_strs_flat[i:i + num_responses]
                                       for i in range(0, len(out_strs_flat), num_responses)])

        return out_strs_per_input, out_ids_all


