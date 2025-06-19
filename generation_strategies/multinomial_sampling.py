import gc
from typing import Union, Callable, Dict, Any, List, Tuple

import torch
from transformers import PreTrainedTokenizer

from data_types import MiniBatch, Episode
from objects import GenerationStrategy
from tokenizer import Tokenizer


class MultinomialSamplingCLM(GenerationStrategy):

    @torch.no_grad()
    def generate(self,
                 model: torch.nn.Module,
                 tokenizer: Union[Tokenizer, PreTrainedTokenizer],
                 batch: MiniBatch,
                 num_responses: int,
                 dtype: torch.dtype) -> Tuple[List[str], List[torch.Tensor]]:
        device = model.device

        end_token_id = tokenizer.eos_token_id
        pad_token_id = tokenizer.pad_token_id
        prefix_token_ids = batch.input_token_ids
        bsz = len(batch.input_strs) * num_responses
        min_prompt_len = min(len(t) for t in prefix_token_ids)
        max_prompt_len = max(len(t) for t in prefix_token_ids)
        total_len = self.max_gen_len + max_prompt_len
        model.module.init_kv_cache(
                max_batch_size=bsz,
                max_seq_len=total_len,
                device=device,
                dtype=dtype,
        )
        tokens = torch.full((bsz, total_len), pad_token_id, dtype=torch.long, device=device)
        for k, t in enumerate(prefix_token_ids):
            offset = k * num_responses
            for i in range(num_responses):
                tokens[offset + i, : len(t)] = torch.tensor(t, dtype=torch.long, device=device)

        prev_pos = 0
        input_text_mask = tokens != pad_token_id
        assert min_prompt_len < total_len
        is_finished = torch.zeros((bsz,), dtype=torch.bool, device=device)

        for cur_pos in range(min_prompt_len, total_len):
            print(
                    f"\r* Generating trajectories: {cur_pos - min_prompt_len:>4d}/{total_len - min_prompt_len:>4d}",
                    flush=True,
                    end="",
            )
            with torch.autocast(device_type=device.type, dtype=dtype):
                logits = model.module.inference(tokens[:, prev_pos:cur_pos], prev_pos)
            probs = torch.softmax(logits[:, -1], dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            next_token = next_token.reshape(-1)
            next_token = torch.where(
                    input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            # if a rollout is finished, we fill the rest of the tokens with pad_token_id
            next_token = torch.where(is_finished, pad_token_id, next_token)
            tokens[:, cur_pos] = next_token
            if end_token_id is not None:
                is_end_token = next_token == end_token_id
                is_generated_token = ~input_text_mask[:, cur_pos]
                is_finished = is_finished | (is_end_token & is_generated_token)
            prev_pos = cur_pos
            if is_finished.all():
                break
        model.module.del_kv_cache()
        gc.collect()
        torch.cuda.empty_cache()

        generated_strs = []
        generated_token_ids_all = []

        for i in range(bsz // num_responses):
            generated_strs_batch = []
            generated_token_ids_batch = []
            for j in range(num_responses):
                idx = i * num_responses + j
                generated_token_ids = tokens[idx, len(batch.input_token_ids[i]):]
                # remove padding tokens
                if pad_token_id in generated_token_ids:
                    first_pad_id_idx = (generated_token_ids != pad_token_id).int().argmin(dim=0)
                    generated_token_ids = generated_token_ids[:first_pad_id_idx]

                generated_strs_batch.append(tokenizer.detokenize(generated_token_ids.tolist()))
                generated_token_ids_batch.append(generated_token_ids)
            generated_strs.append(generated_strs_batch)
            generated_token_ids_all.append(generated_token_ids_batch)

        # clear the output line
        print("\r", end=" " * 100, flush=True)

        return generated_strs, generated_token_ids_all
