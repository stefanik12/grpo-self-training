import gc
import itertools
from typing import Union, List, Tuple

import torch
from transformers import PreTrainedTokenizer

from data_types import MiniBatch
from objects import GenerationStrategy
from tokenizer import Tokenizer


class MultinomialSamplingCLM(GenerationStrategy):

    @torch.no_grad()
    def generate(self,
                 model: torch.nn.Module,
                 tokenizer: Union[Tokenizer, PreTrainedTokenizer],
                 batch: MiniBatch,
                 num_responses: int, dtype: torch.dtype,
                 extra_generate_kwargs=None) -> Tuple[List[str], List[torch.Tensor]]:
        device = model.device

        end_token_id = tokenizer.eos_token_id
        pad_token_id = tokenizer.pad_token_id
        prefix_token_ids = batch.input_token_ids
        bsz = len(batch.input_strs) * num_responses
        min_prompt_len = min(len(t) for t in prefix_token_ids)
        max_prompt_len = max(len(t) for t in prefix_token_ids)
        total_len = self.max_gen_len + max_prompt_len

        if hasattr(model.module, "init_kv_cache"):
            # custom model speedup
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
                if hasattr(model.module, "inference"):
                    # custom model support
                    logits = model.module.inference(tokens[:, prev_pos:cur_pos], prev_pos)
                else:
                    logits = model.module(tokens[:, prev_pos:cur_pos], prev_pos)
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
        if hasattr(model.module, "del_kv_cache"):
            # custom model speedup
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


class MultinomialSamplingHF(GenerationStrategy):
    """
    Support for any generation strategy implemented in HF Transformers generate(),
    `extra_generate_kwargs` are passed to generate() when sampling predictions.
    """

    def generate(self, model: torch.nn.Module,
                 tokenizer: Union[Tokenizer, PreTrainedTokenizer],
                 batch: MiniBatch,
                 num_responses: int,
                 dtype: torch.dtype,
                 extra_generate_kwargs=None) -> Tuple[List[str], List[torch.Tensor]]:

        with torch.autocast(device_type=model.module.device.type, dtype=dtype):
            batch = tokenizer.tokenizer(batch.input_strs, return_tensors="pt", padding=True)

            if extra_generate_kwargs is None:
                extra_generate_kwargs = {}
            out = model.module.generate(**batch, max_new_tokens=self.max_gen_len, num_return_sequences=num_responses,
                                        do_sample=True, temperature=1.0, **extra_generate_kwargs)

        out_ids = out.reshape(-1, num_responses, out.shape[-1])
        out_strs_flat = tokenizer.tokenizer.batch_decode(out, skip_special_tokens=True)
        out_strs_per_input = [out_strs_flat[i:i+num_responses] for i in range(0, len(out_strs_flat), num_responses)]

        return out_strs_per_input, out_ids


class MultinomialAndGreedy(GenerationStrategy):

    def generate(self, model: torch.nn.Module,
                 tokenizer: Union[Tokenizer, PreTrainedTokenizer],
                 batch: MiniBatch,
                 num_responses: int,
                 dtype: torch.dtype,
                 extra_generate_kwargs=None) -> Tuple[List[str], List[torch.Tensor]]:

        with torch.autocast(device_type=model.module.device.type, dtype=dtype):
            batch = tokenizer.tokenizer(batch.input_strs, return_tensors="pt", padding=True)

            if extra_generate_kwargs is None:
                extra_generate_kwargs = {}
            out_sampling = model.module.generate(**batch, max_new_tokens=self.max_gen_len, temperature=1.0,
                                                 num_return_sequences=num_responses+1, do_sample=True,
                                                 **extra_generate_kwargs)
            out_greedy = model.module.generate(**batch, max_new_tokens=self.max_gen_len,
                                               num_return_sequences=1, num_beams=1,
                                               **extra_generate_kwargs)
            if out_greedy.shape[-1] <= out_sampling.shape[-1]:
                # copy the last element as a filler to the out_sampling length
                out_greedy = torch.hstack((out_greedy, out_greedy[:, -1:].expand(-1, out_sampling.shape[-1] - out_greedy.shape[-1])))
            else:
                out_greedy = out_greedy[:, :out_sampling.shape[-1]]
            # replace the last item in each batch with greedy-generated outputs
            last_batch_idx = torch.arange(batch.input_ids.shape[0], len(out_sampling), batch.input_ids.shape[0]+1)
            out_sampling[last_batch_idx] = out_greedy
            out = out_sampling

        out_ids = out.reshape(-1, num_responses, out.shape[-1])
        out_strs_flat = tokenizer.tokenizer.batch_decode(out, skip_special_tokens=True)
        out_strs_per_input = [out_strs_flat[i:i+num_responses] for i in range(0, len(out_strs_flat), num_responses)]

        return out_strs_per_input, out_ids

