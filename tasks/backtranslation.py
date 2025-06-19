from typing import Optional, Tuple, Dict, Callable, List, Any

import datasets
import torch
import transformers
from transformers import NllbTokenizer

from data_types import MiniBatch, Split, Episode
from objects import Task


DATASET_ID = "michal-stefanik/tatoeba_mt_ces-x"

FLORES200_LANGS_MAP = {
    "ces": "ces_Latn",
    "eng": "eng_Latn",
}


class NLLBDataset(torch.utils.data.Dataset):

    def __init__(
            self,
            tokenizer: NllbTokenizer,
            split: Split,
            src_lang: str,
            tgt_lang: str,
            test_size: int = 100,
    ):
        langs_ordered = (src_lang, tgt_lang) if src_lang < tgt_lang else (tgt_lang, src_lang)
        dataset = datasets.load_dataset(DATASET_ID, subset="%s-%s" % langs_ordered, split=str(split))
        if split == Split.test:
            dataset = dataset.select(range(test_size))
        self.src_texts, self.tgt_texts = dataset[langs_ordered[0]], dataset[langs_ordered[1]]

        self.tokenizer = tokenizer
        self.tokenizer.src_lang = FLORES200_LANGS_MAP[src_lang]
        self.tokenizer.tgt_lang = FLORES200_LANGS_MAP[tgt_lang]

    def __getitem__(self, idx):
        # this should produce exclusive indices for each worker:
        worker_idx = idx * self.num_workers + self.worker_rank

        input_text = self.src_texts[worker_idx]
        target_text = self.tgt_texts[worker_idx]

        item = {
            "input_text": input_text,
            "inputs_encoded": self.tokenizer.encode(input_text),  # TODO: check that tokenizer.src_lang is applied
            "target_text": target_text,
        }

        return item  # note: currently, the batching only takes place in Objective -- generation is not batched!

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> MiniBatch:
        """Collate examples into a batch."""
        return MiniBatch(
                input_strs=[item["input_text"] for item in batch],
                input_token_ids=[item["inputs_encoded"] for item in batch],
        )


class Backtranslation(Task):

    def __init__(self,
                 bt_model: transformers.PreTrainedModel,
                 target_lang_model: transformers.PreTrainedModel,
                 tokenizer: transformers.PreTrainedTokenizer,
                 lm_reward_weight: float = 0.5):
        super().__init__()
        self.bt_model = bt_model
        self.target_lang_model = target_lang_model
        self.tokenizer = tokenizer
        self.lm_reward_weight = lm_reward_weight

    def get_dataset(self, split: Split) -> torch.utils.data.Dataset:
        pass

    def collator_function(self) -> Callable[[List[Dict[str, Any]]], MiniBatch]:
        pass

    def _lm_reward(self, orig_str: str, generated_strs: List[str]) -> List[float]:
        # TODO: this will likely need to be contextualized = rescaled on the whole batch of outputs
        #  - this is currently not supported!
        raise NotImplementedError("TODO: LM reward will be implemented later.")

    def _src_lang_similarities(self, texts1: List[str], texts2: List[str]) -> List[float]:
        # TODO: what metric to use to not penalize paraphrases? comet maybe? -- initialize earlier/parametrize
        return [1.0]

    def _bt_rewards(self, orig_strs: List[str], output_ids: List[int]) -> List[float]:
        # assert output_ids[0, 0] == target_lang_id  # this is actually guaranteed by force_bos_token_id!

        backtranslated_ids = self.bt_model.generate(input_ids=output_ids)

        backtranslated_strs = self.tokenizer.batch_decode(backtranslated_ids, skip_special_tokens=True)

        sampled_similarities = self._src_lang_similarities(orig_strs, backtranslated_strs)

        # TODO: return normalized similarities as reward? Maybe one-side rescaled, to penalize all-wrongs?
        return sampled_similarities

    def reward_responses(self,
                         input_batch: MiniBatch,
                         generated_strs: List[str],
                         generated_encodings: List[torch.Tensor]) -> List[Episode]:
        # TODO: no point in decoding ids to string response -> allow arbitrary args of reward_responses
        lm_rewards = self._lm_reward(generated_strs)
        bt_rewards = self._bt_rewards(input_batch.input_strs, generated_encodings)

        out_episodes = []

        iter_args = zip(input_batch.input_strs, input_batch.input_token_ids, generated_strs, generated_encodings, lm_rewards, bt_rewards)
        for input_str, input_ids, gen_str, gen_ids, lm_reward, bt_reward in iter_args:
            combined_reward = self.lm_reward_weight * lm_reward + (1 - self.lm_reward_weight) * bt_reward

            new_episode = Episode(input_str=input_str,
                                  input_ids=input_ids,
                                  generated_str=gen_str,
                                  generated_token_ids=gen_ids.tolist(),
                                  reward=combined_reward,
                                  reward_info={"lm_reward": lm_reward, "bt_reward": bt_reward})
            out_episodes.append(new_episode)

        return out_episodes
