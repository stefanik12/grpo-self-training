import itertools
from typing import Dict, Callable, List, Any, Optional

import datasets
import hydra
import torch
import torch.nn.functional as F
import transformers
from datasets import load_dataset, Dataset

import numpy as np
from rapidfuzz.distance import Levenshtein
from concurrent.futures import ThreadPoolExecutor

from data_types import MiniBatch, Split, Episode
from objects import Task, CollatedDataset

DATASET_ID = "michal-stefanik/tatoeba_mt_ces-x"
DLEX_WORKERS = 8

def pairwise_edit_distance(
    batch_texts: List[List[str]],
    n_workers: Optional[int] = None,
    dtype: torch.dtype = torch.int32,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Compute batch pairwise Levenshtein distances using rapidfuzz in parallel (threads).
    Returns a torch.Tensor of shape (B, N, N) on CPU or moved to `device` if specified.

    - batch_texts: list of B lists, each with N strings (all inner lists must have equal length).
    - n_workers: number of worker threads (defaults to os.cpu_count()).
    - dtype: integer dtype for returned tensor.
    - device: if given (e.g. torch.device('cuda')), the final tensor will be moved to that device.
    """
    B = len(batch_texts)
    if B == 0:
        return torch.empty((0, 0, 0), dtype=dtype, device=device)

    # validate inner lengths
    N = len(batch_texts[0])
    for idx, lst in enumerate(batch_texts):
        if len(lst) != N:
            raise ValueError(f"All inner lists must have same length. index {idx} length {len(lst)} != {N}")

    # trivial cases
    if N == 0:
        return torch.empty((B, 0, 0), dtype=dtype, device=device)
    if N == 1:
        out = torch.zeros((B, 1, 1), dtype=dtype)
        return out.to(device) if device is not None else out

    # prepare shared numpy array (write from threads)
    arr = np.zeros((B, N, N), dtype=np.int32)

    # build task list (upper-triangle only)
    tasks = []
    for b in range(B):
        for i in range(N):
            for j in range(i + 1, N):
                tasks.append((b, i, j))

    # worker computes a single pair and returns the result
    def _worker(task):
        b, i, j = task
        si = batch_texts[b][i]
        sj = batch_texts[b][j]
        if si == sj:
            d = 0
        else:
            d = int(Levenshtein.distance(si, sj))
        return (b, i, j, d)

    # thread pool; rapidfuzz releases GIL so threads are efficient
    if n_workers is None:
        import os
        n_workers = max(1, (os.cpu_count() or 1))

    # Use map to avoid collecting tons of futures; map yields results as workers finish
    with ThreadPoolExecutor(max_workers=n_workers) as exe:
        for b, i, j, d in exe.map(_worker, tasks):
            arr[b, i, j] = d
            arr[b, j, i] = d

    # diagonal is already zero due to zeros initialization
    out = torch.from_numpy(arr).to(dtype=dtype)

    if device is not None:
        out = out.to(device)

    return out            

class NLLBDataset(CollatedDataset):

    def __init__(
            self,
            tokenizer_id: str,
            split: Split,
            src_lang: str,
            tgt_lang: str,
            test_size: int = 100,
            train_size: int = 1000000,
    ):
        super().__init__()
        langs_ordered = (src_lang, tgt_lang) if src_lang < tgt_lang else (tgt_lang, src_lang)
        if split == Split.train:
            # Train split from Tatoeba
            langs_ordered_tatoeba = tuple(x.split("_")[0] for x in langs_ordered)
            dataset = datasets.load_dataset(DATASET_ID, "%s-%s" % langs_ordered_tatoeba, split=split.name, streaming=True)
            dataset_subset = list(itertools.islice(dataset, test_size if split.name == "test" else train_size))
            dataset = datasets.Dataset.from_list(list(dataset_subset))
        if split == Split.test:
            # Test split from Flores
            dataset = load_dataset(
                    "bri25yu/flores200_devtest_translation_pairs",
                    split="devtest", trust_remote_code=True, streaming=True
            )
            dataset = dataset.filter(lambda row: row["source_lang"] == src_lang and row["target_lang"] == tgt_lang)
            dataset_iter = iter(dataset)
            dataset_materialized = [next(dataset_iter) for _ in range(test_size)]
            dataset = Dataset.from_list(dataset_materialized)
            dataset = dataset.map(lambda row: {"source_text": row["source"], "target_text": row["target"]},
                                  batched=True)

        if src_lang.startswith(dataset["source_lang"][0]) and tgt_lang.startswith(dataset["target_lang"][0]):
            self.src_texts, self.tgt_texts = dataset["source_text"], dataset["target_text"]
        elif src_lang.startswith(dataset["target_lang"][0]) and tgt_lang.startswith(dataset["source_lang"][0]):
            self.src_texts, self.tgt_texts = dataset["target_text"], dataset["source_text"]
        else:
            raise ValueError

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_id)

        assert src_lang in self.tokenizer.vocab
        assert tgt_lang in self.tokenizer.vocab

        self.tokenizer.src_lang = src_lang
        self.tokenizer.tgt_lang = tgt_lang

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
    def collator_function(batch: List[Dict[str, Any]]) -> MiniBatch:
        """Collate examples into a batch."""
        return MiniBatch(
                input_strs=[item["input_text"] for item in batch],
                input_token_ids=[item["inputs_encoded"] for item in batch],
                extra_reward_info={"dataset_target": [item["target_text"] for item in batch]}
        )

    def __len__(self):
        return len(self.src_texts)


class ParaphraseEval:

    def __init__(self, encoder_model_id: str, device: str):
        from sentence_transformers import SentenceTransformer

        self.encoder = SentenceTransformer(encoder_model_id, device=device)

    def compute_similarity(self, texts1: List[str], texts2: List[List[str]]) -> List[List[float]]:
        assert isinstance(texts1, list) and isinstance(texts1[0], str)
        assert isinstance(texts2, list) and isinstance(texts2[0], list) and isinstance(texts2[0][0], str)

        bs = len(texts2[0])

        texts1_embs = self.encoder.encode(texts1, convert_to_tensor=True)
        texts2_embs = self.encoder.encode(list(itertools.chain(*texts2)), convert_to_tensor=True)

        texts1_embs = F.normalize(texts1_embs, p=2, dim=1)
        texts2_embs = F.normalize(texts2_embs, p=2, dim=1)

        cosine_sim_matrix = torch.mm(texts1_embs, texts2_embs.T)

        # TODO: consider just picking the best translation out of the candidates, and zero out others
        #  - maybe it's not a good idea as it would spoil the "all bad" batches
        return [cosine_sim_matrix[i, j:j+bs].tolist() for i, j in enumerate(range(0, cosine_sim_matrix.shape[1], bs))]

    def compute_pairwise_similarity(
        self,
        texts: List[List[str]],
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Batched pairwise similarity

        Symbols:
            B - Batch size
            N - Rollout count (number of generations)

        :param texts: Matrix of size BxN with target translations (N translations
        for each of the B sentences in the batch)
        :return: A tensor of shape (B, N, N) (list of B matrices of size NxN each), where for each of
        the B source sentences, we perform the pairwise comparison between its N translations.
        """
        assert isinstance(texts, list) and isinstance(texts[0], list) and isinstance(texts[0][0], str)

        B = len(texts)
        N = len(texts[0])

        flat_texts = list(itertools.chain.from_iterable(texts))

        embs = self.encoder.encode(flat_texts, convert_to_tensor=True)
        embs = F.normalize(embs, p=2, dim=1)

        if device is not None:
            embs.to(device=device)

        result: list[torch.Tensor] = []

        for i in range(B):
            chunk = embs[i * N : (i+1) * N] # (N)
            cosine_sim_matrix = torch.mm(chunk, chunk.T) # (N, N)
            cosine_sim_matrix = cosine_sim_matrix.fill_diagonal_(0) # eliminate self-comparison values

            result.append(cosine_sim_matrix) # result: (i, N, N)

        return torch.stack(result, dim=0) # return tensor of shape (B, N, N)

class Backtranslation(Task):

    def __init__(self,
                 bt_model_cls: str,
                 bt_model_id: str,
                 target_lang_classifier: str,
                 source_lang_sim_encoder: str,
                 src_lang: str,
                 tgt_lang: str,
                 test_size: int,
                 lm_reward_weight: float,
                 cr_reward_weight: float,
                 device: str,
                 dtype: torch.dtype):
        super().__init__()
        self.dtype = dtype
        self.bt_model_id = bt_model_id
        self.test_size = test_size
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.device = device

        self.bt_model_tokenizer = transformers.AutoTokenizer.from_pretrained(bt_model_id)
        self.bt_model: transformers.PreTrainedModel = hydra.utils.get_class(bt_model_cls).from_pretrained(bt_model_id)
        self.bt_model = self.bt_model.to(device)

        self.target_lang_m = transformers.AutoModelForSequenceClassification.from_pretrained(target_lang_classifier)
        self.target_lang_m = self.target_lang_m.to(device)

        if target_lang_classifier != "Mike0307/multilingual-e5-language-detection":
            raise ValueError("We have hardcoded ordering of labels!")

        self.target_lang_index = 8  # TODO: hardcoded -- clean up model config, parametrize, and remove ValueError
        self.target_lang_m_tokenizer = transformers.AutoTokenizer.from_pretrained(target_lang_classifier)

        self.sim_model = ParaphraseEval(source_lang_sim_encoder, device)

        self.lm_reward_weight = lm_reward_weight
        self.cr_reward_weight = cr_reward_weight

    def get_dataset(self, split: Split) -> torch.utils.data.Dataset:
        return NLLBDataset(self.bt_model_id, split, self.src_lang, self.tgt_lang, self.test_size)

    def collator_function(self) -> Callable[[List[Dict[str, Any]]], MiniBatch]:
        pass

    def _target_lang_reward(self, generated_strs: List[List[str]]) -> List[List[float]]:
        device = self.target_lang_m.device
        generated_strs_flat = list(itertools.chain(*generated_strs))

        # no batching for now -- assuming that inference on ~256x512 tokens can fit a single GPU
        with torch.autocast(device_type=device.type):
            batch_inputs = self.target_lang_m_tokenizer(generated_strs_flat, return_tensors="pt", padding=True,
                                                        truncation=True, max_length=512)
            target_lang_m_probs = torch.nn.functional.softmax(self.target_lang_m(**batch_inputs).logits, dim=-1)
            target_lang_probs = target_lang_m_probs[:, self.target_lang_index]

        return target_lang_probs.reshape(len(generated_strs), -1).tolist()

    def _creativity_reward(self, sources: List[str], generated_strs: List[List[str]]) -> List[List[float]]:
        d_sem = self.sim_model.compute_pairwise_similarity(generated_strs, device=self.device) # semantic distances
        d_lex = pairwise_edit_distance(generated_strs, n_workers=DLEX_WORKERS, device=self.device) # lexical distances

        B = len(generated_strs) # batch size
        N = len(generated_strs[0]) # rollout count

        if N == 1: # safeguard to avoid ZeroDivisionError
            print("WARNING! -- Rollout count reached 1")
            return torch.zeros((B, N), dtype=torch.float32).tolist()

        lexical_dispersion = (1 / (N - 1)) * torch.sum(d_lex, dim=-1, keepdim=False) # higher is better
        semantic_dispersion = (1 / (N - 1)) * torch.sum(d_sem, dim=-1, keepdim=False) # lower is better
        
        # semantic distance to source (identify cluster)
        source_faithfulness = torch.tensor(self.sim_model.compute_similarity(sources, generated_strs), dtype=torch.float32, device=self.device) # lower is better

        # keepdim=True for broadcasting
        semantic_std, semantic_mean = torch.std_mean(semantic_dispersion, dim=-1, keepdim=True)
        lexical_std, lexical_mean = torch.std_mean(lexical_dispersion, dim=-1, keepdim=True)
        source_std, source_mean = torch.std_mean(source_faithfulness, dim=-1, keepdim=True)

        # Standardized (z-score) dispersion values
        semantic_dispersion = (semantic_dispersion - semantic_mean) / semantic_std
        lexical_dispersion = (lexical_dispersion - lexical_mean) / lexical_std
        source_faithfulness = (source_faithfulness - source_mean) / source_std

        result = lexical_dispersion - semantic_dispersion - source_faithfulness

        # could add alpha/beta weights to control each score
        return result.tolist()

    def _src_lang_similarities(self, src_texts: List[str], backtranslated_texts: List[List[str]]) -> List[List[float]]:
        similarities_batched = self.sim_model.compute_similarity(src_texts, backtranslated_texts)
        return similarities_batched

    def _bt_rewards(self, orig_strs: List[str], generated_ids: List[torch.Tensor]) -> List[List[float]]:
        # assert output_ids[0, 0] == target_lang_id  # this is actually guaranteed by force_bos_token_id!
        backtranslated_strs = []
        for output_ids_batch in generated_ids:
            src_lang_id = self.bt_model_tokenizer.vocab[self.src_lang]
            backtranslated_ids_batch = self.bt_model.generate(input_ids=output_ids_batch,
                                                              forced_bos_token_id=src_lang_id)
            backtranslated_strs_batch = self.bt_model_tokenizer.batch_decode(backtranslated_ids_batch,
                                                                             skip_special_tokens=True)
            backtranslated_strs.append(backtranslated_strs_batch)

        sampled_similarities = self._src_lang_similarities(orig_strs, backtranslated_strs)

        # TODO: return normalized similarities as reward? Maybe one-side rescaled, to penalize all-wrongs?
        return sampled_similarities

    @property
    def gen_kwargs(self) -> Dict[str, str]:
        # this assumes that the back-translation model is the same as the trained policy
        # NOTE: MAY BE MODEL-SPECIFIC!
        return {"forced_bos_token_id": self.bt_model_tokenizer.vocab[self.tgt_lang]}

    @torch.no_grad()
    def reward_responses(self,
                         input_batch: MiniBatch,
                         generated_strs: List[List[str]],
                         generated_encodings: List[List[torch.Tensor]]) -> List[Episode]:
        lm_rewards = self._target_lang_reward(generated_strs)
        cr_rewards = self._creativity_reward(input_batch.input_strs, generated_strs)
        bt_rewards = self._bt_rewards(input_batch.input_strs, generated_encodings)

        out_episodes = []

        batch_iter_args = zip(input_batch.input_strs, input_batch.input_token_ids)
        for batch_i, (batch_input_str, batch_input_ids) in enumerate(batch_iter_args):

            sample_iter_args = zip(generated_strs[batch_i], generated_encodings[batch_i],
                                   lm_rewards[batch_i], cr_rewards[batch_i], bt_rewards[batch_i])
            for generated_str, generated_ids, lm_reward, cr_reward, bt_reward in sample_iter_args:
                combined_reward = self.lm_reward_weight * lm_reward + self.cr_reward_weight * cr_reward + (1 - self.lm_reward_weight - self.cr_reward_weight) * bt_reward

                new_episode = Episode(input_str=batch_input_str,
                                      input_ids=batch_input_ids,
                                      generated_str=generated_str,
                                      generated_token_ids=generated_ids.tolist(),
                                      reward=combined_reward,
                                      reward_info={"target_lang_reward": lm_reward, "creativity_reward": cr_reward, "bt_reward": bt_reward})
                out_episodes.append(new_episode)

        return out_episodes
