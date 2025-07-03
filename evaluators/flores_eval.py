import abc
from typing import Dict, List
import torch
from tqdm import tqdm
from datasets import load_dataset, Dataset

from tokenizer import Tokenizer

from objects import Evaluator


class FloresEval(Evaluator, abc.ABC):

    def __init__(self, *args, src_langs: List[str], tgt_langs: List[str], dataset_size: int = 100, **kwargs):
        super().__init__(*args, **kwargs)

        self.datasets = {}
        for src_lang, tgt_lang in zip(src_langs, tgt_langs):
            # self.datasets["%s-%s" % (src_lang, tgt_lang)] = load_dataset("Muennighoff/flores200",
            #                                                              "%s-%s" % (src_lang, tgt_lang),
            #                                                              split="dev", trust_remote_code=True)
            dataset = load_dataset(
                    "bri25yu/flores200_devtest_translation_pairs",
                    split="devtest", trust_remote_code=True, streaming=True
            )
            dataset = dataset.filter(lambda row: row["source_lang"] == src_lang and row["target_lang"] == tgt_lang)
            dataset_iter = iter(dataset)
            dataset_mater = [next(dataset_iter) for _ in range(dataset_size)]

            self.datasets["%s-%s" % (src_lang, tgt_lang)] = Dataset.from_list(dataset_mater)


    @abc.abstractmethod
    def evaluate_strs(self, inputs: List[str], expected: List[str], actual: List[str]) -> float:
        pass

    def evaluate(self) -> Dict[str, float]:
        out_dict = {}
        for src_tgt_lang, dataset in self.datasets.items():
            src_lang, tgt_lang = src_tgt_lang.split("-")
            inputs = dataset["sentence_%s" % tgt_lang]
            expected = dataset["sentence_%s" % tgt_lang]
            actual = []
            for batch_i in tqdm(range(0, len(inputs), self.eval_bs, desc="Evaluating lang pair %s" % src_tgt_lang)):
                batch = dataset[batch_i:batch_i + self.eval_bs]
                inputs = self.tokenizer.tokenize(batch)
                outputs = self.model.generate(**inputs)
                output_strs = self.tokenizer.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                actual.extend(output_strs)

            out_dict[src_tgt_lang] = self.evaluate_strs(inputs, expected, actual)


class FloresCometEval(FloresEval):

    def __init__(self, *args, comet_model_id: str = "Unbabel/XCOMET-XL", **kwargs):
        from comet import download_model, load_from_checkpoint
        super().__init__(*args, **kwargs)

        model_path = download_model(comet_model_id)
        self.eval_model = load_from_checkpoint(model_path)

    def evaluate_strs(self, inputs: List[str], expected: List[str], actual: List[str]) -> float:
        data = [{"src": inp, "mt": act, "ref": exp} for inp, exp, act in zip(inputs, expected, actual)]
        eval_outputs = self.eval_model.predict(data, batch_size=self.eval_bs, gpus=1)

        return sum(eval_outputs.scores) / len(eval_outputs.scores)
