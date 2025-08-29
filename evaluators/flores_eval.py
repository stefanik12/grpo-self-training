from typing import List

import torch

from objects import Evaluator


class Comet(Evaluator):

    def __init__(self, *args, comet_model_id: str, device: str, **kwargs):
        from comet import download_model, load_from_checkpoint
        super().__init__(*args, **kwargs)

        model_path = download_model(comet_model_id)
        self.eval_model = load_from_checkpoint(model_path)
        self.eval_model.to(device)
        self.eval_model.eval()

    def evaluate_batch(self, inputs_str: List[str], expected: List[str], actual: List[str]) -> List[float]:
        data = [{"src": inp, "mt": act, "ref": exp} for inp, exp, act in zip(inputs_str, expected, actual)]
        prepared = self.eval_model.prepare_sample(data, stage="predict")
        inputs, *_ = prepared if isinstance(prepared, tuple) else (prepared,)
        inputs_on_device = {k: v.to(self.eval_model.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
        with torch.no_grad():
            output = self.eval_model(**inputs_on_device)

        return output.score.tolist()


class Bleurt(Evaluator):

    def __init__(self, *args, model_id: str, device: str, **kwargs):
        from bleurt_pytorch import BleurtForSequenceClassification, BleurtTokenizer
        super().__init__(*args, **kwargs)

        self.tokenizer = BleurtTokenizer.from_pretrained(model_id)
        self.eval_model = BleurtForSequenceClassification.from_pretrained(model_id)

        self.eval_model.to(device)
        self.eval_model.eval()

    def evaluate_batch(self, inputs_str: List[str], expected: List[str], actual: List[str]) -> List[float]:
        inputs = self.tokenizer(expected, actual, padding="longest", return_tensors="pt")
        with torch.no_grad():
            scores = scores = self.eval_model(**inputs).logits.flatten().tolist()

        return scores
