import json
from pathlib import Path
from typing import List

from tokenizers import Encoding
from tokenizers import Tokenizer as TokenizerBase


class Tokenizer:
    """Tokenizer with chat template supported using jinja2 engine"""

    def __init__(self, path_or_id: str, local_path: bool = False):
        if local_path:
            tokenizer_config_path = Path(path_or_id) / "tokenizer_config.json"
            self.tokenizer_config = json.load(open(tokenizer_config_path))
            self.tokenizer = TokenizerBase.from_file(Path(path_or_id) / "tokenizer.json")

            self.eos_token = self.tokenizer_config["eos_token"]
            self.eos_token_id = self.tokenizer.token_to_id(self.eos_token)
            self.pad_token = self.tokenizer_config["pad_token"]
            self.pad_token_id = self.tokenizer.token_to_id(self.pad_token)
        else:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(path_or_id)
            self.eos_token = self.tokenizer.eos_token
            self.eos_token_id = self.tokenizer.eos_token_id
            self.pad_token = self.tokenizer.pad_token
            self.pad_token_id = self.tokenizer.pad_token_id

    def tokenize(self, text: str) -> Encoding:
        return self.tokenizer.encode(text)

    def detokenize(self, token_ids: List[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)
