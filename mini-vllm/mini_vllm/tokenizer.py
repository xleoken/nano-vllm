import json
import os
from typing import List

class Tokenizer:
    def __init__(self, model_path: str):
        vocab_path = os.path.join(model_path, "tokenizer.json")
        with open(vocab_path, "r", encoding="utf-8") as f:
            tokenizer_data = json.load(f)
        self.encoder = tokenizer_data["model"]["vocab"]
        self.decoder = {v: k for k, v in self.encoder.items()}
        
        # 确保特殊token存在
        self.unk_token_id = self.encoder.get("<unk>", 0)
        self.eos_token_id = tokenizer_data.get("eos_token_id", 1)
        self.pad_token_id = tokenizer_data.get("pad_token_id", 0)

    def encode(self, text: str) -> List[int]:
        return [self.encoder.get(c, self.unk_token_id) for c in text]

    def decode(self, tokens: List[int]) -> str:
        return ''.join([self.decoder.get(t, '') for t in tokens])    