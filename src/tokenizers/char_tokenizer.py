from typing import List, Optional
from src.tokenizers.base import BaseTokenizer


class CharTokenizer(BaseTokenizer):
    def __init__(self, chars: Optional[str] = None):
        self.PAD = "<pad>"
        self.UNK = "<unk>"

        if chars is None:
            chars = (
                "abcdefghijklmnopqrstuvwxyz"
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                "0123456789"
                " .,;:!?¡¿()[]{}\"'`-_/\\@#$%^&*+=<>|~"
                "áéíóúÁÉÍÓÚñÑüÜ"
                "àâæçéèêëîïôœùûüÿ"
                "ÀÂÆÇÉÈÊËÎÏÔŒÙÛÜŸ"
            )

        unique_chars = []
        seen = set()
        for c in chars:
            if c not in seen:
                unique_chars.append(c)
                seen.add(c)

        self.vocab = [self.PAD, self.UNK] + unique_chars
        self.char2id = {c: i for i, c in enumerate(self.vocab)}
        self.id2char = {i: c for c, i in self.char2id.items()}

    def tokenize(self, text: str) -> List[str]:
        return list(text)

    def encode(self, text: str) -> List[int]:
        unk_id = self.char2id[self.UNK]
        return [self.char2id.get(c, unk_id) for c in text]

    def decode(self, ids: List[int]) -> str:
        tokens = [self.id2char.get(i, self.UNK) for i in ids]
        tokens = [t for t in tokens if t != self.PAD]
        return "".join(tokens)

    def get_vocab_size(self) -> int:
        return len(self.vocab)

    def get_vocab(self):
        return self.vocab