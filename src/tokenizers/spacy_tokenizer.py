import spacy
from typing import List
from src.tokenizers.base import BaseTokenizer


class SpacyTokenizer(BaseTokenizer):
    def __init__(self, model_name: str):
        if spacy is None:
            raise ImportError("spaCy is not installed. Please install it with `pip install spacy`.")

        self.model_name = model_name
        self.nlp = spacy.load(model_name)

        self.PAD = "<pad>"
        self.UNK = "<unk>"
        self.vocab = [self.PAD, self.UNK]
        self.token2id = {self.PAD: 0, self.UNK: 1}
        self.id2token = {0: self.PAD, 1: self.UNK}

    def _ensure_tokens_in_vocab(self, tokens: List[str]) -> None:
        for tok in tokens:
            if tok not in self.token2id:
                idx = len(self.vocab)
                self.vocab.append(tok)
                self.token2id[tok] = idx
                self.id2token[idx] = tok

    def tokenize(self, text: str) -> List[str]:
        return [tok.text for tok in self.nlp.tokenizer(text)]

    def encode(self, text: str) -> List[int]:
        tokens = self.tokenize(text)
        self._ensure_tokens_in_vocab(tokens)
        return [self.token2id.get(t, self.token2id[self.UNK]) for t in tokens]

    def decode(self, ids: List[int]) -> str:
        tokens = [self.id2token.get(i, self.UNK) for i in ids]
        tokens = [t for t in tokens if t != self.PAD]
        return " ".join(tokens)

    def get_vocab_size(self) -> int:
        return len(self.vocab)

    def get_vocab(self) -> List[str]:
        return self.vocab