from typing import List, Optional
from src.tokenizers.base import BaseTokenizer


class WordTokenizer(BaseTokenizer):
    def __init__(self, vocab: Optional[List[str]] = None):
        self.PAD = "<pad>"
        self.UNK = "<unk>"
        self.BOS = "<bos>"
        self.EOS = "<eos>"

        base_vocab = [self.PAD, self.UNK, self.BOS, self.EOS]

        if vocab is None:
            vocab = []

        vocab = list(dict.fromkeys(vocab))

        filtered_vocab = [w for w in vocab if w not in base_vocab]
        self.vocab = base_vocab + filtered_vocab

        self.word2id = {w: i for i, w in enumerate(self.vocab)}
        self.id2word = {i: w for w, i in self.word2id.items()}

    def tokenize(self, text: str) -> List[str]:
        return text.strip().split()

    def encode(self, text: str) -> List[int]:
        words = self.tokenize(text)
        unk_id = self.word2id[self.UNK]
        return [self.word2id.get(w, unk_id) for w in words]

    def decode(self, ids: List[int]) -> str:
        words = [self.id2word.get(i, self.UNK) for i in ids]
        # opcional: omitir pad
        words = [w for w in words if w != self.PAD]
        return " ".join(words)

    def get_vocab_size(self) -> int:
        return len(self.vocab)

    def get_vocab(self):
        return self.vocab

    @classmethod
    def from_corpus(cls, corpus: List[str], lowercase: bool = True):
        words = []
        for text in corpus:
            if lowercase:
                text = text.lower()
            words.extend(text.strip().split())
        return cls(vocab=words)