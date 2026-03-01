from abc import ABC, abstractmethod
from typing import List, Dict


class BaseTokenizer(ABC):
    """
    Interface: text <-> ids, tokens, vocab metadata
    """

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """Convert text -> token ids"""
        raise NotImplementedError

    @abstractmethod
    def decode(self, ids: List[int]) -> str:
        """Convert token ids -> text"""
        raise NotImplementedError

    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """Convert text -> tokens (string pieces)"""
        raise NotImplementedError

    @abstractmethod
    def get_vocab_size(self) -> int:
        """Return vocabulary size"""
        raise NotImplementedError

    def get_vocab(self):
        """Optional: return vocab object/dict/list"""
        return None