from typing import List
from transformers import AutoTokenizer
from text_tokenizer.base import BaseTokenizer

class HFWordpieceTokenizer(BaseTokenizer):
    def __init__(self, model_name: str = "bert-base-multilingual-cased"):
        if AutoTokenizer is None:
            raise ImportError("Transformers is not installed. Please install it with `pip install transformers`.")
        
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def tokenize(self, text: str) -> List[str]:
        return self.tokenizer.tokenize(text)
    
    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)
    
    def decode(self, ids: List[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)
    
    def get_vocab_size(self) -> int:
        return self.tokenizer.vocab_size
    
    def get_vocab(self) -> List[str]:
        return self.tokenizer.get_vocab()


        
