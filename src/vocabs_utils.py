from collections import Counter
from typing import List, Dict, Callable, Tuple

SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]

def build_vocab_from_corpus(
    corpus: List[str],
    tokenize_fn: Callable[[str], List[str]],
    min_freq: int = 1
) -> Tuple[Dict[str, int], Dict[int, str]]:
    counter = Counter()
    
    for text in corpus:
        counter.update(tokenize_fn(text))
    
    vocab = SPECIAL_TOKENS.copy()
    for token , freq in counter.items():
        if freq >= min_freq and token not in vocab:
            vocab.append(token)
    
    stoi = {tok: i for i, tok in enumerate(vocab)}
    itos = { i: tok for tok, i in stoi.items()}
    return stoi, itos

def numericalize(text:str, tokenize_fn: Callable[[str], List[str]], stoi: Dict[str, int]) -> List[int]:
    unk_id = stoi["<unk>"]
    return [stoi.get(tok, unk_id) for tok in tokenize_fn(text)]