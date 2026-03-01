from typing import List, Tuple, Dict, Callable
import torch

def pad_sequence(
    sequence: List[int],
    max_len: int,
    pad_id: int
) -> List[int]:
    return sequence + [pad_id] * (max_len - len(sequence))

def collate_batch(
    batch: List[Tuple[str, int]],
    tokenize_fn: Callable[[str], List[str]],
    stoi: Dict[str, int],
):
    """
    batch = [(text, label),...]
    return: 
        x: tensor [batch_size, seq_len]
        y: tensor [batch_size]
    """
    pad_id = stoi["<pad>"]
    unk_id = stoi["<unk>"]
    bos_id = stoi["<bos>"]
    eos_id = stoi["<eos>"]

    encoded_batch = []
    labels = []

    for text, label in batch:
        token_ids = [stoi.get(tok, unk_id) for tok in tokenize_fn(text)]
        token_ids = [bos_id] + token_ids + [eos_id]
        encoded_batch.append(token_ids)
        labels.append(label)
    
    max_len = max(len(ids) for ids in encoded_batch)
    padded_batch = [pad_sequence(seq, max_len, pad_id) for seq in encoded_batch]

    x = torch.tensor(padded_batch, dtype=torch.long)
    y = torch.tensor(labels, dtype=torch.long)

    return x, y