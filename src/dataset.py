from typing import List, Tuple
from torch.utils.data import Dataset

class TextDataset(Dataset):
    """
    Simple dataset for text + labels pairs
    """

    def __init__(self, texts: List[str], labels: List[int]):
        if len(texts) != len(labels):
            raise ValueError("texts and labels must have the same length")
        
        self.texts = texts
        self.labels = labels
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Tuple[str, int]:
        return self.texts[idx], self.labels[idx]