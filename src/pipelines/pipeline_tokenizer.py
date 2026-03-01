from pathlib import Path
from functools import partial
from typing import Dict, List, Optional, Callable, Tuple

from torch.utils.data import DataLoader


from src.config import BATCH_SIZE, SHUFFLE
from src.dataset import TextDataset
from src.collate import collate_batch
from src.vocabs_utils import build_vocab_from_corpus
from src.compare import summarize_tokenizers, collect_examples

def load_lines(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"File {path} does not exist")
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


class DataPipeline:
    def __init__(
        self,
        paths_by_lang: Dict[str, Path],
        tokenize_fn: Callable[[str], List[str]],
        batch_size: int = BATCH_SIZE,
        shuffle: bool = SHUFFLE,
    ):
        self.paths_by_lang = paths_by_lang
        self.tokenize_fn = tokenize_fn
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.texts_by_lang: Dict[str, List[str]] = {}
        self.all_texts: List[str] = []
        self.stoi: Optional[Dict[str, int]] = None
        self.itos: Optional[Dict[int, str]] = None

    def load(self) -> "DataPipeline":
        self.texts_by_lang = {lang: load_lines(p) for lang, p in self.paths_by_lang.items()}
        self.all_texts = []
        for lang in self.texts_by_lang:
            self.all_texts.extend(self.texts_by_lang[lang])
        return self

    def build_vocab(self, min_freq: int = 1) -> "DataPipeline":
        self.stoi, self.itos = build_vocab_from_corpus(self.all_texts, self.tokenize_fn, min_freq=min_freq)
        return self

    def get_loader(
        self,
        lang: str,
        labels: Optional[List[int]] = None,
    ) -> DataLoader:
        if not self.texts_by_lang or self.stoi is None:
            raise RuntimeError("Run .load().build_vocab() first.")
        texts = self.texts_by_lang[lang]
        if labels is None:
            labels = [0] * len(texts)
        dataset = TextDataset(texts, labels)
        return DataLoader(
            dataset,
            batch_size=min(self.batch_size, len(dataset)),
            shuffle=self.shuffle,
            collate_fn=partial(collate_batch, tokenize_fn=self.tokenize_fn, stoi=self.stoi),
        )


class TokenizerEvalPipeline:
    def __init__(self, tokenizers: Dict, samples_by_lang: Dict[str, List[str]]):
        self.tokenizers = tokenizers
        self.samples_by_lang = samples_by_lang

    def summarize(self):
        return summarize_tokenizers(self.tokenizers, self.samples_by_lang)

    def examples(self, max_samples: int = 2):
        return collect_examples(self.tokenizers, self.samples_by_lang, max_samples=max_samples)