import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


class LanguageClassifierPipeline:
    """
    Pipeline for language classification:
    - data loading (CSV / TXT / fallback samples)
    - vocabulary building (if tokenizer supports it)
    - tokenization + padding
    - train/validation DataLoaders

    Expected tokenizer behavior:
    - tokenizer.encode(text) -> List[int]
    Optional (for dynamic vocab tokenizers):
    - tokenizer.encode(text, add_to_vocab=True/False)
    - tokenizer.get_vocab_size() or tokenizer.vocab
    """

    def __init__(
        self,
        tokenizer,
        max_length: int = 100,
        batch_size: int = 32,
        train_ratio: float = 0.8,
        num_workers: int = 0,
        shuffle: bool = True,
        seed: int = 42,
        language_map: Optional[Dict[str, int]] = None,
        stratify_split: bool = True,
    ):
        if not (0 < train_ratio < 1):
            raise ValueError("train_ratio must be between 0 and 1 (exclusive).")

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.seed = seed
        self.rng = random.Random(seed)
        self.stratify_split = stratify_split

        self.language_map = language_map or {"es": 0, "en": 1, "fr": 2}
        self.id_to_language = {v: k for k, v in self.language_map.items()}

        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None
        self.vocab_built = False

        self._texts: List[str] = []
        self._labels: List[int] = []


    def _tokenize_and_pad(self, text: str) -> List[int]:
        """
        Tokenize and pad/truncate text to fixed length.
        Supports tokenizers with:
        - encode(text)
        - encode(text, add_to_vocab=False)
        """
        try:
            token_ids = self.tokenizer.encode(text, add_to_vocab=False)
        except TypeError:
            token_ids = self.tokenizer.encode(text)

        if not isinstance(token_ids, list):
            raise TypeError("tokenizer.encode(text) must return List[int]")

        if len(token_ids) > self.max_length:
            token_ids = token_ids[: self.max_length]
        else:
            token_ids = token_ids + [0] * (self.max_length - len(token_ids))

        return token_ids

    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from texts if tokenizer supports dynamic vocab updates.
        If tokenizer doesn't support add_to_vocab, this method safely skips.
        """
        print(f"Building vocabulary from {len(texts)} texts...")

        updated = False
        for text in texts:
            try:
                self.tokenizer.encode(text, add_to_vocab=True)
                updated = True
            except TypeError:
                break

        self.vocab_built = True

        if hasattr(self.tokenizer, "get_vocab_size"):
            try:
                print("Vocabulary size:", self.tokenizer.get_vocab_size())
            except Exception:
                print("Vocabulary size: unavailable")
        elif hasattr(self.tokenizer, "vocab"):
            try:
                print("Vocabulary size:", len(self.tokenizer.vocab))
            except Exception:
                print("Vocabulary size: unavailable")
        else:
            print("Vocabulary built/skipped (tokenizer has no exposed vocab size).")

        if not updated:
            print("[INFO] Tokenizer does not support add_to_vocab=True. Skipping dynamic vocab building.")

    @staticmethod
    def _read_txt_lines(path: Union[str, Path]) -> List[str]:
        path = Path(path)
        if not path.exists():
            return []
        with open(path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    def load_from_txt_files(
        self,
        spanish_file: Optional[Union[str, Path]] = None,
        english_file: Optional[Union[str, Path]] = None,
        french_file: Optional[Union[str, Path]] = None,
        data_dir: Optional[Union[str, Path]] = None,
    ) -> Tuple[List[str], List[int]]:
        """
        Load multilingual dataset from txt files (one sentence per line).
        """
        texts: List[str] = []
        labels: List[int] = []

        if data_dir is not None:
            data_dir = Path(data_dir)
            spanish_file = data_dir / "sample_es.txt"
            english_file = data_dir / "sample_en.txt"
            french_file = data_dir / "sample_fr.txt"

        file_map = {
            "es": spanish_file,
            "en": english_file,
            "fr": french_file,
        }

        for lang, file_path in file_map.items():
            if file_path is None:
                continue
            lines = self._read_txt_lines(file_path)
            if lines:
                texts.extend(lines)
                labels.extend([self.language_map[lang]] * len(lines))
                print(f"Loaded {len(lines)} texts for '{lang}' from {file_path}")

        return texts, labels

    def load_from_csv(
        self,
        csv_path: Union[str, Path],
        text_col: str = "text",
        language_col: str = "language",
        label_col: Optional[str] = None,
    ) -> Tuple[List[str], List[int]]:
        """
        Load dataset from CSV.

        Supported formats:
        1) text + language (language in {'es','en','fr'})
        2) text + label (integer labels), if label_col is provided
        """
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        df = pd.read_csv(csv_path)
        if text_col not in df.columns:
            raise ValueError(f"Missing required text column '{text_col}' in CSV.")

        texts = df[text_col].fillna("").astype(str).tolist()

        if label_col is not None:
            if label_col not in df.columns:
                raise ValueError(f"Missing label column '{label_col}' in CSV.")
            labels = df[label_col].astype(int).tolist()
            print(f"Loaded {len(texts)} rows from CSV using numeric labels.")
            return texts, labels

        if language_col not in df.columns:
            raise ValueError(f"Missing language column '{language_col}' in CSV.")

        labels: List[int] = []
        for lang in df[language_col].fillna("").astype(str).str.lower():
            if lang not in self.language_map:
                raise ValueError(
                    f"Unknown language label '{lang}'. "
                    f"Expected one of: {list(self.language_map.keys())}"
                )
            labels.append(self.language_map[lang])

        print(f"Loaded {len(texts)} rows from CSV with language labels.")
        return texts, labels

    def create_sample_dataset(self, multiplier: int = 10) -> Tuple[List[str], List[int]]:
        """
        Create synthetic multilingual sample dataset for testing/fallback.
        """
        spanish_texts = [
            "Hola, ¿cómo estás?",
            "El gato está en la casa",
            "Me gusta mucho aprender Python",
            "Hoy hace un día soleado",
            "Voy a viajar a México",
            "La comida mexicana es deliciosa",
            "¿Dónde está la biblioteca?",
            "Tengo que estudiar para el examen",
            "Mañana es mi cumpleaños",
            "El café está muy caliente",
        ] * multiplier

        english_texts = [
            "Hello, how are you?",
            "The cat is in the house",
            "I really like learning Python",
            "Today is a sunny day",
            "I'm going to travel to Mexico",
            "Mexican food is delicious",
            "Where is the library?",
            "I have to study for the exam",
            "Tomorrow is my birthday",
            "The coffee is very hot",
        ] * multiplier

        french_texts = [
            "Bonjour, comment allez-vous?",
            "Le chat est dans la maison",
            "J'aime beaucoup apprendre Python",
            "Aujourd'hui est une journée ensoleillée",
            "Je vais voyager au Mexique",
            "La nourriture mexicaine est délicieuse",
            "Où est la bibliothèque?",
            "Je dois étudier pour l'examen",
            "Demain est mon anniversaire",
            "Le café est très chaud",
        ] * multiplier

        texts: List[str] = []
        labels: List[int] = []

        texts.extend(spanish_texts)
        labels.extend([self.language_map["es"]] * len(spanish_texts))

        texts.extend(english_texts)
        labels.extend([self.language_map["en"]] * len(english_texts))

        texts.extend(french_texts)
        labels.extend([self.language_map["fr"]] * len(french_texts))

        print(f"Created sample dataset with {len(texts)} texts (multiplier={multiplier})")
        return texts, labels

    def load_auto(
        self,
        csv_path: Optional[Union[str, Path]] = None,
        data_dir: Optional[Union[str, Path]] = None,
        txt_files: Optional[Dict[str, Union[str, Path]]] = None,
        csv_text_col: str = "text",
        csv_language_col: str = "language",
        csv_label_col: Optional[str] = None,
        sample_multiplier: int = 10,
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Flexible loading strategy:
        1) Try CSV (if provided and exists)
        2) Try TXT files (if provided or data_dir)
        3) Fallback to generated sample data
        """
        texts: List[str] = []
        labels: List[int] = []

        # 1) CSV
        if csv_path is not None and Path(csv_path).exists():
            print("[INFO] Loading dataset from CSV...")
            texts, labels = self.load_from_csv(
                csv_path=csv_path,
                text_col=csv_text_col,
                language_col=csv_language_col,
                label_col=csv_label_col,
            )

        # 2) TXT files
        elif txt_files is not None or data_dir is not None:
            print("[INFO] Loading dataset from TXT files...")
            txt_files = txt_files or {}
            texts, labels = self.load_from_txt_files(
                spanish_file=txt_files.get("es"),
                english_file=txt_files.get("en"),
                french_file=txt_files.get("fr"),
                data_dir=data_dir,
            )

        # 3) Fallback sample
        if not texts:
            print("[WARN] No valid CSV/TXT data found. Falling back to generated sample dataset.")
            texts, labels = self.create_sample_dataset(multiplier=sample_multiplier)

        return self.create_dataloaders(texts, labels)

    def create_dataloaders(
        self,
        texts: List[str],
        labels: List[int],
    ) -> Tuple[DataLoader, DataLoader]:
        """Create train/validation dataloaders from raw texts and labels."""
        if len(texts) == 0:
            raise ValueError("No texts provided.")
        if len(texts) != len(labels):
            raise ValueError("texts and labels must have the same length.")

        self._texts = texts
        self._labels = labels

        if not self.vocab_built:
            self.build_vocab(texts)

        can_stratify = False
        if self.stratify_split and len(texts) > 1:
            unique_labels = set(labels)
            label_counts = {lbl: labels.count(lbl) for lbl in unique_labels}
            can_stratify = len(unique_labels) > 1 and all(count >= 2 for count in label_counts.values())

        try:
            train_texts, val_texts, train_labels, val_labels = train_test_split(
                texts,
                labels,
                train_size=self.train_ratio,
                random_state=self.seed,
                shuffle=self.shuffle,
                stratify=labels if can_stratify else None,
            )
            if can_stratify:
                print("[INFO] Using stratified train/validation split.")
            else:
                print("[INFO] Using random train/validation split (stratification not possible).")
        except ValueError as e:
            print(f"[WARN] Stratified split failed: {e}")
            print("[INFO] Falling back to manual random split.")

            combined_data = list(zip(texts, labels))
            self.rng.shuffle(combined_data)

            texts_shuffled, labels_shuffled = zip(*combined_data)
            texts_shuffled = list(texts_shuffled)
            labels_shuffled = list(labels_shuffled)

            split_index = int(len(texts_shuffled) * self.train_ratio)
            split_index = max(1, min(split_index, len(texts_shuffled) - 1)) if len(texts_shuffled) > 1 else 1

            train_texts = texts_shuffled[:split_index]
            train_labels = labels_shuffled[:split_index]
            val_texts = texts_shuffled[split_index:]
            val_labels = labels_shuffled[split_index:]

        train_dataset = self._create_dataset(train_texts, train_labels)
        val_dataset = self._create_dataset(val_texts, val_labels)

        pin_memory = torch.cuda.is_available()

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=min(self.batch_size, len(train_dataset)) if len(train_dataset) > 0 else 1,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=pin_memory,
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=min(self.batch_size, len(val_dataset)) if len(val_dataset) > 0 else 1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=pin_memory,
        )

        print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
        print(f"Train batches: {len(self.train_loader)}, Val batches: {len(self.val_loader)}")
        self._print_split_distribution(train_labels, val_labels)

        return self.train_loader, self.val_loader

    def _create_dataset(self, texts: List[str], labels: List[int]) -> Dataset:
        """Create a torch Dataset for tokenized/padded text classification."""

        class _LanguageDataset(Dataset):
            def __init__(self, texts, labels, pipeline):
                self.texts = texts
                self.labels = labels
                self.pipeline = pipeline

            def __len__(self):
                return len(self.texts)

            def __getitem__(self, idx):
                text = self.texts[idx]
                label = self.labels[idx]
                token_ids = self.pipeline._tokenize_and_pad(text)

                return (
                    torch.tensor(token_ids, dtype=torch.long),
                    torch.tensor(label, dtype=torch.long),
                )

        return _LanguageDataset(texts, labels, self)


    def _print_split_distribution(self, train_labels: List[int], val_labels: List[int]) -> None:
        def count_labels(lbls: List[int]) -> Dict[str, int]:
            counts: Dict[str, int] = {}
            for label_id in sorted(set(lbls)):
                lang = self.id_to_language.get(label_id, str(label_id))
                counts[lang] = lbls.count(label_id)
            return counts

        train_counts = count_labels(train_labels)
        val_counts = count_labels(val_labels)

        print("[INFO] Train label distribution:", train_counts)
        print("[INFO] Val   label distribution:", val_counts)



    def predict_batch(self, texts: List[str]) -> torch.Tensor:
        """Prepare a batch of texts for prediction."""
        batch_ids = [self._tokenize_and_pad(text) for text in texts]
        return torch.tensor(batch_ids, dtype=torch.long)

    def get_info(self) -> Dict:
        """Get pipeline metadata."""
        vocab_size = None
        if hasattr(self.tokenizer, "get_vocab_size"):
            try:
                vocab_size = self.tokenizer.get_vocab_size()
            except Exception:
                vocab_size = None
        elif hasattr(self.tokenizer, "vocab"):
            try:
                vocab_size = len(self.tokenizer.vocab)
            except Exception:
                vocab_size = None

        return {
            "vocab_size": vocab_size,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "train_ratio": self.train_ratio,
            "shuffle": self.shuffle,
            "stratify_split": self.stratify_split,
            "language_map": self.language_map,
            "train_samples": len(self.train_loader.dataset) if self.train_loader else 0,
            "val_samples": len(self.val_loader.dataset) if self.val_loader else 0,
            "vocab_built": self.vocab_built,
            "seed": self.seed,
        }