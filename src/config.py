from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Files
SAMPLE_ES = DATA_DIR / "sample_es.txt"
SAMPLE_EN = DATA_DIR / "sample_en.txt"
SAMPLE_FR = DATA_DIR / "sample_fr.txt"

# PyTorch
BATCH_SIZE = 4
SHUFFLE = True

# Special tokens
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"

# HF model
HF_MODEL_NAME = "bert-base-multilingual-cased"

# spaCy models
SPACY_MODELS = {
    "es": "es_core_news_sm",
    "en": "en_core_web_sm",
    "fr": "fr_core_news_sm",
}