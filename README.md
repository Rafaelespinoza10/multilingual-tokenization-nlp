# Tokenization Playground (ES / EN / FR)

A multilingual **NLP tokenization playground** built with **PyTorch**, designed to compare different tokenization strategies across **Spanish**, **English**, and **French**.

This project explores how different tokenizers split text, how that affects sequence length and vocabulary size, and how to build reusable pipelines for:
- **tokenizer comparison**
- **PyTorch data loading / batching**
- **language classification training** (ES / EN / FR)

---

## Features

- ✅ Compare tokenizers across **Spanish / English / French**
- ✅ Supports multiple tokenization strategies:
  - Character-level
  - Word-level
  - Edgegram tokenizer
  - spaCy (language-specific)
  - Hugging Face WordPiece (mBERT)
- ✅ Generate comparison tables:
  - average tokens per text
  - observed vocabulary size
  - examples of tokenization outputs
- ✅ Build reusable **PyTorch data pipelines**
  - text loading
  - vocabulary building
  - numericalization
  - padding
  - batching with `DataLoader`
- ✅ Train a **multilingual language classifier** (ES/EN/FR)
  - configurable tokenizer
  - checkpointing
  - early stopping
  - learning-rate scheduler
  - validation metrics
  - training summary export (`artifacts/training_summary.json`)
- ✅ Unit tests for tokenizers
- ✅ Jupyter notebook for visualization and portfolio presentation

---

## Why this project?

Tokenization is one of the most important steps in NLP and LLM pipelines.  
It directly affects:

- sequence length
- vocabulary size
- model efficiency
- multilingual robustness
- handling of punctuation, accents, and subwords

This project was built as a hands-on way to understand these trade-offs and document them in a reusable, production-style structure.

It also extends the preprocessing pipeline into a simple **language identification task**, showing how tokenizer choices can be integrated into a full training workflow.

---

## What's New

### ✅ Multilingual Language Classifier (ES / EN / FR)

A lightweight neural classifier built with PyTorch:
- Embedding layer
- Mean pooling over token embeddings (padding-aware)
- MLP classifier head
- `predict()` and `predict_proba()` methods

### ✅ Trainer Module

Training workflow includes:
- CrossEntropy loss
- Adam optimizer
- Gradient clipping
- ReduceLROnPlateau scheduler
- Early stopping
- Checkpoint saving/loading
- Training history tracking (loss + accuracy)

### ✅ Language Classification Pipeline

Flexible data ingestion pipeline with:
- CSV loading (`text`, `language`)
- TXT loading (`sample_es.txt`, `sample_en.txt`, `sample_fr.txt`)
- Fallback synthetic sample generation (if no files exist)
- Train/validation split
- Stratified split support (preserves class distribution)
- Tokenizer-based tokenization + padding

### ✅ Training Artifacts

Training run exports:
- `artifacts/best_language_classifier.pth`
- `artifacts/training_summary.json`

---

## Project Structure

```bash
TOKENIZATION_PLAYGROUND/
├── artifacts/
│   └── training_summary.json
│
├── data/
│   ├── sample_es.txt
│   ├── sample_en.txt
│   ├── sample_fr.txt
│   └── multilingual_dataset.csv
│
├── notebooks/
│   └── tokenization_playground.ipynb
│
├── src/
│   ├── __pycache__/
│   ├── models/
│   │   └── classifier.py
│   │
│   ├── pipelines/
│   │   ├── __init__.py
│   │   ├── language_clasification_pipeline.py
│   │   └── pipeline_tokenizer.py
│   │
│   ├── tests/
│   │   ├── __init__.py
│   │   ├── test_char_tokenizer.py
│   │   ├── test_edgegram_tokenizer.py
│   │   ├── test_hf_wordpiece_tokenizer.py
│   │   ├── test_spacy_tokenizer.py
│   │   └── test_word_tokenizer.py
│   │
│   ├── text_tokenizer/
│   │   ├── base.py
│   │   ├── char_tokenizer.py
│   │   ├── edgegram_tokenizer.py
│   │   ├── hf_wordpiece_tokenizer.py
│   │   ├── spacy_tokenizer.py
│   │   └── word_tokenizer.py
│   │
│   ├── collate.py
│   ├── compare.py
│   ├── config.py
│   ├── dataset.py
│   ├── main.py
│   ├── train.py
│   └── vocabs_utils.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

## Installation

### 1) Clone the repository
```bash
git clone https://github.com/Rafaelespinoza10/multilingual-tokenization-nlp.git
cd tokenization_playground
```

### 2)  Create and activate a virtual environment (recommended)

#### Windows (PowerShell):
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
```

#### macOS / Linux:
```bash
python -m venv .venv
source .venv/bin/activate
```

### 3) Install dependencies
```bash
pip install -r requirements.txt
```

### 4) Install spaCy language models (optional but recommended)
```bash
python -m spacy download es_core_news_sm
python -m spacy download en_core_web_sm
python -m spacy download fr_core_news_sm
```

**Note**: If spaCy models are not installed, the tokenizer comparison can still run with character, word, edgegram, and Hugging Face tokenizers (depending on environment setup).

---

## How to Run
### Tokenizer comparison (script)
```bash
python src/main.py
```
### Language classifier training (CSV)
```bash
python src/train.py --csv-path data/multilingual_dataset.csv
```
### Language classifier training (TXT fallback)
```bash
python src/train.py --data-dir data
```
### Automatic fallback (uses synthetic sample dataset if no CSV/TXT is found)
```bash
python src/train.py
```

### Command line arguments for train.py
```bash
python src/train.py --help

# Options:
#   --csv-path PATH       Path to CSV file with 'text' and 'language' columns
#   --data-dir PATH       Directory with sample_es.txt, sample_en.txt, sample_fr.txt
#   --tokenizer {char,word,edgegram,spacy,hf}
#   --max-length LEN      Maximum sequence length (default: 50)
#   --batch-size SIZE     Batch size (default: 32)
#   --epochs N            Number of epochs (default: 20)
#   --embedding-dim DIM   Embedding dimension (default: 50)
#   --hidden-dim DIM      Hidden layer dimension (default: 100)
#   --lr RATE             Learning rate (default: 0.001)
#   --seed SEED           Random seed (default: 42)
```
---

## Example: Training Output Summary
A training run stores metrics in:

```text
artifacts/training_summary.json
```

#### Example output:
```json
{
  "run_args": {
    "tokenizer": "edgegram",
    "max_length": 50,
    "batch_size": 32,
    "epochs": 20,
    "embedding_dim": 50,
    "hidden_dim": 100,
    "learning_rate": 0.001,
    "seed": 42
  },
  "pipeline_info": {
    "vocab_size": 156,
    "train_samples": 240,
    "val_samples": 60,
    "language_map": {"es": 0, "en": 1, "fr": 2}
  },
  "training_history": {
    "train_loss": [1.08, 0.76, 0.52, ...],
    "train_accuracy": [0.38, 0.62, 0.79, ...],
    "val_loss": [0.92, 0.58, 0.41, ...],
    "val_accuracy": [0.45, 0.73, 0.85, ...]
  },
  "final_metrics": {
    "val_accuracy": 0.92,
    "val_loss": 0.28,
    "per_class_accuracy": {
      "es": 0.95,
      "en": 0.90,
      "fr": 0.91
    }
  },
  "best_epoch": 15
}
```
---

## Testing

#### Run all tests

```bash
pytest
```
#### Run tests with verbose output
```bash
pytest -v
```

#### Run specific test file
```bash
pytest src/tests/test_edgegram_tokenizer.py::TestEdgegramTokenizer::test_encode_unknown_token -v
```
---

## Current Limitations

- **Language Support**: The language classifier currently targets only 3 languages (ES/EN/FR)
- **Data Quality**: Performance may be unrealistically high on synthetic or very clean datasets
- **Tokenizer Integration**: Some tokenizers (HF/spaCy) may require adapter logic for pipelines expecting dynamic vocabulary building (`add_to_vocab=True/False`)
- **Edgegram Simplifications**: Edgegram tokenizer uses simple whitespace splitting (no advanced punctuation handling)
- **Dataset Size**: Training data is limited to ~300 samples per language (when using synthetic data)
- **Model Architecture**: Simple MLP classifier may not capture complex linguistic patterns
- **No GPU Acceleration**: Tests and examples default to CPU (GPU supported but optional)

---

## Future Improvements

- [ ] **Confusion Matrix**: Add visualization for language classifier predictions
- [ ] **Tokenizer Benchmarking**: Measure and compare tokenization speed (ms/text)
- [ ] **Additional Languages**: Add German (DE), Italian (IT), Portuguese (PT)
- [ ] **Harder Datasets**: Include typos, code-switching, and noisy text examples
- [ ] **CI/CD**: Add GitHub Actions for automated testing
- [ ] **Web Demo**: Create Streamlit UI for interactive tokenizer and language-ID demos
- [ ] **Cross-Validation**: Implement k-fold cross-validation for more robust evaluation
- [ ] **Custom Tokenizers**: Add support via configuration files (YAML/JSON)
- [ ] **Docker**: Create container for easy reproducibility
- [ ] **Model Export**: Support ONNX format for production deployment
- [ ] **Advanced Architectures**: Add CNN and Transformer-based classifiers
- [ ] **Hyperparameter Tuning**: Integrate Optuna or Ray Tune
- [ ] **Logging**: Add Weights & Biases or TensorBoard integration

---

## Learning Outcomes

By building this project, I practiced:

### NLP Fundamentals
- **Tokenization strategies** (character, word, subword, edge-grams)
- **Multilingual text preprocessing** (Spanish, English, French)
- **Vocabulary construction** and numericalization
- **Sequence padding** and batching strategies

### PyTorch Skills
- **Custom Dataset** and DataLoader implementation
- **Neural network design** (Embedding + MLP)
- **Training loops** with gradient clipping
- **Model checkpointing** and early stopping
- **Learning rate scheduling** (ReduceLROnPlateau)

### Software Engineering
- **Modular project organization** with `src/` structure
- **Abstract base classes** for tokenizer interface
- **Pipeline abstraction** for reusable data processing
- **Unit testing** with pytest
- **Error handling** and edge cases
- **Documentation** best practices

### MLOps Concepts
- **Experiment tracking** with JSON summaries
- **Configuration management** (command-line args)
- **Reproducibility** with random seeds
- **Model versioning** with checkpoints

---

## References

### Official Documentation
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [spaCy Documentation](https://spacy.io/usage)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- [pytest Documentation](https://docs.pytest.org/)

### Academic Papers
- [SentencePiece: A simple and language independent subword tokenizer](https://arxiv.org/abs/1808.06226)
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
- [N-gram Language Models](https://web.stanford.edu/~jurafsky/slp3/3.pdf)

### Tutorials & Articles
- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
- [Hugging Face Course](https://huggingface.co/learn/nlp-course)
- [PyTorch NLP Tutorials](https://pytorch.org/tutorials/beginner/nlp_tutorial.html)
- [spaCy 101](https://spacy.io/usage/spacy-101)

### Related Projects
- [WordPiece Tokenization](https://huggingface.co/learn/nlp-course/chapter6/6)
- [Byte-Pair Encoding (BPE)](https://huggingface.co/learn/nlp-course/chapter6/5)
- [Unigram Language Model Tokenization](https://huggingface.co/learn/nlp-course/chapter6/7)

---

## Author

**Rafa**  
Full-stack developer exploring NLP, LLMs, and AI engineering through hands-on projects.

### Connect with me:
- **GitHub**: [@Rafaelespinoza10](https://github.com/Rafaelespinoza10)
- **Project Repository**: [multilingual-tokenization-nlp](https://github.com/Rafaelespinoza10/multilingual-tokenization-nlp)
- **LinkedIn**: [Rafael Espinoza](https://www.linkedin.com/in/alejandro-rafael-moreno-espinoza10)
- **Email**: rafael.moreno.espinoza10@gmail.com

### Acknowledgments
- Santiago Hernández for the excellent ML/DL courses on Udemy
- Andrej Karpathy for educational content on neural networks
- Hugging Face team for democratizing NLP
- PyTorch community for amazing documentation and tutorials

---

**Happy tokenizing! 🚀**
