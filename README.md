# Tokenization Playground (ES / EN / FR)

A multilingual **NLP tokenization playground** built with **PyTorch**, designed to compare different tokenization strategies across **Spanish**, **English**, and **French**.

This project explores how different tokenizers split text, how that affects sequence length and vocabulary size, and how to build a reusable **data pipeline** (`vocab + padding + DataLoader`) for downstream PyTorch models.

## Features

-  Compare tokenizers across **Spanish / English / French**
- Supports multiple tokenization strategies:
  - Character-level
  - Word-level
  - spaCy (language-specific)
  - Hugging Face WordPiece (mBERT)
-  Generate comparison tables:
  - average tokens per text
  - observed vocabulary size
  - examples of tokenization outputs
-  Build a reusable **PyTorch data pipeline**
  - text loading
  - vocabulary building
  - numericalization
  - padding
  - batching with `DataLoader`
-  Jupyter notebook for visualization and portfolio presentation

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

---

## Project Structure

```bash
tokenization_playgroud/
├── data/
│   ├── sample_es.txt
│   ├── sample_en.txt
│   └── sample_fr.txt
│
├── notebooks/
│   └── tokenization_playground.ipynb
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── compare.py
│   ├── collate.py
│   ├── dataset.py
│   ├── vocab_utils.py           
│   │
│   ├── pipelines/
│   │   ├── __init__.py
│   │   └── pipeline_tokenizer.py
│   │
│   └── tokenizers/
│       ├── __init__.py
│       ├── base.py
│       ├── char_tokenizer.py
│       ├── word_tokenizer.py
│       ├── spacy_tokenizer.py
│       └── hf_wordpiece_tokenizer.py
│
├── requirements.txt
└── README.md