from typing import List, Any, Dict
import pandas as pd


def _safe_run(fn, default=None):
    try:
        return fn()
    except Exception as e:
        return e if default is None else default


def collect_examples(tokenizers: Dict[str, Any], samples_by_lang: Dict[str, List[str]], max_samples: int = 2):
    rows = []
    for lang, texts in samples_by_lang.items():
        for text in texts[:max_samples]:
            for name, tokenizer in tokenizers.items():
                try:
                    tokens = tokenizer.tokenize(text)
                    ids = tokenizer.encode(text)
                    rows.append({
                        "language": lang,
                        "tokenizer": name,
                        "text": text,
                        "tokens": tokens,
                        "n_tokens": len(tokens),
                        "ids_preview": ids[:12],
                    })
                except Exception as e:
                    rows.append({
                        "language": lang,
                        "tokenizer": name,
                        "text": text,
                        "tokens": f"ERROR: {e}",
                        "n_tokens": None,
                        "ids_preview": None,
                    })
    return pd.DataFrame(rows)


def summarize_tokenizers(tokenizers: Dict[str, Any], samples_by_lang: Dict[str, List[str]]):
    rows = []
    for lang, texts in samples_by_lang.items():
        for name, tok in tokenizers.items():
            token_counts = []
            unique_tokens = set()
            error = None

            for text in texts:
                try:
                    toks = tok.tokenize(text)
                    token_counts.append(len(toks))
                    unique_tokens.update(toks)
                except Exception as e:
                    error = str(e)
                    break

            row = {
                "language": lang,
                "tokenizer": name,
                "num_samples": len(texts),
                "avg_tokens": round(sum(token_counts) / len(token_counts), 2) if token_counts else None,
                "min_tokens": min(token_counts) if token_counts else None,
                "max_tokens": max(token_counts) if token_counts else None,
                "observed_vocab_size": len(unique_tokens) if unique_tokens else None,
                "tokenizer_vocab_size": _safe_run(lambda: tok.get_vocab_size()),
                "error": error,
            }
            rows.append(row)

    return pd.DataFrame(rows)