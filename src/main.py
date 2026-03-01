from config import SAMPLE_ES, SAMPLE_EN, SAMPLE_FR, HF_MODEL_NAME, SPACY_MODELS
from pipelines.pipeline_tokenizer import DataPipeline, TokenizerEvalPipeline

from tokenizers.char_tokenizer import CharTokenizer
from tokenizers.word_tokenizer import WordTokenizer
from tokenizers.hf_wordpiece_tokenizer import HFWordpieceTokenizer
from tokenizers.spacy_tokenizer import SpacyTokenizer


def main():
    paths_by_lang = {
        "es": SAMPLE_ES,
        "en": SAMPLE_EN,
        "fr": SAMPLE_FR,
    }

    temp_pipeline = DataPipeline(
        paths_by_lang=paths_by_lang,
        tokenize_fn=lambda x: x.strip().split(),
    ).load()

    all_texts = temp_pipeline.all_texts
    word_tokenizer = WordTokenizer.from_corpus(all_texts)

    data_pipeline = DataPipeline(
        paths_by_lang=paths_by_lang,
        tokenize_fn=word_tokenizer.tokenize,
    ).load().build_vocab()

    samples_by_lang = data_pipeline.texts_by_lang

    tokenizers = {
        "char": CharTokenizer(),
        "word": word_tokenizer,
    }

    try:
        tokenizers["spacy_es"] = SpacyTokenizer(SPACY_MODELS["es"])
    except Exception as e:
        print(f"[WARN] Could not load spacy_es: {e}")

    try:
        tokenizers["spacy_en"] = SpacyTokenizer(SPACY_MODELS["en"])
    except Exception as e:
        print(f"[WARN] Could not load spacy_en: {e}")

    try:
        tokenizers["spacy_fr"] = SpacyTokenizer(SPACY_MODELS["fr"])
    except Exception as e:
        print(f"[WARN] Could not load spacy_fr: {e}")

    try:
        tokenizers["hf_mbert_wordpiece"] = HFWordpieceTokenizer(HF_MODEL_NAME)
    except Exception as e:
        print(f"[WARN] Could not load HF tokenizer: {e}")

    eval_pipeline = TokenizerEvalPipeline(
        tokenizers=tokenizers,
        samples_by_lang=samples_by_lang
    )

    df_summary = eval_pipeline.summarize()
    print(df_summary.to_string(index=False))

    df_examples = eval_pipeline.examples(max_samples=1)
    print(df_examples[["language", "tokenizer", "n_tokens", "tokens"]].to_string(index=False))


    texts_es = samples_by_lang["es"]
    labels_es = [1 if i % 2 == 0 else 0 for i in range(len(texts_es))]  # dummy labels

    loader_es = data_pipeline.get_loader(lang="es", labels=labels_es)

    print("\n=== PYTORCH DATALOADER DEMO (ES) ===")
    print(f"Vocab size: {len(data_pipeline.stoi)}")
    print("stoi preview:", dict(list(data_pipeline.stoi.items())[:10]))

    for batch_x, batch_y in loader_es:
        print("Batch x shape:", batch_x.shape)
        print(batch_x)
        print("Batch y shape:", batch_y.shape)
        print(batch_y)
        break


if __name__ == "__main__":
    main()