import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from models.classifier import LanguageClassifier, LanguageClassifierTrainer
from pipelines.language_clasification_pipeline import LanguageClassifierPipeline
from config import DATA_DIR
from text_tokenizer import EdgegramTokenizer, WordTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Train a language classifier (ES/EN/FR)")

    # Data loading options
    parser.add_argument("--csv-path", type=str, default=None, help="Path to CSV dataset (optional)")
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR), help="Directory with sample_es/en/fr.txt")
    parser.add_argument("--csv-text-col", type=str, default="text", help="CSV text column name")
    parser.add_argument("--csv-language-col", type=str, default="language", help="CSV language column name")
    parser.add_argument("--csv-label-col", type=str, default=None, help="CSV numeric label column name (optional)")
    parser.add_argument("--sample-multiplier", type=int, default=10, help="Fallback synthetic sample multiplier")

    # Pipeline / preprocessing
    parser.add_argument("--max-length", type=int, default=100, help="Max sequence length")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train/val split ratio")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader num_workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-stratify", action="store_true", help="Disable stratified split")

    # Model
    parser.add_argument("--embedding-dim", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden layer dimension")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of hidden layers")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout")
    parser.add_argument("--use-flatten", action="store_true", help="Use flatten mode instead of mean pooling")

    # Training
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--early-stopping", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay")

    # Outputs
    parser.add_argument("--artifacts-dir", type=str, default="artifacts", help="Output directory")
    parser.add_argument("--checkpoint-name", type=str, default="best_language_classifier.pth", help="Checkpoint filename")

    return parser.parse_args()


def set_seed(seed: int):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_model(
    model: LanguageClassifier,
    val_loader: torch.utils.data.DataLoader,
    id_to_language: Dict[int, str],
    device: torch.device,
):
    model.eval()
    total = 0
    correct = 0

    class_correct = {i: 0 for i in id_to_language.keys()}
    class_total = {i: 0 for i in id_to_language.keys()}

    with torch.no_grad():
        for input_ids, labels in val_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            logits = model(input_ids)
            preds = torch.argmax(logits, dim=-1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

            for y_true, y_pred in zip(labels.cpu().tolist(), preds.cpu().tolist()):
                class_total[y_true] += 1
                if y_true == y_pred:
                    class_correct[y_true] += 1

    overall_acc = correct / total if total > 0 else 0.0

    per_class_acc = {}
    for class_id, lang in id_to_language.items():
        denom = class_total[class_id]
        per_class_acc[lang] = (class_correct[class_id] / denom) if denom > 0 else None

    return {
        "overall_accuracy": overall_acc,
        "per_class_accuracy": per_class_acc,
        "total_samples": total,
    }


def save_training_summary(
    output_path: Path,
    args,
    pipeline_info: Dict,
    trainer: LanguageClassifierTrainer,
    eval_metrics: Dict,
):
    summary = {
        "args": vars(args),
        "pipeline_info": pipeline_info,
        "training_history": {
            "train_losses": trainer.train_losses,
            "val_losses": trainer.val_losses,
            "train_accuracies": trainer.train_accuracies,
            "val_accuracies": trainer.val_accuracies,
        },
        "evaluation": eval_metrics,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f" Saved training summary -> {output_path}")


def main():
    args = parse_args()
    set_seed(args.seed)

    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = artifacts_dir / args.checkpoint_name

    tokenizer = EdgegramTokenizer(n=3, use_prefix=True, use_suffix=True)

    pipeline = LanguageClassifierPipeline(
        tokenizer=tokenizer,
        max_length=args.max_length,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        num_workers=args.num_workers,
        shuffle=True,
        seed=args.seed,
        stratify_split=not args.no_stratify,
    )

    train_loader, val_loader = pipeline.load_auto(
        csv_path=args.csv_path,
        data_dir=args.data_dir,
        csv_text_col=args.csv_text_col,
        csv_language_col=args.csv_language_col,
        csv_label_col=args.csv_label_col,
        sample_multiplier=args.sample_multiplier,
    )

    pipeline_info = pipeline.get_info()
    print("\n=== Pipeline Info ===")
    for k, v in pipeline_info.items():
        print(f"{k}: {v}")

    vocab_size = pipeline_info.get("vocab_size")
    if vocab_size is None:
        raise ValueError(
            "Could not determine vocab_size from tokenizer. "
            "Make sure tokenizer implements get_vocab_size() or exposes .vocab"
        )

    num_classes = len(pipeline.language_map)

 
    model = LanguageClassifier(
        vocab_size=vocab_size,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_classes=num_classes,
        num_layers=args.num_layers,
        dropout=args.dropout,
        padding_idx=0,
        use_mean_pooling=not args.use_flatten,
        max_sequence_length=args.max_length,
    )

    trainer = LanguageClassifierTrainer(
        model=model,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        checkpoint_path=str(checkpoint_path),
    )

    
    print("\n=== Training ===")
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        early_stopping=args.early_stopping,
    )

    print("\n=== Final Evaluation (Validation Set) ===")
    eval_metrics = evaluate_model(
        model=trainer.model,
        val_loader=val_loader,
        id_to_language=pipeline.id_to_language,
        device=trainer.device,
    )

    print(f"Overall Accuracy: {eval_metrics['overall_accuracy']:.4f}")
    print("Per-class Accuracy:")
    for lang, acc in eval_metrics["per_class_accuracy"].items():
        if acc is None:
            print(f"  {lang}: N/A")
        else:
            print(f"  {lang}: {acc:.4f}")

    summary_path = artifacts_dir / "training_summary.json"
    save_training_summary(
        output_path=summary_path,
        args=args,
        pipeline_info=pipeline_info,
        trainer=trainer,
        eval_metrics=eval_metrics,
    )

    print("\n Training complete.")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Summary:    {summary_path}")


if __name__ == "__main__":
    main()