from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from dataclasses import asdict
from typing import Dict, List

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

# Allow running this file directly from the model/ folder.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.dataset import load_csv_data


def _train_val_split(texts: List[str], labels: List[int], val_ratio: float = 0.2, seed: int = 42):
    """Deterministic stratified train/validation split without scikit-learn."""
    import random
    rng = random.Random(seed)
    indices_by_class: Dict[int, List[int]] = {}
    for idx, l in enumerate(labels):
        indices_by_class.setdefault(l, []).append(idx)

    train_idx, val_idx = [], []
    for cls, idxs in indices_by_class.items():
        rng.shuffle(idxs)
        n_val = max(1, int(len(idxs) * val_ratio))
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    return (
        [texts[i] for i in train_idx],
        [texts[i] for i in val_idx],
        [labels[i] for i in train_idx],
        [labels[i] for i in val_idx],
    )


MODEL_NAME = "bert-base-uncased"


class NewsDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
        )
        item = {k: torch.tensor(v) for k, v in enc.items()}
        item["labels"] = torch.tensor(int(self.labels[idx]))
        return item


def _make_compute_metrics(debug: bool = False):
    """Return a compute_metrics callback, using PyTorch for all operations."""

    def computing_matrices(eval_pred):
        """
        Evaluation-time metrics callback computed with pure PyTorch tensors:
        - Accuracy, Precision, Recall, F1
        """
        logits, labels = eval_pred
        logits_t = torch.tensor(logits)
        labels_t = torch.tensor(labels)
        preds_t = torch.argmax(logits_t, dim=-1)

        if debug:
            print("===== compute matrices (debug) =====")
            print("logits shape:", logits_t.shape)
            print("labels preview (first 5):", labels_t[:5].tolist())
            print("preds preview (first 5):", preds_t[:5].tolist())
        else:
            print("===== computing matrices =====")

        # Confusion matrix: binary 0 and 1
        tp = int(((preds_t == 1) & (labels_t == 1)).sum().item())
        fp = int(((preds_t == 1) & (labels_t == 0)).sum().item())
        fn = int(((preds_t == 0) & (labels_t == 1)).sum().item())
        tn = int(((preds_t == 0) & (labels_t == 0)).sum().item())

        print(f"confusion_matrix (labels x preds):\n[[{tn}  {fp}]\n [{fn}  {tp}]]")
        print("==============================")

        total = len(labels_t)
        accuracy = (tp + tn) / total if total > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }

    return computing_matrices


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune BERT for fake news detection (English).")
    p.add_argument("--csv", nargs="+", required=True, help="One or more CSV files with columns: text,label")
    p.add_argument("--output_dir", required=True, help="Where to save the trained model")
    p.add_argument("--epochs", type=int, default=3, help="2–4 recommended")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--debug", action="store_true", help="Enable verbose debug output during evaluation")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    texts, labels, stats = load_csv_data(args.csv, shuffle=True, seed=args.seed)
    print("Dataset stats:", asdict(stats))

    X_train, X_val, y_train, y_val = _train_val_split(
        texts,
        labels,
        val_ratio=0.2,
        seed=args.seed,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    train_ds = NewsDataset(X_train, y_train, tokenizer, max_length=args.max_length)
    val_ds = NewsDataset(X_val, y_val, tokenizer, max_length=args.max_length)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    os.makedirs(args.output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to=[],
        seed=args.seed,
    )

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_ds,
        "eval_dataset": val_ds,
        "data_collator": collator,
        "compute_metrics": _make_compute_metrics(debug=args.debug),
    }
    trainer_params = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)

    trainer.train()
    metrics = trainer.evaluate()
    print("Validation metrics:", metrics)

    model_save_dir = args.output_dir
    trainer.save_model(model_save_dir)
    tokenizer.save_pretrained(model_save_dir)

    with open(os.path.join(model_save_dir, "training_summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": MODEL_NAME,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "max_length": args.max_length,
                "seed": args.seed,
                "dataset_stats": asdict(stats),
                "val_metrics": metrics,
            },
            f,
            indent=2,
        )

    print(f"Saved model to: {model_save_dir}")


if __name__ == "__main__":
    main()

