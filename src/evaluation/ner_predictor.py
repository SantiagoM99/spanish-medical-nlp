"""
ner_predictor.py

Orchestrates NER predictions, computes metrics, and saves results.
"""

import json
from pathlib import Path
from typing import Optional

import pandas as pd

from evaluation.metrics import compute_ner_metrics
from models.base_ner import BaseNERModel
from utils.ner_datareader import NERDataset


class NERPredictor:
    """
    Runs a NER model on a dataset split, computes metrics, and saves results.
    """

    def __init__(self, model: BaseNERModel, dataset: NERDataset):
        self.model = model
        self.dataset = dataset

    def predict_split(
        self, split: str, batch_size: Optional[int] = None
    ) -> tuple[list[list[str]], list[list[str]]]:
        """
        Run predictions on a dataset split.

        Returns:
            (true_labels, predicted_labels) — both as lists of label sequences.
        """
        tokens, true_labels = self.dataset.get_tokens_and_labels(split)

        if batch_size:
            predicted = []
            for i in range(0, len(tokens), batch_size):
                batch = tokens[i : i + batch_size]
                predicted.extend(self.model.predict(batch))
        else:
            predicted = self.model.predict(tokens)

        return true_labels, predicted

    def evaluate(self, split: str = "test", batch_size: Optional[int] = None) -> dict:
        """
        Evaluate the model on a split and return metrics.

        Returns:
            Dict with precision, recall, f1, and full report string.
        """
        true_labels, pred_labels = self.predict_split(split, batch_size)
        metrics = compute_ner_metrics(true_labels, pred_labels)
        return metrics

    def save_predictions(
        self,
        split: str,
        output_path: str,
        batch_size: Optional[int] = None,
    ) -> dict:
        """
        Run predictions, compute metrics, and save results.

        Saves:
          - predictions.jsonl: one JSON line per sentence with tokens/true/pred
          - metrics.json: P/R/F1 summary

        Returns:
            Metrics dict.
        """
        tokens_list, _ = self.dataset.get_tokens_and_labels(split)
        true_labels, pred_labels = self.predict_split(split, batch_size)
        metrics = compute_ner_metrics(true_labels, pred_labels)

        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save per-sentence predictions
        predictions_file = output_dir / "predictions.jsonl"
        with open(predictions_file, "w", encoding="utf-8") as f:
            for tokens, true, pred in zip(tokens_list, true_labels, pred_labels):
                record = {
                    "tokens": tokens,
                    "true_labels": true,
                    "pred_labels": pred,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # Save metrics summary
        metrics_file = output_dir / "metrics.json"
        summary = {
            "model": self.model.model_name,
            "split": split,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
        }
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"Predictions saved to: {output_dir}")
        print(metrics["report"])
        print(f"F1: {metrics['f1']:.4f}")

        return metrics
