"""
multilabel_predictor.py

Orchestrates multi-label classification predictions, metrics, and result saving.
"""

import json
from pathlib import Path
from typing import Optional

import pandas as pd

from evaluation.metrics import compute_multilabel_metrics
from models.base_multilabel import BaseMultiLabelModel
from utils.multilabel_datareader import MultiLabelDataset


class MultiLabelPredictor:
    """
    Runs a multi-label model on a dataset split, computes metrics, and saves results.
    """

    def __init__(self, model: BaseMultiLabelModel, dataset: MultiLabelDataset):
        self.model = model
        self.dataset = dataset

    def predict_split(
        self, split: str, batch_size: Optional[int] = None
    ) -> tuple[list[list[str]], list[list[str]]]:
        """
        Run predictions on a dataset split.

        Returns:
            (true_labels, predicted_labels)
        """
        texts, true_labels = self.dataset.get_texts_and_labels(split)

        if batch_size:
            predicted = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                predicted.extend(self.model.predict(batch))
        else:
            predicted = self.model.predict(texts)

        return true_labels, predicted

    def evaluate(self, split: str = "test", batch_size: Optional[int] = None) -> dict:
        """
        Evaluate the model on a split and return metrics.

        Returns:
            Dict with micro/macro F1, precision, recall, hamming loss.
        """
        true_labels, pred_labels = self.predict_split(split, batch_size)
        metrics = compute_multilabel_metrics(true_labels, pred_labels, self.dataset.labels)
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
          - predictions.parquet: columns [input_text, true_labels, predicted_labels]
          - metrics.json: summary metrics

        Returns:
            Metrics dict.
        """
        texts, true_labels = self.dataset.get_texts_and_labels(split)
        _, pred_labels = self.predict_split(split, batch_size)
        metrics = compute_multilabel_metrics(true_labels, pred_labels, self.dataset.labels)

        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save predictions as parquet
        results_df = pd.DataFrame({
            "input_text": texts,
            "true_labels": true_labels,
            "predicted_labels": pred_labels,
        })
        results_df.to_parquet(output_dir / "predictions.parquet", index=False)

        # Save metrics summary
        summary = {
            "model": self.model.model_name,
            "split": split,
            "micro_f1": metrics["micro_f1"],
            "micro_precision": metrics["micro_precision"],
            "micro_recall": metrics["micro_recall"],
            "macro_f1": metrics["macro_f1"],
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "weighted_f1": metrics["weighted_f1"],
            "hamming_loss": metrics["hamming_loss"],
        }
        with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"Predictions saved to: {output_dir}")
        print(metrics["report"])
        print(f"Micro F1: {metrics['micro_f1']:.4f}  Macro F1: {metrics['macro_f1']:.4f}")

        return metrics
