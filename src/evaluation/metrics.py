"""
metrics.py

Evaluation metrics for NER (seqeval) and multi-label classification (sklearn).
"""

from typing import Dict, List

import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    hamming_loss,
)
from seqeval.metrics import (
    classification_report as ner_classification_report,
    f1_score as ner_f1,
    precision_score as ner_precision,
    recall_score as ner_recall,
)


# ---------------------------------------------------------------------------
# NER metrics (seqeval — entity-level, span-based)
# ---------------------------------------------------------------------------

def compute_ner_metrics(
    true_labels: List[List[str]],
    pred_labels: List[List[str]],
    digits: int = 4,
) -> Dict[str, float]:
    """
    Compute entity-level NER metrics using seqeval.

    Args:
        true_labels: List of gold BIO label sequences.
        pred_labels: List of predicted BIO label sequences.
        digits: Decimal places in report.

    Returns:
        Dict with precision, recall, f1 (micro/macro/entity-level).
    """
    # Ensure same length
    assert len(true_labels) == len(pred_labels), (
        f"Length mismatch: {len(true_labels)} true vs {len(pred_labels)} predicted"
    )

    results = {
        "precision": ner_precision(true_labels, pred_labels),
        "recall": ner_recall(true_labels, pred_labels),
        "f1": ner_f1(true_labels, pred_labels),
        "report": ner_classification_report(true_labels, pred_labels, digits=digits),
    }
    return results


def print_ner_report(
    true_labels: List[List[str]],
    pred_labels: List[List[str]],
) -> None:
    """Print entity-level NER classification report."""
    metrics = compute_ner_metrics(true_labels, pred_labels)
    print(metrics["report"])
    print(f"Overall  P: {metrics['precision']:.4f}  R: {metrics['recall']:.4f}  F1: {metrics['f1']:.4f}")


# ---------------------------------------------------------------------------
# Multi-label classification metrics
# ---------------------------------------------------------------------------

def compute_multilabel_metrics(
    true_labels: List[List[str]],
    pred_labels: List[List[str]],
    all_labels: List[str],
    digits: int = 4,
) -> Dict[str, float]:
    """
    Compute multi-label classification metrics.

    Args:
        true_labels: List of true label lists.
        pred_labels: List of predicted label lists.
        all_labels: Ordered list of all possible labels.
        digits: Decimal places in report.

    Returns:
        Dict with micro/macro/weighted F1, precision, recall, hamming loss.
    """
    label2id = {lbl: i for i, lbl in enumerate(all_labels)}
    n_labels = len(all_labels)

    # Convert to binary matrices
    def to_binary(label_lists: List[List[str]]) -> np.ndarray:
        matrix = np.zeros((len(label_lists), n_labels), dtype=int)
        for i, label_list in enumerate(label_lists):
            for lbl in label_list:
                if lbl in label2id:
                    matrix[i, label2id[lbl]] = 1
        return matrix

    y_true = to_binary(true_labels)
    y_pred = to_binary(pred_labels)

    results = {
        # Micro averages
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "micro_precision": precision_score(y_true, y_pred, average="micro", zero_division=0),
        "micro_recall": recall_score(y_true, y_pred, average="micro", zero_division=0),
        # Macro averages
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        # Weighted averages
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        # Hamming loss (lower is better)
        "hamming_loss": hamming_loss(y_true, y_pred),
        # Per-label report
        "report": classification_report(
            y_true, y_pred, target_names=all_labels, digits=digits, zero_division=0
        ),
    }
    return results


def print_multilabel_report(
    true_labels: List[List[str]],
    pred_labels: List[List[str]],
    all_labels: List[str],
) -> None:
    """Print multi-label classification report."""
    metrics = compute_multilabel_metrics(true_labels, pred_labels, all_labels)
    print(metrics["report"])
    print(
        f"Micro   P: {metrics['micro_precision']:.4f}  "
        f"R: {metrics['micro_recall']:.4f}  "
        f"F1: {metrics['micro_f1']:.4f}"
    )
    print(
        f"Macro   P: {metrics['macro_precision']:.4f}  "
        f"R: {metrics['macro_recall']:.4f}  "
        f"F1: {metrics['macro_f1']:.4f}"
    )
    print(f"Hamming Loss: {metrics['hamming_loss']:.4f}")
