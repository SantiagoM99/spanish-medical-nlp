"""
visualizations.py

Plots y reportes para evaluación de modelos NER y multilabel.
"""

import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ------------------------------------------------------------------
# Shared
# ------------------------------------------------------------------

def plot_metrics_heatmap(
    labels: np.ndarray,
    preds: np.ndarray,
    label_names: list,
    out_dir: str,
    title: str = "Per-label Metrics",
) -> None:
    """
    Heatmap con Precision / Recall / F1 por etiqueta.
    Guarda metrics_heatmap.png y per_label_metrics.csv.
    """
    _ensure_dir(out_dir)
    rows = []
    for i, name in enumerate(label_names):
        rows.append({
            "Label": name[:30],
            "Precision": precision_score(labels[:, i], preds[:, i], zero_division=0),
            "Recall": recall_score(labels[:, i], preds[:, i], zero_division=0),
            "F1": f1_score(labels[:, i], preds[:, i], zero_division=0),
        })
    df = pd.DataFrame(rows).set_index("Label")

    plt.figure(figsize=(12, max(6, 0.35 * len(df))))
    sns.heatmap(df[["Precision", "Recall", "F1"]], annot=True, fmt=".3f", cmap="RdYlBu_r")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "metrics_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close()

    df.reset_index().to_csv(os.path.join(out_dir, "per_label_metrics.csv"), index=False)
    print(f"Heatmap guardado en: {out_dir}/metrics_heatmap.png")


def plot_label_distribution(
    labels: np.ndarray,
    preds: np.ndarray,
    label_names: list,
    out_dir: str,
    title: str = "Label Distribution",
    top_n: int = 15,
) -> None:
    """
    Barras comparando frecuencia verdadera vs predicha de las top_n etiquetas.
    """
    _ensure_dir(out_dir)
    true_counts = labels.sum(axis=0)
    pred_counts = preds.sum(axis=0)

    n = min(top_n, len(label_names))
    idx = np.argsort(true_counts)[-n:]

    x = np.arange(n)
    w = 0.4
    plt.figure(figsize=(14, 7))
    plt.bar(x - w / 2, true_counts[idx], width=w, label="True")
    plt.bar(x + w / 2, pred_counts[idx], width=w, label="Pred")
    plt.xticks(x, [label_names[i][:20] for i in idx], rotation=40, ha="right")
    plt.ylabel("Count")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "label_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Label distribution guardada en: {out_dir}/label_distribution.png")


def plot_confidence_distribution(
    probs: np.ndarray,
    out_dir: str,
    title: str = "Confidence Distribution",
) -> None:
    """
    Histograma de probabilidades de salida (útil para clasificación multilabel).
    """
    _ensure_dir(out_dir)
    plt.figure(figsize=(10, 5))
    plt.hist(probs.flatten(), bins=50, edgecolor="black", alpha=0.75)
    plt.axvline(0.5, color="red", linestyle="--", label="threshold=0.5")
    plt.title(title)
    plt.xlabel("Confidence")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confidence_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confidence distribution guardada en: {out_dir}/confidence_distribution.png")


def dump_classification_report(
    labels: np.ndarray,
    preds: np.ndarray,
    label_names: list,
    out_dir: str,
) -> None:
    """
    Guarda classification_report de sklearn como JSON.
    """
    _ensure_dir(out_dir)
    report = classification_report(
        labels, preds, target_names=label_names, output_dict=True, zero_division=0
    )
    out_path = os.path.join(out_dir, "classification_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Classification report guardado en: {out_path}")


# ------------------------------------------------------------------
# NER-specific
# ------------------------------------------------------------------

def plot_ner_entity_counts(
    entity_counts: dict,
    out_dir: str,
    title: str = "Entidades por tipo",
) -> None:
    """
    Gráfico de barras con el conteo de entidades por tipo para NER.

    Args:
        entity_counts: Dict {entity_type: count}.
    """
    _ensure_dir(out_dir)
    types = list(entity_counts.keys())
    counts = [entity_counts[t] for t in types]

    plt.figure(figsize=(max(8, len(types) * 0.7), 5))
    plt.bar(types, counts)
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "ner_entity_counts.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"NER entity counts guardado en: {out_dir}/ner_entity_counts.png")


# ------------------------------------------------------------------
# Training curves
# ------------------------------------------------------------------

def plot_training_curves(
    history: dict,
    out_dir: str,
    title: str = "Training Curves",
) -> None:
    """
    Dibuja curvas de loss y métricas a lo largo de los epochs.

    Args:
        history: Dict con keys como 'train_loss', 'eval_loss', 'eval_f1', etc.
                 Cada valor debe ser una lista de floats (uno por epoch).
    """
    _ensure_dir(out_dir)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title)

    # Loss
    ax = axes[0]
    if "train_loss" in history:
        ax.plot(history["train_loss"], label="Train loss")
    if "eval_loss" in history:
        ax.plot(history["eval_loss"], label="Val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss")
    ax.legend()

    # Metrics
    ax = axes[1]
    metric_keys = [k for k in history if k not in ("train_loss", "eval_loss")]
    for key in metric_keys:
        ax.plot(history[key], label=key)
    ax.set_xlabel("Epoch")
    ax.set_title("Metrics")
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training curves guardadas en: {out_dir}/training_curves.png")
