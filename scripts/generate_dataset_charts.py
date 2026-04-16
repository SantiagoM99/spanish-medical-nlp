"""
generate_dataset_charts.py

Generates two publication-ready charts for the paper:
  1. MeSH Level-1 category distribution (horizontal bar, train/dev/test)
  2. Temporal distribution by publication year (vertical bar)

Usage:
    python scripts/generate_dataset_charts.py
"""

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd

BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MESH_DIR = BASE_DIR / "data" / "pubmed_mesh"
JSON_PATH = BASE_DIR / "data" / "pubmed_mesh_en" / "translated_dataset_with_titles.json"
OUTPUT_DIR = BASE_DIR / "figures"

# ── Style ──
TRAIN_COLOR = "#2C6FAC"
DEV_COLOR = "#69A8DE"
TEST_COLOR = "#B0D4F1"

# MeSH category mapping: column name -> (code, display label)
MESH_CATEGORIES = {
    "category_Anatomy": ("A", "Anatomy"),
    "category_Organisms": ("B", "Organisms"),
    "category_Diseases": ("C", "Diseases"),
    "category_Chemicals_and_Drugs": ("D", "Chemicals & Drugs"),
    "category_Analytical_Diagnostic_and_Therapeutic_Techniques_and_Equipment": ("E", "Techniques"),
    "category_Psychiatry_and_Psychology": ("F", "Psychiatry & Psychology"),
    "category_Phenomena_and_Processes": ("G", "Phenomena & Processes"),
    "category_Disciplines_and_Occupations": ("H", "Disciplines"),
    "category_Anthropology_Education_Sociology_and_Social_Phenomena": ("I", "Anthro/Edu/Sociology"),
    "category_Technology_Industry_Agriculture": ("J", "Technology/Industry"),
    "category_Humanities": ("K", "Humanities"),
    "category_Information_Science": ("L", "Information Science"),
    "category_Named_Groups": ("M", "Named Groups"),
    "category_Health_Care": ("N", "Health Care"),
}


def setup_style():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
    })


def generate_mesh_chart():
    """Horizontal bar chart of MeSH Level-1 categories by split."""
    print("Generating MeSH distribution chart...")

    train_df = pd.read_csv(MESH_DIR / "train_multilabel.csv")
    dev_df = pd.read_csv(MESH_DIR / "val_multilabel.csv")
    test_df = pd.read_csv(MESH_DIR / "test_multilabel.csv")

    rows = []
    for col, (code, label) in MESH_CATEGORIES.items():
        train_c = int(train_df[col].sum()) if col in train_df.columns else 0
        dev_c = int(dev_df[col].sum()) if col in dev_df.columns else 0
        test_c = int(test_df[col].sum()) if col in test_df.columns else 0
        rows.append({
            "code": code,
            "label": f"{code} - {label}",
            "Train": train_c,
            "Dev": dev_c,
            "Test": test_c,
            "total": train_c + dev_c + test_c,
        })

    df = pd.DataFrame(rows).sort_values("total", ascending=True)

    fig, ax = plt.subplots(figsize=(5.5, 3.5), dpi=300)

    y = np.arange(len(df))
    bar_height = 0.25

    ax.barh(y + bar_height, df["Train"], height=bar_height, label="Train", color=TRAIN_COLOR)
    ax.barh(y, df["Dev"], height=bar_height, label="Dev", color=DEV_COLOR)
    ax.barh(y - bar_height, df["Test"], height=bar_height, label="Test", color=TEST_COLOR)

    ax.set_yticks(y)
    ax.set_yticklabels(df["label"])
    ax.set_xlabel("Number of documents")
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    plt.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / "mesh_distribution.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "mesh_distribution.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved to {OUTPUT_DIR / 'mesh_distribution.pdf'}")


def generate_temporal_chart():
    """Vertical bar chart of article count by publication year."""
    print("Generating temporal distribution chart...")

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    years = []
    for art in data["articles"]:
        y = art.get("publication_info", {}).get("year")
        if y is not None:
            years.append(int(y))

    year_counts = pd.Series(years).value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(5.5, 2.5), dpi=300)

    ax.bar(year_counts.index, year_counts.values, color=TRAIN_COLOR, width=0.8)

    ax.set_xlabel("Publication year")
    ax.set_ylabel("Articles")

    # Annotate peak year
    peak_year = year_counts.idxmax()
    peak_count = year_counts.max()
    ax.annotate(
        f"{peak_count:,}",
        xy=(peak_year, peak_count),
        xytext=(0, 6),
        textcoords="offset points",
        ha="center",
        fontsize=8,
        color=TRAIN_COLOR,
    )

    # Rotate x labels if many years
    if len(year_counts) > 20:
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()

    fig.savefig(OUTPUT_DIR / "temporal_distribution.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "temporal_distribution.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved to {OUTPUT_DIR / 'temporal_distribution.pdf'}")


if __name__ == "__main__":
    setup_style()
    generate_mesh_chart()
    generate_temporal_chart()
    print("Done.")
