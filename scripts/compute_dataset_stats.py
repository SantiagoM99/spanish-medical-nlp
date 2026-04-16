"""
Compute dataset statistics for the paper's appendix tables:
  1. MeSH Level-1 category distribution (train/dev/test)
  2. AnatEM entity type distribution (train/dev/test)

Outputs LaTeX-ready table rows to stdout.
"""

import os
import glob
from collections import Counter

import pandas as pd

# ──────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MESH_DIR = os.path.join(BASE_DIR, "data", "pubmed_mesh")
NER_DIR = os.path.join(BASE_DIR, "data", "anat_em")

# ──────────────────────────────────────────────────────────────
# MeSH classification stats
# ──────────────────────────────────────────────────────────────

# Column name → (code, display name) for the paper table
MESH_CATEGORIES = {
    "category_Anatomy": ("A", "Anatomy"),
    "category_Organisms": ("B", "Organisms"),
    "category_Diseases": ("C", "Diseases"),
    "category_Chemicals_and_Drugs": ("D", "Chemicals \\& Drugs"),
    "category_Analytical_Diagnostic_and_Therapeutic_Techniques_and_Equipment": ("E", "Techniques"),
    "category_Psychiatry_and_Psychology": ("F", "Psychiatry \\& Psychology"),
    "category_Phenomena_and_Processes": ("G", "Phenomena \\& Processes"),
    "category_Disciplines_and_Occupations": ("H", "Disciplines"),
    "category_Anthropology_Education_Sociology_and_Social_Phenomena": ("I", "Anthro/Edu/Sociology"),
    "category_Technology_Industry_Agriculture": ("J", "Technology/Industry"),
    "category_Humanities": ("K", "Humanities"),
    "category_Information_Science": ("L", "Information Science"),
    "category_Named_Groups": ("M", "Named Groups"),
    "category_Health_Care": ("N", "Health Care"),
}

# Map split file names
MESH_SPLITS = {
    "Train": "train_multilabel.csv",
    "Dev": "val_multilabel.csv",
    "Test": "test_multilabel.csv",
}


def compute_mesh_stats():
    print("=" * 60)
    print("MeSH Level-1 Category Distribution")
    print("=" * 60)

    split_dfs = {}
    for split_name, fname in MESH_SPLITS.items():
        path = os.path.join(MESH_DIR, fname)
        df = pd.read_csv(path)
        split_dfs[split_name] = df
        print(f"  {split_name}: {len(df)} samples")

    # Check if Geography column exists and should be excluded
    geo_col = "category_Geographicals"
    if geo_col in split_dfs["Train"].columns:
        geo_total = sum(df[geo_col].sum() for df in split_dfs.values())
        print(f"  (Geography column found, total={int(geo_total)}, excluded)")

    # Compute counts per category per split
    print()
    print("LaTeX table rows:")
    print("-" * 60)

    for col, (code, display) in MESH_CATEGORIES.items():
        counts = []
        for split_name in ["Train", "Dev", "Test"]:
            df = split_dfs[split_name]
            if col in df.columns:
                counts.append(int(df[col].sum()))
            else:
                counts.append(0)
        train_c, dev_c, test_c = counts
        print(f"{code} & {display:<25s} & {train_c:,} & {dev_c:,} & {test_c:,} \\\\")

    # Summary stats
    print("\\midrule")
    for split_name in ["Train", "Dev", "Test"]:
        df = split_dfs[split_name]
        cat_cols = [c for c in MESH_CATEGORIES.keys() if c in df.columns]
        labels_per_doc = df[cat_cols].sum(axis=1)
        print(f"  {split_name}: avg labels/doc = {labels_per_doc.mean():.2f}, "
              f"min = {labels_per_doc.min()}, max = {labels_per_doc.max()}")


# ──────────────────────────────────────────────────────────────
# AnatEM NER stats
# ──────────────────────────────────────────────────────────────

NER_SPLITS = {
    "Train": "train",
    "Dev": "devel",
    "Test": "test",
}

# Normalize entity type names (handle Multi-tissue vs Multi_tissue)
def normalize_entity_type(etype: str) -> str:
    return etype.replace("Multi_tissue", "Multi-tissue")

# Display order matching the paper table
ENTITY_DISPLAY_ORDER = [
    "Anatomical_system",
    "Cancer",
    "Cell",
    "Cellular_component",
    "Developing_anatomical_structure",
    "Immaterial_anatomical_entity",
    "Multi-tissue_structure",
    "Organ",
    "Organism_subdivision",
    "Organism_substance",
    "Pathological_formation",
    "Tissue",
]

# Shorter names for LaTeX (matching the paper)
ENTITY_SHORT_NAMES = {
    "Anatomical_system": "Anatomical\\_system",
    "Cancer": "Cancer",
    "Cell": "Cell",
    "Cellular_component": "Cellular\\_component",
    "Developing_anatomical_structure": "Developing\\_anat\\_str.",
    "Immaterial_anatomical_entity": "Immaterial\\_anat\\_ent.",
    "Multi-tissue_structure": "Multi-tissue\\_str.",
    "Organ": "Organ",
    "Organism_subdivision": "Organism\\_subdiv.",
    "Organism_substance": "Organism\\_subst.",
    "Pathological_formation": "Pathological\\_form.",
    "Tissue": "Tissue",
}


def count_entities_in_split(split_dir: str) -> tuple[Counter, int]:
    """Count entity mentions (B- tags) and documents in a split directory."""
    entity_counts = Counter()
    files = glob.glob(os.path.join(split_dir, "*.nersuite"))
    for fpath in files:
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) >= 2 and parts[1].startswith("B-"):
                    etype = normalize_entity_type(parts[1][2:])
                    entity_counts[etype] += 1
    return entity_counts, len(files)


def compute_ner_stats():
    print()
    print("=" * 60)
    print("AnatEM Entity Type Distribution")
    print("=" * 60)

    split_counts = {}
    split_docs = {}
    for split_name, dirname in NER_SPLITS.items():
        split_dir = os.path.join(NER_DIR, dirname)
        counts, n_docs = count_entities_in_split(split_dir)
        split_counts[split_name] = counts
        split_docs[split_name] = n_docs
        total = sum(counts.values())
        print(f"  {split_name}: {n_docs} docs, {total:,} entity mentions")

    print()
    print("LaTeX table rows:")
    print("-" * 60)

    totals = {"Train": 0, "Dev": 0, "Test": 0}
    for etype in ENTITY_DISPLAY_ORDER:
        short = ENTITY_SHORT_NAMES[etype]
        counts = []
        for split_name in ["Train", "Dev", "Test"]:
            c = split_counts[split_name].get(etype, 0)
            counts.append(c)
            totals[split_name] += c
        train_c, dev_c, test_c = counts
        print(f"{short:<25s} & {train_c:,} & {dev_c:,} & {test_c:,} \\\\")

    print("\\midrule")
    total_label = "\\textbf{Total}"
    print(f"{total_label:<25s} & {totals['Train']:,} & {totals['Dev']:,} & {totals['Test']:,} \\\\")

    # Check for entity types not in the display list
    all_types = set()
    for counts in split_counts.values():
        all_types.update(counts.keys())
    unexpected = all_types - set(ENTITY_DISPLAY_ORDER)
    if unexpected:
        print(f"\n  WARNING: Unexpected entity types found: {unexpected}")


if __name__ == "__main__":
    compute_mesh_stats()
    compute_ner_stats()
