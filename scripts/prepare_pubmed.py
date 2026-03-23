"""
prepare_pubmed.py

Converts the PubMed Spanish medical articles dataset into train/dev/test
parquet splits for multi-label classification benchmarking.

Input:
  - Dataset-Creation/dataset/translated_dataset_small.csv
    OR Dataset-Creation/pubmed/data/translated_dataset.json

Output (data/pubmed_mesh/):
  - train.parquet
  - dev.parquet
  - test.parquet
  - label_info.json      -> label names, counts, level mapping
  - dataset_stats.json   -> split sizes, label distribution

Label levels:
  - level1: top-level MeSH categories (A-N, ~26 categories)
  - level2: second-level MeSH (more granular)

Default: uses level1 codes as labels (manageable label space for benchmarking).
"""

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


CSV_PATH = Path("../Dataset-Creation/dataset/translated_dataset_small.csv")
OUTPUT_DIR = Path("../spanish-medical-nlp/data/pubmed_mesh")

# Default split ratios
TRAIN_RATIO = 0.80
DEV_RATIO = 0.10
TEST_RATIO = 0.10

RANDOM_SEED = 42


def load_csv(csv_path: Path) -> pd.DataFrame:
    """Load and basic-validate the CSV dataset."""
    print(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path, encoding="utf-8")
    print(f"  Rows loaded: {len(df):,}")
    print(f"  Columns: {list(df.columns)}")
    return df


def parse_label_column(series: pd.Series) -> list[list[str]]:
    """
    Parse string-represented Python lists in label columns.

    E.g. "['B', 'E']" -> ['B', 'E']
         "[]"          -> []
    """
    result = []
    for val in series:
        try:
            parsed = ast.literal_eval(str(val))
            result.append([str(x) for x in parsed])
        except (ValueError, SyntaxError):
            result.append([])
    return result


def filter_labeled(df: pd.DataFrame, label_col: str = "level1_codes") -> pd.DataFrame:
    """Keep only rows that have at least one label."""
    labels = parse_label_column(df[label_col])
    mask = [len(lbl) > 0 for lbl in labels]
    filtered = df[mask].copy()
    print(f"  Rows with labels: {sum(mask):,} / {len(df):,}")
    return filtered, [lbl for lbl, keep in zip(labels, mask) if keep]


def stratified_split(
    df: pd.DataFrame,
    labels: list[list[str]],
    train_ratio: float,
    dev_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
           list[list[str]], list[list[str]], list[list[str]]]:
    """
    Split into train/dev/test.

    Uses the most frequent label for stratification (approximation for multilabel).
    """
    # Use first label for stratification (most frequent assignment)
    strat_key = [lbl[0] if lbl else "NONE" for lbl in labels]

    indices = np.arange(len(df))
    test_size = 1.0 - train_ratio

    idx_train, idx_temp, labels_train, labels_temp, strat_train, strat_temp = (
        train_test_split(
            indices, labels, strat_key,
            test_size=test_size,
            random_state=seed,
            stratify=strat_key,
        )
    )

    # Split temp into dev and test
    dev_fraction = dev_ratio / (dev_ratio + (1.0 - train_ratio - dev_ratio))
    idx_dev, idx_test, labels_dev, labels_test = train_test_split(
        idx_temp, labels_temp,
        test_size=1.0 - dev_fraction,
        random_state=seed,
        stratify=strat_temp,
    )

    return (
        df.iloc[idx_train], df.iloc[idx_dev], df.iloc[idx_test],
        labels_train, labels_dev, labels_test,
    )


def build_dataframe(
    df: pd.DataFrame,
    labels: list[list[str]],
    label_col: str,
) -> pd.DataFrame:
    """Build the final DataFrame with standardized column names."""
    result = pd.DataFrame({
        "pmid": df["pmid"].values,
        "title": df["title"].values,
        "input_text": df["spanish_abstract"].values,
        "labels": labels,
    })
    return result


def compute_label_stats(
    train_labels: list[list[str]],
    dev_labels: list[list[str]],
    test_labels: list[list[str]],
    all_label_names: dict[str, str],
) -> dict:
    """Compute label frequency statistics across splits."""
    stats = {}
    for split_name, labels in [("train", train_labels), ("dev", dev_labels), ("test", test_labels)]:
        counts = {}
        for label_list in labels:
            for lbl in label_list:
                counts[lbl] = counts.get(lbl, 0) + 1
        stats[split_name] = counts
    return stats


def write_label_info(
    all_labels: list[str],
    label_names: list[list[str]],
    label_counts: dict,
    output_file: Path,
) -> None:
    """Write label info JSON."""
    info = {
        "num_labels": len(all_labels),
        "labels": all_labels,
        "counts_by_split": label_counts,
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Prepare PubMed Spanish multilabel dataset")
    parser.add_argument(
        "--csv_path",
        type=Path,
        default=CSV_PATH,
        help="Path to translated_dataset_small.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory for parquet splits",
    )
    parser.add_argument(
        "--label_level",
        choices=["level1", "level2"],
        default="level1",
        help="MeSH label granularity to use",
    )
    parser.add_argument(
        "--train_ratio", type=float, default=TRAIN_RATIO,
    )
    parser.add_argument(
        "--dev_ratio", type=float, default=DEV_RATIO,
    )
    parser.add_argument(
        "--seed", type=int, default=RANDOM_SEED,
    )
    args = parser.parse_args()

    csv_path = args.csv_path.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    label_col = "level1_codes" if args.label_level == "level1" else "level2_codes"
    name_col = "level1_names" if args.label_level == "level1" else "level2_names"

    print("=" * 60)
    print("PubMed Spanish → Parquet preparation")
    print("=" * 60)
    print(f"Source CSV: {csv_path}")
    print(f"Label level: {args.label_level} ({label_col})")
    print(f"Output: {output_dir}")
    print()

    # Load
    df = load_csv(csv_path)

    # Filter rows without labels
    print("\nFiltering rows with labels...")
    df_labeled, labels = filter_labeled(df, label_col)

    # Get all unique labels
    all_labels = sorted(set(lbl for label_list in labels for lbl in label_list))
    print(f"  Unique labels: {all_labels}")

    # Split
    print("\nSplitting dataset...")
    df_train, df_dev, df_test, labels_train, labels_dev, labels_test = stratified_split(
        df_labeled, labels,
        train_ratio=args.train_ratio,
        dev_ratio=args.dev_ratio,
        seed=args.seed,
    )

    print(f"  Train: {len(df_train):,}")
    print(f"  Dev:   {len(df_dev):,}")
    print(f"  Test:  {len(df_test):,}")

    # Build and save parquet files
    print("\nSaving parquet files...")
    for split_name, df_split, split_labels in [
        ("train", df_train, labels_train),
        ("dev", df_dev, labels_dev),
        ("test", df_test, labels_test),
    ]:
        out_df = build_dataframe(df_split, split_labels, label_col)
        out_path = output_dir / f"{split_name}.parquet"
        out_df.to_parquet(out_path, index=False)
        print(f"  Saved {split_name}.parquet ({len(out_df):,} rows)")

    # Write label info
    label_counts = compute_label_stats(labels_train, labels_dev, labels_test, {})
    label_info_path = output_dir / "label_info.json"
    write_label_info(all_labels, [], label_counts, label_info_path)
    print(f"\nLabel info saved to: {label_info_path}")

    # Write dataset stats
    stats = {
        "label_level": args.label_level,
        "num_labels": len(all_labels),
        "labels": all_labels,
        "split_sizes": {
            "train": len(df_train),
            "dev": len(df_dev),
            "test": len(df_test),
            "total": len(df_train) + len(df_dev) + len(df_test),
        },
    }
    stats_path = output_dir / "dataset_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"Dataset stats saved to: {stats_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
