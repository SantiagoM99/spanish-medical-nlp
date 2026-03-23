"""
multilabel_datareader.py

Loads PubMed Spanish multi-label dataset from parquet files
produced by prepare_pubmed.py.

Parquet schema:
  pmid          int64
  title         string
  input_text    string   (Spanish abstract)
  labels        list     (MeSH level1 or level2 codes)
"""

import json
from pathlib import Path

import pandas as pd


class MultiLabelDataset:
    """
    Loads and serves PubMed Spanish multi-label classification data.

    Attributes:
        data_dir: Path to directory with train/dev/test parquet files
        labels: Sorted list of all unique label codes
        label2id: Mapping from label code to integer index
        id2label: Inverse mapping
    """

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self._validate_files()

        self.train = pd.read_parquet(self.data_dir / "train.parquet")
        self.dev = pd.read_parquet(self.data_dir / "dev.parquet")
        self.test = pd.read_parquet(self.data_dir / "test.parquet")

        self.labels = self._extract_labels()
        self.label2id = {lbl: i for i, lbl in enumerate(self.labels)}
        self.id2label = {i: lbl for lbl, i in self.label2id.items()}

        # Load extra metadata if available
        self._label_info = self._load_label_info()

    def _validate_files(self) -> None:
        for fname in ["train.parquet", "dev.parquet", "test.parquet"]:
            path = self.data_dir / fname
            if not path.exists():
                raise FileNotFoundError(
                    f"{fname} not found in {self.data_dir}. "
                    "Run scripts/prepare_pubmed.py first."
                )

    def _extract_labels(self) -> list[str]:
        """Extract all unique labels across splits, sorted."""
        all_labels: set[str] = set()
        for df in [self.train, self.dev, self.test]:
            for label_list in df["labels"]:
                all_labels.update(label_list)
        return sorted(all_labels)

    def _load_label_info(self) -> dict:
        info_path = self.data_dir / "label_info.json"
        if info_path.exists():
            with open(info_path, encoding="utf-8") as f:
                return json.load(f)
        return {}

    def get_split(self, split: str) -> pd.DataFrame:
        """Return DataFrame for a split ('train', 'dev', 'test')."""
        splits = {"train": self.train, "dev": self.dev, "test": self.test}
        if split not in splits:
            raise ValueError(f"Invalid split '{split}'. Choose from: train, dev, test")
        return splits[split]

    def get_texts_and_labels(
        self, split: str
    ) -> tuple[list[str], list[list[str]]]:
        """
        Return (texts, labels) for a split.

        Returns:
            texts: list of Spanish abstract strings
            labels: list of label code lists (e.g. [['B', 'C'], ['A']])
        """
        df = self.get_split(split)
        texts = df["input_text"].tolist()
        labels = [list(lbl_arr) for lbl_arr in df["labels"]]
        return texts, labels

    def labels_to_multihot(self, label_lists: list[list[str]]) -> "pd.DataFrame":
        """
        Convert list-of-labels to binary multi-hot matrix.

        Returns a DataFrame with shape (n_samples, n_labels).
        """
        import numpy as np
        matrix = pd.DataFrame(
            0,
            index=range(len(label_lists)),
            columns=self.labels,
            dtype=int,
        )
        for i, label_list in enumerate(label_lists):
            for lbl in label_list:
                if lbl in self.label2id:
                    matrix.at[i, lbl] = 1
        return matrix

    def get_label_distribution(self, split: str) -> dict[str, int]:
        """Return per-label count for a split."""
        _, labels = self.get_texts_and_labels(split)
        distribution = {lbl: 0 for lbl in self.labels}
        for label_list in labels:
            for lbl in label_list:
                if lbl in distribution:
                    distribution[lbl] += 1
        return distribution

    def get_stats(self) -> dict:
        """Return dataset statistics."""
        def avg_labels(df: pd.DataFrame) -> float:
            return float(df["labels"].apply(len).mean())

        return {
            "num_labels": len(self.labels),
            "labels": self.labels,
            "train": {"size": len(self.train), "avg_labels": avg_labels(self.train)},
            "dev": {"size": len(self.dev), "avg_labels": avg_labels(self.dev)},
            "test": {"size": len(self.test), "avg_labels": avg_labels(self.test)},
            "total": len(self.train) + len(self.dev) + len(self.test),
        }

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"MultiLabelDataset(\n"
            f"  Labels: {stats['num_labels']} {stats['labels']}\n"
            f"  Train: {stats['train']['size']:,} samples "
            f"(avg {stats['train']['avg_labels']:.1f} labels)\n"
            f"  Dev:   {stats['dev']['size']:,} samples "
            f"(avg {stats['dev']['avg_labels']:.1f} labels)\n"
            f"  Test:  {stats['test']['size']:,} samples "
            f"(avg {stats['test']['avg_labels']:.1f} labels)\n"
            f")"
        )
