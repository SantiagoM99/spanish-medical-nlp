"""
multilabel_datareader.py

Loads PubMed Spanish multi-label dataset.

Accepted formats (auto-detected by filename):
  Parquet (from prepare_pubmed.py):
    train.parquet / dev.parquet / test.parquet
    Columns: pmid, title, input_text, labels (list)

  CSV (existing splits from Dataset-Creation/dataset/splits/):
    train_multilabel.csv / val_multilabel.csv / test_multilabel.csv
    Columns: pmid, title, spanish_abstract, level1_codes, ...

Text modes:
    title_abstract  — título + abstract (default)
    abstract_only   — solo abstract
    title_only      — solo título

Options:
    filter_geographicals=True  — elimina filas con categoría Z (Geographicals)

Usage with existing CSV splits:
    dataset = MultiLabelDataset(
        data_dir="path/to/Dataset-Creation/dataset/splits",
        fmt="csv",
    )

Usage with prepared parquet:
    dataset = MultiLabelDataset(data_dir="data/pubmed_mesh")
"""

import ast
import json
from pathlib import Path
from typing import Literal, Optional

import pandas as pd


class MultiLabelDataset:
    """
    Loads and serves PubMed Spanish multi-label classification data.

    Attributes:
        data_dir: Path to directory containing dataset files
        text_mode: 'title_abstract' | 'abstract_only' | 'title_only'
        filter_geographicals: Remove samples whose only label is Z (Geographicals)
        labels: Sorted list of all unique label codes
        label2id: Mapping from label code to integer index
        id2label: Inverse mapping
    """

    # CSV column names in the existing splits
    _CSV_FILES = {
        "train": "train_multilabel.csv",
        "dev": "val_multilabel.csv",
        "test": "test_multilabel.csv",
    }
    _CSV_TEXT_COL = "spanish_abstract"
    _CSV_TITLE_COL = "title"
    _CSV_LABEL_COL = "level1_codes"

    _TEXT_MODES = ("title_abstract", "abstract_only", "title_only")

    def __init__(
        self,
        data_dir: str,
        fmt: Literal["auto", "parquet", "csv"] = "auto",
        text_mode: str = "title_abstract",
        filter_geographicals: bool = False,
    ):
        if text_mode not in self._TEXT_MODES:
            raise ValueError(f"text_mode must be one of {self._TEXT_MODES}, got '{text_mode}'")

        self.data_dir = Path(data_dir)
        self.text_mode = text_mode
        self.filter_geographicals = filter_geographicals
        self.fmt = self._detect_format(fmt)

        if self.fmt == "parquet":
            self._validate_parquet()
            self.train = self._postprocess(pd.read_parquet(self.data_dir / "train.parquet"))
            self.dev = self._postprocess(pd.read_parquet(self.data_dir / "dev.parquet"))
            self.test = self._postprocess(pd.read_parquet(self.data_dir / "test.parquet"))
        else:
            self._validate_csv()
            self.train = self._postprocess(self._read_csv("train"))
            self.dev = self._postprocess(self._read_csv("dev"))
            self.test = self._postprocess(self._read_csv("test"))

        self.labels = self._extract_labels()
        self.label2id = {lbl: i for i, lbl in enumerate(self.labels)}
        self.id2label = {i: lbl for lbl, i in self.label2id.items()}
        self._label_info = self._load_label_info()

    def _detect_format(self, fmt: str) -> str:
        if fmt != "auto":
            return fmt
        if (self.data_dir / "train.parquet").exists():
            return "parquet"
        if (self.data_dir / self._CSV_FILES["train"]).exists():
            return "csv"
        raise FileNotFoundError(
            f"No dataset files found in {self.data_dir}.\n"
            "Expected train.parquet or train_multilabel.csv."
        )

    def _validate_parquet(self) -> None:
        for split, fname in [("train", "train.parquet"), ("dev", "dev.parquet"), ("test", "test.parquet")]:
            path = self.data_dir / fname
            if not path.exists():
                raise FileNotFoundError(
                    f"{fname} not found in {self.data_dir}. "
                    "Run scripts/prepare_pubmed.py first."
                )

    def _validate_csv(self) -> None:
        for split, fname in self._CSV_FILES.items():
            path = self.data_dir / fname
            if not path.exists():
                raise FileNotFoundError(f"{fname} not found in {self.data_dir}.")

    def _read_csv(self, split: str) -> pd.DataFrame:
        """Read a CSV split and normalize to the standard schema."""
        fname = self._CSV_FILES[split]
        df = pd.read_csv(self.data_dir / fname, encoding="utf-8")

        # Parse the label column from string representation of list
        def parse_labels(val) -> list:
            try:
                return [str(x) for x in ast.literal_eval(str(val))]
            except (ValueError, SyntaxError):
                return []

        df["labels"] = df[self._CSV_LABEL_COL].apply(parse_labels)

        # Normalize text columns
        if self._CSV_TEXT_COL in df.columns:
            df["abstract"] = df[self._CSV_TEXT_COL]
        else:
            df["abstract"] = ""

        if self._CSV_TITLE_COL in df.columns:
            df["title"] = df[self._CSV_TITLE_COL]
        else:
            df["title"] = ""

        # Keep only rows with at least one label
        df = df[df["labels"].apply(len) > 0].reset_index(drop=True)
        return df

    def _postprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply text mode selection and Geographicals filter."""
        df = df.copy()

        # Normalize column names for parquet format (already has input_text)
        if "input_text" in df.columns and "abstract" not in df.columns:
            df["abstract"] = df["input_text"]
        if "title" not in df.columns:
            df["title"] = ""

        # Build input_text according to text_mode
        if self.text_mode == "abstract_only":
            df["input_text"] = df["abstract"].fillna("").str.strip()
        elif self.text_mode == "title_only":
            df["input_text"] = df["title"].fillna("").str.strip()
        else:  # title_abstract
            df["input_text"] = (
                df["title"].fillna("").str.strip()
                + " "
                + df["abstract"].fillna("").str.strip()
            ).str.strip()

        # Filter Geographicals (MeSH level-1 code Z)
        if self.filter_geographicals:
            before = len(df)
            df["labels"] = df["labels"].apply(
                lambda lbls: [l for l in lbls if l != "Z"]
            )
            df = df[df["labels"].apply(len) > 0].reset_index(drop=True)
            removed = before - len(df)
            if removed:
                print(f"Filtrado Geographicals: removidas {removed} filas ({removed/max(before,1)*100:.1f}%).")

        return df

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
            f"  text_mode='{self.text_mode}', filter_geographicals={self.filter_geographicals}\n"
            f"  Labels: {stats['num_labels']} {stats['labels']}\n"
            f"  Train: {stats['train']['size']:,} samples "
            f"(avg {stats['train']['avg_labels']:.1f} labels)\n"
            f"  Dev:   {stats['dev']['size']:,} samples "
            f"(avg {stats['dev']['avg_labels']:.1f} labels)\n"
            f"  Test:  {stats['test']['size']:,} samples "
            f"(avg {stats['test']['avg_labels']:.1f} labels)\n"
            f")"
        )
