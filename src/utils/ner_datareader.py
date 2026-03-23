"""
ner_datareader.py

Loads AnatEM Spanish NER dataset from CoNLL files produced by prepare_anat_em.py.

CoNLL format:
  token\tlabel
  (blank line = sentence boundary)
"""

from pathlib import Path
from typing import Optional


class NERDataset:
    """
    Loads and serves AnatEM Spanish NER data in CoNLL format.

    Attributes:
        data_dir: Path to directory with train.conll, dev.conll, test.conll
        label2id: Mapping from label string to integer id
        id2label: Inverse mapping
    """

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self._validate_files()

        self.train = self._read_conll(self.data_dir / "train.conll")
        self.dev = self._read_conll(self.data_dir / "dev.conll")
        self.test = self._read_conll(self.data_dir / "test.conll")

        self.label2id, self.id2label = self._build_label_vocab()

    def _validate_files(self) -> None:
        for fname in ["train.conll", "dev.conll", "test.conll"]:
            path = self.data_dir / fname
            if not path.exists():
                raise FileNotFoundError(
                    f"{fname} not found in {self.data_dir}. "
                    "Run scripts/prepare_anat_em.py first."
                )

    def _read_conll(self, filepath: Path) -> list[list[tuple[str, str]]]:
        """
        Parse a CoNLL file into sentences.

        Returns:
            List of sentences; each sentence is a list of (token, label) tuples.
        """
        sentences = []
        current: list[tuple[str, str]] = []

        with open(filepath, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if line == "":
                    if current:
                        sentences.append(current)
                        current = []
                else:
                    parts = line.split("\t")
                    token = parts[0]
                    label = parts[1] if len(parts) > 1 else "O"
                    current.append((token, label))

        if current:
            sentences.append(current)

        return sentences

    def _build_label_vocab(self) -> tuple[dict[str, int], dict[int, str]]:
        """Build label vocabulary from all splits."""
        labels = set()
        for split in [self.train, self.dev, self.test]:
            for sentence in split:
                for _, label in sentence:
                    labels.add(label)

        # Always put O first, then sort the rest
        sorted_labels = ["O"] + sorted(lbl for lbl in labels if lbl != "O")
        label2id = {lbl: i for i, lbl in enumerate(sorted_labels)}
        id2label = {i: lbl for lbl, i in label2id.items()}
        return label2id, id2label

    def get_split(self, split: str) -> list[list[tuple[str, str]]]:
        """Return sentences for a split ('train', 'dev', 'test')."""
        splits = {"train": self.train, "dev": self.dev, "test": self.test}
        if split not in splits:
            raise ValueError(f"Invalid split '{split}'. Choose from: train, dev, test")
        return splits[split]

    def get_tokens_and_labels(
        self, split: str
    ) -> tuple[list[list[str]], list[list[str]]]:
        """
        Return (tokens, labels) for a split.

        Returns:
            tokens: list of sentences, each sentence is a list of token strings
            labels: list of sentences, each sentence is a list of BIO label strings
        """
        sentences = self.get_split(split)
        tokens = [[tok for tok, _ in sent] for sent in sentences]
        labels = [[lbl for _, lbl in sent] for sent in sentences]
        return tokens, labels

    def get_label_names(self) -> list[str]:
        """Return sorted label names."""
        return [self.id2label[i] for i in range(len(self.id2label))]

    def get_stats(self) -> dict:
        """Return dataset statistics."""
        def count_entities(sentences: list[list[tuple[str, str]]]) -> int:
            count = 0
            for sent in sentences:
                for _, label in sent:
                    if label.startswith("B-"):
                        count += 1
            return count

        def count_tokens(sentences: list[list[tuple[str, str]]]) -> int:
            return sum(len(sent) for sent in sentences)

        return {
            "num_labels": len(self.label2id),
            "labels": self.get_label_names(),
            "train": {
                "sentences": len(self.train),
                "tokens": count_tokens(self.train),
                "entities": count_entities(self.train),
            },
            "dev": {
                "sentences": len(self.dev),
                "tokens": count_tokens(self.dev),
                "entities": count_entities(self.dev),
            },
            "test": {
                "sentences": len(self.test),
                "tokens": count_tokens(self.test),
                "entities": count_entities(self.test),
            },
        }

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"NERDataset(\n"
            f"  Labels: {stats['num_labels']} {stats['labels']}\n"
            f"  Train: {stats['train']['sentences']} sentences, "
            f"{stats['train']['tokens']:,} tokens, "
            f"{stats['train']['entities']} entities\n"
            f"  Dev:   {stats['dev']['sentences']} sentences, "
            f"{stats['dev']['tokens']:,} tokens, "
            f"{stats['dev']['entities']} entities\n"
            f"  Test:  {stats['test']['sentences']} sentences, "
            f"{stats['test']['tokens']:,} tokens, "
            f"{stats['test']['entities']} entities\n"
            f")"
        )
