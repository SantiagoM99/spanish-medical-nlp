"""
prepare_anat_em.py

Converts AnatEM nersuite-spanish files into CoNLL format train/dev/test splits.

Input:
  - Dataset-Creation/AnatEM/nersuite-spanish/  -> one .nersuite file per document
  - Dataset-Creation/AnatEM/splits/            -> {train,devel,test}-documents.list

Output (data/anat_em/):
  - train.conll
  - dev.conll
  - test.conll
  - label_map.txt

NERSuite-Spanish format (two columns, tab-separated):
  token\tBIO-label
  (blank line = sentence boundary)

CoNLL output format (same, but aggregated per split):
  token\tlabel
  (blank line = sentence boundary)
"""

import argparse
from pathlib import Path


NERSUITE_SPANISH_DIR = Path("../Dataset-Creation/AnatEM/nersuite-spanish")
SPLITS_DIR = Path("../Dataset-Creation/AnatEM/splits")
OUTPUT_DIR = Path("../spanish-medical-nlp/data/anat_em")


def load_document_list(list_file: Path) -> list[str]:
    """Read a document list file and return document IDs."""
    with open(list_file, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def read_nersuite_file(filepath: Path) -> list[list[tuple[str, str]]]:
    """
    Parse a .nersuite file into a list of sentences.

    Returns:
        List of sentences, each sentence is a list of (token, label) tuples.
    """
    sentences = []
    current = []

    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line == "":
                if current:
                    sentences.append(current)
                    current = []
            else:
                parts = line.split("\t")
                if len(parts) >= 2:
                    token = parts[0]
                    label = parts[1]
                    current.append((token, label))
                elif len(parts) == 1 and parts[0]:
                    # Some files may have only the token with O label
                    current.append((parts[0], "O"))

    if current:
        sentences.append(current)

    return sentences


def write_conll(sentences: list[list[tuple[str, str]]], output_file: Path) -> int:
    """Write sentences to CoNLL format. Returns token count."""
    token_count = 0
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        for sentence in sentences:
            for token, label in sentence:
                f.write(f"{token}\t{label}\n")
                token_count += 1
            f.write("\n")

    return token_count


def build_split(
    doc_ids: list[str],
    nersuite_dir: Path,
    output_file: Path,
) -> tuple[int, int, set[str]]:
    """
    Build a CoNLL split from a list of document IDs.

    Returns:
        (doc_count, token_count, label_set)
    """
    all_sentences = []
    labels = set()
    missing = []

    for doc_id in doc_ids:
        filepath = nersuite_dir / f"{doc_id}.nersuite"
        if not filepath.exists():
            missing.append(doc_id)
            continue

        sentences = read_nersuite_file(filepath)
        all_sentences.extend(sentences)
        for sentence in sentences:
            for _, label in sentence:
                labels.add(label)

    if missing:
        print(f"  WARNING: {len(missing)} files not found (e.g. {missing[:3]})")

    token_count = write_conll(all_sentences, output_file)
    return len(doc_ids) - len(missing), token_count, labels


def write_label_map(labels: set[str], output_file: Path) -> None:
    """Write sorted label list to file."""
    sorted_labels = sorted(labels)
    with open(output_file, "w", encoding="utf-8") as f:
        for label in sorted_labels:
            f.write(f"{label}\n")


def main():
    parser = argparse.ArgumentParser(description="Prepare AnatEM Spanish NER dataset")
    parser.add_argument(
        "--nersuite_dir",
        type=Path,
        default=NERSUITE_SPANISH_DIR,
        help="Directory with .nersuite Spanish files",
    )
    parser.add_argument(
        "--splits_dir",
        type=Path,
        default=SPLITS_DIR,
        help="Directory with document list files",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory for CoNLL files",
    )
    args = parser.parse_args()

    nersuite_dir = args.nersuite_dir.resolve()
    splits_dir = args.splits_dir.resolve()
    output_dir = args.output_dir.resolve()

    print("=" * 60)
    print("AnatEM Spanish → CoNLL preparation")
    print("=" * 60)
    print(f"Source: {nersuite_dir}")
    print(f"Splits: {splits_dir}")
    print(f"Output: {output_dir}")
    print()

    all_labels = set()
    split_configs = [
        ("train", "train-documents.list", "train.conll"),
        ("dev", "devel-documents.list", "dev.conll"),
        ("test", "test-documents.list", "test.conll"),
    ]

    for split_name, list_file, out_file in split_configs:
        print(f"[{split_name}]")
        doc_ids = load_document_list(splits_dir / list_file)
        print(f"  Documents in list: {len(doc_ids)}")

        doc_count, token_count, labels = build_split(
            doc_ids=doc_ids,
            nersuite_dir=nersuite_dir,
            output_file=output_dir / out_file,
        )
        all_labels.update(labels)

        print(f"  Documents processed: {doc_count}")
        print(f"  Tokens: {token_count:,}")
        print(f"  Labels seen: {len(labels)}")
        print()

    label_file = output_dir / "label_map.txt"
    write_label_map(all_labels, label_file)
    print(f"Labels saved to: {label_file}")
    print(f"Total unique labels: {len(all_labels)}")
    print("\nDone.")


if __name__ == "__main__":
    main()
