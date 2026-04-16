"""
compute_comet_scores.py

Evaluates translation quality of Opus-MT English→Spanish for both corpora:
  1. PubMed abstracts (document-level)
  2. AnatEM NER sentences (sentence-level)

Model: Unbabel/wmt22-comet-da (reference-based, src used as proxy reference)

Usage:
    python scripts/compute_comet_scores.py                          # both corpora
    python scripts/compute_comet_scores.py --corpus pubmed          # PubMed only
    python scripts/compute_comet_scores.py --corpus anatem          # AnatEM only
    python scripts/compute_comet_scores.py --stats_only             # corpus stats, no COMET
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ──────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────

def load_pubmed_pairs(json_path: Path) -> list[dict]:
    """Load PubMed articles and extract src/mt pairs."""
    print(f"Loading PubMed articles from {json_path}...")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    articles = data["articles"]
    pairs = []
    for art in articles:
        abstract = art.get("abstract", {})
        src = abstract.get("full_text", "")
        translation = abstract.get("spanish_translation", {})
        mt = translation.get("text", "") if isinstance(translation, dict) else ""
        success = translation.get("success", False) if isinstance(translation, dict) else False

        if src and mt and success:
            pairs.append({
                "id": art.get("pmid"),
                "src": src,
                "mt": mt,
            })

    print(f"  Total articles: {len(articles)}")
    print(f"  Valid src+mt pairs: {len(pairs)}")
    return pairs


def _reconstruct_sentences_en(filepath: Path) -> list[str]:
    """Reconstruct sentences from English nersuite file.

    Format: BIO_tag \\t start \\t end \\t token \\t lemma \\t POS \\t chunk
    Blank lines separate sentences.
    """
    sentences = []
    current_tokens = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                if current_tokens:
                    sentences.append(" ".join(current_tokens))
                    current_tokens = []
                continue
            parts = line.split("\t")
            if len(parts) >= 4:
                current_tokens.append(parts[3])  # token column
    if current_tokens:
        sentences.append(" ".join(current_tokens))
    return sentences


def _reconstruct_document_es(filepath: Path) -> str:
    """Reconstruct full document text from Spanish nersuite file.

    Format: token \\t BIO_tag (no blank line separators).
    """
    tokens = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if parts:
                tokens.append(parts[0])
    return " ".join(tokens)


def _reconstruct_document_en(filepath: Path) -> str:
    """Reconstruct full document text from English nersuite file.

    Format: BIO_tag \\t start \\t end \\t token \\t lemma \\t POS \\t chunk
    Blank lines separate sentences.
    """
    tokens = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 4:
                tokens.append(parts[3])
    return " ".join(tokens)


def load_anatem_pairs(en_dir: Path, es_dir: Path) -> list[dict]:
    """Load aligned EN/ES document pairs from AnatEM nersuite files."""
    print(f"Loading AnatEM pairs from {en_dir} / {es_dir}...")

    pairs = []
    skipped_files = 0

    for split in ["train", "devel", "test"]:
        en_split_dir = en_dir / split
        es_split_dir = es_dir / split

        en_files = sorted(glob.glob(str(en_split_dir / "*.nersuite")))

        for en_file in en_files:
            fname = os.path.basename(en_file)
            es_file = es_split_dir / fname

            if not es_file.exists():
                skipped_files += 1
                continue

            src = _reconstruct_document_en(Path(en_file))
            mt = _reconstruct_document_es(es_file)

            if src.strip() and mt.strip():
                pairs.append({
                    "id": fname,
                    "src": src,
                    "mt": mt,
                })

    print(f"  Total document pairs: {len(pairs)}")
    if skipped_files:
        print(f"  Skipped files (no ES match): {skipped_files}")
    return pairs


# ──────────────────────────────────────────────────────────────
# Corpus statistics
# ──────────────────────────────────────────────────────────────

def compute_corpus_stats(pairs: list[dict], corpus_name: str):
    """Compute corpus-level token statistics for EN vs ES."""
    print(f"\n{'=' * 60}")
    print(f"Corpus-level statistics: {corpus_name}")
    print("=" * 60)

    en_lengths = [len(p["src"].split()) for p in pairs]
    es_lengths = [len(p["mt"].split()) for p in pairs]
    en_vocab = set(t.lower() for p in pairs for t in p["src"].split())
    es_vocab = set(t.lower() for p in pairs for t in p["mt"].split())

    en_s = pd.Series(en_lengths)
    es_s = pd.Series(es_lengths)

    print(f"\n  EN:")
    print(f"    Mean tokens: {en_s.mean():.1f} (±{en_s.std():.1f})")
    print(f"    Median: {en_s.median():.0f}, Min: {en_s.min()}, Max: {en_s.max()}")
    print(f"    Unique vocab: {len(en_vocab):,}")

    print(f"\n  ES (Opus-MT):")
    print(f"    Mean tokens: {es_s.mean():.1f} (±{es_s.std():.1f})")
    print(f"    Median: {es_s.median():.0f}, Min: {es_s.min()}, Max: {es_s.max()}")
    print(f"    Unique vocab: {len(es_vocab):,}")

    print(f"\n  Token expansion ratio (ES/EN): {es_s.mean() / en_s.mean():.2f}")


# ──────────────────────────────────────────────────────────────
# COMET evaluation
# ──────────────────────────────────────────────────────────────

def compute_comet_scores(
    pairs: list[dict],
    corpus_name: str,
    sample_size: int,
    seed: int,
    batch_size: int,
    model_name: str,
    model=None,
):
    """Compute COMET scores. Returns (scores, loaded_model) for reuse."""
    from comet import download_model, load_from_checkpoint

    print(f"\n{'=' * 60}")
    print(f"COMET evaluation: {corpus_name} (model={model_name}, n={sample_size})")
    print("=" * 60)

    # Sample
    df = pd.DataFrame(pairs)
    if sample_size >= len(df):
        sample = df
        print(f"  Using all {len(df)} pairs")
    else:
        sample = df.sample(n=sample_size, random_state=seed)
        print(f"  Sampled {sample_size} from {len(df)} pairs (seed={seed})")

    # COMET input: src + mt + ref (src as proxy reference)
    comet_data = [{"src": row["src"], "mt": row["mt"], "ref": row["src"]}
                  for _, row in sample.iterrows()]

    # Load model (reuse if already loaded)
    if model is None:
        print(f"\n  Downloading/loading model: {model_name}...")
        model_path = download_model(model_name)
        model = load_from_checkpoint(model_path)
    else:
        print(f"  Reusing loaded model: {model_name}")

    # Predict
    print(f"  Running inference (batch_size={batch_size})...")
    output = model.predict(comet_data, batch_size=batch_size)
    scores = output.scores

    # Statistics
    ss = pd.Series(scores)
    above_80 = (ss > 0.80).sum() / len(ss) * 100
    above_75 = (ss > 0.75).sum() / len(ss) * 100

    print(f"\n  {'─' * 40}")
    print(f"  COMET Results — {corpus_name}")
    print(f"  {'─' * 40}")
    print(f"  Mean:       {ss.mean():.4f}")
    print(f"  Std:        {ss.std():.4f}")
    print(f"  Min:        {ss.min():.4f}")
    print(f"  Max:        {ss.max():.4f}")
    print(f"  Median:     {ss.median():.4f}")
    print(f"  Above 0.80: {above_80:.1f}%")
    print(f"  Above 0.75: {above_75:.1f}%")

    # LaTeX-ready
    print(f"\n  LaTeX:")
    print(f"  COMET mean = {ss.mean():.2f} (SD = {ss.std():.2f}), "
          f"{above_80:.0f}\\% above 0.80 threshold.")

    # Save per-sample scores
    sample = sample.copy()
    sample["comet_score"] = scores
    safe_name = corpus_name.lower().replace(" ", "_")
    output_path = BASE_DIR / "results" / f"comet_scores_{safe_name}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample[["id", "comet_score"]].to_csv(output_path, index=False)
    print(f"  Scores saved to: {output_path}")

    return scores, model


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compute COMET translation quality scores")
    parser.add_argument("--corpus", choices=["pubmed", "anatem", "both"], default="both",
                        help="Which corpus to evaluate (default: both)")
    parser.add_argument("--sample_size", type=int, default=200,
                        help="Number of samples for COMET (PubMed). AnatEM uses all sentences.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--model", type=str, default="Unbabel/wmt22-comet-da")
    parser.add_argument("--stats_only", action="store_true",
                        help="Only compute corpus stats, skip COMET")
    args = parser.parse_args()

    do_pubmed = args.corpus in ("pubmed", "both")
    do_anatem = args.corpus in ("anatem", "both")

    loaded_model = None

    # ── PubMed ──
    if do_pubmed:
        json_path = BASE_DIR / "data" / "pubmed_mesh_en" / "translated_dataset_with_titles.json"
        if not json_path.exists():
            print(f"ERROR: {json_path} not found")
            sys.exit(1)

        pubmed_pairs = load_pubmed_pairs(json_path)
        compute_corpus_stats(pubmed_pairs, "PubMed Abstracts")

        if not args.stats_only:
            _, loaded_model = compute_comet_scores(
                pubmed_pairs, "PubMed Abstracts",
                args.sample_size, args.seed, args.batch_size, args.model,
            )

    # ── AnatEM ──
    if do_anatem:
        en_dir = BASE_DIR / "data" / "anat_em_en"
        es_dir = BASE_DIR / "data" / "anat_em"

        if not en_dir.exists() or not es_dir.exists():
            print(f"ERROR: AnatEM directories not found ({en_dir}, {es_dir})")
            sys.exit(1)

        anatem_pairs = load_anatem_pairs(en_dir, es_dir)
        compute_corpus_stats(anatem_pairs, "AnatEM NER")

        if not args.stats_only:
            # Use all AnatEM sentences (corpus is small enough)
            anatem_sample = min(len(anatem_pairs), len(anatem_pairs))
            _, loaded_model = compute_comet_scores(
                anatem_pairs, "AnatEM NER",
                anatem_sample, args.seed, args.batch_size, args.model,
                model=loaded_model,
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
