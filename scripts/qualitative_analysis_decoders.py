"""Gold-vs-prediction side-by-side xlsx for every decoder NER run.

Auto-discovers all `results/ner/decoder*/*/predictions.jsonl` files, one sheet
per run plus a summary sheet sorted by micro-F1.
"""
import json
from pathlib import Path

import pandas as pd
from seqeval.metrics import f1_score as seqeval_f1

from qualitative_analysis import extract_spans, render, row_for, summary_row

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "results" / "qualitative_ner_analysis_decoders.xlsx"

STRATEGY_SHORT = {
    "decoder": "FT",
    "decoder_zero_shot": "ZS",
    "decoder_knn_few_shot": "KNN",
    "decoder_knn_few_shot_verify": "KNN+V",
}

MODEL_SHORT = {
    "Llama-3.1-8B-Instruct": "Llama-8B",
    "Qwen2.5-0.5B-Instruct": "Qwen-0.5B",
    "Qwen2.5-7B-Instruct": "Qwen-7B",
    "gemma-2-9b-it": "Gemma-9B",
    "Mistral-7B-Instruct-v0.3": "Mistral-7B",
}


def short_name(strategy_dir, run_dir):
    strategy = STRATEGY_SHORT.get(strategy_dir, strategy_dir)
    base = run_dir.split("_2026")[0]
    model = MODEL_SHORT.get(base, base)
    return f"{model} / {strategy}"


def build_sheet(jsonl_path):
    rows = (json.loads(l) for l in Path(jsonl_path).read_text().splitlines() if l.strip())
    materialized = list(rows)
    data = [
        {"sentence_id": idx, **row_for(r["tokens"], r["true_labels"], r["pred_labels"])}
        for idx, r in enumerate(materialized)
    ]
    y_true = [r["true_labels"] for r in materialized]
    y_pred = [r["pred_labels"] for r in materialized]
    f1 = seqeval_f1(y_true, y_pred)
    return pd.DataFrame(data), f1


def main():
    runs = sorted(ROOT.glob("results/ner/decoder*/*/predictions.jsonl"))
    sheets = {}
    f1s = {}
    for p in runs:
        name = short_name(p.parent.parent.name, p.parent.name)
        df, f1 = build_sheet(p)
        sheets[name] = df
        f1s[name] = f1

    ordered = sorted(sheets, key=lambda k: f1s[k], reverse=True)
    summary = pd.DataFrame(
        {**summary_row(n, sheets[n]), "f1": round(f1s[n], 4)} for n in ordered
    )
    summary = summary[["model", "f1", "sentences", "perfect", "empty", "partial",
                       "miss", "hallucination", "wrong", "total_gold", "total_pred",
                       "total_correct"]]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUT_PATH, engine="openpyxl") as w:
        summary.to_excel(w, sheet_name="summary", index=False)
        for name in ordered:
            sheet_name = name.replace("/", "-")[:31]
            sheets[name].to_excel(w, sheet_name=sheet_name, index=False)

    print(f"wrote {OUT_PATH}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
