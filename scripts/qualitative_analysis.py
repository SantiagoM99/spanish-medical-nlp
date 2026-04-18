"""Build side-by-side gold vs predicted entity spans per sentence for manual inspection.

Writes a single .xlsx with one sheet per model (+ a summary sheet).
Each row = one test-set sentence; entities are rendered as `[TYPE] surface_text`,
joined with ` | `. A status column flags sentences with prediction errors.
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "results" / "qualitative_ner_analysis.xlsx"

MODELS = {
    "RigoBERTa-2.0 (best)": "results/ner/encoder/RigoBERTa-2.0_20260416_110449/predictions.jsonl",
    "Qwen2.5-0.5B (fine-tuned)": "results/ner/decoder/Qwen2.5-0.5B-Instruct/predictions.jsonl",
    "Qwen2.5-7B (knn few-shot)": "results/ner/decoder_knn_few_shot/Qwen2.5-7B-Instruct_20260414_141913/predictions.jsonl",
    "Qwen2.5-7B (zero-shot)":   "results/ner/decoder_zero_shot/Qwen2.5-7B-Instruct_20260414_141913/predictions.jsonl",
}


def extract_spans(tokens, labels):
    """Turn BIO labels into (type, start, end_exclusive, surface) spans."""
    spans, cur_type, cur_start = [], None, None

    def close(end):
        if cur_type is not None:
            surface = " ".join(tokens[cur_start:end])
            spans.append((cur_type, cur_start, end, surface))

    for i, lab in enumerate(labels):
        if lab == "O" or lab is None:
            close(i)
            cur_type, cur_start = None, None
            continue
        prefix, _, etype = lab.partition("-")
        if prefix == "B" or etype != cur_type:
            close(i)
            cur_type, cur_start = etype, i
    close(len(labels))
    return spans


def render(spans):
    return " | ".join(f"[{t}] {s}" for t, _, _, s in spans) if spans else ""


def row_for(tokens, gold_labels, pred_labels):
    gold = extract_spans(tokens, gold_labels)
    pred = extract_spans(tokens, pred_labels)
    gold_keys = {(t, s, e) for t, s, e, _ in gold}
    pred_keys = {(t, s, e) for t, s, e, _ in pred}
    tp = gold_keys & pred_keys
    fn = gold_keys - pred_keys
    fp = pred_keys - gold_keys
    if not gold_keys and not pred_keys:
        status = "empty"
    elif not fp and not fn:
        status = "perfect"
    elif tp and (fp or fn):
        status = "partial"
    elif not gold_keys:
        status = "hallucination"
    elif not pred_keys:
        status = "miss"
    else:
        status = "wrong"
    return {
        "text": " ".join(tokens),
        "gold_entities": render(gold),
        "pred_entities": render(pred),
        "n_gold": len(gold),
        "n_pred": len(pred),
        "n_correct": len(tp),
        "n_false_pos": len(fp),
        "n_false_neg": len(fn),
        "status": status,
    }


def build_sheet(jsonl_path):
    rows = (json.loads(l) for l in Path(jsonl_path).read_text().splitlines() if l.strip())
    data = [
        {"sentence_id": idx, **row_for(r["tokens"], r["true_labels"], r["pred_labels"])}
        for idx, r in enumerate(rows)
    ]
    return pd.DataFrame(data)


def summary_row(name, df):
    total = len(df)
    counts = df["status"].value_counts().to_dict()
    return {
        "model": name,
        "sentences": total,
        "perfect": counts.get("perfect", 0),
        "empty": counts.get("empty", 0),
        "partial": counts.get("partial", 0),
        "miss": counts.get("miss", 0),
        "hallucination": counts.get("hallucination", 0),
        "wrong": counts.get("wrong", 0),
        "total_gold": int(df["n_gold"].sum()),
        "total_pred": int(df["n_pred"].sum()),
        "total_correct": int(df["n_correct"].sum()),
    }


def main():
    sheets = {name: build_sheet(ROOT / path) for name, path in MODELS.items()}
    summary = pd.DataFrame(summary_row(n, df) for n, df in sheets.items())

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUT_PATH, engine="openpyxl") as w:
        summary.to_excel(w, sheet_name="summary", index=False)
        for name, df in sheets.items():
            sheet_name = name.replace("/", "-")[:31]
            df.to_excel(w, sheet_name=sheet_name, index=False)
    print(f"wrote {OUT_PATH}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
