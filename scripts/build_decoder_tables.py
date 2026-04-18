"""Emit expanded LaTeX Tables 4 (classification) and 6 (NER) for the paper.

Cells the paper already reports (via PAPER_VALUES) are preserved verbatim.
Empty cells are filled from the newest matching run under results/.
Cells that are still missing a run are printed as `—`.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MODELS_ORDER = ["Gemma-2-9B", "Llama-3.1-8B", "Qwen2.5-7B", "Mistral-7B"]
MODEL_HF = {
    "Gemma-2-9B":   "gemma-2-9b-it",
    "Llama-3.1-8B": "Llama-3.1-8B-Instruct",
    "Qwen2.5-7B":   "Qwen2.5-7B-Instruct",
    "Mistral-7B":   "Mistral-7B-Instruct-v0.3",
}

# Values already reported in the paper — kept verbatim to avoid changing
# published numbers that disagree with a rerun of seqeval on predictions.jsonl.
PAPER_VALUES = {
    ("multilabel", "zero_shot", "Llama-3.1-8B"): (.398, .459, .426, .178),
    ("multilabel", "zero_shot", "Qwen2.5-7B"):   (.401, .438, .418, .216),
    ("multilabel", "few_shot",  "Llama-3.1-8B"): (.317, .497, .387, .217),
    ("multilabel", "few_shot",  "Qwen2.5-7B"):   (.292, .412, .342, .234),
    ("multilabel", "qlora",     "Qwen2.5-7B"):   (.593, .320, .416, .358),
    ("ner", "zero_shot", "Gemma-2-9B"):    (.243, .295, .267),
    ("ner", "zero_shot", "Llama-3.1-8B"):  (.117, .211, .151),
    ("ner", "zero_shot", "Qwen2.5-7B"):    (.132, .126, .129),
    ("ner", "zero_shot", "Mistral-7B"):    (.115, .129, .122),
    ("ner", "knn",        "Llama-3.1-8B"): (.266, .012, .023),
    ("ner", "knn",        "Qwen2.5-7B"):   (.211, .007, .013),
    ("ner", "knn_verify", "Qwen2.5-7B"):   (.357, .006, .013),
    ("ner", "qlora",      "Llama-3.1-8B"): (.000, .000, .000),
    ("ner", "qlora",      "Qwen2.5-7B"):   (.000, .001, .001),
}

RESULT_DIRS = {
    ("multilabel", "zero_shot"): "results/multilabel/decoder_zero_shot",
    ("multilabel", "few_shot"):  "results/multilabel/decoder_few_shot",
    ("multilabel", "qlora"):     "results/multilabel/peft_qlora",
    ("ner", "knn"):              "results/ner/decoder_knn_few_shot",
    ("ner", "knn_verify"):       "results/ner/decoder_knn_few_shot_verify",
    ("ner", "qlora"):            "results/ner/peft_qlora",
}


def latest_metrics(task, setting, model):
    if (task, setting, model) in PAPER_VALUES:
        return PAPER_VALUES[(task, setting, model)]
    hf = MODEL_HF[model]
    base = ROOT / RESULT_DIRS[(task, setting)]
    if not base.exists():
        return None
    candidates = sorted(
        (p for p in base.glob(f"{hf}*") if (p / "metrics.json").exists()),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    if not candidates:
        return None
    m = json.loads((candidates[0] / "metrics.json").read_text())
    if task == "multilabel":
        return (m["micro_precision"], m["micro_recall"], m["micro_f1"], m["macro_f1"])
    return (m["precision"], m["recall"], m["f1"])


def fmt(x):
    return "---" if x is None else f".{int(round(x * 1000)):03d}"


def fmt_row(values, n):
    if values is None:
        return " & ".join(["---"] * n)
    return " & ".join(fmt(v) for v in values)


def classification_table():
    settings = [("zero_shot", "Zero-shot"), ("few_shot", "Few-shot (3 ex.)"), ("qlora", "QLoRA")]
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{@{}ll cccc@{}}",
        r"\toprule",
        r"\textbf{Setting} & \textbf{Model} & Mi-P & Mi-R & Mi-F1 & Ma-F1 \\",
        r"\midrule",
    ]
    for key, label in settings:
        lines.append(f"\\multicolumn{{6}}{{l}}{{\\textit{{{label}}}}} \\\\")
        for model in MODELS_ORDER:
            vals = latest_metrics("multilabel", key, model)
            lines.append(f" & {model} & {fmt_row(vals, 4)} \\\\")
        if key != "qlora":
            lines.append(r"\addlinespace")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Decoder classification results (T+A input, greedy decoding). "
        r"Dashes indicate configurations not evaluated due to compute budget; "
        r"see \texttt{run\_missing\_experiments.sh} for any remaining runs.}",
        r"\label{tab:decoder_classification}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def ner_table():
    settings = [
        ("zero_shot",  "Zero-shot"),
        ("knn",        "k-NN few-shot (k=5)"),
        ("knn_verify", "k-NN + self-verification"),
        ("qlora",      "QLoRA"),
    ]
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{@{}ll ccc@{}}",
        r"\toprule",
        r"\textbf{Setting} & \textbf{Model} & P & R & F1 \\",
        r"\midrule",
    ]
    for key, label in settings:
        lines.append(f"\\multicolumn{{5}}{{l}}{{\\textit{{{label}}}}} \\\\")
        for model in MODELS_ORDER:
            vals = latest_metrics("ner", key, model)
            lines.append(f" & {model} & {fmt_row(vals, 3)} \\\\")
        if key != "qlora":
            lines.append(r"\addlinespace")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Decoder NER results (entity-level, strict span match). "
        r"k-NN uses FAISS with MiniLM-L6-v2 embeddings. "
        r"Dashes indicate configurations not evaluated due to compute budget; "
        r"see \texttt{run\_missing\_experiments.sh} for any remaining runs.}",
        r"\label{tab:decoder_ner}",
        r"\end{table}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print("% ========== Table 4: Decoder classification ==========\n")
    print(classification_table())
    print("\n% ========== Table 6: Decoder NER ==========\n")
    print(ner_table())
