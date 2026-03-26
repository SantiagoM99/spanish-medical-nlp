# Spanish Medical NLP Benchmark

A benchmarking framework for evaluating NLP models on two Spanish medical tasks:

1. **Multi-label Classification** — MeSH category assignment for PubMed Spanish abstracts
2. **Named Entity Recognition (NER)** — Anatomical entity recognition from the AnatEM dataset

The framework supports three model architectures: fine-tuned encoders (BERT-style), zero/few-shot decoder LLMs, and parameter-efficient fine-tuning (LoRA/QLoRA/DoRA).

---

## Project Structure

```
spanish-medical-nlp/
├── configs/
│   ├── experiment_config.yaml     # Grid search config for multilabel experiments
│   ├── ner_config.yaml            # Detailed NER benchmark config
│   └── models.yaml                # Model registry with training hyperparameters
│
├── data/
│   ├── pubmed_mesh/               # Multi-label dataset (parquet splits, populated by prepare_pubmed.py)
│   └── anat_em/                   # NER dataset (CoNLL splits, populated by prepare_anat_em.py)
│
├── results/
│   ├── multilabel/                # Multi-label experiment outputs
│   └── ner/                       # NER experiment outputs
│
├── scripts/
│   ├── prepare_pubmed.py          # Convert PubMed CSV to train/dev/test parquet splits
│   ├── prepare_anat_em.py         # Convert AnatEM nersuite files to CoNLL format + splits
│   └── tune.py                    # Bayesian hyperparameter search via W&B Sweeps
│
├── src/
│   ├── evaluation/
│   │   ├── metrics.py             # NER (seqeval) and multilabel (sklearn) metrics
│   │   ├── multilabel_predictor.py  # Inference orchestration for multilabel
│   │   └── ner_predictor.py       # Inference orchestration for NER
│   │
│   ├── models/
│   │   ├── base_llm.py            # Abstract base for language models
│   │   ├── base_multilabel.py     # Abstract base for multilabel models
│   │   ├── base_ner.py            # Abstract base for NER models
│   │   ├── encoder_multilabel.py  # Fine-tuned BERT-style classifier (sigmoid head)
│   │   ├── encoder_ner.py         # Fine-tuned BERT-style token classifier
│   │   ├── huggingface_llm.py     # Wrapper for causal LMs (Qwen, Llama, Mistral)
│   │   ├── llm_multilabel_model.py  # Zero/few-shot multilabel with LLMs
│   │   ├── llm_ner_model.py       # Zero/few-shot/k-NN NER with LLMs
│   │   ├── openai_llm.py          # Optional OpenAI API wrapper
│   │   ├── peft_multilabel_model.py  # LoRA/QLoRA/DoRA multilabel fine-tuning
│   │   └── peft_ner_model.py      # LoRA/QLoRA/DoRA NER fine-tuning
│   │
│   ├── prompts/
│   │   ├── base_prompt.py         # Abstract prompt template
│   │   ├── multilabel_prompt.py   # MeSH category prompts (zero-shot and few-shot)
│   │   └── ner_prompt.py          # Anatomical entity prompts (zero, few, k-NN)
│   │
│   └── utils/
│       ├── experiment_runner.py   # Grid search orchestration for multilabel
│       ├── knn_retrieval.py       # FAISS-based k-NN retrieval for few-shot examples
│       ├── multilabel_datareader.py  # Dataset loader for multilabel task
│       ├── ner_datareader.py      # Dataset loader for NER task
│       └── visualizations.py     # Plotting utilities for results
│
├── run_multilabel_benchmark.py    # Main entry point — multilabel experiments
├── run_ner_benchmark.py           # Main entry point — NER experiments
└── requirements.txt               # Python dependencies
```

---

## Installation

```bash
git clone <repo-url>
cd spanish-medical-nlp

pip install -r requirements.txt
```

For GPU support with quantization (required for 7B+ models):

```bash
pip install bitsandbytes accelerate
```

> **Note:** `bitsandbytes` requires a CUDA-capable GPU. For CPU-only setups use small models (e.g., `Qwen/Qwen2.5-0.5B-Instruct`) without quantization flags.

---

## Data Preparation

Before running any experiment, populate the `data/` directories using the preparation scripts.

### Multi-label (PubMed MeSH)

```bash
python scripts/prepare_pubmed.py \
    --csv_path ../Dataset-Creation/dataset/translated_dataset_small.csv \
    --output_dir data/pubmed_mesh
```

Produces `train.parquet`, `dev.parquet`, `test.parquet` with columns: `pmid`, `title`, `input_text`, `labels`.

### NER (AnatEM)

```bash
python scripts/prepare_anat_em.py \
    --nersuite_dir ../Dataset-Creation/AnatEM/nersuite-spanish \
    --splits_dir ../Dataset-Creation/AnatEM/splits \
    --output_dir data/anat_em
```

Produces CoNLL-format splits and a label mapping from BIO tags.

---

## Running Experiments

Both benchmark scripts share the same three `--model_type` options:

| `--model_type` | Description |
|---|---|
| `encoder` | Full fine-tuning of a BERT-style encoder |
| `decoder` | Zero/few-shot inference with a causal LLM |
| `peft` | Parameter-efficient fine-tuning (LoRA / QLoRA / DoRA) |

### Multi-label Classification

```bash
python run_multilabel_benchmark.py \
    --model_type <encoder|decoder|peft> \
    --model_name <hf-model-id> \
    [options]
```

**All arguments:**

| Argument | Default | Description |
|---|---|---|
| `--model_type` | required | `encoder`, `decoder`, or `peft` |
| `--model_name` | required | HuggingFace model ID or local path |
| `--data_dir` | `data/pubmed_mesh` | Directory with parquet splits |
| `--text_mode` | `title_abstract` | Input text: `title_abstract`, `abstract_only`, `title_only` |
| `--filter_geographicals` | `False` | Remove samples whose only label is Z (Geographicals) |
| `--num_epochs` | `3` | Training epochs (encoder/peft only) |
| `--batch_size` | `8` | Batch size |
| `--learning_rate` | `2e-5` | Learning rate |
| `--max_length` | `512` | Max token length |
| `--threshold` | `0.5` | Sigmoid threshold for encoder predictions |
| `--load_in_4bit` | `False` | 4-bit quantization (decoder/peft) |
| `--load_in_8bit` | `False` | 8-bit quantization (decoder/peft) |
| `--peft_method` | `lora` | `lora`, `qlora`, or `dora` (peft only) |
| `--lora_r` | `16` | LoRA rank |
| `--lora_alpha` | `32` | LoRA alpha scaling |
| `--few_shot` | `False` | Enable few-shot prompting (decoder only) |
| `--wandb_project` | `None` | W&B project name for logging |
| `--wandb_run_name` | `None` | W&B run name (auto-generated if unset) |

**Examples:**

```bash
# Fine-tune RoBERTa biomedical encoder
python run_multilabel_benchmark.py \
    --model_type encoder \
    --model_name PlanTL-GOB-ES/roberta-base-biomedical-es \
    --num_epochs 3 \
    --batch_size 8 \
    --learning_rate 2e-5

# Zero-shot Qwen with 4-bit quantization
python run_multilabel_benchmark.py \
    --model_type decoder \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --load_in_4bit

# Few-shot Qwen
python run_multilabel_benchmark.py \
    --model_type decoder \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --load_in_4bit \
    --few_shot

# QLoRA fine-tuning — abstract only, no Geographicals, W&B logging
python run_multilabel_benchmark.py \
    --model_type peft \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --load_in_4bit \
    --peft_method qlora \
    --text_mode abstract_only \
    --filter_geographicals \
    --num_epochs 3 \
    --wandb_project spanish-multilabel
```

---

### Named Entity Recognition

```bash
python run_ner_benchmark.py \
    --model_type <encoder|decoder|peft> \
    --model_name <hf-model-id> \
    [options]
```

**All arguments:**

| Argument | Default | Description |
|---|---|---|
| `--model_type` | required | `encoder`, `decoder`, or `peft` |
| `--model_name` | required | HuggingFace model ID or local path |
| `--data_dir` | `data/anat_em` | Directory with nersuite files and split lists |
| `--num_epochs` | `3` | Training epochs (encoder/peft only) |
| `--batch_size` | `8` | Batch size |
| `--learning_rate` | `2e-5` | Learning rate |
| `--max_length` | `512` | Max token length |
| `--load_in_4bit` | `False` | 4-bit quantization (decoder/peft) |
| `--load_in_8bit` | `False` | 8-bit quantization (decoder/peft) |
| `--peft_method` | `lora` | `lora`, `qlora`, or `dora` (peft only) |
| `--lora_r` | `16` | LoRA rank |
| `--lora_alpha` | `32` | LoRA alpha scaling |
| `--prompt_strategy` | `zero_shot` | `zero_shot`, `few_shot`, or `knn_few_shot` (decoder only) |
| `--self_verification` | `False` | Run a verification pass on extracted entities (+2–5% F1) |
| `--knn_k` | `5` | Number of k-NN examples to retrieve |
| `--wandb_project` | `None` | W&B project name for logging |
| `--wandb_run_name` | `None` | W&B run name (auto-generated if unset) |

**Examples:**

```bash
# Fine-tune BETO encoder
python run_ner_benchmark.py \
    --model_type encoder \
    --model_name dccuchile/bert-base-spanish-wwm-cased \
    --num_epochs 5 \
    --batch_size 16 \
    --learning_rate 3e-5

# Zero-shot Qwen with 4-bit quantization
python run_ner_benchmark.py \
    --model_type decoder \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --load_in_4bit

# k-NN few-shot with self-verification
python run_ner_benchmark.py \
    --model_type decoder \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --load_in_4bit \
    --prompt_strategy knn_few_shot \
    --knn_k 5 \
    --self_verification

# LoRA fine-tuning with W&B
python run_ner_benchmark.py \
    --model_type peft \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --load_in_4bit \
    --peft_method lora \
    --num_epochs 3 \
    --wandb_project spanish-ner
```

> **k-NN index:** When using `--prompt_strategy knn_few_shot`, the script automatically builds a FAISS index from the training set on first run and saves it to `data/anat_em/knn_index/` for reuse.

---

## Grid Search (Multi-label)

Use `ExperimentRunner` to run a systematic sweep across models, text modes, and filter options:

```python
from src.utils.experiment_runner import ExperimentRunner

runner = ExperimentRunner(
    data_dir="data/pubmed_mesh",
    output_base="results/multilabel",
    models=[
        "dccuchile/bert-base-spanish-wwm-cased",
        "PlanTL-GOB-ES/roberta-base-biomedical-es",
        "xlm-roberta-base",
    ],
    text_modes=["title_abstract", "abstract_only"],
    filter_geographicals_options=[False, True],
    num_epochs=3,
    batch_size=8,
    learning_rate=2e-5,
    wandb_project="spanish-multilabel",
)
results = runner.run()
runner.print_summary()
```

The configuration in `configs/experiment_config.yaml` documents the default grid:
- **Models:** BETO, RoBERTa-biomedical (base/large), XLM-RoBERTa
- **Text modes:** `title_abstract`, `abstract_only`, `title_only`
- **Filter geographicals:** `false`, `true`

---

## Hyperparameter Tuning

Bayesian hyperparameter search via Weights & Biases Sweeps:

```bash
python scripts/tune.py \
    --task multilabel \
    --model_name dccuchile/bert-base-spanish-wwm-cased \
    --data_dir data/pubmed_mesh \
    --wandb_project spanish-multilabel-tuning
```

The sweep searches over:
- `learning_rate`: log-uniform `[1e-5, 5e-4]`
- `batch_size`: `[8, 16, 32]`
- `num_epochs`: `[3, 5, 8]`
- `threshold`: `[0.3, 0.4, 0.5, 0.6]`

---

## Supported Models

### Encoders (fine-tuning)

| Model | HuggingFace ID |
|---|---|
| BETO | `dccuchile/bert-base-spanish-wwm-cased` |
| RoBERTa Biomedical (base) | `PlanTL-GOB-ES/roberta-base-biomedical-es` |
| RoBERTa Biomedical (large) | `PlanTL-GOB-ES/roberta-large-biomedical-es` |
| XLM-RoBERTa | `xlm-roberta-base` |
| mBERT | `bert-base-multilingual-cased` |

### Decoders (zero/few-shot or PEFT)

| Model | HuggingFace ID | Quantization |
|---|---|---|
| Qwen2.5 0.5B | `Qwen/Qwen2.5-0.5B-Instruct` | None |
| Qwen2.5 7B | `Qwen/Qwen2.5-7B-Instruct` | 4-bit |
| Llama 3.1 8B | `meta-llama/Llama-3.1-8B-Instruct` | 4-bit |
| Mistral 7B | `mistralai/Mistral-7B-Instruct-v0.3` | 4-bit |

---

## Output Structure

Each run writes to `results/{task}/{model_type}/{model_name}/`:

```
results/
└── multilabel/
│   └── encoder/
│       └── roberta-base-biomedical-es/
│           ├── predictions.parquet   # Predictions and gold labels
│           ├── metrics.json          # F1, precision, recall, hamming loss
│           └── checkpoint-*/         # Saved model checkpoints
└── ner/
    └── peft_lora/
        └── Llama-3.1-8B-Instruct/
            ├── predictions.parquet
            ├── metrics.json
            └── final/                # Final LoRA adapter weights
```

Grid search results are aggregated to:
- `results/multilabel/experiment_summary.json`
- `results/ner/experiment_summary.json`

---

## Configuration Files

| File | Purpose |
|---|---|
| `configs/experiment_config.yaml` | Default grid for multilabel ExperimentRunner |
| `configs/ner_config.yaml` | Encoder/PEFT training hyperparameters and decoder inference settings for NER |
| `configs/models.yaml` | Full model registry with LoRA and quantization defaults |

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `torch` | >=2.0.0 | Deep learning backend |
| `transformers` | >=4.40.0 | Model loading and training |
| `datasets` | >=2.18.0 | Data loading |
| `peft` | >=0.10.0 | LoRA / QLoRA / DoRA |
| `bitsandbytes` | >=0.43.0 | 4-bit / 8-bit quantization |
| `accelerate` | >=0.29.0 | Distributed and mixed-precision training |
| `seqeval` | >=1.2.2 | Entity-level NER metrics |
| `scikit-learn` | >=1.3.0 | Multilabel metrics |
| `sentence-transformers` | >=3.0.0 | Embeddings for k-NN retrieval |
| `faiss-cpu` | >=1.8.0 | k-NN index |
| `wandb` | >=0.17.0 | Experiment tracking |
| `openai` | >=1.30.0 | Optional OpenAI / Azure API support |
