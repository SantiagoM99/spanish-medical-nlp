import os
import gc
import json
import numpy as np
import pandas as pd
import torch
import evaluate
from glob import glob
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)

# ============================================================
# GPU config
# ============================================================
# No sobreescribir CUDA_VISIBLE_DEVICES: el scheduler del cluster (SLURM)
# ya asigna la GPU correcta. Solo setear si no viene del entorno.
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
torch.backends.cuda.matmul.allow_tf32 = True
device = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs("results", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ============================================================
# Lectura del formato .nersuite
# ============================================================
# Mapa de normalización para etiquetas ruidosas en anat_em
_TAG_NORM = {
    "o": "O",                           # lowercase typo
    "=": "O",                           # annotation artifact
    "B-Multi_tissue_structure": "B-Multi-tissue_structure",  # underscore vs hyphen
    "I-Multi_tissue_structure": "I-Multi-tissue_structure",  # (precaución)
}
# Solo '.' es límite de oración en este corpus biomédico.
# '!' es inexistente, '?' ultra-raro (2 casos en >600 archivos) y ';'
# se usa dentro de referencias parentéticas — ninguno es límite real.
# Verificado empíricamente: ningún archivo tiene >512 tokens entre puntos,
# por lo que el corte duro de 512 es solo seguro de emergencia.
_SENT_END = {"."}


def _normalize_tag(tag: str) -> str:
    return _TAG_NORM.get(tag, tag)


def _split_into_sentences(tokens, tags, max_len=512):
    """
    Divide una secuencia de tokens en oraciones usando '.' como límite.

    Regla: se emite una oración cuando se encuentra un '.' y la oración
    tiene al menos 5 tokens (para no cortar en puntos de abreviaturas sueltos
    al inicio de un segmento).

    El límite duro de max_len=512 es solo de emergencia: en anat_em ningún
    archivo tiene un tramo >512 tokens entre puntos, por lo que en la
    práctica nunca se activa.
    """
    all_sents, all_labels = [], []
    curr, curr_lbl = [], []

    for tok, tag in zip(tokens, tags):
        curr.append(tok)
        curr_lbl.append(tag)

        end_of_sent = (tok in _SENT_END and len(curr) >= 5)
        too_long    = (len(curr) >= max_len)

        if end_of_sent or too_long:
            all_sents.append(curr[:])
            all_labels.append(curr_lbl[:])
            curr, curr_lbl = [], []

    if curr:
        all_sents.append(curr)
        all_labels.append(curr_lbl)

    return all_sents, all_labels


def read_nersuite_folder(folder_path):
    """
    Lee todos los archivos .nersuite de una carpeta.
    Formato por línea: word<TAB>tag
    Los archivos sin líneas en blanco (99 % del dataset) se dividen en
    oraciones usando _split_into_sentences para evitar truncación masiva.
    """
    sentences, labels = [], []
    files = sorted(glob(os.path.join(folder_path, "*.nersuite")))
    if not files:
        files = sorted(f for f in glob(os.path.join(folder_path, "*")) if os.path.isfile(f))

    print(f"  {folder_path}: {len(files)} archivos")

    for fpath in files:
        tokens, tags = [], []
        with open(fpath, encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.rstrip("\r\n")
                if not line.strip():
                    if tokens:
                        sents, lbls = _split_into_sentences(tokens, tags)
                        sentences.extend(sents)
                        labels.extend(lbls)
                        tokens, tags = [], []
                else:
                    parts = line.split("\t")
                    if len(parts) >= 2:
                        word = parts[0].strip()
                        tag  = parts[1].strip()
                    else:
                        parts = line.split()
                        if len(parts) >= 2:
                            word = parts[0].strip()
                            tag  = parts[1].strip()
                        else:
                            continue

                    tag = _normalize_tag(tag) if tag else "O"
                    if word:
                        tokens.append(word)
                        tags.append(tag)

        if tokens:
            sents, lbls = _split_into_sentences(tokens, tags)
            sentences.extend(sents)
            labels.extend(lbls)

    print(f"  → {len(sentences)} oraciones")
    if sentences:
        lens = [len(s) for s in sentences]
        print(f"  → tokens/oración: min={min(lens)}, max={max(lens)}, "
              f"median={int(np.median(lens))}, mean={np.mean(lens):.1f}")

    return pd.DataFrame({"tokens": sentences, "ner_tags": labels})


MAX_LEN = 512


# ============================================================
# Carga de datos
# ============================================================
def load_anat_em(data_dir="data/anat_em"):
    print("Cargando dataset anat_em...")
    train_df = read_nersuite_folder(os.path.join(data_dir, "train"))
    val_df   = read_nersuite_folder(os.path.join(data_dir, "devel"))
    test_df  = read_nersuite_folder(os.path.join(data_dir, "test"))

    # Recoger etiquetas de TODOS los splits para evitar KeyError en eval
    all_tags = set()
    for df in (train_df, val_df, test_df):
        for tags in df["ner_tags"]:
            all_tags.update(tags)

    unique_tags = sorted(all_tags)
    label_to_id = {tag: i for i, tag in enumerate(unique_tags)}
    id_to_label = {i: tag for tag, i in label_to_id.items()}
    print(f"Etiquetas ({len(unique_tags)}): {unique_tags}")

    dataset = DatasetDict({
        "train":      Dataset.from_pandas(train_df, preserve_index=False),
        "validation": Dataset.from_pandas(val_df,   preserve_index=False),
        "test":       Dataset.from_pandas(test_df,  preserve_index=False),
    })

    def tags_to_ids(example):
        return {"ner_tags": [label_to_id[t] for t in example["ner_tags"]]}

    for split in dataset:
        dataset[split] = dataset[split].map(tags_to_ids)

    return dataset, unique_tags, label_to_id, id_to_label


# ============================================================
# Tokenización y alineación de labels
# ============================================================
def tokenize_and_align(batch, tokenizer, label_all_tokens=False):
    tok = tokenizer(
        batch["tokens"],
        is_split_into_words=True,
        truncation=True,
        padding=False,
        max_length=MAX_LEN,
    )
    labels = []
    for i in range(len(batch["tokens"])):
        word_ids = tok.word_ids(batch_index=i)
        prev = None
        lab_seq = []
        for wid in word_ids:
            if wid is None:
                lab_seq.append(-100)
            elif wid != prev:
                lab_seq.append(batch["ner_tags"][i][wid])
            else:
                lab_seq.append(batch["ner_tags"][i][wid] if label_all_tokens else -100)
            prev = wid
        labels.append(lab_seq)
    tok["labels"] = labels
    return tok


# ============================================================
# Métricas — con protección contra tags vacíos para seqeval
# ============================================================
def make_compute_metrics(id_to_label, metric):
    """
    Crea compute_metrics con closure.
    Protege contra:
      - tags vacíos ("") que causan IndexError en seqeval
      - secuencias vacías después de filtrar -100
    """
    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_labels = []
        true_preds  = []

        for pred_seq, label_seq in zip(predictions, labels):
            t_labels = []
            t_preds  = []
            for pr, lb in zip(pred_seq, label_seq):
                if lb == -100:
                    continue
                tag_true = id_to_label.get(int(lb), "O")
                tag_pred = id_to_label.get(int(pr), "O")
                # Protección contra string vacío → seqeval crash
                if not tag_true:
                    tag_true = "O"
                if not tag_pred:
                    tag_pred = "O"
                t_labels.append(tag_true)
                t_preds.append(tag_pred)

            if t_labels:
                true_labels.append(t_labels)
                true_preds.append(t_preds)

        if not true_labels:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0}

        results = metric.compute(predictions=true_preds, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall":    results["overall_recall"],
            "f1":        results["overall_f1"],
            "accuracy":  results["overall_accuracy"],
        }
    return compute_metrics


# ============================================================
# Entrenamiento BETO
# ============================================================
def train_beto(batchs, lrs, epochs, weight_decay, warmup, lr_decay):
    all_results = []
    metric = evaluate.load("seqeval")

    dataset, unique_tags, label_to_id, id_to_label = load_anat_em()

    model_checkpoint = "dccuchile/bert-base-spanish-wwm-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    tokenized = dataset.map(
        lambda ex: tokenize_and_align(ex, tokenizer, label_all_tokens=False),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )
    data_collator = DataCollatorForTokenClassification(tokenizer)
    compute_metrics = make_compute_metrics(id_to_label, metric)

    for batch in batchs:
        for lr in lrs:
            print(f"\n{'='*60}")
            print(f"BETO — lr={lr}, batch={batch}")
            print(f"{'='*60}")

            out_dir = f"models/anat_em_beto_bs{batch}_lr{lr}".replace(".", "p")

            model = AutoModelForTokenClassification.from_pretrained(
                model_checkpoint,
                num_labels=len(unique_tags),
                id2label=id_to_label,
                label2id=label_to_id,
            )

            args = TrainingArguments(
                output_dir=out_dir,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="eval_f1",
                greater_is_better=True,
                learning_rate=lr,
                per_device_train_batch_size=batch,
                per_device_eval_batch_size=batch,
                num_train_epochs=epochs,
                weight_decay=weight_decay,
                logging_dir="./logs",
                logging_steps=50,
                save_total_limit=2,
                seed=444,
                fp16=True,
                warmup_ratio=warmup,
                lr_scheduler_type=lr_decay,
                report_to=["none"],
            )

            trainer = Trainer(
                model=model,
                args=args,
                train_dataset=tokenized["train"],
                eval_dataset=tokenized["validation"],
                tokenizer=tokenizer,
                data_collator=data_collator,
                compute_metrics=compute_metrics,
            )

            trainer.train()

            best_dir = os.path.join(out_dir, "best")
            os.makedirs(best_dir, exist_ok=True)
            trainer.save_model(best_dir)
            tokenizer.save_pretrained(best_dir)

            dev_metrics  = trainer.evaluate(eval_dataset=tokenized["validation"])
            test_metrics = trainer.evaluate(eval_dataset=tokenized["test"])

            result = {
                "model_name": "beto",
                "batch_size": batch,
                "learning_rate": lr,
                "epochs": epochs,
                "weight_decay": weight_decay,
                "warmup_ratio": warmup,
                "lr_scheduler": lr_decay,
                "best_checkpoint": trainer.state.best_model_checkpoint,
                "val_accuracy":  float(dev_metrics.get("eval_accuracy", np.nan)),
                "val_f1":        float(dev_metrics.get("eval_f1", np.nan)),
                "val_precision": float(dev_metrics.get("eval_precision", np.nan)),
                "val_recall":    float(dev_metrics.get("eval_recall", np.nan)),
                "test_accuracy":  float(test_metrics.get("eval_accuracy", np.nan)),
                "test_f1":        float(test_metrics.get("eval_f1", np.nan)),
                "test_precision": float(test_metrics.get("eval_precision", np.nan)),
                "test_recall":    float(test_metrics.get("eval_recall", np.nan)),
            }

            all_results[:] = sorted(
                all_results + [result], key=lambda r: r["test_f1"], reverse=True
            )

            with open("results/anat_em_beto.json", "w", encoding="utf-8") as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)

            print(f"  VAL  F1={result['val_f1']:.4f}  P={result['val_precision']:.4f}  R={result['val_recall']:.4f}")
            print(f"  TEST F1={result['test_f1']:.4f}  P={result['test_precision']:.4f}  R={result['test_recall']:.4f}")

            del model, trainer
            gc.collect()
            torch.cuda.empty_cache()

    return all_results


# ============================================================
# Entrenamiento RoBERTas
# ============================================================
def train_robertas(models, batchs, lrs, epochs, weight_decay, warmup, lr_decay):
    all_results = []
    metric = evaluate.load("seqeval")

    LARGE_MODELS = {"Flaglab/SciBETO-large", "FacebookAI/xlm-roberta-large"}

    for name_model in models:
        print(f"\n{'#'*60}")
        print(f"Modelo: {name_model}")
        print(f"{'#'*60}")

        dataset, unique_tags, label_to_id, id_to_label = load_anat_em()

        # XLM-RoBERTa usa SentencePiece, no BPE → no soporta add_prefix_space
        # ni RobertaTokenizerFast; usamos AutoTokenizer para todos los modelos.
        xlm_model = "xlm-roberta" in name_model.lower()
        tok_kwargs = {} if xlm_model else {"add_prefix_space": True}
        tokenizer = AutoTokenizer.from_pretrained(name_model, **tok_kwargs)

        tokenized = dataset.map(
            lambda ex: tokenize_and_align(ex, tokenizer, label_all_tokens=False),
            batched=True,
            remove_columns=dataset["train"].column_names,
        )
        data_collator = DataCollatorForTokenClassification(tokenizer)
        compute_metrics = make_compute_metrics(id_to_label, metric)

        for batch in batchs:
            for lr in lrs:
                print(f"\n{'='*60}")
                print(f"{name_model} — lr={lr}, batch={batch}")
                print(f"{'='*60}")

                safe_name = name_model.replace("/", "_").replace(".", "p")
                out_dir = f"models/anat_em_{safe_name}_bs{batch}_lr{lr}".replace(".", "p")

                model = AutoModelForTokenClassification.from_pretrained(
                    name_model,
                    num_labels=len(unique_tags),
                    id2label=id_to_label,
                    label2id=label_to_id,
                )

                if name_model in LARGE_MODELS:
                    per_device_bs = 8
                    grad_acc = max(1, batch // per_device_bs)
                else:
                    per_device_bs = batch
                    grad_acc = 1

                args = TrainingArguments(
                    output_dir=out_dir,
                    evaluation_strategy="epoch",
                    save_strategy="epoch",
                    load_best_model_at_end=True,
                    metric_for_best_model="eval_f1",
                    greater_is_better=True,
                    learning_rate=lr,
                    per_device_train_batch_size=per_device_bs,
                    gradient_accumulation_steps=grad_acc,
                    per_device_eval_batch_size=min(per_device_bs, 16),
                    eval_accumulation_steps=64,
                    num_train_epochs=epochs,
                    weight_decay=weight_decay,
                    logging_dir="./logs",
                    logging_steps=50,
                    save_total_limit=2,
                    seed=444,
                    fp16=True,
                    warmup_ratio=warmup,
                    lr_scheduler_type=lr_decay,
                    report_to=["none"],
                )

                trainer = Trainer(
                    model=model,
                    args=args,
                    train_dataset=tokenized["train"],
                    eval_dataset=tokenized["validation"],
                    tokenizer=tokenizer,
                    data_collator=data_collator,
                    compute_metrics=compute_metrics,
                )

                trainer.train()

                best_dir = os.path.join(out_dir, "best")
                os.makedirs(best_dir, exist_ok=True)
                trainer.save_model(best_dir)
                tokenizer.save_pretrained(best_dir)

                dev_metrics  = trainer.evaluate(eval_dataset=tokenized["validation"])
                test_metrics = trainer.evaluate(eval_dataset=tokenized["test"])

                result = {
                    "model_name": name_model,
                    "batch_size": batch,
                    "learning_rate": lr,
                    "epochs": epochs,
                    "weight_decay": weight_decay,
                    "warmup_ratio": warmup,
                    "lr_scheduler": lr_decay,
                    "best_checkpoint": trainer.state.best_model_checkpoint,
                    "val_accuracy":  float(dev_metrics.get("eval_accuracy", np.nan)),
                    "val_f1":        float(dev_metrics.get("eval_f1", np.nan)),
                    "val_precision": float(dev_metrics.get("eval_precision", np.nan)),
                    "val_recall":    float(dev_metrics.get("eval_recall", np.nan)),
                    "test_accuracy":  float(test_metrics.get("eval_accuracy", np.nan)),
                    "test_f1":        float(test_metrics.get("eval_f1", np.nan)),
                    "test_precision": float(test_metrics.get("eval_precision", np.nan)),
                    "test_recall":    float(test_metrics.get("eval_recall", np.nan)),
                }

                all_results[:] = sorted(
                    all_results + [result], key=lambda r: r["test_f1"], reverse=True
                )

                with open("results/anat_em_robertas_sci.json", "w", encoding="utf-8") as f:
                    json.dump(all_results, f, ensure_ascii=False, indent=2)

                print(f"  VAL  F1={result['val_f1']:.4f}  P={result['val_precision']:.4f}  R={result['val_recall']:.4f}")
                print(f"  TEST F1={result['test_f1']:.4f}  P={result['test_precision']:.4f}  R={result['test_recall']:.4f}")

                del model, trainer
                gc.collect()
                torch.cuda.empty_cache()

    return all_results




batchs = [16, 32]
lrs = [1e-5, 2e-5, 3e-5]
epochs = 10
weight_decay = 0.1
warmup = 0.06
lr_decay = "linear"

print("\n" + "█" * 60)
print("  ENTRENANDO BETO — anat_em NER")
print("█" * 60)
# results_beto = train_beto(batchs, lrs, epochs, weight_decay, warmup, lr_decay)

roberta_models = [
    # "Flaglab/SciBETO-base",
    # "bertin-project/bertin-roberta-base-spanish",
    "Flaglab/SciBETO-large",
    # "FacebookAI/xlm-roberta-base",
    # "FacebookAI/xlm-roberta-large"
]

print("\n" + "█" * 60)
print("  ENTRENANDO ROBERTAS — anat_em NER")
print("█" * 60)
results_roberta = train_robertas(
    roberta_models, batchs, lrs, epochs, weight_decay, warmup, lr_decay
)

#print("\n" + "=" * 60)
#print("MEJORES RESULTADOS BETO:")
#for r in results_beto[:3]:
#    print(f"  lr={r['learning_rate']} bs={r['batch_size']}  "
#          f"F1={r['test_f1']:.4f}  P={r['test_precision']:.4f}  R={r['test_recall']:.4f}")

print("\nMEJORES RESULTADOS ROBERTAS:")
for r in results_roberta[:5]:
    print(f"  {r['model_name']:40s} lr={r['learning_rate']} bs={r['batch_size']}  "
          f"F1={r['test_f1']:.4f}  P={r['test_precision']:.4f}  R={r['test_recall']:.4f}")