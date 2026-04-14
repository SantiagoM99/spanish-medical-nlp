import os
import gc
import json
import ast
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
from datasets import Dataset, DatasetDict, Sequence, Value
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)
from scipy.special import expit

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
torch.backends.cuda.matmul.allow_tf32 = True
device = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs("results", exist_ok=True)
os.makedirs("models", exist_ok=True)

MAX_LEN = 512
THRESH = 0.5


def to_list_labels(x):
    """Parsea level1_codes desde string de lista o valor suelto."""
    if isinstance(x, (list, np.ndarray)):
        return [str(i) for i in x]
    if pd.isna(x):
        return []
    if isinstance(x, str):
        try:
            parsed = ast.literal_eval(x)
            if isinstance(parsed, (list, tuple, set)):
                return [str(i) for i in parsed]
        except (ValueError, SyntaxError):
            pass
        return [x.strip()]
    return [str(x)]


def load_and_prepare_data(data_dir="data/pubmed_mesh"):
    """Carga los CSV, crea input_text y parsea labels."""
    df_train = pd.read_csv(os.path.join(data_dir, "train_multilabel.csv"))
    df_dev   = pd.read_csv(os.path.join(data_dir, "val_multilabel.csv"))
    df_test  = pd.read_csv(os.path.join(data_dir, "test_multilabel.csv"))

    for df in (df_train, df_dev, df_test):
        # Crear input_text = title + abstract
        df["input_text"] = (
            df["title"].fillna("") + " " + df["spanish_abstract"].fillna("")
        ).str.strip()
        # Parsear labels
        df["labels"] = df["level1_codes"].apply(to_list_labels)

    # Quitar filas sin texto o sin etiquetas
    clean = []
    for name, df in (("train", df_train), ("dev", df_dev), ("test", df_test)):
        n0 = len(df)
        df = df.dropna(subset=["input_text"])
        df = df[df["input_text"].str.len() > 0]
        df = df[df["labels"].map(len) > 0]
        print(f"{name}: {n0} → {len(df)} (filas válidas)")
        clean.append(df)

    df_train, df_dev, df_test = clean

    # MultiLabelBinarizer (fit solo en train)
    mlb = MultiLabelBinarizer()
    Y_train = mlb.fit_transform(df_train["labels"])
    Y_dev   = mlb.transform(df_dev["labels"])
    Y_test  = mlb.transform(df_test["labels"])

    id2label = {i: lab for i, lab in enumerate(mlb.classes_)}
    label2id = {lab: i for i, lab in id2label.items()}
    num_labels = len(mlb.classes_)
    print(f"Número de etiquetas: {num_labels} → {list(mlb.classes_)}")

    # Columna target como float32
    df_train = df_train.assign(target=[row.astype("float32").tolist() for row in Y_train])
    df_dev   = df_dev.assign(target=[row.astype("float32").tolist() for row in Y_dev])
    df_test  = df_test.assign(target=[row.astype("float32").tolist() for row in Y_test])

    return df_train, df_dev, df_test, id2label, label2id, num_labels


def build_datasets(df_train, df_dev, df_test, tokenizer):
    """Tokeniza y crea DatasetDict listo para Trainer."""
    cols_keep = ["input_text", "target"]
    ds = DatasetDict({
        "train":      Dataset.from_pandas(df_train[cols_keep], preserve_index=False),
        "validation": Dataset.from_pandas(df_dev[cols_keep],   preserve_index=False),
        "test":       Dataset.from_pandas(df_test[cols_keep],  preserve_index=False),
    })

    def tok_fn(batch):
        return tokenizer(batch["input_text"], truncation=True, max_length=MAX_LEN)

    ds_tok = ds.map(tok_fn, batched=True, remove_columns=["input_text"])

    for split in ds_tok:
        ds_tok[split] = ds_tok[split].add_column("labels", ds[split]["target"])
        ds_tok[split] = ds_tok[split].remove_columns(["target"])

    ds_tok = ds_tok.cast_column("labels", Sequence(Value("float32")))
    return ds_tok


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    labels = (np.asarray(labels) > 0.5).astype(int)
    probs  = expit(logits)
    preds  = (probs >= THRESH).astype(int)

    f1_micro    = f1_score(labels, preds, average="micro", zero_division=0)
    f1_macro    = f1_score(labels, preds, average="macro", zero_division=0)
    exact_match = (preds == labels).all(axis=1).mean().item()

    return {"f1_micro": f1_micro, "f1_macro": f1_macro, "exact_match": exact_match}


def train_beto(batchs, lrs, epochs, weight_decay, warmup, lr_decay):
    all_results = []

    df_train, df_dev, df_test, id2label, label2id, num_labels = load_and_prepare_data()

    tokenizer = AutoTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")
    ds_tok = build_datasets(df_train, df_dev, df_test, tokenizer)
    data_collator = DataCollatorWithPadding(tokenizer, return_tensors="pt", pad_to_multiple_of=8)

    for batch in batchs:
        for lr in lrs:
            print(f"\n{'='*60}")
            print(f"BETO — lr={lr}, batch={batch}")
            print(f"{'='*60}")

            out_dir = f"models/pubmed_beto_bs{batch}_lr{lr}".replace(".", "p")

            config = AutoConfig.from_pretrained(
                "dccuchile/bert-base-spanish-wwm-cased",
                num_labels=num_labels,
                id2label=id2label,
                label2id=label2id,
                problem_type="multi_label_classification",
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                "dccuchile/bert-base-spanish-wwm-cased", config=config
            )
            model.gradient_checkpointing_enable()
            model.config.use_cache = False
            model.to(device)

            args = TrainingArguments(
                output_dir=out_dir,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="f1_micro",
                greater_is_better=True,
                learning_rate=lr,
                per_device_train_batch_size=batch,
                per_device_eval_batch_size=batch,
                num_train_epochs=epochs,
                weight_decay=weight_decay,
                logging_dir="./logs",
                logging_steps=50,
                save_total_limit=1,
                seed=444,
                fp16=True,
                warmup_ratio=warmup,
                lr_scheduler_type=lr_decay,
                report_to=["none"],
            )

            trainer = Trainer(
                model=model,
                args=args,
                train_dataset=ds_tok["train"],
                eval_dataset=ds_tok["validation"],
                tokenizer=tokenizer,
                data_collator=data_collator,
                compute_metrics=compute_metrics,
            )

            trainer.train()

            # Guardar mejor modelo
            best_dir = os.path.join(out_dir, "best")
            os.makedirs(best_dir, exist_ok=True)
            trainer.save_model(best_dir)
            tokenizer.save_pretrained(best_dir)

            # Evaluar
            dev_metrics  = trainer.evaluate(eval_dataset=ds_tok["validation"])
            test_metrics = trainer.evaluate(eval_dataset=ds_tok["test"])

            result = {
                "model_name": "beto",
                "batch_size": batch,
                "learning_rate": lr,
                "epochs": epochs,
                "weight_decay": weight_decay,
                "warmup_ratio": warmup,
                "lr_scheduler": lr_decay,
                "best_checkpoint": trainer.state.best_model_checkpoint,
                "val_exact_match": float(dev_metrics.get("eval_exact_match", np.nan)),
                "val_f1_micro":    float(dev_metrics.get("eval_f1_micro", np.nan)),
                "val_f1_macro":    float(dev_metrics.get("eval_f1_macro", np.nan)),
                "test_exact_match": float(test_metrics.get("eval_exact_match", np.nan)),
                "test_f1_micro":    float(test_metrics.get("eval_f1_micro", np.nan)),
                "test_f1_macro":    float(test_metrics.get("eval_f1_macro", np.nan)),
            }

            all_results[:] = sorted(
                all_results + [result],
                key=lambda r: r["test_f1_micro"],
                reverse=True,
            )

            json_path = "results/pubmed_beto.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)

            print(f"  VAL  F1-micro={result['val_f1_micro']:.4f}  F1-macro={result['val_f1_macro']:.4f}")
            print(f"  TEST F1-micro={result['test_f1_micro']:.4f}  F1-macro={result['test_f1_macro']:.4f}")

            del model, trainer
            gc.collect()
            torch.cuda.empty_cache()

    return all_results


# ============================================================
# Entrenamiento de RoBERTas
# ============================================================
def train_robertas(models, batchs, lrs, epochs, weight_decay, warmup, lr_decay):
    all_results = []

    # Modelos large que necesitan batch más pequeño
    LARGE_MODELS = {"sci-roberta-large", "FacebookAI/xlm-roberta-large"}

    for name_model in models:
        print(f"\n{'#'*60}")
        print(f"Modelo: {name_model}")
        print(f"{'#'*60}")

        df_train, df_dev, df_test, id2label, label2id, num_labels = load_and_prepare_data()

        tokenizer = AutoTokenizer.from_pretrained(name_model)
        ds_tok = build_datasets(df_train, df_dev, df_test, tokenizer)
        data_collator = DataCollatorWithPadding(tokenizer, return_tensors="pt", pad_to_multiple_of=8)

        for batch in batchs:
            for lr in lrs:
                print(f"\n{'='*60}")
                print(f"{name_model} — lr={lr}, batch={batch}")
                print(f"{'='*60}")

                safe_name = name_model.replace("/", "_").replace(".", "p")
                out_dir = f"models/pubmed_{safe_name}_bs{batch}_lr{lr}".replace(".", "p")

                config = AutoConfig.from_pretrained(
                    name_model,
                    num_labels=num_labels,
                    id2label=id2label,
                    label2id=label2id,
                    problem_type="multi_label_classification",
                )
                model = AutoModelForSequenceClassification.from_pretrained(
                    name_model, config=config
                )
                model.gradient_checkpointing_enable()
                model.config.use_cache = False
                model.to(device)

                # Ajustar batch real y gradient accumulation para modelos large
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
                    metric_for_best_model="f1_micro",
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
                    save_total_limit=1,
                    seed=444,
                    fp16=True,
                    warmup_ratio=warmup,
                    lr_scheduler_type=lr_decay,
                    report_to=["none"],
                )

                trainer = Trainer(
                    model=model,
                    args=args,
                    train_dataset=ds_tok["train"],
                    eval_dataset=ds_tok["validation"],
                    tokenizer=tokenizer,
                    data_collator=data_collator,
                    compute_metrics=compute_metrics,
                )

                trainer.train()

                # Guardar mejor modelo
                best_dir = os.path.join(out_dir, "best")
                os.makedirs(best_dir, exist_ok=True)
                trainer.save_model(best_dir)
                tokenizer.save_pretrained(best_dir)

                # Evaluar
                dev_metrics  = trainer.evaluate(eval_dataset=ds_tok["validation"])
                test_metrics = trainer.evaluate(eval_dataset=ds_tok["test"])

                result = {
                    "model_name": name_model,
                    "batch_size": batch,
                    "learning_rate": lr,
                    "epochs": epochs,
                    "weight_decay": weight_decay,
                    "warmup_ratio": warmup,
                    "lr_scheduler": lr_decay,
                    "best_checkpoint": trainer.state.best_model_checkpoint,
                    "val_exact_match": float(dev_metrics.get("eval_exact_match", np.nan)),
                    "val_f1_micro":    float(dev_metrics.get("eval_f1_micro", np.nan)),
                    "val_f1_macro":    float(dev_metrics.get("eval_f1_macro", np.nan)),
                    "test_exact_match": float(test_metrics.get("eval_exact_match", np.nan)),
                    "test_f1_micro":    float(test_metrics.get("eval_f1_micro", np.nan)),
                    "test_f1_macro":    float(test_metrics.get("eval_f1_macro", np.nan)),
                }

                all_results[:] = sorted(
                    all_results + [result],
                    key=lambda r: r["test_f1_micro"],
                    reverse=True,
                )

                json_path = "results/pubmed_robertas_2.json"
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(all_results, f, ensure_ascii=False, indent=2)

                print(f"  VAL  F1-micro={result['val_f1_micro']:.4f}  F1-macro={result['val_f1_macro']:.4f}")
                print(f"  TEST F1-micro={result['test_f1_micro']:.4f}  F1-macro={result['test_f1_macro']:.4f}")

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

# 1) Entrenar BETO
print("\n" + "█" * 60)
print("  ENTRENANDO BETO")
print("█" * 60)
# results_beto = train_beto(batchs, lrs, epochs, weight_decay, warmup, lr_decay)

# 2) Entrenar RoBERTas
roberta_models = [
    # "Flaglab/SciBETO-large",
    "Flaglab/SciBETO-base",
    "bertin-project/bertin-roberta-base-spanish",
    "FacebookAI/xlm-roberta-base"
    "FacebookAI/xlm-roberta-large"
    # "continue_roberta_base/checkpoint-55000",
]

print("\n" + "█" * 60)
print("  ENTRENANDO ROBERTAS")
print("█" * 60)
results_roberta = train_robertas(
    roberta_models, batchs, lrs, epochs, weight_decay, warmup, lr_decay
)

# Resumen final
# print("\n" + "=" * 60)
# print("MEJORES RESULTADOS BETO:")
# print("=" * 60)
# for r in results_beto[:3]:
#     print(f"  lr={r['learning_rate']} bs={r['batch_size']}  "
#           f"test_F1m={r['test_f1_micro']:.4f}  test_F1M={r['test_f1_macro']:.4f}")

print("\nMEJORES RESULTADOS ROBERTAS:")
print("=" * 60)
for r in results_roberta[:5]:
    print(f"  {r['model_name']:40s} lr={r['learning_rate']} bs={r['batch_size']}  "
          f"test_F1m={r['test_f1_micro']:.4f}  test_F1M={r['test_f1_macro']:.4f}")