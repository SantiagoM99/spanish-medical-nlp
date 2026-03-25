"""
tune.py

Bayesian hyperparameter search via W&B Sweeps para clasificación multilabel.

Uso:
  1. Configura WANDB_PROJECT en el entorno o en sweep_config abajo.
  2. Lanza el sweep:
       python scripts/tune.py --task multilabel --model_name dccuchile/bert-base-spanish-wwm-cased
  3. Opcionalmente arranca agentes adicionales:
       wandb agent <sweep_id>

El sweep busca el learning_rate, batch_size, y num_epochs óptimos.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ------------------------------------------------------------------
# Sweep configs
# ------------------------------------------------------------------

MULTILABEL_SWEEP_CONFIG = {
    "method": "bayes",
    "metric": {"name": "f1_micro", "goal": "maximize"},
    "parameters": {
        "learning_rate": {
            "distribution": "log_uniform_values",
            "min": 1e-5,
            "max": 5e-4,
        },
        "batch_size": {"values": [8, 16, 32]},
        "num_epochs": {"values": [3, 5, 8]},
        "threshold": {"values": [0.3, 0.4, 0.5, 0.6]},
        "text_mode": {"values": ["title_abstract", "abstract_only"]},
    },
    "early_terminate": {
        "type": "hyperband",
        "min_iter": 2,
    },
}

NER_SWEEP_CONFIG = {
    "method": "bayes",
    "metric": {"name": "f1", "goal": "maximize"},
    "parameters": {
        "learning_rate": {
            "distribution": "log_uniform_values",
            "min": 5e-6,
            "max": 1e-4,
        },
        "batch_size": {"values": [4, 8, 16]},
        "num_epochs": {"values": [3, 5, 10]},
    },
    "early_terminate": {
        "type": "hyperband",
        "min_iter": 2,
    },
}


# ------------------------------------------------------------------
# Train functions (llamadas por el agente W&B)
# ------------------------------------------------------------------

def train_multilabel(config, args):
    """Un trial de fine-tuning multilabel."""
    from utils.multilabel_datareader import MultiLabelDataset
    from models.encoder_multilabel import EncoderMultiLabelModel
    from evaluation.multilabel_predictor import MultiLabelPredictor
    import wandb

    dataset = MultiLabelDataset(
        data_dir=args.data_dir,
        text_mode=config.text_mode,
        filter_geographicals=args.filter_geographicals,
    )

    output_dir = f"results/tune/multilabel/{args.model_name.split('/')[-1]}"

    model = EncoderMultiLabelModel(
        model_name=args.model_name,
        label2id=dataset.label2id,
        id2label=dataset.id2label,
        max_length=args.max_length,
        threshold=config.threshold,
    )

    train_texts, train_labels = dataset.get_texts_and_labels("train")
    dev_texts, dev_labels = dataset.get_texts_and_labels("dev")

    model.train(
        train_texts=train_texts,
        train_labels=train_labels,
        dev_texts=dev_texts,
        dev_labels=dev_labels,
        output_dir=output_dir,
        num_epochs=config.num_epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
    )

    predictor = MultiLabelPredictor(model, dataset)
    metrics = predictor.compute_metrics(split="dev")
    wandb.log(metrics)


def train_ner(config, args):
    """Un trial de fine-tuning NER."""
    from utils.ner_datareader import NERDataset
    from models.encoder_ner import EncoderNERModel
    from evaluation.ner_predictor import NERPredictor
    import wandb

    dataset = NERDataset(data_dir=args.data_dir)
    output_dir = f"results/tune/ner/{args.model_name.split('/')[-1]}"

    model = EncoderNERModel(
        model_name=args.model_name,
        label2id=dataset.label2id,
        id2label=dataset.id2label,
        max_length=args.max_length,
    )

    train_tokens, train_labels = dataset.get_tokens_and_labels("train")
    dev_tokens, dev_labels = dataset.get_tokens_and_labels("dev")

    model.train(
        train_sentences=train_tokens,
        train_labels=train_labels,
        dev_sentences=dev_tokens,
        dev_labels=dev_labels,
        output_dir=output_dir,
        num_epochs=config.num_epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
    )

    predictor = NERPredictor(model, dataset)
    metrics = predictor.compute_metrics(split="dev")
    wandb.log(metrics)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="W&B Bayesian Hyperparameter Sweep")
    parser.add_argument(
        "--task",
        choices=["multilabel", "ner"],
        required=True,
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Data directory (defaults to data/pubmed_mesh or data/anat_em)",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=os.getenv("WANDB_PROJECT", "spanish-medical-nlp-tune"),
    )
    parser.add_argument("--count", type=int, default=20, help="Number of sweep trials")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--filter_geographicals", action="store_true")
    args = parser.parse_args()

    if args.data_dir is None:
        args.data_dir = "data/pubmed_mesh" if args.task == "multilabel" else "data/anat_em"

    try:
        import wandb
    except ImportError:
        print("ERROR: wandb no instalado. Ejecuta: pip install wandb")
        sys.exit(1)

    sweep_config = MULTILABEL_SWEEP_CONFIG if args.task == "multilabel" else NER_SWEEP_CONFIG
    sweep_config["name"] = f"{args.task}_{args.model_name.split('/')[-1]}"

    sweep_id = wandb.sweep(sweep_config, project=args.wandb_project)
    print(f"Sweep ID: {sweep_id}")
    print(f"Dashboard: https://wandb.ai/{wandb.api.default_entity}/{args.wandb_project}/sweeps/{sweep_id}")

    train_fn = train_multilabel if args.task == "multilabel" else train_ner

    def agent_fn():
        with wandb.init() as run:
            train_fn(run.config, args)

    wandb.agent(sweep_id, function=agent_fn, count=args.count)


if __name__ == "__main__":
    main()
