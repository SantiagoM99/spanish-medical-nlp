"""
run_multilabel_benchmark.py

Main entry point for benchmarking multi-label classification models
on the PubMed Spanish MeSH dataset.

Supports three model types:
  --model_type encoder   → Fine-tune BERT-style encoder (BETO, RoBERTa-biomedical, etc.)
  --model_type decoder   → Zero-shot / few-shot with decoder LLM (Qwen, Llama, Mistral)
  --model_type peft      → LoRA/QLoRA/DoRA fine-tuning of a decoder

Text modes:
  --text_mode title_abstract | abstract_only | title_only

Optional filters:
  --filter_geographicals   → Remove MeSH category Z samples

Optional W&B logging:
  --wandb_project my_project

Usage examples:
  # Fine-tune RoBERTa biomedical encoder
  python run_multilabel_benchmark.py --model_type encoder --model_name PlanTL-GOB-ES/roberta-base-biomedical-es

  # Zero-shot Qwen with few-shot prompting
  python run_multilabel_benchmark.py --model_type decoder --model_name Qwen/Qwen2.5-7B-Instruct --load_in_4bit --few_shot

  # QLoRA fine-tuning, abstract only, no Geographicals, W&B
  python run_multilabel_benchmark.py --model_type peft --model_name Qwen/Qwen2.5-7B-Instruct \\
      --load_in_4bit --peft_method qlora --text_mode abstract_only --filter_geographicals \\
      --wandb_project spanish-multilabel
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils.multilabel_datareader import MultiLabelDataset
from evaluation.multilabel_predictor import MultiLabelPredictor


def get_output_dir(model_type: str, model_name: str, peft_method: str = "") -> str:
    model_short = model_name.split("/")[-1]
    suffix = f"_{peft_method}" if peft_method else ""
    return f"results/multilabel/{model_type}{suffix}/{model_short}"


def init_wandb(args, dataset: MultiLabelDataset):
    """Inicializa W&B si se especificó --wandb_project."""
    if not args.wandb_project:
        return None
    try:
        import wandb
        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f"{args.model_type}_{args.model_name.split('/')[-1]}",
            config={
                "model_type": args.model_type,
                "model_name": args.model_name,
                "dataset": "PubMed-MeSH-Spanish",
                "text_mode": args.text_mode,
                "filter_geographicals": args.filter_geographicals,
                "num_labels": len(dataset.label2id),
                "train_size": len(dataset.train),
                "num_epochs": args.num_epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "max_length": args.max_length,
                "threshold": args.threshold,
                "peft_method": args.peft_method,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
            },
        )
        print(f"W&B run: {run.url}")
        return run
    except ImportError:
        print("AVISO: wandb no instalado. Omitiendo logging.")
        return None


def log_metrics_to_wandb(wandb_run, metrics: dict) -> None:
    if wandb_run is None:
        return
    import wandb
    flat = {}
    for k, v in metrics.items():
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                flat[f"{k}/{sub_k}"] = sub_v
        else:
            flat[k] = v
    wandb_run.log(flat)
    wandb_run.finish()


def run_encoder(args, dataset: MultiLabelDataset, wandb_run=None) -> None:
    from models.encoder_multilabel import EncoderMultiLabelModel

    print(f"\n[Encoder Multi-label] {args.model_name}")
    model = EncoderMultiLabelModel(
        model_name=args.model_name,
        label2id=dataset.label2id,
        id2label=dataset.id2label,
        max_length=args.max_length,
        threshold=args.threshold,
    )

    train_texts, train_labels = dataset.get_texts_and_labels("train")
    dev_texts, dev_labels = dataset.get_texts_and_labels("dev")

    output_dir = get_output_dir("encoder", args.model_name)
    print(f"Training for {args.num_epochs} epochs...")
    model.train(
        train_texts=train_texts,
        train_labels=train_labels,
        dev_texts=dev_texts,
        dev_labels=dev_labels,
        output_dir=output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )

    predictor = MultiLabelPredictor(model, dataset)
    predictor.save_predictions(split="test", output_path=output_dir)

    metrics_path = Path(output_dir) / "metrics.json"
    if wandb_run and metrics_path.exists():
        with open(metrics_path) as f:
            log_metrics_to_wandb(wandb_run, json.load(f))


def run_decoder(args, dataset: MultiLabelDataset, wandb_run=None) -> None:
    from models.huggingface_llm import HuggingFaceLLM
    from models.llm_multilabel_model import LLMMultiLabelModel
    from prompts.multilabel_prompt import MultiLabelPromptTemplate

    print(f"\n[Decoder Multi-label zero-shot] {args.model_name}")
    llm = HuggingFaceLLM(
        model_name=args.model_name,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
    )
    prompt_template = MultiLabelPromptTemplate(
        available_labels=dataset.labels,
        few_shot=args.few_shot,
    )
    model = LLMMultiLabelModel(
        llm=llm,
        available_labels=dataset.labels,
        prompt_template=prompt_template,
        batch_size=args.batch_size,
    )

    predictor = MultiLabelPredictor(model, dataset)
    output_dir = get_output_dir("decoder", args.model_name)
    predictor.save_predictions(split="test", output_path=output_dir)

    metrics_path = Path(output_dir) / "metrics.json"
    if wandb_run and metrics_path.exists():
        with open(metrics_path) as f:
            log_metrics_to_wandb(wandb_run, json.load(f))


def run_peft(args, dataset: MultiLabelDataset, wandb_run=None) -> None:
    from models.peft_multilabel_model import PEFTMultiLabelModel

    use_dora = args.peft_method == "dora"
    print(f"\n[PEFT Multi-label — {args.peft_method.upper()}] {args.model_name}")

    model = PEFTMultiLabelModel(
        model_name=args.model_name,
        available_labels=dataset.labels,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        use_dora=use_dora,
        max_length=args.max_length,
    )

    train_texts, train_labels = dataset.get_texts_and_labels("train")
    dev_texts, dev_labels = dataset.get_texts_and_labels("dev")

    output_dir = get_output_dir("peft", args.model_name, args.peft_method)
    print(f"Fine-tuning for {args.num_epochs} epochs...")
    model.train(
        train_texts=train_texts,
        train_labels=train_labels,
        dev_texts=dev_texts,
        dev_labels=dev_labels,
        output_dir=output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    model.save(output_dir + "/final")

    predictor = MultiLabelPredictor(model, dataset)
    predictor.save_predictions(split="test", output_path=output_dir)

    metrics_path = Path(output_dir) / "metrics.json"
    if wandb_run and metrics_path.exists():
        with open(metrics_path) as f:
            log_metrics_to_wandb(wandb_run, json.load(f))


def main():
    parser = argparse.ArgumentParser(
        description="Multi-label Benchmark — PubMed Spanish MeSH"
    )
    parser.add_argument(
        "--model_type",
        choices=["encoder", "decoder", "peft"],
        required=True,
        help="Type of model to benchmark",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="HuggingFace model ID or local path",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/pubmed_mesh",
        help="Directory with train/dev/test parquet files",
    )

    # Dataset options
    parser.add_argument(
        "--text_mode",
        choices=["title_abstract", "abstract_only", "title_only"],
        default="title_abstract",
        help="Input text construction mode",
    )
    parser.add_argument(
        "--filter_geographicals",
        action="store_true",
        help="Remove samples whose only label is Z (Geographicals)",
    )

    # Training args
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Sigmoid threshold for encoder predictions",
    )

    # Quantization
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--load_in_8bit", action="store_true")

    # PEFT-specific
    parser.add_argument(
        "--peft_method",
        choices=["lora", "qlora", "dora"],
        default="lora",
        help="PEFT method (only for --model_type peft)",
    )
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)

    # Decoder-specific
    parser.add_argument("--few_shot", action="store_true")

    # W&B
    parser.add_argument("--wandb_project", type=str, default=None, help="W&B project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="W&B run name")

    args = parser.parse_args()

    print("=" * 60)
    print(f"Multi-label Benchmark — PubMed Spanish MeSH")
    print(f"Model type : {args.model_type}")
    print(f"Model      : {args.model_name}")
    print(f"Data dir   : {args.data_dir}")
    print(f"Text mode  : {args.text_mode}")
    print(f"Filter geo : {args.filter_geographicals}")
    print("=" * 60)

    print("\nLoading dataset...")
    dataset = MultiLabelDataset(
        data_dir=args.data_dir,
        text_mode=args.text_mode,
        filter_geographicals=args.filter_geographicals,
    )
    print(dataset)

    wandb_run = init_wandb(args, dataset)

    if args.model_type == "encoder":
        run_encoder(args, dataset, wandb_run)
    elif args.model_type == "decoder":
        run_decoder(args, dataset, wandb_run)
    elif args.model_type == "peft":
        run_peft(args, dataset, wandb_run)


if __name__ == "__main__":
    main()
