"""
run_ner_benchmark.py

Main entry point for benchmarking NER models on the AnatEM Spanish dataset.

Supports three model types:
  --model_type encoder   → Fine-tune a BERT-style encoder (BETO, RoBERTa-biomedical, etc.)
  --model_type decoder   → Zero-shot with a decoder LLM (Qwen, Llama, Mistral)
  --model_type peft      → LoRA/QLoRA/DoRA fine-tuning of a decoder

Usage examples:
  # Fine-tune BETO encoder
  python run_ner_benchmark.py --model_type encoder --model_name dccuchile/bert-base-spanish-wwm-cased

  # Zero-shot with Qwen
  python run_ner_benchmark.py --model_type decoder --model_name Qwen/Qwen2.5-7B-Instruct --load_in_4bit

  # QLoRA fine-tuning
  python run_ner_benchmark.py --model_type peft --model_name Qwen/Qwen2.5-7B-Instruct --load_in_4bit --peft_method lora
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils.ner_datareader import NERDataset
from evaluation.ner_predictor import NERPredictor


def get_output_dir(model_type: str, model_name: str, peft_method: str = "") -> str:
    model_short = model_name.split("/")[-1]
    suffix = f"_{peft_method}" if peft_method else ""
    return f"results/ner/{model_type}{suffix}/{model_short}"


def run_encoder(args, dataset: NERDataset) -> None:
    from models.encoder_ner import EncoderNERModel

    print(f"\n[Encoder NER] {args.model_name}")
    model = EncoderNERModel(
        model_name=args.model_name,
        label2id=dataset.label2id,
        id2label=dataset.id2label,
        max_length=args.max_length,
    )

    train_tokens, train_labels = dataset.get_tokens_and_labels("train")
    dev_tokens, dev_labels = dataset.get_tokens_and_labels("dev")

    output_dir = get_output_dir("encoder", args.model_name)
    print(f"Training for {args.num_epochs} epochs...")
    model.train(
        train_sentences=train_tokens,
        train_labels=train_labels,
        dev_sentences=dev_tokens,
        dev_labels=dev_labels,
        output_dir=output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )

    predictor = NERPredictor(model, dataset)
    predictor.save_predictions(
        split="test",
        output_path=output_dir,
    )


def run_decoder(args, dataset: NERDataset) -> None:
    from models.huggingface_llm import HuggingFaceLLM
    from models.llm_ner_model import LLMNERModel
    from prompts.ner_prompt import NERPromptTemplate

    print(f"\n[Decoder NER zero-shot] {args.model_name}")
    llm = HuggingFaceLLM(
        model_name=args.model_name,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
    )
    prompt_template = NERPromptTemplate(
        entity_types=dataset.get_label_names(),
        few_shot=args.few_shot,
    )
    model = LLMNERModel(
        llm=llm,
        entity_types=dataset.get_label_names(),
        prompt_template=prompt_template,
        batch_size=args.batch_size,
    )

    predictor = NERPredictor(model, dataset)
    output_dir = get_output_dir("decoder", args.model_name)
    predictor.save_predictions(
        split="test",
        output_path=output_dir,
        batch_size=args.batch_size,
    )


def run_peft(args, dataset: NERDataset) -> None:
    from models.peft_ner_model import PEFTNERModel

    use_dora = args.peft_method == "dora"
    print(f"\n[PEFT NER — {args.peft_method.upper()}] {args.model_name}")

    model = PEFTNERModel(
        model_name=args.model_name,
        entity_types=dataset.get_label_names(),
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        use_dora=use_dora,
        max_length=args.max_length,
    )

    train_tokens, train_labels = dataset.get_tokens_and_labels("train")
    dev_tokens, dev_labels = dataset.get_tokens_and_labels("dev")

    output_dir = get_output_dir("peft", args.model_name, args.peft_method)
    print(f"Fine-tuning for {args.num_epochs} epochs...")
    model.train(
        train_sentences=train_tokens,
        train_labels=train_labels,
        dev_sentences=dev_tokens,
        dev_labels=dev_labels,
        output_dir=output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    model.save(output_dir + "/final")

    predictor = NERPredictor(model, dataset)
    predictor.save_predictions(split="test", output_path=output_dir)


def main():
    parser = argparse.ArgumentParser(description="NER Benchmark — AnatEM Spanish")
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
        default="data/anat_em",
        help="Directory with train/dev/test CoNLL files",
    )

    # Training args
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=512)

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

    args = parser.parse_args()

    print("=" * 60)
    print(f"NER Benchmark — AnatEM Spanish")
    print(f"Model type : {args.model_type}")
    print(f"Model      : {args.model_name}")
    print(f"Data dir   : {args.data_dir}")
    print("=" * 60)

    # Load dataset
    print("\nLoading dataset...")
    dataset = NERDataset(data_dir=args.data_dir)
    print(dataset)

    # Run benchmark
    if args.model_type == "encoder":
        run_encoder(args, dataset)
    elif args.model_type == "decoder":
        run_decoder(args, dataset)
    elif args.model_type == "peft":
        run_peft(args, dataset)


if __name__ == "__main__":
    main()
