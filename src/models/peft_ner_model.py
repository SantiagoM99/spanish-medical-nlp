"""
peft_ner_model.py

PEFT fine-tuning (LoRA / QLoRA / DoRA) of a decoder LLM for NER.

The model is fine-tuned to output BIO-labeled sequences given input token sequences,
using a generative format:
  Input:  "fibrilación ventricular debida a síndrome de QT"
  Output: "O B-Multi-tissue_structure O O O O O"
"""

from typing import List, Optional

import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)
from torch.utils.data import Dataset

from models.base_ner import BaseNERModel


class NERSeq2SeqDataset(Dataset):
    """Formats NER examples as instruction-following sequences."""

    PROMPT_TEMPLATE = (
        "Eres un sistema de reconocimiento de entidades médicas. "
        "Etiqueta cada token con BIO labels. Tipos de entidades: {entity_types}.\n\n"
        "Tokens: {tokens}\n"
        "Etiquetas:"
    )

    def __init__(
        self,
        sentences: List[List[str]],
        labels: List[List[str]],
        tokenizer,
        entity_types: List[str],
        max_length: int = 512,
    ):
        self.examples = []
        entity_types_str = ", ".join(
            sorted(set(et.replace("B-", "").replace("I-", "") for et in entity_types if et != "O"))
        )
        for tokens, token_labels in zip(sentences, labels):
            prompt = self.PROMPT_TEMPLATE.format(
                entity_types=entity_types_str,
                tokens=" ".join(tokens),
            )
            target = " ".join(token_labels)
            full_text = prompt + " " + target
            encoding = tokenizer(
                full_text,
                truncation=True,
                max_length=max_length,
                padding=False,
            )
            # Mask the prompt tokens in the labels
            prompt_encoding = tokenizer(prompt, truncation=True, max_length=max_length)
            prompt_len = len(prompt_encoding["input_ids"])
            input_ids = encoding["input_ids"]
            labels_ids = [-100] * prompt_len + input_ids[prompt_len:]
            encoding["labels"] = labels_ids
            self.examples.append(encoding)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return {k: torch.tensor(v) for k, v in self.examples[idx].items()}


class PEFTNERModel(BaseNERModel):
    """
    LoRA / QLoRA / DoRA fine-tuned decoder for NER.

    Usage:
        model = PEFTNERModel(
            model_name="Qwen/Qwen2.5-7B-Instruct",
            entity_types=dataset.get_label_names(),
            lora_r=16,
            load_in_4bit=True,   # QLoRA
        )
        model.train(train_sentences, train_labels, dev_sentences, dev_labels)
        predictions = model.predict(test_sentences)
    """

    PROMPT_TEMPLATE = (
        "Eres un sistema de reconocimiento de entidades médicas. "
        "Etiqueta cada token con BIO labels. Tipos de entidades: {entity_types}.\n\n"
        "Tokens: {tokens}\n"
        "Etiquetas:"
    )

    def __init__(
        self,
        model_name: str,
        entity_types: List[str],
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        target_modules: Optional[List[str]] = None,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        use_dora: bool = False,
        max_length: int = 512,
        device: Optional[str] = None,
    ):
        super().__init__(model_name)
        self.entity_types = entity_types
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.entity_types_str = ", ".join(
            sorted(set(et.replace("B-", "").replace("I-", "") for et in entity_types if et != "O"))
        )

        # Quantization
        bnb_config = None
        if load_in_4bit or load_in_8bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            device_map="auto" if bnb_config else None,
        )

        if bnb_config:
            base_model = prepare_model_for_kbit_training(base_model)

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules or ["q_proj", "v_proj", "k_proj", "o_proj"],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            use_dora=use_dora,
        )
        self.model = get_peft_model(base_model, lora_config)
        self.model.print_trainable_parameters()

        if not bnb_config:
            self.model.to(self.device)

    def train(
        self,
        train_sentences: List[List[str]],
        train_labels: List[List[str]],
        dev_sentences: List[List[str]],
        dev_labels: List[List[str]],
        output_dir: str = "results/ner/peft",
        num_epochs: int = 3,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        learning_rate: float = 2e-4,
    ) -> None:
        train_dataset = NERSeq2SeqDataset(
            train_sentences, train_labels, self.tokenizer,
            self.entity_types, self.max_length
        )
        dev_dataset = NERSeq2SeqDataset(
            dev_sentences, dev_labels, self.tokenizer,
            self.entity_types, self.max_length
        )
        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer, model=self.model, pad_to_multiple_of=8
        )

        args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            fp16=True,
            logging_steps=50,
            report_to="none",
            optim="paged_adamw_8bit",
            warmup_ratio=0.05,
        )

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            data_collator=data_collator,
        )
        trainer.train()

    def predict(self, sentences: List[List[str]]) -> List[List[str]]:
        """Generate BIO labels via autoregressive decoding."""
        self.model.eval()
        all_predictions = []

        for tokens in sentences:
            prompt = self.PROMPT_TEMPLATE.format(
                entity_types=self.entity_types_str,
                tokens=" ".join(tokens),
            )
            inputs = self.tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=self.max_length
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=len(tokens) * 4,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            ).strip()

            # Parse space-separated label sequence
            predicted_labels = response.split()
            # Align length with input tokens
            if len(predicted_labels) < len(tokens):
                predicted_labels += ["O"] * (len(tokens) - len(predicted_labels))
            else:
                predicted_labels = predicted_labels[: len(tokens)]

            all_predictions.append(predicted_labels)

        return all_predictions

    def save(self, output_dir: str) -> None:
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
