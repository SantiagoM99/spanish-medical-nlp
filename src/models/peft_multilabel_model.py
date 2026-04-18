"""
peft_multilabel_model.py

PEFT fine-tuning (LoRA / QLoRA / DoRA) of a decoder LLM for multi-label classification.

Generative format:
  Input:  "[abstract text]"
  Output: "B, C, D"   (comma-separated MeSH label codes)
"""

from typing import Callable, List, Optional, Tuple

import torch
from tqdm import tqdm
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

from models.base_multilabel import BaseMultiLabelModel


class MultiLabelSeq2SeqDataset(Dataset):
    """Formats multi-label examples as instruction-following sequences."""

    PROMPT_TEMPLATE = (
        "Eres un experto en clasificación de artículos médicos.\n"
        "Clasifica el siguiente abstract con los códigos MeSH aplicables de la lista: {labels}.\n\n"
        "Abstract: {text}\n"
        "Códigos MeSH:"
    )

    def __init__(
        self,
        texts: List[str],
        labels: List[List[str]],
        tokenizer,
        available_labels: List[str],
        max_length: int = 512,
        format_fn: Optional[Callable[[str], str]] = None,
    ):
        self.examples = []
        labels_str = ", ".join(available_labels)

        for text, label_list in zip(texts, labels):
            raw_prompt = self.PROMPT_TEMPLATE.format(labels=labels_str, text=text)
            prompt = format_fn(raw_prompt) if format_fn else raw_prompt
            target = ", ".join(sorted(label_list)) if label_list else "ninguno"
            full_text = prompt + target + tokenizer.eos_token

            encoding = tokenizer(
                full_text, truncation=True, max_length=max_length, padding=False
            )
            prompt_encoding = tokenizer(prompt, truncation=True, max_length=max_length)
            prompt_len = len(prompt_encoding["input_ids"])
            input_ids = encoding["input_ids"]
            label_ids = [-100] * prompt_len + input_ids[prompt_len:]
            encoding["labels"] = label_ids
            self.examples.append(encoding)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return {k: torch.tensor(v) for k, v in self.examples[idx].items()}


class PEFTMultiLabelModel(BaseMultiLabelModel):
    """
    LoRA / QLoRA / DoRA fine-tuned decoder for multi-label classification.

    Usage:
        model = PEFTMultiLabelModel(
            model_name="Qwen/Qwen2.5-7B-Instruct",
            available_labels=dataset.labels,
            load_in_4bit=True,   # QLoRA
        )
        model.train(train_texts, train_labels, dev_texts, dev_labels)
        predictions = model.predict(test_texts)
    """

    PROMPT_TEMPLATE = (
        "Eres un experto en clasificación de artículos médicos.\n"
        "Clasifica el siguiente abstract con los códigos MeSH aplicables de la lista: {labels}.\n\n"
        "Abstract: {text}\n"
        "Códigos MeSH:"
    )

    def __init__(
        self,
        model_name: str,
        available_labels: List[str],
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
        self.available_labels = available_labels
        self.labels_str = ", ".join(available_labels)
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

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

        n_gpus = torch.cuda.device_count()
        use_device_map = bnb_config or n_gpus > 1
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            device_map="auto" if use_device_map else None,
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

    def _format_prompt(self, prompt: str) -> str:
        """Wrap a raw prompt in the model's chat template when available."""
        if not hasattr(self.tokenizer, "chat_template") or not self.tokenizer.chat_template:
            return prompt
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def train(
        self,
        train_texts: List[str],
        train_labels: List[List[str]],
        dev_texts: List[str],
        dev_labels: List[List[str]],
        output_dir: str = "results/multilabel/peft",
        num_epochs: int = 3,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        learning_rate: float = 2e-4,
    ) -> None:
        train_dataset = MultiLabelSeq2SeqDataset(
            train_texts, train_labels, self.tokenizer,
            self.available_labels, self.max_length,
            format_fn=self._format_prompt,
        )
        dev_dataset = MultiLabelSeq2SeqDataset(
            dev_texts, dev_labels, self.tokenizer,
            self.available_labels, self.max_length,
            format_fn=self._format_prompt,
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
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            fp16=torch.cuda.is_available(),
            gradient_checkpointing=True,
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

    def predict(self, texts: List[str]) -> List[List[str]]:
        self.model.eval()
        all_predictions = []

        for text in tqdm(texts, desc="PEFT multilabel inference"):
            prompt = self._format_prompt(
                self.PROMPT_TEMPLATE.format(labels=self.labels_str, text=text)
            )
            inputs = self.tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=self.max_length
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            ).strip()

            # Parse comma-separated codes, filter to valid labels
            predicted = []
            for token in response.replace(",", " ").split():
                code = token.strip().upper()
                if code in self.available_labels and code not in predicted:
                    predicted.append(code)
            all_predictions.append(predicted)

        return all_predictions

    def predict_with_scores(
        self, texts: List[str]
    ) -> List[Tuple[List[str], List[float]]]:
        predictions = self.predict(texts)
        return [(labels, [1.0] * len(labels)) for labels in predictions]

    def save(self, output_dir: str) -> None:
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
