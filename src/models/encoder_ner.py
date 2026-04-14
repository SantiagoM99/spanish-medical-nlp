"""
encoder_ner.py

BERT-style encoder fine-tuned for token classification (NER).
Uses HuggingFace AutoModelForTokenClassification with a linear head.

Supports: BETO, RoBERTa-biomedical-es, XLM-RoBERTa, mBERT, etc.
"""

from typing import List, Optional

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from models.base_ner import BaseNERModel


class TokenClassificationDataset(Dataset):

    @staticmethod
    def _split_long_sentences(
        sentences: List[List[str]],
        labels: List[List[str]],
        tokenizer,
        max_length: int,
    ) -> tuple:
        new_sentences, new_labels = [], []
        for tokens, token_labels in zip(sentences, labels):
            enc = tokenizer(tokens, is_split_into_words=True, truncation=False)
            if len(enc["input_ids"]) <= max_length:
                new_sentences.append(tokens)
                new_labels.append(token_labels)
                continue
            n_subwords = len(enc["input_ids"]) - 2
            ratio = n_subwords / len(tokens)
            words_per_chunk = max(1, int((max_length - 2) / ratio * 0.9))
            start = 0
            while start < len(tokens):
                end = min(start + words_per_chunk, len(tokens))
                while end < len(tokens) and token_labels[end].startswith("I-"):
                    end += 1
                new_sentences.append(tokens[start:end])
                new_labels.append(token_labels[start:end])
                start = end
        return new_sentences, new_labels

    def __init__(
        self,
        sentences: List[List[str]],
        labels: List[List[str]],
        tokenizer,
        label2id: dict,
        max_length: int = 512,
    ):
        sentences, labels = self._split_long_sentences(
            sentences, labels, tokenizer, max_length
        )
        self.encodings = []
        for tokens, token_labels in zip(sentences, labels):
            encoding = tokenizer(
                tokens,
                is_split_into_words=True,
                truncation=True,
                max_length=max_length,
                padding=False,
            )
            word_ids = encoding.word_ids()
            aligned_labels = []
            prev_word_id = None
            for word_id in word_ids:
                if word_id is None:
                    aligned_labels.append(-100)
                elif word_id != prev_word_id:
                    aligned_labels.append(label2id.get(token_labels[word_id], 0))
                else:
                    aligned_labels.append(-100)
                prev_word_id = word_id
            encoding["labels"] = aligned_labels
            self.encodings.append(encoding)

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        return {k: torch.tensor(v) for k, v in self.encodings[idx].items()}


class EncoderNERModel(BaseNERModel):
    """
    Fine-tuned encoder model for Spanish medical NER.

    Usage:
        model = EncoderNERModel(
            model_name="PlanTL-GOB-ES/roberta-base-biomedical-es",
            label2id=dataset.label2id,
            id2label=dataset.id2label,
        )
        model.train(train_sentences, train_labels, dev_sentences, dev_labels)
        predictions = model.predict(test_sentences)
    """

    def __init__(
        self,
        model_name: str,
        label2id: dict,
        id2label: dict,
        max_length: int = 512,
        device: Optional[str] = None,
    ):
        super().__init__(model_name)
        self.label2id = label2id
        self.id2label = id2label
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=len(label2id),
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        )
        self.model.to(self.device)

    def train(
        self,
        train_sentences: List[List[str]],
        train_labels: List[List[str]],
        dev_sentences: List[List[str]],
        dev_labels: List[List[str]],
        output_dir: str = "results/ner/encoder",
        num_epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
    ) -> None:
        train_dataset = TokenClassificationDataset(
            train_sentences, train_labels, self.tokenizer, self.label2id, self.max_length
        )
        dev_dataset = TokenClassificationDataset(
            dev_sentences, dev_labels, self.tokenizer, self.label2id, self.max_length
        )

        data_collator = DataCollatorForTokenClassification(self.tokenizer)

        import evaluate as hf_evaluate
        seqeval_metric = hf_evaluate.load("seqeval")

        def compute_metrics_fn(p):
            import numpy as np
            predictions, labels = p
            predictions = np.argmax(predictions, axis=2)
            true_labels, true_preds = [], []
            for pred_seq, label_seq in zip(predictions, labels):
                t_labels, t_preds = [], []
                for pr, lb in zip(pred_seq, label_seq):
                    if lb == -100:
                        continue
                    t_labels.append(self.id2label.get(int(lb), "O"))
                    t_preds.append(self.id2label.get(int(pr), "O"))
                if t_labels:
                    true_labels.append(t_labels)
                    true_preds.append(t_preds)
            if not true_labels:
                return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0}
            results = seqeval_metric.compute(predictions=true_preds, references=true_labels)
            return {
                "precision": results["overall_precision"],
                "recall": results["overall_recall"],
                "f1": results["overall_f1"],
                "accuracy": results["overall_accuracy"],
            }

        args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=0.1,
            warmup_ratio=0.06,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            logging_steps=50,
            save_total_limit=2,
            fp16=torch.cuda.is_available(),
            report_to="none",
        )

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            processing_class=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics_fn,
        )
        trainer.train()

    def _predict_chunk(self, tokens: List[str]) -> List[str]:
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        word_ids = encoding.word_ids(batch_index=0)
        encoding = {k: v.to(self.device) for k, v in encoding.items()}

        with torch.no_grad():
            logits = self.model(**encoding).logits[0]

        predicted_ids = torch.argmax(logits, dim=-1).cpu().tolist()

        first_subword_labels = {}
        for word_id, pred_id in zip(word_ids, predicted_ids):
            if word_id is not None and word_id not in first_subword_labels:
                first_subword_labels[word_id] = self.id2label[pred_id]
        return list(first_subword_labels.values())

    def _subword_safe_chunk_size(self, tokens: List[str]) -> int:
        subword_count = len(self.tokenizer(
            tokens, is_split_into_words=True, truncation=False
        )["input_ids"]) - 2
        subword_ratio = subword_count / len(tokens)
        return max(1, int((self.max_length - 2) / subword_ratio * 0.9))

    def _predict_with_sliding_window(self, tokens: List[str]) -> List[str]:
        chunk_size = self._subword_safe_chunk_size(tokens)
        labels = [None] * len(tokens)
        start = 0
        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            chunk_labels = self._predict_chunk(tokens[start:end])
            labels[start:start + len(chunk_labels)] = [
                existing or predicted
                for existing, predicted in zip(labels[start:start + len(chunk_labels)], chunk_labels)
            ]
            start += chunk_size
        return [label or "O" for label in labels]

    def _needs_sliding_window(self, tokens: List[str]) -> bool:
        subword_length = len(self.tokenizer(
            tokens, is_split_into_words=True, truncation=False
        )["input_ids"])
        return subword_length > self.max_length

    def predict(self, sentences: List[List[str]]) -> List[List[str]]:
        self.model.eval()
        return [
            self._predict_with_sliding_window(tokens) if self._needs_sliding_window(tokens)
            else self._predict_chunk(tokens)
            for tokens in sentences
        ]

    def save(self, output_dir: str) -> None:
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    @classmethod
    def load(cls, model_dir: str, label2id: dict, id2label: dict) -> "EncoderNERModel":
        instance = cls.__new__(cls)
        instance.model_name = model_dir
        instance.label2id = label2id
        instance.id2label = id2label
        instance.max_length = 512
        instance.device = "cuda" if torch.cuda.is_available() else "cpu"
        instance.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        instance.model = AutoModelForTokenClassification.from_pretrained(model_dir)
        instance.model.to(instance.device)
        return instance
