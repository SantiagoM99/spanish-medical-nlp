"""
encoder_multilabel.py

BERT-style encoder fine-tuned for multi-label classification.
Uses AutoModelForSequenceClassification with sigmoid activation (problem_type=multi_label_classification).

Supports: BETO, RoBERTa-biomedical-es, XLM-RoBERTa, mBERT, etc.
"""

from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from models.base_multilabel import BaseMultiLabelModel


class MultiLabelTextDataset(Dataset):
    """Dataset for multi-label text classification."""

    def __init__(
        self,
        texts: List[str],
        labels: List[List[str]],
        tokenizer,
        label2id: dict,
        max_length: int = 512,
    ):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        # Build binary multi-hot label vectors
        n_labels = len(label2id)
        self.label_matrix = torch.zeros((len(texts), n_labels), dtype=torch.float32)
        for i, label_list in enumerate(labels):
            for lbl in label_list:
                if lbl in label2id:
                    self.label_matrix[i, label2id[lbl]] = 1.0

    def __len__(self):
        return self.label_matrix.shape[0]

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.label_matrix[idx]
        return item


class EncoderMultiLabelModel(BaseMultiLabelModel):
    """
    Fine-tuned encoder model for Spanish medical multi-label classification.

    Usage:
        model = EncoderMultiLabelModel(
            model_name="PlanTL-GOB-ES/roberta-base-biomedical-es",
            label2id=dataset.label2id,
            id2label=dataset.id2label,
        )
        model.train(train_texts, train_labels, dev_texts, dev_labels)
        predictions = model.predict(test_texts)
    """

    def __init__(
        self,
        model_name: str,
        label2id: dict,
        id2label: dict,
        max_length: int = 512,
        threshold: float = 0.5,
        device: Optional[str] = None,
    ):
        super().__init__(model_name)
        self.label2id = label2id
        self.id2label = id2label
        self.max_length = max_length
        self.threshold = threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(label2id),
            id2label=id2label,
            label2id=label2id,
            problem_type="multi_label_classification",
            ignore_mismatched_sizes=True,
        )
        self.model.to(self.device)

    def train(
        self,
        train_texts: List[str],
        train_labels: List[List[str]],
        dev_texts: List[str],
        dev_labels: List[List[str]],
        output_dir: str = "results/multilabel/encoder",
        num_epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
    ) -> None:
        """Fine-tune the encoder on training data."""
        train_dataset = MultiLabelTextDataset(
            train_texts, train_labels, self.tokenizer, self.label2id, self.max_length
        )
        dev_dataset = MultiLabelTextDataset(
            dev_texts, dev_labels, self.tokenizer, self.label2id, self.max_length
        )

        args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            logging_steps=100,
            fp16=torch.cuda.is_available(),
            report_to="none",
        )

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            tokenizer=self.tokenizer,
        )
        trainer.train()

    def predict(self, texts: List[str]) -> List[List[str]]:
        """Predict labels using threshold on sigmoid probabilities."""
        _, scores_list = zip(*self.predict_with_scores(texts))
        predictions = []
        for label_names, scores in zip(
            [list(self.label2id.keys())] * len(texts), scores_list
        ):
            predicted = [
                lbl for lbl, score in zip(self.id2label.values(), scores)
                if score >= self.threshold
            ]
            predictions.append(predicted)
        return predictions

    def predict_with_scores(
        self, texts: List[str], batch_size: int = 32
    ) -> List[Tuple[List[str], List[float]]]:
        """Return (labels, sigmoid_scores) for each text."""
        self.model.eval()
        all_results = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch_texts,
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            probs = torch.sigmoid(outputs.logits).cpu().numpy()
            for prob_row in probs:
                labels = [self.id2label[i] for i in range(len(prob_row))]
                scores = prob_row.tolist()
                all_results.append((labels, scores))

        return all_results

    def save(self, output_dir: str) -> None:
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    @classmethod
    def load(
        cls,
        model_dir: str,
        label2id: dict,
        id2label: dict,
        threshold: float = 0.5,
    ) -> "EncoderMultiLabelModel":
        instance = cls.__new__(cls)
        instance.model_name = model_dir
        instance.label2id = label2id
        instance.id2label = id2label
        instance.max_length = 512
        instance.threshold = threshold
        instance.device = "cuda" if torch.cuda.is_available() else "cpu"
        instance.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        instance.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        instance.model.to(instance.device)
        return instance
