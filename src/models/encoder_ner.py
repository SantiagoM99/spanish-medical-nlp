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
    """Tokenizes and aligns labels for sub-word tokenization."""

    def __init__(
        self,
        sentences: List[List[str]],
        labels: List[List[str]],
        tokenizer,
        label2id: dict,
        max_length: int = 512,
    ):
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
                    # For sub-word pieces: use -100 to ignore in loss
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
        """Fine-tune the encoder on training data."""
        train_dataset = TokenClassificationDataset(
            train_sentences, train_labels, self.tokenizer, self.label2id, self.max_length
        )
        dev_dataset = TokenClassificationDataset(
            dev_sentences, dev_labels, self.tokenizer, self.label2id, self.max_length
        )

        data_collator = DataCollatorForTokenClassification(self.tokenizer)

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
            logging_steps=50,
            fp16=torch.cuda.is_available(),
            report_to="none",
        )

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )
        trainer.train()

    def predict(self, sentences: List[List[str]]) -> List[List[str]]:
        """
        Predict BIO labels for tokenized sentences.

        Args:
            sentences: List of tokenized sentences (list of token strings).

        Returns:
            List of label lists, one per sentence (aligned to input tokens).
        """
        self.model.eval()
        all_predictions = []

        for tokens in sentences:
            encoding = self.tokenizer(
                tokens,
                is_split_into_words=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoding = {k: v.to(self.device) for k, v in encoding.items()}
            word_ids = self.tokenizer(
                tokens, is_split_into_words=True
            ).word_ids()

            with torch.no_grad():
                outputs = self.model(**encoding)

            logits = outputs.logits[0]
            predictions = torch.argmax(logits, dim=-1).cpu().tolist()

            # Map back to original tokens (first sub-word only)
            token_labels = []
            prev_word_id = None
            for word_id, pred_id in zip(word_ids, predictions):
                if word_id is None:
                    continue
                if word_id != prev_word_id:
                    token_labels.append(self.id2label[pred_id])
                prev_word_id = word_id

            all_predictions.append(token_labels)

        return all_predictions

    def save(self, output_dir: str) -> None:
        """Save model and tokenizer."""
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    @classmethod
    def load(cls, model_dir: str, label2id: dict, id2label: dict) -> "EncoderNERModel":
        """Load a saved model."""
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
