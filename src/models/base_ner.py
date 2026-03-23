"""Base class for NER models."""

from abc import ABC, abstractmethod
from typing import List


class BaseNERModel(ABC):
    """Abstract base for NER models (encoder fine-tuned, decoder zero-shot, PEFT)."""

    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def predict(
        self, sentences: List[List[str]]
    ) -> List[List[str]]:
        """
        Predict BIO labels for a list of tokenized sentences.

        Args:
            sentences: List of sentences, each is a list of token strings.

        Returns:
            List of label lists (same shape as input).
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_name='{self.model_name}')"
