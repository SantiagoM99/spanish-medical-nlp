"""Base class for multi-label classification models."""

from abc import ABC, abstractmethod
from typing import List, Tuple


class BaseMultiLabelModel(ABC):
    """Abstract base for multi-label classification models."""

    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def predict(self, texts: List[str]) -> List[List[str]]:
        """
        Predict labels for a list of texts.

        Returns:
            List of lists of predicted label codes.
        """
        pass

    @abstractmethod
    def predict_with_scores(
        self, texts: List[str]
    ) -> List[Tuple[List[str], List[float]]]:
        """
        Predict labels with confidence scores.

        Returns:
            List of (labels, scores) tuples.
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_name='{self.model_name}')"
