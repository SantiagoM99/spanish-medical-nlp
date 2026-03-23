"""Base class for all language models (encoder-decoder agnostic)."""

from abc import ABC, abstractmethod
from typing import List


class BaseLLM(ABC):
    """Abstract base for any generative language model."""

    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.config = kwargs

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 512, **kwargs) -> str:
        """Generate a single response."""
        pass

    @abstractmethod
    def batch_generate(
        self, prompts: List[str], max_tokens: int = 512, **kwargs
    ) -> List[str]:
        """Generate responses for multiple prompts."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_name='{self.model_name}')"
