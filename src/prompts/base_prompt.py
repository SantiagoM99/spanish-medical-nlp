"""Base class for all prompt templates."""

from abc import ABC, abstractmethod


class BasePromptTemplate(ABC):
    """Abstract base for prompt templates."""

    def __init__(self, language: str = "es"):
        self.language = language

    @abstractmethod
    def create_prompt(self, **kwargs) -> str:
        pass

    @abstractmethod
    def parse_response(self, response: str, **kwargs):
        pass
