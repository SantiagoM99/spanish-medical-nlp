"""
llm_ner_model.py

Zero-shot / few-shot NER using a decoder LLM.
The model is prompted to output token-label pairs or a list of entities.
"""

from typing import List, Optional

from models.base_llm import BaseLLM
from models.base_ner import BaseNERModel
from prompts.ner_prompt import NERPromptTemplate


class LLMNERModel(BaseNERModel):
    """
    NER model using a decoder LLM with prompt-based extraction.

    The LLM receives a sentence and outputs the named entities found,
    which are then mapped back to BIO labels aligned with the input tokens.
    """

    def __init__(
        self,
        llm: BaseLLM,
        entity_types: List[str],
        prompt_template: Optional[NERPromptTemplate] = None,
        batch_size: int = 4,
    ):
        super().__init__(model_name=f"{llm.model_name}_ner")
        self.llm = llm
        self.entity_types = entity_types
        self.prompt_template = prompt_template or NERPromptTemplate(
            entity_types=entity_types
        )
        self.batch_size = batch_size

    def predict(self, sentences: List[List[str]]) -> List[List[str]]:
        """
        Predict BIO labels for tokenized sentences.

        Strategy:
          1. Convert token list to a plain string for the prompt.
          2. Ask the LLM to identify entities and their types.
          3. Parse the response and map entities back to BIO labels.
        """
        all_predictions = []

        for i in range(0, len(sentences), self.batch_size):
            batch = sentences[i : i + self.batch_size]
            prompts = [
                self.prompt_template.create_prompt(tokens=tokens)
                for tokens in batch
            ]
            responses = self.llm.batch_generate(
                prompts=prompts, max_tokens=256, do_sample=False
            )
            for tokens, response in zip(batch, responses):
                bio_labels = self.prompt_template.parse_response(
                    response=response, tokens=tokens
                )
                all_predictions.append(bio_labels)

        return all_predictions

    def __repr__(self) -> str:
        return (
            f"LLMNERModel(\n"
            f"  llm={self.llm.model_name},\n"
            f"  entity_types={self.entity_types},\n"
            f"  batch_size={self.batch_size}\n"
            f")"
        )
