"""
llm_multilabel_model.py

Zero-shot / few-shot multi-label classification using a decoder LLM.
"""

from typing import List, Optional, Tuple

from tqdm import tqdm

from models.base_llm import BaseLLM
from models.base_multilabel import BaseMultiLabelModel
from prompts.multilabel_prompt import MultiLabelPromptTemplate


class LLMMultiLabelModel(BaseMultiLabelModel):
    """
    Multi-label classification model using a decoder LLM with prompting.
    """

    def __init__(
        self,
        llm: BaseLLM,
        available_labels: List[str],
        prompt_template: Optional[MultiLabelPromptTemplate] = None,
        batch_size: int = 4,
    ):
        super().__init__(model_name=f"{llm.model_name}_multilabel")
        self.llm = llm
        self.available_labels = available_labels
        self.prompt_template = prompt_template or MultiLabelPromptTemplate(
            available_labels=available_labels
        )
        self.batch_size = batch_size

    def predict(self, texts: List[str]) -> List[List[str]]:
        self.llm.model.eval()
        all_predictions = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Multilabel inference"):
            batch = texts[i : i + self.batch_size]
            prompts = [self.prompt_template.create_prompt(text=text) for text in batch]
            responses = self.llm.batch_generate(
                prompts=prompts,
                max_tokens=100,
                do_sample=False,
                batch_size=self.batch_size,
            )
            all_predictions.extend([self.prompt_template.parse_response(r) for r in responses])
        return all_predictions

    def predict_with_scores(
        self, texts: List[str]
    ) -> List[Tuple[List[str], List[float]]]:
        predictions = self.predict(texts)
        return [(labels, [1.0] * len(labels)) for labels in predictions]

    def __repr__(self) -> str:
        return (
            f"LLMMultiLabelModel(\n"
            f"  llm={self.llm.model_name},\n"
            f"  n_labels={len(self.available_labels)},\n"
            f"  batch_size={self.batch_size}\n"
            f")"
        )
