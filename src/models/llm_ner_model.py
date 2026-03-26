"""
llm_ner_model.py

Zero-shot / few-shot NER using a decoder LLM.

Soporta:
  - zero_shot / few_shot / knn_few_shot  (via NERPromptTemplate)
  - self_verification: segunda llamada al LLM por cada entidad extraída
                       para filtrar falsos positivos (+2-5% F1 según GPT-NER)
"""

from typing import List, Optional

from models.base_llm import BaseLLM
from models.base_ner import BaseNERModel
from prompts.ner_prompt import NERPromptTemplate


class LLMNERModel(BaseNERModel):
    """
    NER model using a decoder LLM with prompt-based extraction.

    Args:
        llm: Cualquier instancia que implemente BaseLLM.
        entity_types: Lista de etiquetas BIO válidas (e.g. ["B-Organ", "I-Organ", "O"]).
        prompt_template: Instancia de NERPromptTemplate. Si None, se crea con defaults.
        strategy: 'zero_shot' | 'few_shot' | 'knn_few_shot'.
        knn_retriever: Instancia de KNNRetriever ya inicializada (requerida si strategy='knn_few_shot').
        knn_k: Número de ejemplos k-NN a recuperar.
        self_verification: Si True, verifica cada entidad con una segunda llamada al LLM.
        batch_size: Número de oraciones por batch.
    """

    def __init__(
        self,
        llm: BaseLLM,
        entity_types: List[str],
        prompt_template: Optional[NERPromptTemplate] = None,
        strategy: str = "zero_shot",
        knn_retriever=None,
        knn_k: int = 5,
        self_verification: bool = False,
        batch_size: int = 4,
    ):
        super().__init__(model_name=f"{llm.model_name}_ner")
        self.llm = llm
        self.entity_types = entity_types
        self.strategy = strategy
        self.knn_retriever = knn_retriever
        self.knn_k = knn_k
        self.self_verification = self_verification
        self.batch_size = batch_size

        self.prompt_template = prompt_template or NERPromptTemplate(
            entity_types=entity_types,
            strategy=strategy,
        )

    def predict(self, sentences: List[List[str]]) -> List[List[str]]:
        """
        Predice BIO labels para una lista de oraciones tokenizadas.

        Returns:
            Lista de listas de BIO labels, alineada token a token con `sentences`.
        """
        all_predictions = []

        for i in range(0, len(sentences), self.batch_size):
            batch = sentences[i : i + self.batch_size]

            # Construir prompts (con k-NN si aplica)
            prompts = []
            knn_examples_batch = []
            for tokens in batch:
                knn_examples = None
                if self.strategy == "knn_few_shot" and self.knn_retriever is not None:
                    query = " ".join(tokens)
                    knn_examples = self.knn_retriever.retrieve_examples(query, k=self.knn_k)
                knn_examples_batch.append(knn_examples)
                prompts.append(
                    self.prompt_template.create_prompt(
                        tokens=tokens, knn_examples=knn_examples
                    )
                )

            responses = self.llm.batch_generate(
                prompts=prompts, max_tokens=512
            )

            for tokens, response in zip(batch, responses):
                bio_labels = self.prompt_template.parse_response(
                    response=response, tokens=tokens
                )

                if self.self_verification:
                    bio_labels = self._verify_entities(tokens, bio_labels)

                all_predictions.append(bio_labels)

        return all_predictions

    # ------------------------------------------------------------------
    # Self-verification
    # ------------------------------------------------------------------

    def _verify_entities(
        self, tokens: List[str], bio_labels: List[str]
    ) -> List[str]:
        """
        Verifica cada entidad extraída con una segunda llamada al LLM.
        Las entidades rechazadas se convierten en 'O'.

        Args:
            tokens: Tokens originales.
            bio_labels: BIO labels predichos antes de verificación.

        Returns:
            BIO labels corregidos.
        """
        sentence = " ".join(tokens)
        entities = self._extract_spans(tokens, bio_labels)
        if not entities:
            return bio_labels

        # Verificar cada entidad individualmente
        verified: dict[tuple, bool] = {}
        for start, end, entity_text, entity_type in entities:
            prompt = self.prompt_template.build_verification_prompt(
                sentence=sentence,
                entity=entity_text,
                entity_type=entity_type,
            )
            response = self.llm.generate(prompt=prompt, max_tokens=8).strip().upper()
            verified[(start, end)] = response.startswith("SÍ") or response.startswith("SI") or response == "YES"

        # Aplicar decisiones: eliminar entidades rechazadas
        corrected = list(bio_labels)
        for start, end, _, _ in entities:
            if not verified.get((start, end), True):
                for idx in range(start, end + 1):
                    corrected[idx] = "O"

        return corrected

    @staticmethod
    def _extract_spans(
        tokens: List[str], bio_labels: List[str]
    ) -> List[tuple]:
        """
        Extrae spans (start, end, texto, tipo) desde etiquetas BIO.

        Returns:
            Lista de tuplas (start_idx, end_idx, entity_text, entity_type).
        """
        spans = []
        i = 0
        while i < len(bio_labels):
            label = bio_labels[i]
            if label.startswith("B-"):
                entity_type = label[2:]
                start = i
                end = i
                j = i + 1
                while j < len(bio_labels) and bio_labels[j] == f"I-{entity_type}":
                    end = j
                    j += 1
                entity_text = " ".join(tokens[start : end + 1])
                spans.append((start, end, entity_text, entity_type))
                i = j
            else:
                i += 1
        return spans

    def __repr__(self) -> str:
        return (
            f"LLMNERModel(\n"
            f"  llm={self.llm.model_name},\n"
            f"  strategy='{self.strategy}',\n"
            f"  self_verification={self.self_verification},\n"
            f"  knn_k={self.knn_k},\n"
            f"  batch_size={self.batch_size}\n"
            f")"
        )
