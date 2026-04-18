"""
llm_ner_model.py

Zero-shot / few-shot NER using a decoder LLM.

Soporta:
  - zero_shot / few_shot / knn_few_shot  (via NERPromptTemplate)
  - self_verification: segunda llamada al LLM por cada entidad extraída
                       para filtrar falsos positivos (+2-5% F1 según GPT-NER)
  - Chunking automático: textos largos se parten en chunks solapados
                         para mejorar recall en párrafos extensos.
"""

from typing import List, Optional, Tuple

from tqdm import tqdm

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
        max_chunk_tokens: Máximo de tokens (palabras) por chunk. Textos más largos
                          se parten en chunks con solapamiento.
        chunk_overlap: Número de tokens de solapamiento entre chunks consecutivos.
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
        max_chunk_tokens: int = 80,
        chunk_overlap: int = 10,
        structured_generation: bool = False,
    ):
        super().__init__(model_name=f"{llm.model_name}_ner")
        self.llm = llm
        self.entity_types = entity_types
        self.strategy = strategy
        self.knn_retriever = knn_retriever
        self.knn_k = knn_k
        self.self_verification = self_verification
        self.batch_size = batch_size
        self.max_chunk_tokens = max_chunk_tokens
        self.chunk_overlap = chunk_overlap
        self.structured_generation = structured_generation

        self.prompt_template = prompt_template or NERPromptTemplate(
            entity_types=entity_types,
            strategy=strategy,
        )

        # Build JSON schema for constrained decoding (entity types as enum)
        self._ner_json_schema = self._build_ner_schema()

        # Stores raw LLM responses for the last predict() call (debugging)
        self.last_raw_responses: List[List[str]] = []

    def _build_ner_schema(self) -> dict:
        """Build a JSON schema that constrains output to valid NER entities."""
        type_names = self.prompt_template.type_names
        return {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "entity": {"type": "string"},
                    "type": {"type": "string", "enum": type_names},
                },
                "required": ["entity", "type"],
            },
        }

    # ------------------------------------------------------------------
    # Chunking helpers
    # ------------------------------------------------------------------

    def _chunk_sentence(
        self, tokens: List[str]
    ) -> List[Tuple[int, int]]:
        """Split a long token list into overlapping (start, end) spans."""
        n = len(tokens)
        if n <= self.max_chunk_tokens:
            return [(0, n)]
        spans = []
        start = 0
        step = self.max_chunk_tokens - self.chunk_overlap
        while start < n:
            end = min(start + self.max_chunk_tokens, n)
            spans.append((start, end))
            if end == n:
                break
            start += step
        return spans

    @staticmethod
    def _merge_chunk_labels(
        total_len: int,
        spans: List[Tuple[int, int]],
        chunk_labels_list: List[List[str]],
    ) -> List[str]:
        """Merge BIO labels from overlapping chunks.

        For overlapping regions, prefer non-O labels.  When both chunks
        assign an entity tag the earlier chunk wins (it saw more left
        context for that span).
        """
        merged = ["O"] * total_len
        for (start, _end), chunk_labels in zip(spans, chunk_labels_list):
            for j, label in enumerate(chunk_labels):
                idx = start + j
                if idx < total_len and merged[idx] == "O":
                    merged[idx] = label
        return merged

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, sentences: List[List[str]]) -> List[List[str]]:
        """Predice BIO labels para una lista de oraciones tokenizadas.

        Textos más largos que ``max_chunk_tokens`` se parten
        automáticamente en chunks solapados para mejorar el recall.

        Returns:
            Lista de listas de BIO labels, alineada token a token con `sentences`.
        """
        self.llm.model.eval()
        self.last_raw_responses = []

        # Flatten: expand each sentence into its chunks
        flat_tokens: List[List[str]] = []
        flat_knn: List[Optional[list]] = []
        sentence_plan: List[List[Tuple[int, int]]] = []  # spans per sentence

        for tokens in sentences:
            spans = self._chunk_sentence(tokens)
            sentence_plan.append(spans)
            knn_examples = None
            if self.strategy == "knn_few_shot" and self.knn_retriever is not None:
                query = " ".join(tokens)
                knn_examples = self.knn_retriever.retrieve_examples(query, k=self.knn_k)
            for start, end in spans:
                flat_tokens.append(tokens[start:end])
                flat_knn.append(knn_examples)

        # Build prompts
        prompts = [
            self.prompt_template.create_prompt(tokens=chunk_toks, knn_examples=knn_ex)
            for chunk_toks, knn_ex in zip(flat_tokens, flat_knn)
        ]

        # Generate responses
        flat_responses: List[str] = []
        use_structured = (
            self.structured_generation
            and hasattr(self.llm, "generate_structured")
        )

        if use_structured:
            import json as json_mod
            schema = json_mod.dumps(self._ner_json_schema)
            for prompt in tqdm(prompts, desc="NER inference (structured)"):
                resp = self.llm.generate_structured(
                    prompt=prompt, json_schema=schema, max_tokens=512
                )
                flat_responses.append(resp if isinstance(resp, str) else json_mod.dumps(resp, ensure_ascii=False))
        else:
            n_batches = (len(prompts) + self.batch_size - 1) // self.batch_size
            for i in tqdm(range(0, len(prompts), self.batch_size), total=n_batches, desc="NER inference"):
                batch_prompts = prompts[i : i + self.batch_size]
                batch_responses = self.llm.batch_generate(
                    prompts=batch_prompts, max_tokens=512
                )
                flat_responses.extend(batch_responses)

        # Parse responses and re-assemble per sentence
        resp_idx = 0
        all_predictions: List[List[str]] = []
        for tokens, spans in zip(sentences, sentence_plan):
            chunk_labels_list = []
            raw_for_sentence = []
            for start, end in spans:
                response = flat_responses[resp_idx]
                raw_for_sentence.append(response)
                chunk_toks = tokens[start:end]
                bio = self.prompt_template.parse_response(
                    response=response, tokens=chunk_toks
                )
                chunk_labels_list.append(bio)
                resp_idx += 1

            self.last_raw_responses.append(raw_for_sentence)

            if len(spans) == 1:
                bio_labels = chunk_labels_list[0]
            else:
                bio_labels = self._merge_chunk_labels(
                    len(tokens), spans, chunk_labels_list
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
            f"  structured_generation={self.structured_generation},\n"
            f"  knn_k={self.knn_k},\n"
            f"  max_chunk_tokens={self.max_chunk_tokens},\n"
            f"  batch_size={self.batch_size}\n"
            f")"
        )
