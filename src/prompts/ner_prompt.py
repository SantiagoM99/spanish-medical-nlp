"""
ner_prompt.py

Prompt template for zero-shot NER with decoder LLMs on the AnatEM Spanish dataset.

The prompt asks the model to identify anatomical entities and return them
as a JSON list of {"entity": ..., "type": ...} objects.
The response is then parsed and aligned back to the input tokens.
"""

import json
import re
from typing import List

from prompts.base_prompt import BasePromptTemplate


class NERPromptTemplate(BasePromptTemplate):
    """
    Prompt for anatomical NER in Spanish medical text.

    Entity types from AnatEM:
      - Anatomical_entity (covering all sub-types for zero-shot)
      OR granular types: Organism_subdivision, Anatomical_system,
        Organ, Multi-tissue_structure, Tissue, Cell, Cellular_component,
        Developing_anatomical_structure, Organism_substance, Immaterial_anatomical_entity,
        Pathological_formation, Cancer
    """

    ZERO_SHOT_ES = """\
Eres un sistema de reconocimiento de entidades de anatomía médica.

Tipos de entidades anatómicas a identificar: {entity_types}

Dado el siguiente texto médico en español, identifica todas las entidades anatómicas.
Responde con una lista JSON de objetos con los campos "entidad" y "tipo".
Si no hay entidades, responde con [].

Texto: {sentence}

Entidades (JSON):"""

    ZERO_SHOT_EN = """\
You are a medical anatomy named entity recognition system.

Anatomical entity types to identify: {entity_types}

Given the following Spanish medical text, identify all anatomical entities.
Respond with a JSON list of objects with fields "entity" and "type".
If there are no entities, respond with [].

Text: {sentence}

Entities (JSON):"""

    FEW_SHOT_ES = """\
Eres un sistema de reconocimiento de entidades de anatomía médica.

Tipos de entidades: {entity_types}

Ejemplos:
Texto: "Se observó una lesión en el ventrículo izquierdo y el miocardio."
Entidades: [{{"entidad": "ventrículo izquierdo", "tipo": "Organ"}}, {{"entidad": "miocardio", "tipo": "Tissue"}}]

Texto: "El paciente presentó metástasis en el hígado y el pulmón derecho."
Entidades: [{{"entidad": "hígado", "tipo": "Organ"}}, {{"entidad": "pulmón derecho", "tipo": "Organ"}}]

Ahora analiza:
Texto: {sentence}
Entidades (JSON):"""

    def __init__(
        self,
        entity_types: List[str],
        language: str = "es",
        few_shot: bool = False,
    ):
        super().__init__(language=language)
        # Extract unique type names from BIO labels
        self.raw_entity_types = entity_types
        type_names = sorted(set(
            et.replace("B-", "").replace("I-", "")
            for et in entity_types
            if et != "O"
        ))
        self.entity_types_str = ", ".join(type_names)
        self.few_shot = few_shot

    def create_prompt(self, tokens: List[str], **kwargs) -> str:
        """
        Create a NER prompt for a tokenized sentence.

        Args:
            tokens: List of token strings.

        Returns:
            Formatted prompt string.
        """
        sentence = " ".join(tokens)

        if self.few_shot and self.language == "es":
            template = self.FEW_SHOT_ES
        elif self.language == "es":
            template = self.ZERO_SHOT_ES
        else:
            template = self.ZERO_SHOT_EN

        return template.format(
            entity_types=self.entity_types_str,
            sentence=sentence,
        )

    def parse_response(self, response: str, tokens: List[str]) -> List[str]:
        """
        Parse LLM response and align entities to BIO labels for input tokens.

        Strategy:
          1. Extract JSON from response.
          2. For each entity span, find matching tokens and assign BIO labels.
          3. Return label list aligned to input tokens.

        Args:
            response: Raw LLM output.
            tokens: Original input tokens.

        Returns:
            BIO label list aligned to tokens.
        """
        labels = ["O"] * len(tokens)
        sentence = " ".join(tokens).lower()

        # Extract JSON array from response
        entities = self._extract_json(response)
        if not entities:
            return labels

        for item in entities:
            entity_text = str(item.get("entidad", item.get("entity", ""))).lower().strip()
            entity_type = str(item.get("tipo", item.get("type", "Anatomical_entity"))).strip()

            if not entity_text:
                continue

            # Find entity span in token sequence
            entity_tokens = entity_text.split()
            matched = self._find_token_span(tokens, entity_tokens)
            if matched is not None:
                start, end = matched
                labels[start] = f"B-{entity_type}"
                for i in range(start + 1, end + 1):
                    labels[i] = f"I-{entity_type}"

        return labels

    def _extract_json(self, response: str) -> list:
        """Try to extract a JSON list from the response."""
        # Look for JSON array in response
        match = re.search(r"\[.*?\]", response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        # Fallback: try whole response
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            return []

    def _find_token_span(
        self, tokens: List[str], entity_tokens: List[str]
    ) -> tuple | None:
        """Find the start/end token indices of an entity span (case-insensitive)."""
        n = len(entity_tokens)
        tokens_lower = [t.lower() for t in tokens]
        for i in range(len(tokens_lower) - n + 1):
            if tokens_lower[i : i + n] == entity_tokens:
                return (i, i + n - 1)
        return None
