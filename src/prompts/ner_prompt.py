"""
ner_prompt.py

Prompt templates para NER en textos médicos en español (AnatEM).

Estrategias disponibles:
  zero_shot          — Sin ejemplos
  few_shot           — Ejemplos fijos hardcodeados
  knn_few_shot       — Ejemplos dinámicos recuperados por k-NN (requiere KNNRetriever)

Self-verification:
  build_verification_prompt() — Verifica cada entidad extraída con una segunda llamada al LLM.
  +2-5% F1 según GPT-NER paper.
"""

import json
import re
from typing import Dict, List, Optional

from prompts.base_prompt import BasePromptTemplate


class NERPromptTemplate(BasePromptTemplate):
    """
    Prompt para NER anatómico en español.

    Soporta zero-shot, few-shot fijo y k-NN few-shot.
    """

    _ZERO_SHOT = """\
Eres un sistema experto en reconocimiento de entidades de anatomía médica en español.

Tipos de entidades válidos: {entity_types}

Extrae TODAS las entidades del texto. Responde SOLO con un array JSON:
[{{"entity": "texto_exacto", "type": "tipo"}}]
Si no hay entidades, responde: []

Texto: "{sentence}"

JSON:"""

    _FEW_SHOT = """\
Eres un sistema experto en reconocimiento de entidades de anatomía médica en español.

Tipos de entidades válidos: {entity_types}

EJEMPLOS:

Texto: "Se observó una lesión en el ventrículo izquierdo y el miocardio."
JSON: [{{"entity": "ventrículo izquierdo", "type": "Organ"}}, {{"entity": "miocardio", "type": "Tissue"}}]

Texto: "El paciente presentó metástasis en el hígado y el pulmón derecho."
JSON: [{{"entity": "hígado", "type": "Organ"}}, {{"entity": "pulmón derecho", "type": "Organ"}}]

Texto: "Se detectó actividad en la corteza cerebral y células T."
JSON: [{{"entity": "corteza cerebral", "type": "Multi-tissue_structure"}}, {{"entity": "células T", "type": "Cell"}}]

Ahora analiza:
Texto: "{sentence}"
JSON:"""

    _KNN_FEW_SHOT = """\
Eres un sistema experto en reconocimiento de entidades de anatomía médica en español.

Tipos de entidades válidos: {entity_types}

EJEMPLOS DE ENTRENAMIENTO (ordenados por similitud con el texto):
{examples_block}

TEXTO A ANALIZAR:
Texto: "{sentence}"
JSON:"""

    _VERIFICATION = """\
Eres un verificador experto de entidades biomédicas en español.

Tipos válidos: {entity_types}

Verifica si la entidad extraída es correcta en el contexto del texto.

Texto: "{sentence}"
Entidad extraída: "{entity}" (tipo: {entity_type})

¿Es "{entity}" realmente una entidad de tipo {entity_type} en este texto?
Responde SOLO: SÍ o NO

Respuesta:"""

    def __init__(
        self,
        entity_types: List[str],
        language: str = "es",
        strategy: str = "zero_shot",
    ):
        super().__init__(language=language)
        self.raw_entity_types = entity_types
        self.strategy = strategy

        # Nombres limpios sin prefijos BIO
        type_names = sorted(set(
            et.replace("B-", "").replace("I-", "")
            for et in entity_types
            if et != "O"
        ))
        self.entity_types_str = ", ".join(type_names)
        self.type_names = type_names

    def create_prompt(
        self,
        tokens: List[str],
        knn_examples: Optional[List[Dict]] = None,
        **kwargs,
    ) -> str:
        """
        Crea el prompt NER para una oración tokenizada.

        Args:
            tokens: Lista de tokens de la oración.
            knn_examples: Ejemplos recuperados por k-NN (solo para strategy='knn_few_shot').

        Returns:
            Prompt formateado.
        """
        sentence = " ".join(tokens)

        if self.strategy == "knn_few_shot" and knn_examples:
            examples_block = self._format_knn_examples(knn_examples)
            return self._KNN_FEW_SHOT.format(
                entity_types=self.entity_types_str,
                examples_block=examples_block,
                sentence=sentence,
            )
        elif self.strategy == "few_shot":
            return self._FEW_SHOT.format(
                entity_types=self.entity_types_str,
                sentence=sentence,
            )
        else:
            return self._ZERO_SHOT.format(
                entity_types=self.entity_types_str,
                sentence=sentence,
            )

    def build_verification_prompt(
        self, sentence: str, entity: str, entity_type: str
    ) -> str:
        """
        Construye un prompt de auto-verificación para una entidad.

        Args:
            sentence: Oración original.
            entity: Texto de la entidad a verificar.
            entity_type: Tipo predicho.

        Returns:
            Prompt de verificación.
        """
        return self._VERIFICATION.format(
            entity_types=self.entity_types_str,
            sentence=sentence,
            entity=entity,
            entity_type=entity_type,
        )

    def parse_response(self, response: str, tokens: List[str]) -> List[str]:
        """
        Parsea la respuesta del LLM y la alinea como BIO labels a los tokens.

        Args:
            response: Respuesta cruda del modelo.
            tokens: Tokens originales de la oración.

        Returns:
            Lista de BIO labels alineada con tokens.
        """
        labels = ["O"] * len(tokens)
        entities = self._extract_json(response)
        if not entities:
            return labels

        for item in entities:
            entity_text = str(item.get("entity", "")).lower().strip()
            entity_type = str(item.get("type", "Anatomical_entity")).strip()
            if not entity_text:
                continue

            matched = self._find_token_span(tokens, entity_text.split())
            if matched is not None:
                start, end = matched
                labels[start] = f"B-{entity_type}"
                for i in range(start + 1, end + 1):
                    labels[i] = f"I-{entity_type}"

        return labels

    # ------------------------------------------------------------------
    # Helpers privados
    # ------------------------------------------------------------------

    def _format_knn_examples(self, examples: List[Dict]) -> str:
        lines = []
        for i, ex in enumerate(examples, 1):
            sentence = ex.get("sentence", "")
            entities = ex.get("entities", [])
            entities_json = json.dumps(
                [{"entity": e[0], "type": e[1]} for e in entities],
                ensure_ascii=False,
            )
            lines.append(f'Ejemplo {i}:\nTexto: "{sentence}"\nJSON: {entities_json}')
        return "\n\n".join(lines)

    def _extract_json(self, response: str) -> list:
        """Extrae un array JSON de la respuesta."""
        # Limpiar markdown fences
        response = re.sub(r"```json|```", "", response).strip()

        match = re.search(r"\[.*?\]", response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return []

    @staticmethod
    def _find_token_span(
        tokens: List[str], entity_tokens: List[str]
    ) -> Optional[tuple]:
        """Busca el span de una entidad en la lista de tokens (case-insensitive)."""
        n = len(entity_tokens)
        tokens_lower = [t.lower() for t in tokens]
        for i in range(len(tokens_lower) - n + 1):
            if tokens_lower[i: i + n] == entity_tokens:
                return (i, i + n - 1)
        return None
