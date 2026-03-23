"""
multilabel_prompt.py

Prompt template for zero-shot multi-label classification of Spanish medical abstracts
using MeSH level-1 category codes (A-N).

MeSH level-1 categories:
  A  Anatomy
  B  Organisms
  C  Diseases
  D  Chemicals and Drugs
  E  Analytical, Diagnostic and Therapeutic Techniques and Equipment
  F  Psychiatry and Psychology
  G  Phenomena and Processes
  H  Disciplines and Occupations
  I  Anthropology, Education, Sociology and Social Phenomena
  J  Technology, Industry, and Agriculture
  K  Humanities
  L  Information Science
  M  Named Groups
  N  Health Care
  V  Publication Characteristics
  Z  Geographicals
"""

import re
from typing import List

from prompts.base_prompt import BasePromptTemplate

# Human-readable MeSH level-1 category names for the prompt
MESH_LEVEL1_NAMES = {
    "A": "Anatomía",
    "B": "Organismos",
    "C": "Enfermedades",
    "D": "Sustancias Químicas y Fármacos",
    "E": "Técnicas Analíticas, Diagnósticas y Terapéuticas",
    "F": "Psiquiatría y Psicología",
    "G": "Fenómenos y Procesos",
    "H": "Disciplinas y Ocupaciones",
    "I": "Antropología, Educación y Sociología",
    "J": "Tecnología, Industria y Agricultura",
    "K": "Humanidades",
    "L": "Ciencias de la Información",
    "M": "Grupos Nombrados",
    "N": "Cuidados de Salud",
    "V": "Características de Publicación",
    "Z": "Geografía",
}


class MultiLabelPromptTemplate(BasePromptTemplate):
    """
    Prompt template for MeSH multi-label classification of medical abstracts.
    """

    ZERO_SHOT_ES = """\
Eres un experto en indexación biomédica con conocimiento profundo de MeSH (Medical Subject Headings).

Categorías MeSH disponibles:
{categories}

Clasifica el siguiente abstract médico en español asignándole los códigos MeSH de nivel 1 que correspondan.
Responde ÚNICAMENTE con los códigos separados por comas (ejemplo: A, C, D).
Si corresponde solo una categoría, escribe solo ese código.

Abstract:
{text}

Códigos MeSH:"""

    FEW_SHOT_ES = """\
Eres un experto en indexación biomédica con conocimiento profundo de MeSH (Medical Subject Headings).

Categorías MeSH disponibles:
{categories}

Ejemplos de clasificación:

Abstract: "Se estudió el efecto del ibuprofeno en el tratamiento de la artritis reumatoide en pacientes adultos."
Códigos: C, D

Abstract: "Análisis epidemiológico de la prevalencia de diabetes tipo 2 en América Latina entre 2010 y 2020."
Códigos: C, N

Abstract: "Técnicas de resonancia magnética para el diagnóstico de tumores cerebrales en edad pediátrica."
Códigos: A, C, E, M

Clasifica ahora:

Abstract:
{text}

Códigos MeSH:"""

    def __init__(
        self,
        available_labels: List[str],
        language: str = "es",
        few_shot: bool = False,
    ):
        super().__init__(language=language)
        self.available_labels = available_labels
        self.few_shot = few_shot

        # Build category description string
        lines = []
        for code in sorted(available_labels):
            name = MESH_LEVEL1_NAMES.get(code, code)
            lines.append(f"  {code}: {name}")
        self.categories_str = "\n".join(lines)

    def create_prompt(self, text: str, **kwargs) -> str:
        template = self.FEW_SHOT_ES if self.few_shot else self.ZERO_SHOT_ES
        return template.format(categories=self.categories_str, text=text)

    def parse_response(self, response: str) -> List[str]:
        """
        Parse the model response to extract valid label codes.

        Handles formats like: "A, C, D" / "A,C,D" / "Los códigos son: A, C"
        """
        response = response.strip().upper()

        # Extract single uppercase letters that are valid labels
        tokens = re.split(r"[,\s]+", response)
        labels = []
        seen = set()
        for token in tokens:
            code = re.sub(r"[^A-Z]", "", token)
            if code in self.available_labels and code not in seen:
                labels.append(code)
                seen.add(code)

        return labels

    def __repr__(self) -> str:
        return (
            f"MultiLabelPromptTemplate(\n"
            f"  n_labels={len(self.available_labels)},\n"
            f"  labels={self.available_labels},\n"
            f"  few_shot={self.few_shot}\n"
            f")"
        )
