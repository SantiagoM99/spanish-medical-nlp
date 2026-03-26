"""
openai_llm.py

OpenAI / Azure OpenAI wrapper que implementa BaseLLM.
Soporta tanto la API estándar de OpenAI como Azure OpenAI.

Configuración via variables de entorno o argumentos explícitos:
  OPENAI_API_KEY            → clave OpenAI estándar
  AZURE_OPENAI_ENDPOINT     → URL del endpoint Azure
  AZURE_OPENAI_API_KEY      → clave Azure
  AZURE_OPENAI_API_VERSION  → versión API Azure (default: 2024-10-21)

Uso:
    # OpenAI estándar
    llm = OpenAILLM(model_name="gpt-4o-mini")

    # Azure OpenAI
    llm = OpenAILLM(
        model_name="gpt-4o-mini",
        use_azure=True,
        azure_endpoint="https://...",
        api_key="...",
    )
"""

import json
import os
import time
from typing import List, Optional

from models.base_llm import BaseLLM


class OpenAILLM(BaseLLM):
    """
    Wrapper unificado para OpenAI y Azure OpenAI.
    Implementa la misma interfaz que HuggingFaceLLM.
    """

    def __init__(
        self,
        model_name: str,
        use_azure: bool = False,
        azure_endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        api_version: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        system_prompt: str = "You are a helpful medical NLP assistant.",
        **kwargs,
    ):
        super().__init__(model_name, **kwargs)
        self.temperature = temperature
        self.default_max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.use_azure = use_azure

        if use_azure:
            from openai import AzureOpenAI
            self.client = AzureOpenAI(
                azure_endpoint=azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=api_key or os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=api_version or os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
            )
        else:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=api_key or os.getenv("OPENAI_API_KEY"),
            )

        backend = "Azure" if use_azure else "OpenAI"
        print(f"OpenAILLM initialized | backend={backend} | model={model_name}")

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Generate a single response via chat completions.

        Args:
            prompt: User message content.
            max_tokens: Override default max tokens.
            temperature: Override default temperature.
            system_prompt: Override default system prompt.

        Returns:
            Model response as string.
        """
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt or self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=max_tokens or self.default_max_tokens,
            top_p=1.0,
            **kwargs,
        )
        return response.choices[0].message.content.strip()

    def batch_generate(
        self,
        prompts: List[str],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None,
        sleep_between: float = 0.03,
        **kwargs,
    ) -> List[str]:
        """
        Generate responses for multiple prompts (sequential with rate-limit delay).

        Args:
            prompts: List of user prompts.
            sleep_between: Seconds to sleep between requests (avoid rate limiting).

        Returns:
            List of response strings.
        """
        responses = []
        for prompt in prompts:
            response = self.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                system_prompt=system_prompt,
                **kwargs,
            )
            responses.append(response)
            if sleep_between > 0:
                time.sleep(sleep_between)
        return responses

    def generate_json(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
    ) -> dict | list:
        """
        Generate and parse a JSON response.
        Strips markdown code fences if present.

        Returns:
            Parsed Python object (dict or list).

        Raises:
            json.JSONDecodeError if response is not valid JSON.
        """
        raw = self.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            system_prompt=system_prompt or "You are a precise assistant. Respond only with valid JSON.",
        )
        return json.loads(self._clean_json(raw))

    @staticmethod
    def _clean_json(content: str) -> str:
        """Remove markdown code fences from a JSON response."""
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        return content.strip()

    def __repr__(self) -> str:
        backend = "Azure" if self.use_azure else "OpenAI"
        return (
            f"OpenAILLM(\n"
            f"  model_name='{self.model_name}',\n"
            f"  backend='{backend}',\n"
            f"  temperature={self.temperature}\n"
            f")"
        )
