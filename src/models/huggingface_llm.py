"""
HuggingFace decoder LLM wrapper for zero-shot / few-shot inference.
Supports quantization (8-bit, 4-bit/QLoRA) via BitsAndBytes.
"""

from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from models.base_llm import BaseLLM


class HuggingFaceLLM(BaseLLM):
    """
    Generic wrapper for HuggingFace causal language models.

    Supports Llama, Mistral, Qwen, Gemma, etc. with optional 4/8-bit quantization.
    """

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        torch_dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = False,
        use_flash_attention: bool = False,
        **kwargs,
    ):
        super().__init__(model_name, **kwargs)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        quantization_config = None
        if load_in_8bit or load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=load_in_8bit,
                load_in_4bit=load_in_4bit,
                bnb_4bit_compute_dtype=torch_dtype or torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        model_kwargs = {
            "quantization_config": quantization_config,
            "torch_dtype": torch_dtype or torch.float16,
            "trust_remote_code": trust_remote_code,
            "device_map": "auto" if quantization_config else None,
        }
        if use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        quant_str = "8-bit" if load_in_8bit else "4-bit" if load_in_4bit else "None"
        print(f"Loading {model_name} | device={self.device} | quant={quant_str}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

        if not quantization_config:
            self.model.to(self.device)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model.eval()
        print(f"Model loaded successfully.")

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        do_sample: bool = False,
        top_p: float = 0.9,
        **kwargs,
    ) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=do_sample,
                top_p=top_p if do_sample else None,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **kwargs,
            )
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        return response.strip()

    def batch_generate(
        self,
        prompts: List[str],
        max_tokens: int = 512,
        do_sample: bool = False,
        batch_size: int = 4,
        **kwargs,
    ) -> List[str]:
        responses = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    **kwargs,
                )
            for j, output in enumerate(outputs):
                prompt_len = inputs["input_ids"][j].shape[0]
                response = self.tokenizer.decode(
                    output[prompt_len:], skip_special_tokens=True
                )
                responses.append(response.strip())
        return responses
