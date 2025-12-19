from openai import OpenAI
from keys import OPENAI_API_KEY

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

import time
from tqdm import tqdm

class OpenAIGenerator:

    def __init__(self, model_name : str, logger = None):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model_name = model_name
        self.logger = logger
    
    def generate(self, prompts: list[str]) -> list[str]:
        """Generate responses for a list of prompts using OpenAI API."""
        if isinstance(prompts, str):
            prompts = [prompts]
        responses = []
        for i, prompt in enumerate(prompts):
            try_cnt = 0
            while(1):
                try:
                    response = self.client.responses.create(
                                    model=self.model_name,
                                    input=prompt
                    )
                    responses.append(response.output_text)
                    break
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"[OpenAIGenerator] Error on prompt {i+1}/{len(prompts)}: {e}, retrying in 30 seconds...")
                    try_cnt += 1
                    if try_cnt >= 120:
                        if self.logger:
                            self.logger.error(f"[OpenAIGenerator] Failed to get response after {try_cnt} attempts, skipping this prompt.")
                        responses.append("")  # Append empty response on failure
                        break
                    time.sleep(30)
                    continue

            if self.logger:
                self.logger.info(f"[OpenAIGenerator] ({i+1}/{len(prompts)}) generation done\n- Prompt:\n{prompt}\n- Response:\n{response.output_text}")
        return responses

class VLLMGenerator:
    """Generate responses using vLLM for open-source LLM models."""

    def __init__(self, model_name: str, api_base: str = "http://localhost:8000/v1", logger=None):
        from openai import OpenAI

        self.client = OpenAI(api_key="EMPTY", base_url=api_base)
        if model_name == "exaone":
            self.model_name = "LGAI-EXAONE/EXAONE-4.0-1.2B"
        elif model_name == "midm":
            self.model_name = "K-intelligence/Midm-2.0-Mini-Instruct"
        elif model_name == "hyperclovax":
            self.model_name = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B"
        elif model_name == "qwen":
            self.model_name = "Qwen/Qwen2.5-7B-Instruct"
        else:
            raise ValueError(f"Unsupported model_name: {model_name}. Supported models are: exaone, midm, hyperclovax, qwen.")
        self.logger = logger

        if self.logger:
            self.logger.info(
                f"[VLLMGenerator] Initialized with model: {model_name}, API base: {api_base}"
            )

    def generate(
        self,
        prompts: list[str],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> list[str]:

        if isinstance(prompts, str):
            prompts = [prompts]

        responses = []

        for i, prompt in enumerate(prompts):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{
                        "role": "user",
                        "content": [{"type": "text", "text": prompt}],
                    }],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                generated_text = (
                    response.choices[0].message.content or ""
                ).strip()
                responses.append(generated_text)

                if self.logger:
                    self.logger.info(
                        f"[VLLMGenerator] ({i+1}/{len(prompts)}) done\n"
                        f"- Prompt:\n{prompt}\n- Response:\n{generated_text}"
                    )

            except Exception as e:
                if self.logger:
                    self.logger.error(
                        f"[VLLMGenerator] Error on prompt {i+1}/{len(prompts)}: {e}"
                    )
                raise

        return responses


