from openai import OpenAI
from keys import OPENAI_API_KEY

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
            response = self.client.responses.create(
                            model=self.model_name,
                            input=prompt
            )
            responses.append(response.output_text)

            if self.logger:
                self.logger.info(f"[OpenAIGenerator] ({i+1}/{len(prompts)}) generation done\n- Prompt:\n{prompt}\n- Response:\n{response.output_text}")
        return responses
