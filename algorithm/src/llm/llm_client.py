from openai import OpenAI
from typing import List, Dict, Optional


class LLMClient:
    def __init__(
        self,
        base_url: str = "http://localhost:8081/v1",
        api_key: str = "sk-no-key-required",
        model: str = "qwen",
        temperature: float = 0.2,
    ) -> None:
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.temperature = temperature

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = 256,
        temperature: Optional[float] = None,
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()
