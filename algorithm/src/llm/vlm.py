from openai import OpenAI
import re
from typing import Optional

from .base64 import image_to_data_url

# ./llama.cpp/llama-server \
#     -hf unsloth/Qwen3-VL-8B-Instruct-GGUF:UD-Q4_K_XL \
#     --host 0.0.0.0 \
#     --port 8081 \
#     --n-gpu-layers 99 \
#     --jinja \
#     --top-p 0.8 \
#     --top-k 20 \
#     --temp 0.7 \
#     --min-p 0.0 \
#     --flash-attn on \
#     --presence-penalty 1.5 \
#     --ctx-size 8192


class VLMClient:
    def __init__(
        self,
        base_url: str = "http://localhost:8081/v1",
        api_key: str = "sk-no-key-required",
        model: str = "qwen-vl",
        temperature: float = 0.2,
    ) -> None:
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.temperature = temperature

    def describe_image(self, image, prompt: str = "Describe the scene.") -> str:
        image_url = image_to_data_url(image)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
        )
        return response.choices[0].message.content.strip()

    def judge_relevance(
        self,
        current_image,
        target_image: Optional[object] = None,
        target_description: Optional[str] = None,
    ) -> bool:
        if target_image is None and not target_description:
            raise ValueError("Provide target_image or target_description.")

        content = [
            {
                "type": "text",
                "text": (
                    "Answer only with 'yes' or 'no'. "
                    "Is the current view relevant to the target goal?"
                ),
            },
            {"type": "image_url", "image_url": {"url": image_to_data_url(current_image)}},
        ]
        if target_image is not None:
            content.append(
                {"type": "image_url", "image_url": {"url": image_to_data_url(target_image)}}
            )
        if target_description:
            content.append({"type": "text", "text": f"Target description: {target_description}"})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            temperature=0.0,
        )
        text = response.choices[0].message.content.strip().lower()
        return bool(re.search(r"\byes\b", text))


if __name__ == "__main__":
    print("VLMClient ready. Provide images to describe or judge relevance.")
