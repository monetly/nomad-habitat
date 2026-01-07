from dataclasses import dataclass
from typing import Dict, List, Optional

from .logging import PromptLogger


@dataclass
class PromptBlock:
    name: str
    content: str


class StructuredPrompt:
    def __init__(self, system: Optional[str] = None, logger: Optional[PromptLogger] = None) -> None:
        self.system = system or ""
        self._sections: Dict[str, str] = {}
        self._order: List[str] = []
        self._increments: List[PromptBlock] = []
        self._logger = logger
        if self.system:
            self._log("system_set", {"content": self.system})

    def set_section(self, name: str, content: str) -> None:
        if name not in self._sections:
            self._order.append(name)
        self._sections[name] = content
        self._log("section_set", {"name": name, "content": content})

    def add_increment(self, name: str, content: str) -> None:
        self._increments.append(PromptBlock(name=name, content=content))
        self._log("increment_added", {"name": name, "content": content})

    def build_user_content(self) -> str:
        lines: List[str] = []
        for name in self._order:
            lines.append(f"## {name}")
            lines.append(self._sections[name])
        for block in self._increments:
            lines.append(f"## Update: {block.name}")
            lines.append(block.content)
        return "\n".join(lines).strip()

    def to_messages(self) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = []
        if self.system:
            messages.append({"role": "system", "content": self.system})
        user_content = self.build_user_content()
        if user_content:
            messages.append({"role": "user", "content": user_content})
        self._log("messages_built", {"message_count": len(messages)})
        return messages

    def _log(self, event: str, data: Dict[str, str]) -> None:
        if self._logger:
            self._logger.log(event, data)


class PromptRegistry:
    def __init__(self, logger: Optional[PromptLogger] = None) -> None:
        self._prompts: Dict[str, StructuredPrompt] = {}
        self._active_name: Optional[str] = None
        self._logger = logger

    def register(self, name: str, prompt: StructuredPrompt) -> None:
        self._prompts[name] = prompt
        if self._active_name is None:
            self._active_name = name
        if self._logger:
            self._logger.log("prompt_registered", {"name": name})

    def activate(self, name: str) -> None:
        if name not in self._prompts:
            raise KeyError(f"Prompt not found: {name}")
        self._active_name = name
        if self._logger:
            self._logger.log("prompt_activated", {"name": name})

    def get(self, name: str) -> StructuredPrompt:
        return self._prompts[name]

    def active(self) -> StructuredPrompt:
        if self._active_name is None:
            raise ValueError("No active prompt registered.")
        return self._prompts[self._active_name]

    def names(self) -> List[str]:
        return list(self._prompts.keys())
