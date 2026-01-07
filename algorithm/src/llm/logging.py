import json
import time
from typing import Any, Dict, Optional


class PromptLogger:
    def __init__(self, log_path: str) -> None:
        self.log_path = log_path

    def log(self, event: str, data: Optional[Dict[str, Any]] = None) -> None:
        payload: Dict[str, Any] = {
            "ts": time.time(),
            "event": event,
        }
        if data:
            payload.update(data)
        line = json.dumps(payload, ensure_ascii=True)
        with open(self.log_path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
