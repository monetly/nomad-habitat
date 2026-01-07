from dataclasses import dataclass
from typing import Dict, List, Optional
import hashlib

from .base64 import image_to_data_url


@dataclass
class ImageEntry:
    index: int
    data_url: str
    description: Optional[str] = None
    tag: Optional[str] = None


class ImageMemory:
    def __init__(self) -> None:
        self._entries: Dict[int, ImageEntry] = {}
        self._hash_to_index: Dict[str, int] = {}
        self._next_index = 0

    def add_image(
        self,
        image,
        description: Optional[str] = None,
        tag: Optional[str] = None,
    ) -> int:
        data_url = image_to_data_url(image)
        digest = hashlib.sha1(data_url.encode("utf-8")).hexdigest()
        if digest in self._hash_to_index:
            idx = self._hash_to_index[digest]
            entry = self._entries[idx]
            if description and not entry.description:
                entry.description = description
            if tag and not entry.tag:
                entry.tag = tag
            return idx

        idx = self._next_index
        self._next_index += 1
        self._hash_to_index[digest] = idx
        self._entries[idx] = ImageEntry(
            index=idx,
            data_url=data_url,
            description=description,
            tag=tag,
        )
        return idx

    def get_entry(self, index: int) -> Optional[ImageEntry]:
        return self._entries.get(index)

    def get_data_url(self, index: int) -> Optional[str]:
        entry = self._entries.get(index)
        return entry.data_url if entry else None

    def list_entries(self) -> List[ImageEntry]:
        return [self._entries[idx] for idx in sorted(self._entries)]

    def find_index(self, image) -> Optional[int]:
        data_url = image_to_data_url(image)
        digest = hashlib.sha1(data_url.encode("utf-8")).hexdigest()
        return self._hash_to_index.get(digest)

    def set_description(self, index: int, description: str) -> None:
        entry = self._entries.get(index)
        if not entry:
            raise KeyError(f"Unknown image index: {index}")
        entry.description = description
