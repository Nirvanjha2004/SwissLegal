from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class TextChunk:
    chunk_id: int
    source_id: str
    text: str


def chunk_text(text: str, chunk_size: int = 4000, overlap: int = 200) -> list[str]:
    if not text:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: list[str] = []
    start = 0
    length = len(text)
    step = chunk_size - overlap

    while start < length:
        end = min(length, start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= length:
            break
        start += step

    return chunks


def chunk_records(records: Iterable[tuple[str, str]], chunk_size: int = 4000, overlap: int = 200) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    chunk_id = 0
    for source_id, text in records:
        for chunk in chunk_text(text, chunk_size=chunk_size, overlap=overlap):
            chunks.append(TextChunk(chunk_id=chunk_id, source_id=source_id, text=chunk))
            chunk_id += 1
    return chunks
