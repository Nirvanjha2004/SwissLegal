from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class AgentConfig:
    temperature: float = 0.0
    max_context_chunks: int = 5


def build_prompt(question: str, contexts: Iterable[str]) -> str:
    context_block = "\n\n".join(contexts)
    return (
        "You are a legal reasoning agent.\n"
        "Use only the provided context to answer the question.\n"
        "Return a concise answer.\n\n"
        f"Question:\n{question}\n\n"
        f"Context:\n{context_block}\n\n"
        "Answer:"
    )


def parse_model_output(text: str) -> str:
    return " ".join(str(text).split()).strip()


def run_agent(question: str, contexts: Iterable[str], llm) -> str:
    prompt = build_prompt(question, contexts)
    response = llm.invoke(prompt) if hasattr(llm, "invoke") else llm(prompt)
    return parse_model_output(response)
