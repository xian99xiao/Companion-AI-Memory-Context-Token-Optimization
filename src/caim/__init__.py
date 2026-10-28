"""Companion-AI-Memory-Context-Token-Optimization

基于人脑记忆机制（工作记忆 / 情景记忆 / 语义记忆 + 遗忘曲线 + 巩固）
的 LLM 上下文管理：省 token，且不失忆。
"""

from __future__ import annotations

__version__ = "0.1.0"

from .activation import activation, retention, stability_hours
from .compress import (
    ExtractiveSummarizer,
    NullSummarizer,
    Summarizer,
    extract_facts,
)
from .context import ContextBuilder
from .llm import LLM, RuleLLM, grade
from .memory import CompanionMemory
from .models import (
    KIND_EPISODE,
    KIND_FACT,
    KIND_RAW,
    ContextBundle,
    MemoryItem,
)
from .retrieval import bigrams, relevance, retrieve, similarity
from .store import MemoryStore
from .tokens import estimate_tokens

__all__ = [
    "__version__",
    "CompanionMemory",
    "MemoryStore",
    "ContextBuilder",
    "MemoryItem",
    "ContextBundle",
    "KIND_FACT",
    "KIND_EPISODE",
    "KIND_RAW",
    "ExtractiveSummarizer",
    "NullSummarizer",
    "Summarizer",
    "extract_facts",
    "activation",
    "retention",
    "stability_hours",
    "retrieve",
    "relevance",
    "similarity",
    "bigrams",
    "estimate_tokens",
    "LLM",
    "RuleLLM",
    "grade",
]
