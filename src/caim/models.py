"""核心数据结构。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

KIND_FACT = "fact"
"""语义记忆：原子事实，长期驻留，importance 高。"""
KIND_EPISODE = "episode"
"""情景记忆：一段对话的摘要。"""
KIND_RAW = "raw"
"""原始片段：尚未被巩固的对话原文（通常只在工作记忆里，落库较少）。"""


@dataclass(slots=True)
class MemoryItem:
    """一条记忆。

    ``stability`` 相关字段由 :mod:`caim.activation` 读取：
    ``access_count`` 越大、``importance`` 越高，遗忘越慢。
    """

    id: int
    kind: str
    text: str
    tokens: int
    importance: float
    created_at: float
    last_access: float
    access_count: int = 0
    parent_id: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def short(self, limit: int = 60) -> str:
        text = self.text.replace("\n", " ")
        return text if len(text) <= limit else text[: limit - 1] + "…"


@dataclass(slots=True)
class ContextBundle:
    """组装好的上下文，附带用量明细，便于观测与调参。"""

    text: str
    tokens: int
    sections: dict[str, int]
    items: list[MemoryItem]
    dropped: list[str] = field(default_factory=list)

    def __str__(self) -> str:  # pragma: no cover - 便于调试
        parts = [f"tokens={self.tokens}"]
        parts += [f"{k}={v}" for k, v in self.sections.items()]
        if self.dropped:
            parts.append(f"dropped={len(self.dropped)}")
        return "<ContextBundle " + " ".join(parts) + ">"
