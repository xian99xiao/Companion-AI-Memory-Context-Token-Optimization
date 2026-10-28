"""门面：把「存储 / 巩固 / 装配」串成一个可用的对象。

对外只需要四个方法：``add_turn`` / ``build_context`` / ``recall`` / ``stats``。
"""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Callable, Iterable, Sequence

from .compress import ExtractiveSummarizer, Summarizer, extract_facts
from .context import ContextBuilder
from .models import KIND_EPISODE, KIND_FACT, ContextBundle, MemoryItem
from .retrieval import similarity
from .store import MemoryStore
from .tokens import TokenCounter, default_counter

FACT_IMPORTANCE = 0.82
EPISODE_IMPORTANCE = 0.42

# 这些话题在记忆里天然更重要（健康、身份、关系），提高保留优先级
_SENSITIVE_HINTS = ("过敏", "病", "药", "医院", "身份证", "密码", "生日", "家人", "妈妈", "爸爸")

# 助手的口头确认语，抽事实时要排除
_ASSISTANT_NOISE = ("记住了", "记下了", "知道了", "收到啦", "好的", "嗯嗯")


class CompanionMemory:
    """三层记忆 + 预算化上下文。

    Parameters
    ----------
    db_path:
        SQLite 路径；``":memory:"`` 表示不落盘（测试用）。
    context_budget:
        每轮注入上下文的 token 上限（估算值）。
    working_capacity:
        工作记忆容量（保留多少轮逐字原文），超出触发巩固。
    summarizer:
        摘要器，默认离线抽取式；接真实 LLM 时实现 ``Summarizer`` 协议。
    token_counter:
        自定义 token 计算函数。
    """

    def __init__(
        self,
        db_path: str | Path = ":memory:",
        *,
        context_budget: int = 1200,
        working_capacity: int = 6,
        summarizer: Summarizer | None = None,
        token_counter: TokenCounter | None = None,
        persona: str | None = None,
    ) -> None:
        self.store = MemoryStore(db_path)
        self.count = token_counter or default_counter()
        self.working_capacity = max(1, working_capacity)
        self.summarizer: Summarizer = summarizer or ExtractiveSummarizer()
        self.recent: deque[tuple[str, str]] = deque()
        self.builder = ContextBuilder(
            budget=context_budget,
            recent_turns=working_capacity,
            token_counter=self.count,
            **({"persona": persona} if persona else {}),
        )
        self._last_consolidation = 0.0

    # ------------------------------------------------------------------ 写入

    def add_turn(self, user: str, assistant: str = "", *, now: float | None = None) -> None:
        """记录一轮对话；工作记忆超出容量时自动巩固。"""
        ts = time.time() if now is None else now
        self.recent.append((user, assistant))
        # 明显是陈述句的用户输入，立即抽事实（不必等巩固）
        for fact in extract_facts(user):
            self._remember_fact(fact, ts)
        self._consolidate_if_needed(ts)
    def _consolidate_if_needed(self, now: float) -> None:
        if len(self.recent) <= self.working_capacity:
            return
        overflow = len(self.recent) - self.working_capacity
        chunk = [self.recent.popleft() for _ in range(overflow)]
        self.consolidate(chunk, now=now)

    def consolidate(
        self, turns: Sequence[tuple[str, str]], *, now: float | None = None
    ) -> MemoryItem | None:
        """把一段对话压成情景摘要 + 抽事实（= 睡眠巩固）。"""
        ts = time.time() if now is None else now
        # 用户的话和助手的话分开，避免拼成一句后抽出「… 嗯嗯」这种脏事实
        texts = [f"{u}\n{a}".strip() for u, a in turns if (u or a)]
        if not texts:
            return None

        summary_budget = 120
        summary = self.summarizer.summarize(texts, max_tokens=summary_budget)
        episode = None
        if summary.strip():
            episode = self.store.add(
                KIND_EPISODE,
                summary.strip(),
                self.count(summary),
                EPISODE_IMPORTANCE,
                meta={"turns": len(turns)},
                now=ts,
            )
            for fact in extract_facts(summary):
                self._remember_fact(fact, ts, parent_id=episode.id)

        self.store.merge_similar_facts(similarity)
        self._last_consolidation = ts
        return episode

    def _remember_fact(
        self, text: str, now: float, *, parent_id: int | None = None
    ) -> MemoryItem | None:
        # 只保留"关于用户"的事实；助手的话（"记住了""好的"）不是事实
        if "用户" not in text:
            return None
        if any(noise in text for noise in _ASSISTANT_NOISE):
            return None

        importance = FACT_IMPORTANCE
        if any(h in text for h in _SENSITIVE_HINTS):
            importance = min(1.0, importance + 0.12)

        # 已有高度相似的事实 → 加固而不是重复存
        for existing in self.store.all((KIND_FACT,)):
            if similarity(existing.text, text) >= 0.9:
                self.store.touch(existing.id, now=now)
                return existing

        return self.store.add(
            KIND_FACT,
            text,
            self.count(text),
            importance,
            parent_id=parent_id,
            now=now,
        )

    # ------------------------------------------------------------------ 读取

    def build_context(
        self,
        query: str,
        *,
        budget: int | None = None,
        now: float | None = None,
    ) -> ContextBundle:
        """组装本轮要发给 LLM 的上下文。"""
        if budget is not None:
            self.builder.budget = budget
        return self.builder.build(
            self.store, query, list(self.recent), now=now, touch=True
        )

    def recall(self, query: str, k: int = 5, *, now: float | None = None) -> list[MemoryItem]:
        """按激活度召回（可直接当检索 API）。"""
        ts = time.time() if now is None else now
        return [m for m, _ in self.store_search(query, k=k, now=ts)]

    def store_search(
        self, query: str, k: int = 5, *, now: float | None = None
    ) -> list[tuple[MemoryItem, float]]:
        from .retrieval import retrieve

        ts = time.time() if now is None else now
        items: Iterable[MemoryItem] = self.store.all()
        return retrieve(items, query, k=k, now=ts)

    def stats(self) -> dict[str, int]:
        info = self.store.stats()
        info["working_turns"] = len(self.recent)
        info["context_budget"] = self.builder.budget
        return info

    def close(self) -> None:
        self.store.close()

    # 上下文管理器
    def __enter__(self) -> "CompanionMemory":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


__all__ = ["CompanionMemory", "FACT_IMPORTANCE", "EPISODE_IMPORTANCE"]
