"""预算化上下文装配：**先分配额度，再装内容**。

这是"省 token 且不失忆"的落点。装配顺序即优先级：

1. ``persona``   —— 固定的角色设定（几行，永远在）
2. ``facts``     —— 语义记忆：与当前线索相关的召回 + 高重要性常驻
3. ``episodes``  —— 情景记忆：与当前线索最相关的往事摘要
4. ``recent``    —— 工作记忆：最近若干轮**逐字**原文（越近越优先）
5. ``query``     —— 当前这轮输入（必须保留）

任何一段装不进预算，就记进 ``dropped`` 并跳过 —— **不是删除存储，只是这次不注入**。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Sequence

from .models import KIND_EPISODE, KIND_FACT, ContextBundle, MemoryItem
from .retrieval import relevance, retrieve
from .store import MemoryStore
from .tokens import estimate_tokens

DEFAULT_PERSONA = "你是用户的长期陪伴助手，记得你们之前聊过的事情。"


@dataclass
class ContextBuilder:
    """按预算装配上下文。"""

    budget: int = 1200
    recent_turns: int = 6
    max_pinned_facts: int = 12
    max_episodes: int = 3
    persona: str = DEFAULT_PERSONA
    token_counter: Callable[[str], int] = estimate_tokens

    def build(
        self,
        store: MemoryStore,
        query: str,
        recent: Sequence[tuple[str, str]],
        *,
        now: float | None = None,
        touch: bool = True,
    ) -> ContextBundle:
        ts = time.time() if now is None else now
        count = self.token_counter

        sections: dict[str, int] = {}
        dropped: list[str] = []
        picked: list[MemoryItem] = []

        # query 必须保留，先扣掉它的额度
        query_block = f"【当前输入】\n用户：{query}"
        remaining = self.budget - count(query_block) - 4

        # ---------- 1. persona ----------
        persona_block = f"【设定】{self.persona}"
        if count(persona_block) <= remaining:
            remaining -= count(persona_block)
            sections["persona"] = count(persona_block)
        else:
            persona_block = ""
            dropped.append("persona")

        # ---------- 2. facts ----------
        facts = store.all((KIND_FACT,))
        related_ids = [m.id for m, _ in retrieve(facts, query, k=self.max_pinned_facts, now=ts)]
        pinned_ids = [
            m.id
            for m in sorted(facts, key=lambda m: m.importance, reverse=True)[
                : self.max_pinned_facts
            ]
        ]
        order: list[int] = []
        for fid in related_ids + pinned_ids:
            if fid not in order:
                order.append(fid)
        by_id = {m.id: m for m in facts}
        fact_items = [by_id[i] for i in order if i in by_id]

        facts_block, remaining, used_items = self._pack(
            fact_items,
            header="【关于用户的事实】",
            remaining=remaining,
            prefix="- ",
            count=count,
            dropped=dropped,
            drop_tag="fact",
        )
        if facts_block:
            sections["facts"] = count(facts_block)
            picked.extend(used_items)
            if touch:                      # 被召回 = 被复习，遗忘曲线往后推
                for item in used_items:
                    store.touch(item.id, now=ts)

        # ---------- 3. episodes ----------
        episodes = store.all((KIND_EPISODE,))
        # 只有与当前线索确实相关的往事才值得占额度
        ep_hits = [
            m
            for m, _ in retrieve(episodes, query, k=self.max_episodes * 2, now=ts)
            if relevance(query, m.text) >= 0.08
        ][: self.max_episodes]
        episodes_block, remaining, used_items = self._pack(
            ep_hits,
            header="【往事回顾】",
            remaining=remaining,
            prefix="- ",
            count=count,
            dropped=dropped,
            drop_tag="episode",
        )
        if episodes_block:
            sections["episodes"] = count(episodes_block)
            picked.extend(used_items)

        # ---------- 4. recent（从最近往回装） ----------
        recent_block, remaining, _ = self._pack(
            [
                (f"用户：{u}\n助手：{a}", 0.0)
                for u, a in list(recent)[-self.recent_turns :][::-1]
            ],
            header="【最近对话】",
            remaining=remaining,
            prefix="",
            count=count,
            dropped=dropped,
            drop_tag="recent",
            item_attr=None,
        )
        if recent_block:
            sections["recent"] = count(recent_block)

        # ---------- 5. query ----------
        sections["query"] = count(query_block)

        parts = [
            p
            for p in (persona_block, facts_block, episodes_block, recent_block, query_block)
            if p
        ]
        text = "\n\n".join(parts)
        return ContextBundle(
            text=text,
            tokens=count(text),
            sections=sections,
            items=picked,
            dropped=dropped,
        )

    # ------------------------------------------------------------------

    def _pack(
        self,
        entries,
        *,
        header: str,
        remaining: int,
        prefix: str,
        count: Callable[[str], int],
        dropped: list[str],
        drop_tag: str,
        item_attr: str | None = "text",
    ):
        """把条目按顺序塞进额度，返回 (块文本, 剩余额度, 实际采用的 MemoryItem)。

        ``entries`` 既可以是 ``MemoryItem`` 列表，也可以是 ``(text, importance)`` 列表。
        """
        header_cost = count(header + "\n")
        if header_cost > remaining:
            dropped.append(header)
            return "", remaining, []

        lines: list[str] = []
        used_items: list[MemoryItem] = []
        spent = header_cost

        for entry in entries:
            if item_attr is not None:
                text = getattr(entry, item_attr)
                item = entry
                tag = f"{drop_tag}:{item.id}"
            else:
                text = entry[0]
                item = None
                tag = f"{drop_tag}:older"

            line = prefix + text
            cost = count(line) + 1
            if spent + cost > remaining:
                dropped.append(tag)
                continue
            lines.append(line)
            spent += cost
            if item is not None:
                used_items.append(item)

        if not lines:
            return "", remaining, []

        block = header + "\n" + "\n".join(lines)
        return block, remaining - spent, used_items


def rank_memories(
    items: Sequence[MemoryItem], query: str
) -> list[tuple[MemoryItem, float]]:
    """便利函数：只按相关度排序（调试/外部检索用）。"""
    scored = [(m, relevance(query, m.text)) for m in items]
    scored.sort(key=lambda p: p[1], reverse=True)
    return scored
