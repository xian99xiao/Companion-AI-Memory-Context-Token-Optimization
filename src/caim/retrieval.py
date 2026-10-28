"""线索检索：按当前 query 找相关记忆。

为什么先做"词面相似"而不是向量：
- 零依赖、可解释、冷启动即可用（不需要 embedding 服务）；
- 中文用字符 bigram，对「过敏」↔「花生过敏」这种包含关系已经够用；
- 真正的语义泛化留给 ``embedder`` 钩子：传入 ``Callable[[str], list[float]]``
  即可切换到向量相似度，接口不变。
"""

from __future__ import annotations

import math
import re
import time
from typing import Callable, Iterable, Sequence

from .activation import activation
from .models import MemoryItem

_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_WORD_RE = re.compile(r"[A-Za-z0-9_]+")
_STOP = {
    "的", "了", "是", "在", "我", "你", "他", "她", "它", "们", "吗", "呢", "啊",
    "和", "与", "也", "就", "都", "不", "有", "这", "那", "个", "什么", "怎么",
}


def bigrams(text: str) -> set[str]:
    """字符 bigram 集合（中文友好）。"""
    chars = [c for c in text if not c.isspace()]
    if len(chars) < 2:
        return set(chars)
    return {chars[i] + chars[i + 1] for i in range(len(chars) - 1)}


def keywords(text: str) -> set[str]:
    """关键词：英文词 + 中文双字片段，去掉高频虚词。"""
    words = {w.lower() for w in _WORD_RE.findall(text)}
    grams = bigrams(text)
    return {g for g in (words | grams) if g not in _STOP}


def similarity(a: str, b: str) -> float:
    """Jaccard 相似度（字符 bigram 层面）。"""
    sa, sb = bigrams(a), bigrams(b)
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    if inter == 0:
        return 0.0
    return inter / len(sa | sb)


def relevance(query: str, text: str, embedder: Callable[[str], list[float]] | None = None) -> float:
    """查询与记忆的相关度，0~1。"""
    if embedder is not None:
        return _cosine(embedder(query), embedder(text))

    q = keywords(query)
    t = keywords(text)
    if not q or not t:
        return 0.0
    overlap = len(q & t)
    if overlap == 0:
        return 0.0
    # 用 query 覆盖率衡量：query 里有多少线索能在记忆里找到
    coverage = overlap / len(q)
    # 再用 Jaccard 抑制"记忆很长但只沾一点边"的情况
    jaccard = overlap / len(q | t)
    return 0.65 * coverage + 0.35 * jaccard


def retrieve(
    items: Iterable[MemoryItem],
    query: str,
    *,
    k: int = 5,
    now: float | None = None,
    embedder: Callable[[str], list[float]] | None = None,
    min_score: float = 0.0,
    touch_store=None,
) -> list[tuple[MemoryItem, float]]:
    """按激活度召回 top-k。

    ``touch_store`` 传入 :class:`~caim.store.MemoryStore` 时，
    被成功召回的条目会 ``touch()`` —— 即"复述加固"。
    """
    ts = time.time() if now is None else now
    scored: list[tuple[MemoryItem, float]] = []
    for item in items:
        rel = relevance(query, item.text, embedder)
        score = activation(item, ts, rel)
        if score > min_score:
            scored.append((item, score))
    scored.sort(key=lambda pair: pair[1], reverse=True)
    top = scored[:k]
    if touch_store is not None:
        for item, _ in top:
            touch_store.touch(item.id, now=ts)
    return top


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return max(0.0, min(1.0, dot / (na * nb)))
