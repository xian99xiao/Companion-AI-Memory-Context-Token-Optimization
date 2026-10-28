"""激活度：把"遗忘曲线"和"重要性"变成可计算的召回权重。

人脑记忆的三个可观察现象，这里都做了对应：
1. **遗忘曲线**（Ebbinghaus）：不被提起的记忆，保留强度指数衰减。
2. **稳定度差异**：被反复提起、或情绪上重要的记忆，衰减更慢（S 更大）。
3. **复述加固**：每次被成功召回，稳定度就提升一档。

公式：
    S(t)  = BASE * (1 + 0.5 * access_count) * (1 + 2 * importance)   # 小时
    R(t)  = exp(-ln2 * Δt / S)                                        # ∈(0,1]
"""

from __future__ import annotations

import math

from .models import MemoryItem

BASE_STABILITY_HOURS = 6.0
"""从未被再次访问、importance=0.5 的记忆，6 小时后保留强度减半。"""


def stability_hours(item: MemoryItem) -> float:
    """记忆的稳定度（半衰期，单位小时）。"""
    return (
        BASE_STABILITY_HOURS
        * (1.0 + 0.5 * item.access_count)
        * (1.0 + 2.0 * item.importance)
    )


def retention(item: MemoryItem, now: float) -> float:
    """当前时刻的保留强度，0~1。"""
    elapsed_h = max(0.0, (now - item.last_access) / 3600.0)
    s = stability_hours(item)
    if s <= 0:
        return 0.0
    return math.exp(-math.log(2) * elapsed_h / s)


def activation(
    item: MemoryItem,
    now: float,
    relevance: float,
    *,
    w_relevance: float = 0.55,
    w_importance: float = 0.30,
    w_recency: float = 0.15,
) -> float:
    """综合激活度：与当前线索的相关度 × 重要性 × 保留强度。

    三者的权重刻意让 **relevance 占主导**：检索是"线索驱动"的，
    不能因为某条记忆很新或很重要就无条件注入无关内容。
    """
    score = (
        w_relevance * max(0.0, min(1.0, relevance))
        + w_importance * max(0.0, min(1.0, item.importance))
        + w_recency * retention(item, now)
    )
    return score
