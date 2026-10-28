"""巩固（consolidation）：把原文压成摘要、把摘要里的原子事实抽出来。

对应人脑的"睡眠巩固"：
- 原文不会长期驻留，**摘要**进情景记忆；
- 摘要里能独立成立的信息（"我对花生过敏"）提为**事实**，长期驻留；
- 事实之间高度重复的合并去重。

``Summarizer`` 是协议：默认用离线抽取式实现，接真实 LLM 只需换一个类。
"""

from __future__ import annotations

import re
from typing import Protocol, Sequence

from .tokens import estimate_tokens

_SENT_SPLIT = re.compile(r"(?<=[。！？!?；;])|\n+")

# 值得记的事实线索（中文口语里的常见断言模式）
_FACT_HINTS = (
    "叫", "是", "在", "住", "养", "喜欢", "讨厌", "过敏", "做", "从事",
    "生日", "岁", "手机", "邮箱", "地址", "习惯", "需要", "不要", "别",
)
_NEGATIVE_HINTS = ("吗", "呢", "?", "？", "假如", "如果", "听说")


class Summarizer(Protocol):
    """摘要器协议。接真实 LLM 时实现这个方法即可。"""

    def summarize(self, texts: Sequence[str], max_tokens: int) -> str: ...


class ExtractiveSummarizer:
    """离线抽取式摘要：按信息密度挑句子，不产生幻觉。

    排序逻辑：含事实线索 > 含数字/专名 > 长度适中 > 靠前。
    这比"取前 N 句"好，因为长对话里真正有用的信息往往不在开头。
    """

    def summarize(self, texts: Sequence[str], max_tokens: int) -> str:
        sents: list[tuple[float, int, str]] = []
        for idx, text in enumerate(texts):
            for sent in _split_sentences(text):
                score = _sentence_score(sent)
                sents.append((score, idx, sent))

        sents.sort(key=lambda x: (-x[0], x[1]))
        picked: list[str] = []
        used = 0
        for score, idx, sent in sents:
            if score <= 0:
                continue
            cost = estimate_tokens(sent)
            if used + cost > max_tokens and picked:
                break
            picked.append(sent)
            used += cost
            if used >= max_tokens:
                break

        # 输出时按原始顺序，读起来才像一段对话回顾
        order = {s: i for i, (_, _, s) in enumerate(sents)}
        picked.sort(key=lambda s: order.get(s, 0))
        out = "；".join(p.strip("。；;") for p in picked)
        if estimate_tokens(out) > max_tokens:      # 硬截断，预算不能被单句冲破
            budget = max(1, max_tokens)
            acc = ""
            for ch in out:
                if estimate_tokens(acc + ch) > budget:
                    break
                acc += ch
            out = acc.rstrip("；;，,。 ")
        return out


class NullSummarizer:
    """不摘要，直接拼接（用于对照实验/调试）。"""

    def summarize(self, texts: Sequence[str], max_tokens: int) -> str:
        out: list[str] = []
        used = 0
        for t in texts:
            cost = estimate_tokens(t)
            if used + cost > max_tokens and out:
                break
            out.append(t)
            used += cost
        return " ".join(out)


def extract_facts(text: str, *, min_chars: int = 4, max_chars: int = 80) -> list[str]:
    """从一段话里抽原子事实（启发式）。

    目标不是抽全，而是**低误报**：宁可漏，不能把闲聊当事实长期存起来。
    """
    facts: list[str] = []
    seen: set[str] = set()

    for sent in _split_sentences(text):
        sent = sent.strip()
        if not (min_chars <= len(sent) <= max_chars):
            continue
        if any(neg in sent for neg in _NEGATIVE_HINTS):
            continue
        if not any(h in sent for h in _FACT_HINTS):
            continue
        # 去掉纯灌水（如"第12轮闲聊"）
        if "轮闲聊" in sent or "闲聊" == sent:
            continue
        key = re.sub(r"\s+", "", sent)
        if key in seen:
            continue
        seen.add(key)
        facts.append(_normalize_fact(sent.rstrip("。；;，,")))
    return facts


def _normalize_fact(sentence: str) -> str:
    """把第一人称陈述改写成「用户…」，便于检索与去重。

    存储层用第三人称、检索层用原话，两者对齐后召回更稳。
    """
    s = sentence.strip()
    if s.startswith("本人"):
        s = "用户" + s[2:]
    # 第一人称一律改写成「用户」，这样存储层与检索层都在同一个主语上比对
    if "我" in s and not s.startswith("用户"):
        s = s.replace("我们", "用户").replace("我", "用户")

    # 「用户叫林晚」→「用户的名字叫林晚」：补出答案槽，检索时才对得上「名字」
    m = re.match(r"^用户叫(?!什么|啥)(.+)$", s)
    if m:
        s = "用户的名字叫" + m.group(1)
    return s


def _split_sentences(text: str) -> list[str]:
    parts = [p for p in _SENT_SPLIT.split(text) if p and p.strip()]
    return parts


def _sentence_score(sent: str) -> float:
    if not sent.strip():
        return 0.0
    score = 0.0
    if any(h in sent for h in _FACT_HINTS):
        score += 2.0
    if re.search(r"\d", sent):
        score += 0.8
    if re.search(r"[A-Za-z]{2,}", sent):
        score += 0.4
    # 太短没信息，太长可能是闲聊
    n = len(sent)
    if 8 <= n <= 60:
        score += 0.6
    elif n > 120:
        score -= 0.8
    if any(neg in sent for neg in ("吗", "呢", "？", "?")):
        score -= 1.2
    return score
