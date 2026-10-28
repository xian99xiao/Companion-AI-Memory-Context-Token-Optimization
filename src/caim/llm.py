"""LLM 协议 + 一个"只看上下文"的规则 LLM（评测用）。

为什么评测要自己写一个 RuleLLM：
如果直接用真模型打分，"答对了"可能来自模型自身的常识或推理补全，
就测不出**上下文里到底有没有这条信息**。
RuleLLM 只在上文里做字符级匹配，找不到就回答"不知道"——
这样召回率衡量的是记忆系统本身，而不是模型能力。
"""

from __future__ import annotations

import re
from typing import Protocol, Sequence

_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_WORD_RE = re.compile(r"[A-Za-z0-9_]+")

_QUESTION_STOP = (
    "什么", "哪个", "哪儿", "哪里", "怎么", "多少", "记得", "告诉",
    "请问", "一下", "时候", "吗", "呢", "啊", "吧", "了", "的", "是",
    "在", "我", "你", "他", "她", "它", "还", "有", "和", "与", "说",
    "过", "请", "问", "哪",
)
"""从问题里去掉的虚词。注意：**叫、名字、猫**这类实词必须保留，
它们是「答案槽」唯一可用的线索。"""


class LLM(Protocol):
    def answer(self, context: str, question: str) -> str: ...


def question_keywords(question: str) -> dict[str, float]:
    """抽问题线索：单字权重 1，相邻双字权重 2（双字更有区分度）。"""
    q = question
    for stop in _QUESTION_STOP:
        q = q.replace(stop, " ")
    chars = [c for c in q if _CJK_RE.match(c) or c.isalnum()]
    keys: dict[str, float] = {}
    for c in chars:
        keys[c] = max(keys.get(c, 0.0), 1.0)
    for i in range(len(chars) - 1):
        bg = chars[i] + chars[i + 1]
        keys[bg] = max(keys.get(bg, 0.0), 2.0)
    for w in _WORD_RE.findall(question):
        keys[w.lower()] = max(keys.get(w.lower(), 0.0), 2.0)
    return keys


class RuleLLM:
    """只在 context 里找答案的规则模型。"""

    def __init__(self, *, min_score: float = 1.5) -> None:
        self.min_score = min_score

    def answer(self, context: str, question: str) -> str:
        keys = question_keywords(question)
        if not keys:
            return "不知道"

        best_line = ""
        best_score = 0.0
        for line in context.splitlines():
            line = line.strip()
            if line.startswith("【"):
                if "当前输入" in line:      # 问题自己永远不是答案
                    break
                continue
            if not line:
                continue
            score = sum(w for k, w in keys.items() if k in line)
            if score > best_score or (score == best_score and 0 < len(line) < len(best_line)):
                best_score = score
                best_line = line

        if best_score < self.min_score or not best_line:
            return "不知道"
        return best_line


def grade(answer: str, expected: Sequence[str]) -> bool:
    """答案里出现任一期望片段即算命中。"""
    return any(e and e in answer for e in expected)
