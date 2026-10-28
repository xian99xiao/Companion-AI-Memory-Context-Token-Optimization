"""Token 估算：不依赖第三方 tokenizer，够用且可替换。

设计取舍：
- 估算只用于「预算控制」，不需要和真实 tokenizer 完全一致，只需要单调、稳定。
- 中文按字计（CJK 每个字约 1 token），英文/数字按词计并乘一个经验系数。
- 生产环境请通过 ``CompanionMemory(token_counter=...)`` 注入真实 tokenizer。
"""

from __future__ import annotations

import re
from typing import Callable

_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
# 英文单词/数字/符号串
_WORD_RE = re.compile(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]")

# 实测经验：英文一个词平均约 1.3 token
_WORD_FACTOR = 1.3


def estimate_tokens(text: str) -> int:
    """估算一段文本的 token 数。

    >>> estimate_tokens("")
    0
    >>> estimate_tokens("你好")
    2
    """
    if not text:
        return 0

    cjk_count = len(_CJK_RE.findall(text))
    rest = _CJK_RE.sub(" ", text)
    word_count = len(_WORD_RE.findall(rest))
    return cjk_count + int(word_count * _WORD_FACTOR + 0.9999)


TokenCounter = Callable[[str], int]


def default_counter() -> TokenCounter:
    return estimate_tokens
