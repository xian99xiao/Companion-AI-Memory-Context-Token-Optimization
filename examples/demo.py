"""演示 / 基准：同一段对话，三种上下文策略的 token 成本与召回率。

    python examples/demo.py

三种策略：
  * full    —— 历史全文都塞进上下文（现有多数框架的做法）
  * window  —— 只留最近 8 轮（常见滑动窗口）
  * caim    —— 本项目：工作记忆 + 情景摘要 + 语义事实
 
一次性写入约 70 轮对话，其中 4 条是关键事实，其余是闲聊干扰。
写完后问 4 个只有靠长期记忆才能答的问题。
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

try:  # Windows 控制台默认 GBK，中文/符号会炸
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass

from caim import CompanionMemory, RuleLLM, estimate_tokens, grade  # noqa: E402

# --------------------------------------------------------------------------- 场景

FACT_TURNS = [
    ("我叫林晚，在杭州做后端开发", "林晚，杭州后端，记下了。"),
    ("我养了只猫叫豆豆，是只橘猫", "豆豆这名字挺可爱。"),
    ("我对花生过敏，上次差点进医院", "花生我记住了，会提醒你。"),
    ("我每周三晚上要上日语课", "周三日语课，记下了。"),
]

FILLER = [
    "今天杭州下雨了，好烦",
    "晚上吃什么呢",
    "这家外卖味道一般",
    "地铁又挤爆了",
    "周末想去看电影",
    "新出的那个游戏怎么样",
    "困死了，想睡觉",
    "公司楼下开了家咖啡店",
    "我手机快没电了",
    "这个月工资还没发",
]

QUESTIONS = [
    ("我叫什么名字？", ["林晚"]),
    ("我对什么过敏？", ["花生"]),
    ("我养的猫叫什么？", ["豆豆"]),
    ("我在杭州做什么工作？", ["后端"]),
]

BUDGET = 900
WINDOW_TURNS = 8


def build_turns(n_filler: int = 66) -> list[tuple[str, str]]:
    """事实 + 连续闲聊，还原真实使用里的「记忆稀释」。"""
    turns: list[tuple[str, str]] = []
    for i in range(n_filler):
        u = FILLER[i % len(FILLER)]
        turns.append((u, "嗯嗯。"))
        # 把关键事实分散在闲聊中间
        if i < len(FACT_TURNS):
            turns.append(FACT_TURNS[i])
    turns.extend(FACT_TURNS[len(FACT_TURNS):])
    return turns


def full_history_text(turns) -> str:
    return "\n".join(f"用户：{u}\n助手：{a}" for u, a in turns)


# --------------------------------------------------------------------------- 三种策略

def run_full(turns, questions):
    hist = full_history_text(turns)
    per_turn = estimate_tokens(hist)  # 最坏情况：每轮都带全量
    answers = [RuleLLM().answer(hist, q) for q, _ in questions]
    return per_turn, answers


def run_window(turns, questions):
    hist = full_history_text(turns[-WINDOW_TURNS:])
    per_turn = estimate_tokens(hist)
    answers = [RuleLLM().answer(hist, q) for q, _ in questions]
    return per_turn, answers


def run_caim(turns, questions):
    mem = CompanionMemory(context_budget=BUDGET, working_capacity=6)
    total = 0
    for u, a in turns:
        bundle = mem.build_context(u)
        total += bundle.tokens
        mem.add_turn(u, a)
    per_turn = total // max(1, len(turns))

    answers = []
    for q, _ in questions:
        ctx = mem.build_context(q).text
        answers.append(RuleLLM().answer(ctx, q))
    return per_turn, answers, mem


# --------------------------------------------------------------------------- 输出

def main() -> None:
    turns = build_turns()
    print(f"对话轮数：{len(turns)}（其中关键事实 {len(FACT_TURNS)} 条）\n")

    rows = []
    per_turn_full, ans_full = run_full(turns, QUESTIONS)
    per_turn_win, ans_win = run_window(turns, QUESTIONS)
    per_turn_caim, ans_caim, mem = run_caim(turns, QUESTIONS)

    for name, per_turn, answers in [
        ("full（全量历史）", per_turn_full, ans_full),
        ("window（最近8轮）", per_turn_win, ans_win),
        ("caim（三层记忆）", per_turn_caim, ans_caim),
    ]:
        hit = sum(1 for (q, exp), a in zip(QUESTIONS, answers) if grade(a, exp))
        total_tokens = per_turn * len(turns)
        rows.append((name, per_turn, total_tokens, hit, answers))

    print(f"{'策略':<18}{'每轮 tokens':>12}{'累计 tokens':>14}{'召回':>8}")
    print("-" * 54)
    for name, per_turn, total, hit, _ in rows:
        print(f"{name:<18}{per_turn:>12,}{total:>14,}{hit:>5}/{len(QUESTIONS)}")

    base = rows[0][2]
    caim_total = rows[2][2]
    print(f"\ncaim 相对 full 节省：{100 * (1 - caim_total / base):.1f}%")

    print("\n问答明细（caim）：")
    for (q, exp), ans in zip(QUESTIONS, rows[2][4]):
        ok = "✓" if grade(ans, exp) else "✗"
        print(f"  {ok} {q}\n      → {ans}")

    print("\n记忆库统计：", mem.stats())

    print("\n一次真实装配出的上下文（节选）：")
    sample = mem.build_context("我对什么过敏？").text
    for line in sample.splitlines()[:14]:
        print("  " + line)


if __name__ == "__main__":
    main()
