"""核心能力测试：预算、遗忘、巩固、检索、白名单式安全边界。"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import pytest  # noqa: E402

from caim import (  # noqa: E402
    KIND_EPISODE,
    KIND_FACT,
    CompanionMemory,
    MemoryStore,
    extract_facts,
    relevance,
    retention,
    similarity,
)


# --------------------------------------------------------------------------- 基础

def test_token_estimate_is_sane():
    from caim import estimate_tokens

    assert estimate_tokens("") == 0
    assert 3 <= estimate_tokens("你好世界") <= 12          # 中文 1 字 ≈ 1~1.5 token
    assert 3 <= estimate_tokens("hello world foo") <= 20   # 英文 ~4 字符/token


def test_fact_extraction():
    facts = extract_facts("我叫林晚，我对花生过敏")
    assert any("林晚" in f for f in facts)
    assert any("过敏" in f for f in facts)
    # 纯寒暄不该被当成事实
    assert extract_facts("你好啊") == []
    assert extract_facts("哈哈") == []


# --------------------------------------------------------------------------- 遗忘曲线

def _item(**kw):
    from caim import MemoryItem

    base = dict(
        id=1,
        kind=KIND_FACT,
        text="x",
        tokens=1,
        importance=0.5,
        created_at=0.0,
        last_access=0.0,
        access_count=0,
        parent_id=None,
        meta={},
    )
    base.update(kw)
    return MemoryItem(**base)


def test_retention_decays_over_time():
    from caim.activation import retention

    now = 1_700_000_000.0
    fresh = _item(last_access=now)
    stale = _item(last_access=now - 30 * 86400)
    assert retention(fresh, now) > retention(stale, now)
    assert 0 < retention(stale, now) < 1


def test_access_strengthens_memory():
    from caim.activation import retention

    now = 1_700_000_000.0
    weak = _item(last_access=now - 86400, access_count=0)
    strong = _item(last_access=now - 86400, access_count=5)
    assert retention(strong, now) > retention(weak, now)


# --------------------------------------------------------------------------- 存储

def test_store_roundtrip(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    item = store.add(KIND_FACT, "用户对花生过敏", 12, 0.9, now=1_700_000_000.0)
    loaded = store.get(item.id)
    assert loaded is not None
    assert loaded.text == "用户对花生过敏"
    assert loaded.kind == KIND_FACT

    store.touch(item.id, now=1_700_000_100.0)
    touched = store.get(item.id)
    assert touched.access_count == 1
    assert touched.last_access > 1_700_000_000.0
    store.close()


def test_store_persists_across_reopen(tmp_path):
    path = tmp_path / "m.db"
    store = MemoryStore(path)
    store.add(KIND_FACT, "用户住在杭州", 8, 0.7, now=1_700_000_000.0)
    store.close()

    again = MemoryStore(path)
    texts = [m.text for m in again.all()]
    assert "用户住在杭州" in texts
    again.close()


# --------------------------------------------------------------------------- 检索

def test_relevance_prefers_matching_fact():
    assert relevance("我对什么过敏", "用户对花生过敏") > relevance("我对什么过敏", "用户在杭州工作")


def test_retrieval_ranks_by_activation():
    mem = CompanionMemory()
    mem.add_turn("我对花生过敏", "记住了")
    mem.add_turn("我住在杭州", "好的")
    top = mem.recall("过敏的东西是什么")
    assert "花生" in top[0].text


# --------------------------------------------------------------------------- 预算

def test_context_respects_budget():
    mem = CompanionMemory(context_budget=150, working_capacity=10)
    for i in range(30):
        mem.add_turn(f"第{i}轮：这是一句比较长的闲聊内容用来占位", "好的，收到")
    bundle = mem.build_context("随便问问")
    assert bundle.tokens <= 150 + 20          # 允许极小的估算误差
    assert bundle.dropped                    # 一定发生了裁剪


def test_budget_too_small_degrades_gracefully():
    mem = CompanionMemory(context_budget=20)
    mem.add_turn("我叫林晚", "好的")
    bundle = mem.build_context("你好")
    assert bundle.tokens > 0
    assert "你好" in bundle.text             # 当前输入永不丢


# --------------------------------------------------------------------------- 不失忆

def test_old_fact_survives_many_turns():
    mem = CompanionMemory(context_budget=600, working_capacity=6)
    mem.add_turn("我对花生过敏，很严重", "记住了")
    for i in range(80):
        mem.add_turn(f"第{i}次闲聊，天气不错", "嗯")
    ctx = mem.build_context("我想吃点点心，有什么要注意的吗").text
    assert "花生" in ctx or "过敏" in ctx


def test_fact_extracted_from_episode_summary():
    mem = CompanionMemory(context_budget=800, working_capacity=2)
    mem.add_turn("顺便说下我生日是三月十二号", "记下了")
    mem.add_turn("哦对，我妈妈的生日是五月一号", "好")
    # 超出工作记忆容量触发巩固
    mem.add_turn("今天天气不错", "是的")
    mem.add_turn("再见", "再见")
    facts = [m.text for m in mem.store.all((KIND_FACT,))]
    assert any("三月十二" in f or "生日" in f for f in facts)
    episodes = mem.store.all((KIND_EPISODE,))
    assert episodes, "应当生成情景摘要"


def test_duplicate_facts_are_merged():
    mem = CompanionMemory()
    mem.add_turn("我对花生过敏", "记住了")
    n_before = len(mem.store.all((KIND_FACT,)))
    mem.add_turn("我对花生过敏", "又记住了")
    n_after = len(mem.store.all((KIND_FACT,)))
    assert n_after <= n_before + 1
    assert n_after < 4


def test_repeated_access_keeps_memory_prominent():
    mem = CompanionMemory(context_budget=500)
    mem.add_turn("我养了只猫叫豆豆", "可爱")
    for _ in range(3):
        mem.build_context("豆豆最近怎么样")
    item = [m for m in mem.store.all((KIND_FACT,)) if "豆豆" in m.text][0]
    assert item.access_count >= 2


# --------------------------------------------------------------------------- 安全边界

def test_summarizer_respects_token_cap():
    from caim import ExtractiveSummarizer

    s = ExtractiveSummarizer()
    long_text = ["这是一句很长的话用来测试摘要是否会超限" * 3 for _ in range(20)]
    summary = s.summarize(long_text, max_tokens=30)
    from caim import estimate_tokens

    assert estimate_tokens(summary) <= 30
