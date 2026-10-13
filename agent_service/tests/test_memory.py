import asyncio

from memory.extractor import is_sensitive, validate_candidates
from memory.service import MemoryService
from memory.session import KEEP_RECENT, SUMMARIZE_AFTER, SessionContext, needs_rewrite
from memory.store import MemoryStore


def _store(tmp_path):
    return MemoryStore(tmp_path / "m.sqlite3")


def _active(store, user="u"):
    return sorted((m["key"], m["value"]) for m in store.list(user, ("active",)))


# ---------------- 写入规则 ----------------
def test_single_key_new_value_supersedes_old(tmp_path):
    s = _store(tmp_path)
    s.upsert("u", "companion", "带父母", 0.9)
    change = s.upsert("u", "companion", "和女朋友", 0.9)
    assert change["action"] == "updated"
    assert _active(s) == [("companion", "和女朋友")]
    assert any(m["status"] == "superseded" for m in s.list("u", ("superseded",)))


def test_low_confidence_does_not_override_confirmed_value(tmp_path):
    s = _store(tmp_path)
    s.upsert("u", "budget", "预算低", 0.9)
    assert s.upsert("u", "budget", "预算宽裕", 0.5)["action"] == "pending"
    assert _active(s) == [("budget", "预算低")]  # 已确认的值仍然有效


def test_pending_becomes_active_when_mentioned_again(tmp_path):
    s = _store(tmp_path)
    assert s.upsert("u", "pace", "慢节奏", 0.5)["action"] == "pending"
    assert _active(s) == []
    assert s.upsert("u", "pace", "慢 节奏", 0.5)["action"] == "confirmed"  # 归一化后同值
    assert _active(s) == [("pace", "慢节奏")]


def test_multi_value_dedupe_and_opposite_override(tmp_path):
    s = _store(tmp_path)
    s.upsert("u", "likes", "博物馆", 0.9)
    s.upsert("u", "likes", "夜市", 0.9)
    assert s.upsert("u", "likes", "博物馆", 0.9) is None  # 重复不产生变化
    s.upsert("u", "dislikes", "博物馆", 0.9)                # 改口：不喜欢博物馆
    assert _active(s) == [("dislikes", "博物馆"), ("likes", "夜市")]


def test_delete_and_user_isolation(tmp_path):
    s = _store(tmp_path)
    m = s.upsert("u", "diet", "不吃辣", 0.9)["memory"]
    s.upsert("other", "diet", "素食", 0.9)
    assert s.delete("other", m["id"]) == 0  # 不能删别人的
    assert s.delete("u", m["id"]) == 1
    assert _active(s) == [] and _active(s, "other") == [("diet", "素食")]
    assert s.prompt_lines("other") == ["饮食：素食"]


# ---------------- 抽取结果的确定性过滤 ----------------
def test_validate_candidates_rules():
    msg = "我膝盖不好，不能爬山，电话 13812345678"
    cands = [
        {"key": "mobility", "value": "膝盖不好不能爬山", "evidence": "膝盖不好，不能爬山", "confidence": 0.9},
        {"key": "diet", "value": "不吃辣", "evidence": "我不吃辣", "confidence": 0.9},          # 原话里没有 → 幻觉
        {"key": "phone", "value": "x", "evidence": "电话", "confidence": 0.9},                 # 非法键
        {"key": "home_city", "value": "13812345678", "evidence": "电话 13812345678", "confidence": 0.9},  # 敏感
    ]
    out = validate_candidates(msg, cands)
    assert [c["key"] for c in out] == ["mobility"]
    assert is_sensitive("身份证 11010119900307123X") and not is_sensitive("预算 3000 元")


# ---------------- 会话上下文 ----------------
def test_needs_rewrite_heuristic():
    assert needs_rewrite("那附近呢") and needs_rewrite("第二天呢") and needs_rewrite("还有吗")
    assert not needs_rewrite("成都吃火锅有什么要注意的地方")


class FakeLLM:
    def __init__(self, reply="摘要：目的地成都"):
        self.reply = reply
        self.calls = 0

    async def chat(self, messages, **kw):
        self.calls += 1
        return self.reply


def test_summary_folds_older_turns_and_keeps_recent(tmp_path):
    s = _store(tmp_path)
    for i in range(SUMMARIZE_AFTER + 2):
        s.add_turn("sess", "u", "user" if i % 2 == 0 else "assistant", f"第{i}句")
    ctx = SessionContext(s, FakeLLM())
    assert asyncio.run(ctx.maybe_summarize("sess")) is True
    text = ctx.context_text("sess")
    assert "摘要：目的地成都" in text
    assert text.count("第") == KEEP_RECENT  # 只保留最近几轮原文
    assert asyncio.run(ctx.maybe_summarize("sess")) is False  # 不足阈值不再摘要


def test_standalone_question_only_when_needed(tmp_path):
    s = _store(tmp_path)
    ctx = SessionContext(s, FakeLLM("武侯祠附近有什么好吃的"))
    assert asyncio.run(ctx.standalone_question("sess", "那附近呢")) == "那附近呢"  # 没有上下文
    s.add_turn("sess", "u", "user", "武侯祠门票多少钱")
    assert asyncio.run(ctx.standalone_question("sess", "那附近呢")) == "武侯祠附近有什么好吃的"
    assert asyncio.run(ctx.standalone_question("sess", "成都吃火锅有什么要注意的地方")) == "成都吃火锅有什么要注意的地方"


def test_service_learn_writes_validated_memories(tmp_path, monkeypatch):
    s = _store(tmp_path)
    svc = MemoryService(s, FakeLLM('{"memories": [{"key": "diet", "value": "不吃辣", "evidence": "我不吃辣", "confidence": 0.9}]}'))
    updates = asyncio.run(svc.learn("u", "我不吃辣，帮我换掉火锅"))
    assert updates[0]["action"] == "added" and updates[0]["key_cn"] == "饮食"
    assert svc.prompt_lines("u") == ["饮食：不吃辣"]
    assert asyncio.run(svc.learn("", "我不吃辣")) == []  # 没有 user_id 不记


def test_dislike_overrides_like_even_when_values_include_verbs(tmp_path):
    """回归：LLM 把动词写进 value（“喜欢逛博物馆” / “不喜欢博物馆”），改口规则仍要生效。"""
    s = _store(tmp_path)
    s.upsert("u", "likes", "喜欢逛博物馆", 0.9)
    s.upsert("u", "dislikes", "不喜欢博物馆", 0.9)
    assert _active(s) == [("dislikes", "不喜欢博物馆")]
    assert s.upsert("u", "dislikes", "博物馆", 0.9) is None  # 同一对象视为重复
