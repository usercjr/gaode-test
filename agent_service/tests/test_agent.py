"""Agent 单元测试：用假的 LLM / 高德客户端，离线、确定性地验证校验规则与 Function Calling 循环。"""
import asyncio
import json

from agent.orchestrator import PlanAgent
from agent.schemas import PlanRequest
from agent.tools import PlanContext, ToolExecutor
from agent.validator import ItineraryValidator
from config import AgentConfig

CHENGDU = {"adcode": "510100", "level": "市", "city": "成都", "location": {"lng": 104.06, "lat": 30.65}}
POIS = {
    "P1": {"poi_id": "P1", "name": "宽窄巷子", "adcode": "510105", "city": "成都", "district": "青羊区",
           "address": "", "type": "风景名胜", "rating": "4.8", "cost": "", "location": {"lng": 104.05, "lat": 30.67}},
    "P2": {"poi_id": "P2", "name": "人民公园", "adcode": "510105", "city": "成都", "district": "青羊区",
           "address": "", "type": "公园", "rating": "4.7", "cost": "", "location": {"lng": 104.06, "lat": 30.66}},
    "P3": {"poi_id": "P3", "name": "都江堰", "adcode": "510181", "city": "成都", "district": "都江堰市",
           "address": "", "type": "风景名胜", "rating": "4.8", "cost": "", "location": {"lng": 103.61, "lat": 31.0}},
    "CQ": {"poi_id": "CQ", "name": "洪崖洞", "adcode": "500103", "city": "重庆", "district": "渝中区",
           "address": "", "type": "风景名胜", "rating": "4.8", "cost": "", "location": {"lng": 106.58, "lat": 29.56}},
}


class FakeAmap:
    """路线时长按两点经度差粗算：经度差 0.1 ≈ 10 分钟。"""

    def __init__(self):
        self.route_calls = 0

    async def geocode(self, address, city=""):
        return CHENGDU

    async def search_poi(self, keyword, city="", limit=5):
        return [p for p in POIS.values() if p["name"] == keyword][:limit]

    async def route(self, origin, dest, mode="driving", city=""):
        self.route_calls += 1
        minutes = round(abs(origin["lng"] - dest["lng"]) * 100) + 5
        return {"mode": mode, "duration_min": minutes, "distance_m": minutes * 300}

    async def weather(self, adcode):
        return []


def _ctx(*poi_ids):
    ctx = PlanContext(destination="成都", dest_geo=CHENGDU)
    for pid in poi_ids:
        ctx.pois[pid] = POIS[pid]
    return ctx


def _stop(pid, name=None, mode="walking"):
    return {"name": name or POIS.get(pid, {}).get("name", "某地"), "poi_id": pid, "stay_min": 60,
            "transport_to_next": {"mode": mode}}


def _check(raw, ctx, days=1):
    validator = ItineraryValidator(FakeAmap(), AgentConfig())
    return asyncio.run(validator.check(raw, ctx, days))


def test_valid_itinerary_passes_and_legs_are_recomputed():
    raw = {"title": "t", "city": "成都", "days": [{"day": 1, "stops": [_stop("P1"), _stop("P2")]}]}
    r = _check(raw, _ctx("P1", "P2"))
    assert r.passed, r.issues
    first, last = r.itinerary.days[0].stops
    assert first.location.lng == 104.05  # 坐标由服务端按 poi_id 回填
    assert first.transport_to_next.verified and first.transport_to_next.duration_min == 6
    assert last.transport_to_next is None  # 当天最后一站不保留交通
    assert r.stats["stops_resolved"] == 2 and r.stats["legs_verified"] == 1


def test_hallucinated_poi_id_is_rejected():
    raw = {"title": "t", "city": "成都", "days": [{"day": 1, "stops": [_stop("P1"), _stop("FAKE", "编造的店")]}]}
    r = _check(raw, _ctx("P1"))
    assert not r.passed
    assert any("编造的店" in i and "poi_search" in i for i in r.issues)


def test_out_of_city_and_long_leg_and_wrong_day_count():
    raw = {"title": "t", "city": "成都", "days": [{"day": 1, "stops": [_stop("P1", mode="driving"), _stop("P3"), _stop("CQ")]}]}
    r = _check(raw, _ctx("P1", "P3", "CQ"), days=2)
    text = "\n".join(r.issues)
    assert "要求 2 天" in text
    assert "洪崖洞" in text and "不在目的地" in text
    assert "超过 60 分钟" in text  # 都江堰 → 洪崖洞 按假路线约 300 分钟
    assert r.stats["long_legs"] >= 1


def test_duplicate_stop_and_schema_error():
    raw = {"title": "t", "city": "成都", "days": [{"day": 1, "stops": [_stop("P1")]}, {"day": 2, "stops": [_stop("P1")]}]}
    r = _check(raw, _ctx("P1"), days=2)
    assert any("重复" in i for i in r.issues)

    bad = _check({"title": "t", "days": []}, _ctx())
    assert bad.itinerary is None and any("结构错误" in i for i in bad.issues)


# ---------------- Function Calling 循环 ----------------
def _tool_call(cid, name, args):
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}


class ScriptedLLM:
    """按顺序返回预设的 assistant 消息，并记录每轮收到的 messages。"""

    def __init__(self, replies):
        self.replies = list(replies)
        self.seen = []

    async def complete(self, messages, **kwargs):
        self.seen.append([dict(m) for m in messages])
        return {"message": self.replies.pop(0), "usage": {"prompt_tokens": 10, "completion_tokens": 5}}


def test_agent_loop_feeds_validation_issues_back_and_fixes():
    good = {"title": "成都一日", "city": "成都", "days": [{"day": 1, "stops": [_stop("P1"), _stop("P2")]}]}
    bad = {"title": "成都一日", "city": "成都", "days": [{"day": 1, "stops": [_stop("P1"), _stop("XX", "不存在的店")]}]}
    llm = ScriptedLLM([
        {"content": "", "tool_calls": [_tool_call("c1", "poi_search", {"keyword": "宽窄巷子"}),
                                       _tool_call("c2", "poi_search", {"keyword": "人民公园"})]},
        {"content": "", "tool_calls": [_tool_call("c3", "submit_itinerary", bad)]},
        {"content": "", "tool_calls": [_tool_call("c4", "submit_itinerary", good)]},
    ])
    amap = FakeAmap()
    agent = PlanAgent(llm, amap, ToolExecutor(amap, retriever=None), AgentConfig())
    events = []

    async def emit(ev, data):
        events.append(ev)

    result = asyncio.run(agent.plan(PlanRequest(destination="成都", days=1), emit))

    assert result.passed and not result.first_pass and result.fix_rounds == 1
    assert result.llm_rounds == 3
    assert [c["name"] for c in result.tool_calls] == ["poi_search", "poi_search"]
    # 第 3 轮收到的上下文里：每个工具调用都有对应的 tool 消息，且校验问题被回填
    third = llm.seen[2]
    tool_msgs = {m["tool_call_id"]: m["content"] for m in third if m["role"] == "tool"}
    assert set(tool_msgs) == {"c1", "c2", "c3"}
    assert "不存在的店" in tool_msgs["c3"]
    assert "validation" in events and "tool_call" in events
    assert "宽窄巷子" in result.text


class RecordingLLM(ScriptedLLM):
    def __init__(self, replies):
        super().__init__(replies)
        self.kwargs = []

    async def complete(self, messages, **kwargs):
        self.kwargs.append(kwargs)
        return await super().complete(messages, **kwargs)


def test_budget_guard_forces_submit_before_steps_run_out():
    good = {"title": "t", "city": "成都", "days": [{"day": 1, "stops": [_stop("P1"), _stop("P2")]}]}
    search = lambda cid, kw: {"content": "", "tool_calls": [_tool_call(cid, "poi_search", {"keyword": kw})]}
    llm = RecordingLLM([search("a", "宽窄巷子"), search("b", "人民公园"),
                        {"content": "", "tool_calls": [_tool_call("s", "submit_itinerary", good)]}])
    amap = FakeAmap()
    cfg = AgentConfig(max_steps=4, max_fix_rounds=1)  # 第 3 轮时只剩 2 轮 = 1 次提交 + 1 次修正
    result = asyncio.run(PlanAgent(llm, amap, ToolExecutor(amap, retriever=None), cfg).plan(
        PlanRequest(destination="成都", days=1)))
    assert result.passed
    assert llm.kwargs[0].get("tool_choice") is None and llm.kwargs[1].get("tool_choice") is None
    assert llm.kwargs[2]["tool_choice"]["function"]["name"] == "submit_itinerary"


def test_failed_submit_then_drifting_is_forced_to_resubmit_and_returns_with_warnings():
    """回归：提交未通过后模型又去调别的工具，步数耗尽前应被强制重新提交；仍不通过则带 warnings 返回而不是报错。"""
    bad = {"title": "t", "city": "成都", "days": [{"day": 1, "stops": [_stop("P1"), _stop("XX", "不存在的店")]}]}
    search = lambda cid, kw: {"content": "", "tool_calls": [_tool_call(cid, "poi_search", {"keyword": kw})]}
    llm = RecordingLLM([
        search("a", "宽窄巷子"),
        {"content": "", "tool_calls": [_tool_call("s1", "submit_itinerary", bad)]},
        search("b", "人民公园"),                                                   # 提交失败后跑去搜别的
        {"content": "", "tool_calls": [_tool_call("s2", "submit_itinerary", bad)]},  # 被强制重新提交
    ])
    amap = FakeAmap()
    cfg = AgentConfig(max_steps=4, max_fix_rounds=1)
    result = asyncio.run(PlanAgent(llm, amap, ToolExecutor(amap, retriever=None), cfg).plan(
        PlanRequest(destination="成都", days=1)))
    assert llm.kwargs[3]["tool_choice"]["function"]["name"] == "submit_itinerary"
    assert not result.passed and any("不存在的店" in w for w in result.itinerary["warnings"])
