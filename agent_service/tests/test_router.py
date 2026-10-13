"""动态路由单元测试：用假的分类器 / RAG / Agent / LLM 验证分发逻辑。"""
import asyncio

from agent.router import ChatRouter
from agent.schemas import ChatRequest, Itinerary
from agent.tools import ToolExecutor
from tests.test_agent import FakeAmap


class FakeClassifier:
    def __init__(self, route, score=0.95):
        self._r = None if route is None else {"route": route, "score": score}

    async def predict_route(self, text):
        return self._r


class FakeLLM:
    def __init__(self):
        self.purposes = []

    async def chat(self, messages, **kw):
        self.purposes.append(kw.get("purpose"))
        if kw.get("purpose") == "route.llm":
            return '{"label": "realtime"}'
        return "不客气～"

    async def complete(self, messages, **kw):
        self.purposes.append(kw.get("purpose"))
        return {"message": {"content": "宽窄巷子附近推荐：A、B、C"}, "usage": {}}


class FakeRag:
    def __init__(self):
        self.queries = []

    async def search(self, query, top_k=5, rerank_k=10):
        self.queries.append(query)
        return {"answer": "要提前预约 [1]", "hits": [{"title": "t"}], "degraded": []}


class FakePlanResult:
    text = "第1天 ..."

    def to_dict(self):
        return {"itinerary": {}, "text": self.text}


class FakeAgent:
    def __init__(self):
        self.requests = []

    async def plan(self, req, emit=None):
        self.requests.append(req)
        return FakePlanResult()


PREV = Itinerary.model_validate({"title": "成都", "city": "成都", "days": [
    {"day": 1, "stops": [{"name": "宽窄巷子", "poi_id": "P1", "location": {"lng": 104.05, "lat": 30.67}}]}]})


def _router(route):
    llm, rag, agent, amap = FakeLLM(), FakeRag(), FakeAgent(), FakeAmap()
    router = ChatRouter(FakeClassifier(route), llm, rag, agent, ToolExecutor(amap, retriever=None), amap)
    return router, llm, rag, agent


def test_notes_qa_goes_to_rag_and_keeps_itinerary():
    router, llm, rag, agent = _router("notes_qa")
    out = asyncio.run(router.handle(ChatRequest(message="要注意什么", previous_itinerary=PREV)))
    assert out["route"] == "notes_qa" and "plan" not in out
    assert rag.queries == ["成都 要注意什么"]  # 用行程城市补全检索词
    assert agent.requests == []


def test_plan_route_revises_previous_itinerary():
    router, _, _, agent = _router("plan")
    out = asyncio.run(router.handle(ChatRequest(message="第二天轻松点", previous_itinerary=PREV)))
    assert out["plan"] and agent.requests[0].previous_itinerary is PREV
    assert agent.requests[0].destination == "成都"


def test_plan_without_destination_asks_for_it():
    router, _, _, agent = _router("plan")
    out = asyncio.run(router.handle(ChatRequest(message="帮我排个行程")))
    assert "目的地" in out["answer"] and agent.requests == []


def test_chitchat_and_llm_fallback_routing():
    router, llm, _, _ = _router("chitchat")
    assert asyncio.run(router.handle(ChatRequest(message="谢谢")))["answer"] == "不客气～"

    router, llm, _, _ = _router(None)  # 本地路由不可用 → LLM 判定为 realtime
    out = asyncio.run(router.handle(ChatRequest(message="明天下雨吗", destination="成都")))
    assert out["route"] == "realtime" and out["route_source"] == "llm"
    assert "chat.tool_qa" in llm.purposes


def test_spatial_question_uses_tool_qa_with_itinerary_pois():
    router, llm, _, _ = _router("graph_qa")
    out = asyncio.run(router.handle(ChatRequest(message="宽窄巷子附近有什么吃的", previous_itinerary=PREV)))
    assert out["route"] == "graph_qa" and "推荐" in out["answer"]
    assert llm.purposes == ["chat.tool_qa"]


def test_router_injects_memory_and_returns_updates(tmp_path):
    from memory.service import MemoryService
    from memory.store import MemoryStore

    class MemLLM(FakeLLM):
        async def chat(self, messages, **kw):
            if kw.get("purpose") == "memory.extract":
                return '{"memories": [{"key": "mobility", "value": "不能爬山", "evidence": "不能爬山", "confidence": 0.9}]}'
            return await super().chat(messages, **kw)

    store = MemoryStore(tmp_path / "m.sqlite3")
    store.upsert("u1", "diet", "不吃辣", 0.9)
    llm, rag, agent, amap = MemLLM(), FakeRag(), FakeAgent(), FakeAmap()
    router = ChatRouter(FakeClassifier("plan"), llm, rag, agent, ToolExecutor(amap, retriever=None), amap,
                        MemoryService(store, llm))
    out = asyncio.run(router.handle(ChatRequest(message="我爸不能爬山，第二天换一下", previous_itinerary=PREV,
                                                user_id="u1", session_id="s1")))
    assert agent.requests[0].memory == ["饮食：不吃辣"]  # 规划时带上已有长期记忆
    assert out["memory_updates"][0]["value"] == "不能爬山"  # 本轮新学到的记忆
    assert [t["role"] for t in store.turns("s1")] == ["user", "assistant"]


def test_destination_inferred_from_conversation_when_no_trip(tmp_path):
    from memory.service import MemoryService
    from memory.store import MemoryStore
    from rag.metadata import PlaceMatcher

    store = MemoryStore(tmp_path / "m.sqlite3")
    store.add_turn("s1", "u1", "user", "这次去成都想吃火锅")
    llm, rag, agent, amap = FakeLLM(), FakeRag(), FakeAgent(), FakeAmap()
    rag.retriever = type("R", (), {"matcher": PlaceMatcher(["成都", "重庆"])})()
    router = ChatRouter(FakeClassifier("graph_qa"), llm, rag, agent, ToolExecutor(amap, retriever=None), amap,
                        MemoryService(store, llm))
    out = asyncio.run(router.handle(ChatRequest(message="宽窄巷子附近有什么吃的", user_id="u1", session_id="s1")))
    assert "推荐" in out["answer"]  # 识别出成都后走工具问答，而不是反问城市


def test_tool_markup_never_reaches_the_user():
    """回归：DeepSeek 在禁止调用工具时偶尔把内部工具调用语法写进正文。"""
    from agent.qa import strip_tool_markup

    leaked = ('Now let me check the routes.\n<｜｜DSML｜｜ calls> <｜｜DSML｜｜ invoke name="route_plan">\n'
              '<｜｜DSML｜｜ parameter name="origin_poi_id" string="true">B001C7WE5S</｜｜DSML｜｜ parameter>')
    assert "DSML" not in strip_tool_markup(leaked)
    assert strip_tool_markup("<｜｜DSML｜｜ calls>B001</｜｜DSML｜｜ calls>") == ""
    assert strip_tool_markup("宽窄巷子附近推荐：A、B") == "宽窄巷子附近推荐：A、B"


def test_last_round_has_no_tools_and_answer_is_sanitized():
    from agent.qa import FALLBACK, answer_with_tools
    from agent.tools import PlanContext

    class LoopingLLM:
        def __init__(self):
            self.calls = []

        async def complete(self, messages, **kw):
            self.calls.append(kw)
            if kw.get("tools"):  # 一直想调工具
                return {"message": {"content": "", "tool_calls": [
                    {"id": f"c{len(self.calls)}", "type": "function",
                     "function": {"name": "poi_search", "arguments": '{"keyword": "宽窄巷子"}'}}]}, "usage": {}}
            return {"message": {"content": "<｜｜DSML｜｜ calls>B1</｜｜DSML｜｜ calls>"}, "usage": {}}

    llm, amap = LoopingLLM(), FakeAmap()
    out = asyncio.run(answer_with_tools(llm, ToolExecutor(amap, retriever=None), PlanContext(destination="成都"),
                                        "sys", "宽窄巷子附近有什么", ["poi_search"], max_rounds=3))
    assert "tools" not in llm.calls[-1]          # 最后一轮不给工具
    assert out["answer"] == FALLBACK             # 只剩标记时返回兜底文案，而不是把标记给用户
