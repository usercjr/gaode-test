"""动态路由：聊天消息先识别意图，再交给对应的处理流程，而不是每句话都重新规划行程。

    plan      → PlanAgent（有上一版行程时在其基础上修改）
    notes_qa  → RAG 问答（混合检索 + 重排 + 引用回答）
    graph_qa  → 工具问答：地点搜索 / 周边搜索 / 路线（知识图谱在 P7 接入前的空间查询实现）
    realtime  → 工具问答：天气 / 路线时长
    chitchat  → 直接简短回复
"""
import asyncio
import time
from datetime import date
from typing import Any, Awaitable, Callable, Dict, List, Optional

from agent.orchestrator import PlanAgent
from agent.qa import answer_with_tools
from agent.schemas import ChatRequest, PlanRequest
from agent.tools import PlanContext, ToolExecutor
from classifier.llm_zero_shot import classify_route
from classifier.service import ClassifierService
from core.amap import AmapClient
from core.errors import AppError, ErrorCode
from core.llm import LLMClient
from core.logging import get_logger
from memory.service import MemoryService
from rag.pipeline import RagPipeline

log = get_logger("agent.router")
EventSink = Callable[[str, Dict[str, Any]], Awaitable[None]]

ROUTE_CN = {"plan": "规划/修改行程", "notes_qa": "游记经验问答", "graph_qa": "地点与周边查询",
            "realtime": "实时信息", "chitchat": "闲聊"}

SPATIAL_SYSTEM = """你是旅行助手，回答地点关系类问题（某地附近有什么、两地顺不顺路、去了 A 还能去哪）。
目的地：{dest}。{trip}
必须用工具查证，按问题类型选工具：
- “附近 / 旁边 / 顺路”：先用 graph_query 的 nearby（游记推荐过的近处地点）；结果少于 3 个时，再用 nearby_search 补充高德周边结果。
- “去了 A 的人一般还去哪 / 常和 A 一起去”：用 graph_query 的 co_visit；结果带直线距离，距离超过 5 公里的要说明“常一起去但不顺路”。
- “某城市有哪些推荐的 X”：用 graph_query 的 by_tag，再用 search_notes 看游记原文补充理由。
- 两地实际交通时长用 route_plan（先用 poi_search 拿到 poi_id）。
图谱推荐只有 1 篇游记提到时不要强调“推荐度”；2 篇及以上可以注明“X 篇游记提到”。
不要编造距离、评分和人均；回答简洁，列出 3~5 个具体结果（名称、距离、评分或人均），最后给一句建议。"""

REALTIME_SYSTEM = """你是旅行助手，回答实时信息类问题（天气、两地之间交通多久）。今天是 {today}。
目的地：{dest}。{trip}
天气用 get_weather（只能查未来约 4 天，超出范围要直说）；交通时长用 route_plan（先用 poi_search 拿到 poi_id）。
回答简洁，给出具体数字和出行建议。"""

CHITCHAT_SYSTEM = "你是旅行助手“行界”。用一两句话友好回复；如果用户在道谢或闲聊，顺带提示可以继续问行程、美食或天气。"


async def _noop(_e, _d):
    return None


def _trip_summary(req: ChatRequest) -> str:
    if not req.previous_itinerary:
        return ""
    days = [f"第{d.day}天：" + "、".join(s.name for s in d.stops) for d in req.previous_itinerary.days]
    return "当前行程：" + "；".join(days)


class ChatRouter:
    def __init__(self, classifier: ClassifierService, llm: LLMClient, rag: RagPipeline, agent: PlanAgent,
                 executor: ToolExecutor, amap: AmapClient, memory: Optional[MemoryService] = None):
        self.classifier = classifier
        self.llm = llm
        self.rag = rag
        self.agent = agent
        self.executor = executor
        self.amap = amap
        self.memory = memory

    async def route(self, message: str) -> Dict[str, Any]:
        res = await self.classifier.predict_route(message)
        if res:
            return {"route": res["route"], "source": "local", "score": res["score"]}
        llm = await classify_route(self.llm, message)  # 本地模型缺失或不确定
        return {"route": llm["route"], "source": llm["source"], "score": None}

    async def handle(self, req: ChatRequest, emit: EventSink = _noop) -> Dict[str, Any]:
        started = time.perf_counter()
        mem = self.memory
        memory_lines = mem.prompt_lines(req.user_id) if mem else []
        # 记忆抽取与回答并行：抽的是用户原话，不依赖回答结果，所以不增加延迟
        learn_task = asyncio.create_task(mem.learn(req.user_id, req.message, "chat")) if mem and req.user_id else None
        question = await mem.session.standalone_question(req.session_id, req.message) if mem else req.message
        if question != req.message:
            await emit("status", {"stage": "rewrite", "message": f"理解为：{question}"})

        r = await self.route(question)
        route = r["route"]
        destination = (req.destination or (req.previous_itinerary.city if req.previous_itinerary else "")
                       or self._infer_destination(question, req.session_id))
        await emit("route", {**r, "route_cn": ROUTE_CN[route]})
        log.info("chat routed", extra={"route": route, "source": r["source"], "score": r["score"],
                                       "rewritten": question != req.message})

        out: Dict[str, Any] = {"route": route, "route_source": r["source"], "route_score": r["score"]}
        if question != req.message:
            out["question"] = question
        if route == "plan":
            if not destination:
                out["answer"] = "想去哪里呢？告诉我目的地和天数，我来帮你排行程。"
            else:
                plan = await self.agent.plan(PlanRequest(
                    destination=destination, days=req.days, start_date=req.start_date,
                    preferences=req.preferences, style_prompt=req.style_prompt, message=question,
                    previous_itinerary=req.previous_itinerary, memory=memory_lines,
                    user_id=req.user_id, session_id=req.session_id), emit)
                out["plan"] = plan.to_dict()
                out["answer"] = plan.text
        elif route == "notes_qa":
            query = question if (not destination or destination in question) else f"{destination} {question}"
            res = await self.rag.search(query)
            out.update(answer=res["answer"], hits=res["hits"][:3], degraded=res.get("degraded", []))
        elif route in ("graph_qa", "realtime"):
            if not destination:
                out["answer"] = "你想问的是哪个城市呢？告诉我目的地我再帮你查。"
            else:
                out.update(await self._tool_qa(route, req, question, destination, memory_lines, emit))
        else:
            out["answer"] = await self.llm.chat(
                [{"role": "system", "content": CHITCHAT_SYSTEM}, {"role": "user", "content": req.message}],
                temperature=0.7, max_tokens=150, purpose="chat.chitchat")

        if mem:
            mem.record(req.session_id, req.user_id, "user", req.message, route)
            mem.record(req.session_id, req.user_id, "assistant", out.get("answer", ""), route)
            mem.summarize_later(req.session_id)
        if learn_task is not None:
            try:
                out["memory_updates"] = await learn_task
            except Exception:  # 记忆失败不影响回答
                log.exception("memory learning failed")
                out["memory_updates"] = []
        out["latency_ms"] = round((time.perf_counter() - started) * 1000)
        return out

    def _infer_destination(self, question: str, session_id: str) -> str:
        """没有行程上下文时，从问题本身、再从本会话最近的对话里识别城市（用检索器的地名词典）。"""
        matcher = getattr(getattr(self.rag, "retriever", None), "matcher", None)
        if matcher is None:
            return ""
        found = matcher.find(question)
        if not found and self.memory and session_id:
            found = matcher.find(self.memory.session.context_text(session_id))
        return found[0] if found else ""

    async def _tool_qa(self, route: str, req: ChatRequest, question: str, destination: str,
                       memory_lines: List[str], emit: EventSink) -> Dict[str, Any]:
        ctx = PlanContext(destination=destination)
        ctx.dest_geo = await self.amap.geocode(destination)
        if ctx.dest_geo is None:
            raise AppError(ErrorCode.BAD_REQUEST, f"无法识别目的地：{destination}")
        if req.previous_itinerary:  # 行程里的地点可以直接用 poi_id
            for d in req.previous_itinerary.days:
                for s in d.stops:
                    if s.location:
                        ctx.pois[s.poi_id] = {"poi_id": s.poi_id, "name": s.name, "location": s.location.model_dump(),
                                              "adcode": ctx.dest_geo.get("adcode", ""), "city": destination}
        trip = _trip_summary(req)
        if memory_lines:
            trip += "\n用户长期偏好（推荐时要考虑）：" + "；".join(memory_lines)
        if route == "graph_qa":
            system = SPATIAL_SYSTEM.format(dest=destination, trip=trip)
            tools = ["graph_query", "poi_search", "nearby_search", "route_plan", "search_notes"]
        else:
            system = REALTIME_SYSTEM.format(dest=destination, trip=trip, today=date.today().isoformat())
            tools = ["get_weather", "poi_search", "route_plan"]
        res = await answer_with_tools(self.llm, self.executor, ctx, system, question, tools, emit=emit)
        return {"answer": res["answer"] or "暂时没查到，换个问法试试？",
                "tool_calls": [{"name": c.name, "ok": c.ok} for c in ctx.records]}
