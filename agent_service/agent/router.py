"""动态路由：聊天消息先识别意图，再交给对应的处理流程，而不是每句话都重新规划行程。

    plan      → PlanAgent（有上一版行程时在其基础上修改）
    notes_qa  → RAG 问答（混合检索 + 重排 + 引用回答）
    graph_qa  → 工具问答：地点搜索 / 周边搜索 / 路线（知识图谱在 P7 接入前的空间查询实现）
    realtime  → 工具问答：天气 / 路线时长
    chitchat  → 直接简短回复
"""
import time
from datetime import date
from typing import Any, Awaitable, Callable, Dict

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
from rag.pipeline import RagPipeline

log = get_logger("agent.router")
EventSink = Callable[[str, Dict[str, Any]], Awaitable[None]]

ROUTE_CN = {"plan": "规划/修改行程", "notes_qa": "游记经验问答", "graph_qa": "地点与周边查询",
            "realtime": "实时信息", "chitchat": "闲聊"}

SPATIAL_SYSTEM = """你是旅行助手，回答地点关系类问题（某地附近有什么、两地顺不顺路、去了 A 还能去哪）。
目的地：{dest}。{trip}
必须用工具查证：附近的店用 nearby_search，两地距离和时长用 route_plan（先用 poi_search 拿到 poi_id）。
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
                 executor: ToolExecutor, amap: AmapClient):
        self.classifier = classifier
        self.llm = llm
        self.rag = rag
        self.agent = agent
        self.executor = executor
        self.amap = amap

    async def route(self, message: str) -> Dict[str, Any]:
        res = await self.classifier.predict_route(message)
        if res:
            return {"route": res["route"], "source": "local", "score": res["score"]}
        llm = await classify_route(self.llm, message)  # 本地模型缺失或不确定
        return {"route": llm["route"], "source": llm["source"], "score": None}

    async def handle(self, req: ChatRequest, emit: EventSink = _noop) -> Dict[str, Any]:
        started = time.perf_counter()
        r = await self.route(req.message)
        route = r["route"]
        destination = req.destination or (req.previous_itinerary.city if req.previous_itinerary else "")
        await emit("route", {**r, "route_cn": ROUTE_CN[route]})
        log.info("chat routed", extra={"route": route, "source": r["source"], "score": r["score"]})

        out: Dict[str, Any] = {"route": route, "route_source": r["source"], "route_score": r["score"]}
        if route == "plan":
            if not destination:
                out["answer"] = "想去哪里呢？告诉我目的地和天数，我来帮你排行程。"
            else:
                plan = await self.agent.plan(PlanRequest(
                    destination=destination, days=req.days, start_date=req.start_date,
                    preferences=req.preferences, style_prompt=req.style_prompt, message=req.message,
                    previous_itinerary=req.previous_itinerary), emit)
                out["plan"] = plan.to_dict()
                out["answer"] = plan.text
        elif route == "notes_qa":
            query = req.message if (not destination or destination in req.message) else f"{destination} {req.message}"
            res = await self.rag.search(query)
            out.update(answer=res["answer"], hits=res["hits"][:3], degraded=res.get("degraded", []))
        elif route in ("graph_qa", "realtime"):
            if not destination:
                out["answer"] = "你想问的是哪个城市呢？告诉我目的地我再帮你查。"
            else:
                out.update(await self._tool_qa(route, req, destination, emit))
        else:
            out["answer"] = await self.llm.chat(
                [{"role": "system", "content": CHITCHAT_SYSTEM}, {"role": "user", "content": req.message}],
                temperature=0.7, max_tokens=150, purpose="chat.chitchat")
        out["latency_ms"] = round((time.perf_counter() - started) * 1000)
        return out

    async def _tool_qa(self, route: str, req: ChatRequest, destination: str, emit: EventSink) -> Dict[str, Any]:
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
        if route == "graph_qa":
            system = SPATIAL_SYSTEM.format(dest=destination, trip=trip)
            tools = ["poi_search", "nearby_search", "route_plan", "search_notes"]
        else:
            system = REALTIME_SYSTEM.format(dest=destination, trip=trip, today=date.today().isoformat())
            tools = ["get_weather", "poi_search", "route_plan"]
        res = await answer_with_tools(self.llm, self.executor, ctx, system, req.message, tools, emit=emit)
        return {"answer": res["answer"] or "暂时没查到，换个问法试试？",
                "tool_calls": [{"name": c.name, "ok": c.ok} for c in ctx.records]}
