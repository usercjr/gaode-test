"""Agent 工具：定义（OpenAI tools 格式）与执行。

工具结果无论成功失败都以 JSON 回填给模型：失败时带 error + hint，模型可以换关键词重试，
这就是“工具调用失败自动纠错”。每次调用都记入 trace，供日志、前端进度展示和评测统计。
"""
import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional

from starlette.concurrency import run_in_threadpool

from core.amap import AmapClient, region_prefix
from core.errors import AppError
from core.logging import get_logger
from rag.retriever import Retriever

log = get_logger("agent.tools")

_STOP_PROPS = {
    "name": {"type": "string"},
    "poi_id": {"type": "string", "description": "必须是 poi_search 返回的 poi_id"},
    "category": {"type": "string", "enum": ["attraction", "food", "shopping", "hotel", "other"]},
    "arrive": {"type": "string", "description": "建议到达时间 HH:MM"},
    "stay_min": {"type": "integer", "description": "建议停留分钟数"},
    "cost_per_person": {"type": "number", "description": "人均花费（元），不确定可省略"},
    "tips": {"type": "array", "items": {"type": "string"}, "description": "1-3 条实用建议，口语化"},
    "source_note_ids": {"type": "array", "items": {"type": "string"}, "description": "参考过的笔记 note_id"},
    "transport_to_next": {
        "type": "object",
        "description": "到当天下一站的交通方式；当天最后一站省略",
        "properties": {"mode": {"type": "string", "enum": ["walking", "driving", "transit"]}},
        "required": ["mode"],
    },
}

TOOL_DEFS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_notes",
            "description": "检索真实旅行者笔记，获取具体店名、玩法、避坑经验。返回笔记 note_id、标题和摘录。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "检索词，建议包含目的地和关注点，如“成都 火锅 老店”"},
                    "top_k": {"type": "integer", "description": "返回条数，默认 4"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "poi_search",
            "description": "在高德地图搜索地点，返回 poi_id、坐标、地址、评分、人均。行程里的每个地点都必须先用它查到 poi_id。",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {"type": "string", "description": "地点名称，尽量用标准名，如“宽窄巷子”“陕西历史博物馆”"},
                    "city": {"type": "string", "description": "城市，默认为目的地"},
                },
                "required": ["keyword"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "route_plan",
            "description": "查询两个地点之间的实际交通时长和距离，用于判断顺序是否合理。",
            "parameters": {
                "type": "object",
                "properties": {
                    "from_poi_id": {"type": "string"},
                    "to_poi_id": {"type": "string"},
                    "mode": {"type": "string", "enum": ["walking", "driving", "transit"]},
                },
                "required": ["from_poi_id", "to_poi_id", "mode"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询目的地未来几天天气预报（高德只提供约 4 天）。",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_itinerary",
            "description": "提交最终行程。服务端会校验：地点必须来自 poi_search、位于目的地、相邻两站交通时长与每日总时长合理。"
                           "不通过时会返回问题列表，请修正后再次提交。",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "city": {"type": "string"},
                    "summary": {"type": "string", "description": "一两句话概括行程亮点"},
                    "days": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "day": {"type": "integer"},
                                "theme": {"type": "string", "description": "当天主题，有画面感"},
                                "stops": {
                                    "type": "array",
                                    "items": {"type": "object", "properties": _STOP_PROPS,
                                              "required": ["name", "poi_id", "stay_min"]},
                                },
                            },
                            "required": ["day", "stops"],
                        },
                    },
                },
                "required": ["title", "city", "days"],
            },
        },
    },
]

SUBMIT_TOOL = "submit_itinerary"


@dataclass
class ToolCallRecord:
    name: str
    args: Dict[str, Any]
    ok: bool
    latency_ms: int
    empty: bool = False
    error: str = ""


@dataclass
class PlanContext:
    """一次规划的会话状态：目的地信息、本次查到过的 POI/笔记、工具调用记录。"""

    destination: str
    dest_geo: Optional[Dict[str, Any]] = None
    pois: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    notes: Dict[str, str] = field(default_factory=dict)
    records: List[ToolCallRecord] = field(default_factory=list)

    @property
    def dest_prefix(self) -> str:
        if not self.dest_geo:
            return ""
        return region_prefix(self.dest_geo["adcode"], self.dest_geo.get("level", ""))

    def in_destination(self, poi: Dict[str, Any]) -> Optional[bool]:
        if not self.dest_prefix or not poi.get("adcode"):
            return None
        return region_prefix(poi["adcode"]).startswith(self.dest_prefix) or str(poi["adcode"]).startswith(self.dest_prefix)


class ToolExecutor:
    def __init__(self, amap: AmapClient, retriever: Retriever, notes_top_k: int = 4):
        self.amap = amap
        self.retriever = retriever
        self.notes_top_k = notes_top_k
        self._handlers: Dict[str, Callable[[PlanContext, Dict[str, Any]], Awaitable[Dict[str, Any]]]] = {
            "search_notes": self._search_notes,
            "poi_search": self._poi_search,
            "route_plan": self._route_plan,
            "get_weather": self._get_weather,
        }

    async def run(self, ctx: PlanContext, name: str, raw_args: str) -> Dict[str, Any]:
        """执行一次工具调用，返回要回填给模型的结果（总是 dict，不抛异常）。"""
        started = time.perf_counter()
        try:
            args = json.loads(raw_args or "{}")
            if not isinstance(args, dict):
                raise ValueError("arguments must be an object")
        except ValueError as e:
            result = {"ok": False, "error": f"参数不是合法 JSON：{e}", "hint": "请按工具定义重新调用"}
            self._record(ctx, name, {}, result, started)
            return result

        handler = self._handlers.get(name)
        if handler is None:
            result = {"ok": False, "error": f"未知工具 {name}"}
        else:
            try:
                result = await asyncio.wait_for(handler(ctx, args), timeout=30)
            except AppError as e:
                result = {"ok": False, "error": e.message, "hint": "可以换个关键词或稍后重试"}
            except asyncio.TimeoutError:
                result = {"ok": False, "error": "工具超时", "hint": "可以稍后重试"}
            except Exception as e:  # 工具内部错误不应中断整个规划
                log.exception("tool failed", extra={"tool": name})
                result = {"ok": False, "error": f"工具内部错误：{type(e).__name__}"}
        self._record(ctx, name, args, result, started)
        return result

    def _record(self, ctx: PlanContext, name: str, args: Dict[str, Any], result: Dict[str, Any], started: float):
        rec = ToolCallRecord(
            name=name,
            args=args,
            ok=bool(result.get("ok")),
            empty=bool(result.get("empty")),
            error=str(result.get("error", "")),
            latency_ms=round((time.perf_counter() - started) * 1000),
        )
        ctx.records.append(rec)
        log.info("tool call", extra={"tool": name, "ok": rec.ok, "empty": rec.empty, "latency_ms": rec.latency_ms,
                                     "error": rec.error or None})

    # ---------- 各工具 ----------
    async def _search_notes(self, ctx: PlanContext, args: Dict[str, Any]) -> Dict[str, Any]:
        query = str(args.get("query") or ctx.destination).strip()
        top_k = max(1, min(int(args.get("top_k") or self.notes_top_k), 8))
        # 规划场景用不重排的检索路径（P2 评测：nDCG@5 0.826，P95 约 36ms）
        res = await run_in_threadpool(self.retriever.retrieve, query, top_k * 3, rerank="none")
        notes, seen = [], set()
        for h in res["hits"]:
            nid = h.get("row_id")
            if not nid or nid in seen:
                continue
            seen.add(nid)
            ctx.notes[nid] = h.get("title", "")
            notes.append({"note_id": nid, "title": h.get("title", ""), "excerpt": h.get("text", "")[:300]})
            if len(notes) >= top_k:
                break
        return {"ok": True, "empty": not notes, "notes": notes}

    async def _poi_search(self, ctx: PlanContext, args: Dict[str, Any]) -> Dict[str, Any]:
        keyword = str(args.get("keyword") or "").strip()
        if not keyword:
            return {"ok": False, "error": "keyword 不能为空"}
        city = str(args.get("city") or ctx.destination)
        pois = await self.amap.search_poi(keyword, city, limit=3)
        out = []
        for p in pois:
            ctx.pois[p["poi_id"]] = p
            out.append({
                "poi_id": p["poi_id"], "name": p["name"], "type": p["type"].split(";")[0],
                "district": p["district"], "address": p["address"], "rating": p["rating"], "cost": p["cost"],
                "in_destination": ctx.in_destination(p),
            })
        if not out:
            return {"ok": True, "empty": True, "pois": [],
                    "hint": "没有找到，换更标准的名称（去掉“打卡”“附近”等词）或换一个同类地点"}
        return {"ok": True, "pois": out}

    async def _route_plan(self, ctx: PlanContext, args: Dict[str, Any]) -> Dict[str, Any]:
        a, b = ctx.pois.get(str(args.get("from_poi_id"))), ctx.pois.get(str(args.get("to_poi_id")))
        if not a or not b:
            return {"ok": False, "error": "poi_id 不存在", "hint": "先用 poi_search 查询地点再规划路线"}
        mode = str(args.get("mode") or "transit")
        route = await self.amap.route(a["location"], b["location"], mode, ctx.destination)
        if not route:
            return {"ok": True, "empty": True, "hint": f"没有可行的{mode}路线，换一种交通方式"}
        return {"ok": True, **route}

    async def _get_weather(self, ctx: PlanContext, args: Dict[str, Any]) -> Dict[str, Any]:
        city = str(args.get("city") or "").strip()
        geo = ctx.dest_geo if not city or city == ctx.destination else await self.amap.geocode(city)
        if not geo:
            return {"ok": True, "empty": True, "hint": "未识别该城市"}
        casts = await self.amap.weather(geo["adcode"])
        return {"ok": True, "empty": not casts, "forecast": casts}
