"""行程校验：纯规则 + 高德实算，不依赖 LLM，结果确定、可解释。

不通过的问题以自然语言列表返回给模型修正；同时产出统计（地点匹配、城市一致、超长路段等）供评测使用。
"""
import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from pydantic import ValidationError

from agent.schemas import Itinerary, Leg, Location
from agent.tools import PlanContext
from config import AgentConfig
from core.amap import AmapClient
from core.errors import AppError
from core.logging import get_logger

log = get_logger("agent.validator")


@dataclass
class ValidationResult:
    itinerary: Optional[Itinerary]
    issues: List[str] = field(default_factory=list)
    stats: Dict[str, int] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.itinerary is not None and not self.issues


def _pydantic_issues(e: ValidationError) -> List[str]:
    out = []
    for err in e.errors()[:8]:
        loc = ".".join(str(p) for p in err.get("loc", []))
        out.append(f"结构错误 {loc}：{err.get('msg')}")
    return out


class ItineraryValidator:
    def __init__(self, amap: AmapClient, cfg: AgentConfig):
        self.amap = amap
        self.cfg = cfg

    async def check(self, raw: Dict[str, Any], ctx: PlanContext, expected_days: int) -> ValidationResult:
        try:
            itinerary = Itinerary.model_validate(raw)
        except ValidationError as e:
            return ValidationResult(None, _pydantic_issues(e), {"schema_ok": 0})

        issues: List[str] = []
        stats = {"schema_ok": 1, "stops": 0, "stops_resolved": 0, "stops_in_city": 0,
                 "legs": 0, "legs_verified": 0, "long_legs": 0, "over_days": 0}

        if len(itinerary.days) != expected_days:
            issues.append(f"用户要求 {expected_days} 天，当前是 {len(itinerary.days)} 天")

        seen: Dict[str, str] = {}
        for day in itinerary.days:
            for stop in day.stops:
                stats["stops"] += 1
                poi = ctx.pois.get(stop.poi_id)
                if poi is None:
                    issues.append(f"第{day.day}天「{stop.name}」的 poi_id 不是 poi_search 返回的结果，请先搜索该地点并使用返回的 poi_id")
                    continue
                stats["stops_resolved"] += 1
                stop.location = Location(**poi["location"])
                stop.address = poi.get("address", "")
                stop.district = poi.get("district", "")
                in_city = ctx.in_destination(poi)
                if in_city is False:
                    issues.append(f"第{day.day}天「{stop.name}」位于{poi.get('city') or '其他城市'}，不在目的地{ctx.destination}")
                else:
                    stats["stops_in_city"] += 1
                if stop.poi_id in seen:
                    issues.append(f"「{stop.name}」在{seen[stop.poi_id]}和第{day.day}天重复出现")
                else:
                    seen[stop.poi_id] = f"第{day.day}天"

        await self._verify_legs(itinerary, ctx, issues, stats)
        return ValidationResult(itinerary, issues, stats)

    async def _verify_legs(self, itinerary: Itinerary, ctx: PlanContext, issues: List[str], stats: Dict[str, int]):
        """用高德实算相邻两站交通时长，覆盖模型填写的值，并检查超长路段与每日总时长。"""
        jobs: List[Tuple[Any, Any, Any]] = []
        for day in itinerary.days:
            stops = day.stops
            if stops and stops[-1].transport_to_next:
                stops[-1].transport_to_next = None  # 当天最后一站不需要交通
            for a, b in zip(stops, stops[1:]):
                if a.location and b.location:
                    if a.transport_to_next is None:
                        a.transport_to_next = Leg(mode="transit")
                    jobs.append((day, a, b))

        results = await asyncio.gather(*[self._route_for(a, b, ctx) for _, a, b in jobs])
        for (day, a, b), route in zip(jobs, results):
            stats["legs"] += 1
            if route is None:
                itinerary.warnings.append(f"第{day.day}天 {a.name}→{b.name} 未能计算路线")
                continue
            stats["legs_verified"] += 1
            a.transport_to_next = Leg(mode=route["mode"], duration_min=route["duration_min"],
                                      distance_m=route["distance_m"], verified=True)
            if route["duration_min"] > self.cfg.max_leg_minutes:
                stats["long_legs"] += 1
                issues.append(
                    f"第{day.day}天 {a.name}→{b.name} {self._mode_cn(route['mode'])}需 {route['duration_min']} 分钟，"
                    f"超过 {self.cfg.max_leg_minutes} 分钟，请调整顺序或换成更近的地点"
                )

        for day in itinerary.days:
            total = sum(s.stay_min for s in day.stops) + sum(
                (s.transport_to_next.duration_min or 0) for s in day.stops if s.transport_to_next
            )
            if total > self.cfg.max_day_minutes:
                stats["over_days"] += 1
                issues.append(f"第{day.day}天停留加交通共 {total} 分钟，超过 {self.cfg.max_day_minutes} 分钟，请减少地点或缩短停留")

    async def _route_for(self, a, b, ctx: PlanContext) -> Optional[Dict[str, Any]]:
        mode = a.transport_to_next.mode if a.transport_to_next else "transit"
        try:
            route = await self.amap.route(a.location.model_dump(), b.location.model_dump(), mode, ctx.destination)
            # 公交无方案（常见于距离很近）时退回步行
            if route is None and mode == "transit":
                route = await self.amap.route(a.location.model_dump(), b.location.model_dump(), "walking")
            return route
        except AppError as e:
            log.warning("route verify failed", extra={"code": e.code.value, "from": a.name, "to": b.name})
            return None

    @staticmethod
    def _mode_cn(mode: str) -> str:
        return {"walking": "步行", "driving": "驾车", "transit": "公交/地铁"}.get(mode, mode)
