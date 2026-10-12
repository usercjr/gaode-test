"""Agent 编排：Function Calling 循环 + 校验自纠。

    messages = [system, user]
    循环（最多 max_steps 轮）：
        LLM(messages, tools) →
          有工具调用：并发执行普通工具；submit_itinerary 交给校验器，
                      不通过则把问题列表作为该工具的结果回填，让模型修正（最多 max_fix_rounds 次）
          无工具调用：提醒模型调用 submit_itinerary
    校验通过 / 修正次数用完 → 返回行程（未通过的问题写入 warnings）
"""
import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional

from agent.prompts import build_system_prompt, build_user_prompt
from agent.render import render_text
from agent.schemas import PlanRequest
from agent.tools import PLAN_TOOLS, SUBMIT_TOOL, PlanContext, ToolExecutor, tool_defs
from agent.validator import ItineraryValidator, ValidationResult
from config import AgentConfig
from core.amap import AmapClient
from core.errors import AppError, ErrorCode
from core.llm import LLMClient
from core.logging import get_logger

log = get_logger("agent")

EventSink = Callable[[str, Dict[str, Any]], Awaitable[None]]


async def _noop(_event: str, _data: Dict[str, Any]) -> None:
    return None


@dataclass
class PlanResult:
    itinerary: Dict[str, Any]
    text: str
    passed: bool
    first_pass: bool
    fix_rounds: int
    llm_rounds: int
    tool_calls: List[Dict[str, Any]]
    validation: Dict[str, int]
    tokens: Dict[str, int]
    latency_ms: int
    issues: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


def _brief(args: Dict[str, Any]) -> str:
    for key in ("keyword", "query", "city"):
        if args.get(key):
            return str(args[key])[:40]
    return ""


class PlanAgent:
    def __init__(self, llm: LLMClient, amap: AmapClient, tools: ToolExecutor, cfg: AgentConfig):
        self.llm = llm
        self.amap = amap
        self.tools = tools
        self.cfg = cfg
        self.validator = ItineraryValidator(amap, cfg)

    async def plan(self, req: PlanRequest, emit: EventSink = _noop) -> PlanResult:
        started = time.perf_counter()
        days = req.days or (len(req.previous_itinerary.days) if req.previous_itinerary else self.cfg.default_days)
        ctx = PlanContext(destination=req.destination)

        await emit("status", {"stage": "geocode", "message": f"识别目的地 {req.destination}"})
        ctx.dest_geo = await self.amap.geocode(req.destination)
        if ctx.dest_geo is None:
            raise AppError(ErrorCode.BAD_REQUEST, f"无法识别目的地：{req.destination}")
        if req.previous_itinerary:
            # 多轮修改：上一版里的地点视为已查询过，模型可以直接沿用 poi_id
            await self._preload_previous(ctx, req)

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": build_system_prompt(req, days, self.cfg.max_leg_minutes)},
            {"role": "user", "content": build_user_prompt(req, days, req.previous_itinerary)},
        ]
        tokens = {"prompt": 0, "completion": 0}
        fix_rounds = 0
        first_result: Optional[ValidationResult] = None
        final: Optional[ValidationResult] = None
        nudges = 0

        submitted = False
        for step in range(1, self.cfg.max_steps + 1):
            await emit("status", {"stage": "thinking", "message": f"第 {step} 轮规划"})
            # 预算保护：剩余轮数只够“提交 + 修正”时，强制模型提交，避免信息收集耗尽步数
            steps_left = self.cfg.max_steps - step + 1
            force_submit = not submitted and steps_left <= self.cfg.max_fix_rounds + 1
            if force_submit:
                messages.append({"role": "user", "content": "信息已经足够，请立即调用 submit_itinerary 提交行程。"})
            resp = await self.llm.complete(
                messages, tools=tool_defs(PLAN_TOOLS), temperature=self.cfg.temperature, purpose="agent.plan",
                tool_choice={"type": "function", "function": {"name": SUBMIT_TOOL}} if force_submit else None,
            )
            usage = resp["usage"]
            tokens["prompt"] += int(usage.get("prompt_tokens") or 0)
            tokens["completion"] += int(usage.get("completion_tokens") or 0)
            msg = resp["message"]
            tool_calls = msg.get("tool_calls") or []
            messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": tool_calls}
                            if tool_calls else {"role": "assistant", "content": msg.get("content") or ""})

            if not tool_calls:
                nudges += 1
                if nudges > 2:
                    break
                messages.append({"role": "user", "content": "请通过调用 submit_itinerary 工具提交行程，不要直接输出文字。"})
                continue

            submits = [c for c in tool_calls if c["function"]["name"] == SUBMIT_TOOL]
            submitted = submitted or bool(submits)
            normal = [c for c in tool_calls if c["function"]["name"] != SUBMIT_TOOL]
            for c in normal:
                args = _safe_args(c)
                await emit("tool_call", {"tool": c["function"]["name"], "brief": _brief(args)})
            results = await asyncio.gather(*[self.tools.run(ctx, c["function"]["name"], c["function"]["arguments"])
                                             for c in normal])
            for c, result in zip(normal, results):
                messages.append({"role": "tool", "tool_call_id": c["id"],
                                 "content": json.dumps(result, ensure_ascii=False)})

            for i, c in enumerate(submits):
                if i < len(submits) - 1:  # 同一轮多次提交只校验最后一次
                    messages.append({"role": "tool", "tool_call_id": c["id"],
                                     "content": json.dumps({"ok": False, "error": "已忽略，只处理最后一次提交"})})
                    continue
                await emit("status", {"stage": "validate", "message": "校验行程（地点、城市、路线时长）"})
                result = await self.validator.check(_safe_args(c), ctx, days)
                if first_result is None:
                    first_result = result
                await emit("validation", {"passed": result.passed, "issues": result.issues[:10],
                                          "round": fix_rounds})
                if result.passed or fix_rounds >= self.cfg.max_fix_rounds:
                    final = result
                    messages.append({"role": "tool", "tool_call_id": c["id"],
                                     "content": json.dumps({"ok": True}, ensure_ascii=False)})
                else:
                    fix_rounds += 1
                    messages.append({"role": "tool", "tool_call_id": c["id"], "content": json.dumps(
                        {"ok": False, "issues": result.issues, "hint": "请修正以上问题后重新调用 submit_itinerary"},
                        ensure_ascii=False)})
            if final is not None:
                break

        if final is None or final.itinerary is None:
            raise AppError(ErrorCode.PLAN_INVALID, "规划未能在限定步数内完成，请简化需求后重试",
                           detail={"steps": step, "fix_rounds": fix_rounds})

        itinerary = final.itinerary
        if not final.passed:
            itinerary.warnings.extend(final.issues)
        result = PlanResult(
            itinerary=itinerary.model_dump(),
            text=render_text(itinerary),
            passed=final.passed,
            first_pass=bool(first_result and first_result.passed),
            fix_rounds=fix_rounds,
            llm_rounds=step,
            tool_calls=[r.__dict__ for r in ctx.records],
            validation=final.stats,
            tokens=tokens,
            latency_ms=round((time.perf_counter() - started) * 1000),
            issues=final.issues,
        )
        log.info("plan done", extra={k: getattr(result, k) for k in
                                      ("passed", "first_pass", "fix_rounds", "llm_rounds", "latency_ms")}
                 | {"tool_calls": len(ctx.records), "tokens": tokens})
        return result

    async def _preload_previous(self, ctx: PlanContext, req: PlanRequest) -> None:
        for day in req.previous_itinerary.days:
            for stop in day.stops:
                if stop.location and stop.poi_id not in ctx.pois:
                    ctx.pois[stop.poi_id] = {
                        "poi_id": stop.poi_id, "name": stop.name, "location": stop.location.model_dump(),
                        "address": stop.address, "district": stop.district,
                        # 上一版已通过城市校验，用目的地 adcode 代表其所在城市
                        "adcode": ctx.dest_geo.get("adcode", ""), "city": req.destination,
                    }


def _safe_args(call: Dict[str, Any]) -> Dict[str, Any]:
    try:
        args = json.loads(call["function"].get("arguments") or "{}")
        return args if isinstance(args, dict) else {}
    except ValueError:
        return {}
