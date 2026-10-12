"""轻量工具问答：给模型一小组工具，最多几轮查证后直接用文字回答（不产出行程）。

用于地点关系类（附近有什么、顺不顺路）与实时类（天气、交通时长）问题。
"""
import asyncio
import json
from typing import Any, Awaitable, Callable, Dict, List

from agent.tools import PlanContext, ToolExecutor, tool_defs
from core.llm import LLMClient

EventSink = Callable[[str, Dict[str, Any]], Awaitable[None]]


async def _noop(_e, _d):
    return None


async def answer_with_tools(
    llm: LLMClient,
    executor: ToolExecutor,
    ctx: PlanContext,
    system: str,
    question: str,
    tool_names: List[str],
    max_rounds: int = 4,
    emit: EventSink = _noop,
) -> Dict[str, Any]:
    messages: List[Dict[str, Any]] = [{"role": "system", "content": system}, {"role": "user", "content": question}]
    tools = tool_defs(tool_names)
    for round_no in range(1, max_rounds + 1):
        last = round_no == max_rounds
        resp = await llm.complete(messages, tools=tools, temperature=0.3, purpose="chat.tool_qa",
                                  tool_choice="none" if last else None)  # 最后一轮禁止调工具，必须作答
        msg = resp["message"]
        calls = msg.get("tool_calls") or []
        if not calls:
            return {"answer": (msg.get("content") or "").strip(), "rounds": round_no}
        messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls})
        for c in calls:
            await emit("tool_call", {"tool": c["function"]["name"], "brief": c["function"].get("arguments", "")[:60]})
        results = await asyncio.gather(*[executor.run(ctx, c["function"]["name"], c["function"]["arguments"])
                                         for c in calls])
        for c, r in zip(calls, results):
            messages.append({"role": "tool", "tool_call_id": c["id"], "content": json.dumps(r, ensure_ascii=False)})
    return {"answer": "", "rounds": max_rounds}
