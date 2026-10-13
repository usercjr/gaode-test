"""轻量工具问答：给模型一小组工具，最多几轮查证后直接用文字回答（不产出行程）。

用于地点关系类（附近有什么、顺不顺路）与实时类（天气、交通时长）问题。
"""
import asyncio
import json
import re
from typing import Any, Awaitable, Callable, Dict, List

from agent.tools import PlanContext, ToolExecutor, tool_defs
from core.llm import LLMClient
from core.logging import get_logger

log = get_logger("agent.qa")
EventSink = Callable[[str, Dict[str, Any]], Awaitable[None]]

# 模型想调工具却被禁止时，DeepSeek 偶尔会把内部的工具调用语法当正文输出（<｜DSML｜invoke ...>）
_TOOL_MARKUP = re.compile(r"<[｜|][^>]*>|</[｜|][^>]*>")
FINAL_INSTRUCTION = "请直接根据以上查询结果，用中文文字回答用户的问题，不要再调用任何工具。"
FALLBACK = "暂时没查到可靠的结果，换个问法试试？"


async def _noop(_e, _d):
    return None


def strip_tool_markup(text: str) -> str:
    """去掉泄漏到正文里的工具调用标记；只剩标记（或空）时返回空字符串。"""
    if not text or "<｜" not in text and "<|" not in text:
        return (text or "").strip()
    cleaned = _TOOL_MARKUP.sub("", text)
    # 标记之间常夹着参数值（poi_id 等），只保留像句子的行
    lines = [l.strip() for l in cleaned.splitlines()
             if l.strip() and not re.fullmatch(r"[\w\-.,:/\s]+", l.strip())]
    return "\n".join(lines).strip()


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
        if round_no == max_rounds:
            # 最后一轮：不提供工具并明确要求作答。只用 tool_choice="none" 时模型仍可能把工具调用写进正文
            messages.append({"role": "user", "content": FINAL_INSTRUCTION})
            resp = await llm.complete(messages, temperature=0.3, purpose="chat.tool_qa")
        else:
            resp = await llm.complete(messages, tools=tools, temperature=0.3, purpose="chat.tool_qa")
        msg = resp["message"]
        calls = msg.get("tool_calls") or []
        if not calls or round_no == max_rounds:
            raw = msg.get("content") or ""
            answer = strip_tool_markup(raw)
            if answer != raw.strip():
                log.warning("tool markup leaked into answer", extra={"round": round_no, "raw": raw[:200]})
            return {"answer": answer or FALLBACK, "rounds": round_no}
        messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls})
        for c in calls:
            await emit("tool_call", {"tool": c["function"]["name"], "brief": c["function"].get("arguments", "")[:60]})
        results = await asyncio.gather(*[executor.run(ctx, c["function"]["name"], c["function"]["arguments"])
                                         for c in calls])
        for c, r in zip(calls, results):
            messages.append({"role": "tool", "tool_call_id": c["id"], "content": json.dumps(r, ensure_ascii=False)})
    return {"answer": FALLBACK, "rounds": max_rounds}
