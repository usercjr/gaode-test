"""压测用的模拟 LLM：OpenAI 兼容 /chat/completions，固定延迟后返回一句话。

压测要测的是“我们系统”的并发与保护能力，不应把 DeepSeek 的速度和费用算进来；
延迟和故障模式可在运行中通过 POST /mode 切换（用于熔断测试）。

python eval/mock_llm.py --port 8099 --delay 1.0
"""
import argparse
import asyncio

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

state = {"delay": 1.0, "fail": False, "calls": 0, "in_flight": 0, "max_in_flight": 0}


async def completions(request: Request):
    await request.body()
    state["calls"] += 1
    state["in_flight"] += 1
    state["max_in_flight"] = max(state["max_in_flight"], state["in_flight"])
    try:
        await asyncio.sleep(state["delay"])
        if state["fail"]:
            return JSONResponse({"error": {"message": "mock upstream failure"}}, status_code=500)
        return JSONResponse({
            "choices": [{"message": {"role": "assistant", "content": "你好！想去哪里玩？可以告诉我目的地和天数。"}}],
            "usage": {"prompt_tokens": 50, "completion_tokens": 20},
        })
    finally:
        state["in_flight"] -= 1


async def mode(request: Request):
    body = await request.json()
    for k in ("delay", "fail"):
        if k in body:
            state[k] = body[k]
    if body.get("reset_stats"):
        state.update(calls=0, max_in_flight=0)
    return JSONResponse(state)


async def stats(_request: Request):
    return JSONResponse(state)


app = Starlette(routes=[Route("/chat/completions", completions, methods=["POST"]),
                        Route("/mode", mode, methods=["POST"]), Route("/stats", stats)])

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--delay", type=float, default=1.0)
    args = ap.parse_args()
    state["delay"] = args.delay
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
