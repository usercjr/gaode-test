"""异步 LLM 客户端（OpenAI 兼容 /chat/completions）。

职责：超时、按错误类型重试、熔断、并发闸门、调用日志（耗时/token），
并把供应商的各种失败统一翻译成 ErrorCode。
"""
import asyncio
import time
from typing import Any, Dict, List, Optional

import httpx

from config import Settings
from core.errors import AppError, ErrorCode
from core.logging import get_logger
from core.metrics import metrics
from core.resilience import CircuitBreaker, CircuitOpenError, retry_async

log = get_logger("llm")

# 不同供应商表示“额度/余额不足”的方式不同：DeepSeek 402，DashScope 403 + AllocationQuota.*
_QUOTA_MARKERS = ("insufficient", "quota", "balance", "allocationquota", "arrearage")


def _classify_http_error(status: int, body: str) -> ErrorCode:
    lowered = body.lower()
    if status == 402 or any(m in lowered for m in _QUOTA_MARKERS):
        return ErrorCode.UPSTREAM_LLM_QUOTA
    if status in (401, 403):
        return ErrorCode.UPSTREAM_LLM_AUTH
    if status == 429:
        return ErrorCode.UPSTREAM_LLM_RATE
    if status >= 500:
        return ErrorCode.UPSTREAM_LLM_UNAVAILABLE
    return ErrorCode.UPSTREAM_LLM_BAD_RESPONSE


class LLMClient:
    def __init__(self, settings: Settings):
        self._cfg = settings.llm
        self._api_key = settings.llm_api_key.get_secret_value().strip()
        self._endpoint = f"{self._cfg.base_url}/chat/completions"
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(self._cfg.timeout_s, connect=10.0))
        self._gate = asyncio.Semaphore(self._cfg.max_concurrency)
        self._waiting = 0  # 等待并发闸门的调用数（排队长度），用于监控
        self._in_use = 0
        # 进程内累计用量（评测时取前后差值即可得到单次请求的 token 消耗）
        self.usage_total = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
        self.breaker = CircuitBreaker(
            "llm",
            failure_threshold=self._cfg.breaker_failure_threshold,
            recovery_timeout_s=self._cfg.breaker_recovery_s,
        )

    @property
    def enabled(self) -> bool:
        return bool(self._api_key)

    @property
    def default_model(self) -> str:
        return self._cfg.model

    async def aclose(self) -> None:
        await self._client.aclose()

    def concurrency(self) -> Dict[str, int]:
        return {"limit": self._cfg.max_concurrency, "in_use": self._in_use, "waiting": self._waiting}

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        *,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        purpose: str = "chat",
    ) -> str:
        """返回 assistant 文本；失败时抛 AppError。purpose 用于日志区分调用场景。"""
        result = await self.complete(
            messages, model=model, temperature=temperature, max_tokens=max_tokens, purpose=purpose
        )
        return (result["message"].get("content") or "").strip()

    async def complete(
        self,
        messages: List[Dict[str, Any]],
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        purpose: str = "chat",
    ) -> Dict[str, Any]:
        """完整的 chat/completions 调用，返回 {message, usage}。message 可能包含 tool_calls。"""
        if not self.enabled:
            raise AppError(ErrorCode.UPSTREAM_LLM_AUTH, "未配置 LLM_API_KEY")
        payload: Dict[str, Any] = {
            "model": model or self._cfg.model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens
        if tools:
            payload["tools"] = tools
            if tool_choice is not None:
                payload["tool_choice"] = tool_choice

        return await retry_async(
            lambda: self._call_once(payload, purpose),
            attempts=self._cfg.retry_attempts,
            op=f"llm.{purpose}",
        )

    async def _call_once(self, payload: Dict[str, Any], purpose: str) -> Dict[str, Any]:
        if not self.breaker.allow():
            raise CircuitOpenError(ErrorCode.UPSTREAM_LLM_UNAVAILABLE, "llm")

        started = time.perf_counter()
        status: Optional[int] = None
        try:
            self._waiting += 1
            try:
                await self._gate.acquire()
            finally:
                self._waiting -= 1
            self._in_use += 1
            try:
                resp = await self._client.post(
                    self._endpoint,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    json=payload,
                )
            finally:
                self._in_use -= 1
                self._gate.release()
            status = resp.status_code
            if status != 200:
                code = _classify_http_error(status, resp.text)
                raise AppError(code, detail=resp.text[:500])
            data = resp.json()
            message = (data.get("choices") or [{}])[0].get("message") or {}
            # 工具调用轮的 content 可以为空；既没有文本也没有工具调用才算异常
            if not (message.get("content") or "").strip() and not message.get("tool_calls"):
                raise AppError(ErrorCode.UPSTREAM_LLM_BAD_RESPONSE, detail="empty content")
        except httpx.TimeoutException as e:
            self._finish(payload, purpose, started, status, ErrorCode.UPSTREAM_LLM_TIMEOUT)
            raise AppError(ErrorCode.UPSTREAM_LLM_TIMEOUT, detail=str(e)) from e
        except httpx.HTTPError as e:
            self._finish(payload, purpose, started, status, ErrorCode.UPSTREAM_LLM_UNAVAILABLE)
            raise AppError(ErrorCode.UPSTREAM_LLM_UNAVAILABLE, detail=str(e)) from e
        except AppError as e:
            self._finish(payload, purpose, started, status, e.code, detail=e.detail)
            raise
        except ValueError as e:  # 非 JSON 响应
            self._finish(payload, purpose, started, status, ErrorCode.UPSTREAM_LLM_BAD_RESPONSE)
            raise AppError(ErrorCode.UPSTREAM_LLM_BAD_RESPONSE, detail=str(e)) from e

        self._finish(payload, purpose, started, status, None, usage=data.get("usage"))
        return {"message": message, "usage": data.get("usage") or {}}

    def _finish(self, payload, purpose, started, status, code: Optional[ErrorCode], usage=None, detail=None):
        # 只有“服务端不可用”类错误计入熔断；额度/鉴权/参数问题重试也没用，但不代表服务挂了
        if code is None:
            self.breaker.record_success()
        elif code in (ErrorCode.UPSTREAM_LLM_TIMEOUT, ErrorCode.UPSTREAM_LLM_UNAVAILABLE, ErrorCode.UPSTREAM_LLM_RATE):
            self.breaker.record_failure()

        fields = {
            "purpose": purpose,
            "model": payload.get("model"),
            "http_status": status,
            "latency_ms": round((time.perf_counter() - started) * 1000),
            "ok": code is None,
        }
        metrics.llm_call(purpose, code is None, fields["latency_ms"],
                         int((usage or {}).get("prompt_tokens") or 0), int((usage or {}).get("completion_tokens") or 0))
        if code is None:
            self.usage_total["calls"] += 1
            self.usage_total["prompt_tokens"] += int((usage or {}).get("prompt_tokens") or 0)
            self.usage_total["completion_tokens"] += int((usage or {}).get("completion_tokens") or 0)
        if usage:
            fields["prompt_tokens"] = usage.get("prompt_tokens")
            fields["completion_tokens"] = usage.get("completion_tokens")
        if code is not None:
            fields["code"] = code.value
            if detail:
                fields["detail"] = str(detail)[:300]
            log.warning("llm call failed", extra=fields)
        else:
            log.info("llm call", extra=fields)
