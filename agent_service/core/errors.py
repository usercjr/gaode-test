"""统一错误码。

所有接口失败时都返回同一结构：
    {"error": {"code", "message", "retryable", "request_id"}, "degraded": [...]}
前端只根据 code 决定提示语与是否允许重试。
"""
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class ErrorSpec:
    http_status: int
    retryable: bool
    message: str  # 面向用户的默认提示语


class ErrorCode(str, Enum):
    BAD_REQUEST = "BAD_REQUEST"
    NOT_FOUND = "NOT_FOUND"
    RATE_LIMITED = "RATE_LIMITED"
    UPSTREAM_LLM_TIMEOUT = "UPSTREAM_LLM_TIMEOUT"
    UPSTREAM_LLM_RATE = "UPSTREAM_LLM_RATE"
    UPSTREAM_LLM_QUOTA = "UPSTREAM_LLM_QUOTA"
    UPSTREAM_LLM_AUTH = "UPSTREAM_LLM_AUTH"
    UPSTREAM_LLM_UNAVAILABLE = "UPSTREAM_LLM_UNAVAILABLE"
    UPSTREAM_LLM_BAD_RESPONSE = "UPSTREAM_LLM_BAD_RESPONSE"
    UPSTREAM_AMAP_ERROR = "UPSTREAM_AMAP_ERROR"
    RAG_UNAVAILABLE = "RAG_UNAVAILABLE"
    GRAPH_UNAVAILABLE = "GRAPH_UNAVAILABLE"
    AGENT_UNAVAILABLE = "AGENT_UNAVAILABLE"
    GATEWAY_TIMEOUT = "GATEWAY_TIMEOUT"
    PLAN_INVALID = "PLAN_INVALID"
    INTERNAL = "INTERNAL"

    @property
    def spec(self) -> ErrorSpec:
        return ERROR_SPECS[self]


ERROR_SPECS: Dict[ErrorCode, ErrorSpec] = {
    ErrorCode.BAD_REQUEST: ErrorSpec(400, False, "请求参数有误"),
    ErrorCode.NOT_FOUND: ErrorSpec(404, False, "资源不存在"),
    ErrorCode.RATE_LIMITED: ErrorSpec(429, True, "请求太频繁，请稍后再试"),
    ErrorCode.UPSTREAM_LLM_TIMEOUT: ErrorSpec(504, True, "AI 生成超时，请稍后重试"),
    ErrorCode.UPSTREAM_LLM_RATE: ErrorSpec(503, True, "AI 服务繁忙，请稍后重试"),
    ErrorCode.UPSTREAM_LLM_QUOTA: ErrorSpec(502, False, "AI 服务额度不足，请联系管理员"),
    ErrorCode.UPSTREAM_LLM_AUTH: ErrorSpec(502, False, "AI 服务鉴权失败，请检查 API Key 配置"),
    ErrorCode.UPSTREAM_LLM_UNAVAILABLE: ErrorSpec(503, True, "AI 服务暂时不可用，请稍后重试"),
    ErrorCode.UPSTREAM_LLM_BAD_RESPONSE: ErrorSpec(502, True, "AI 返回内容异常，请重试"),
    ErrorCode.UPSTREAM_AMAP_ERROR: ErrorSpec(502, True, "地图服务暂时不可用"),
    ErrorCode.RAG_UNAVAILABLE: ErrorSpec(503, True, "笔记检索服务暂时不可用"),
    ErrorCode.GRAPH_UNAVAILABLE: ErrorSpec(503, True, "知识图谱服务暂时不可用"),
    ErrorCode.AGENT_UNAVAILABLE: ErrorSpec(503, True, "智能服务未启动或暂时不可用"),
    ErrorCode.GATEWAY_TIMEOUT: ErrorSpec(504, True, "请求处理超时，请稍后重试"),
    ErrorCode.PLAN_INVALID: ErrorSpec(422, False, "行程生成未通过校验，请调整需求后重试"),
    ErrorCode.INTERNAL: ErrorSpec(500, False, "服务内部错误"),
}


class AppError(Exception):
    """业务异常：携带错误码，由全局异常处理器转换为统一响应。"""

    def __init__(self, code: ErrorCode, message: Optional[str] = None, detail: Any = None):
        self.code = code
        self.message = message or code.spec.message
        self.detail = detail  # 仅写日志，不返回给前端
        super().__init__(f"{code.value}: {self.message}")

    @property
    def retryable(self) -> bool:
        return self.code.spec.retryable

    @property
    def http_status(self) -> int:
        return self.code.spec.http_status


def error_body(code: ErrorCode, message: str, request_id: str, degraded=None) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "error": {
            "code": code.value,
            "message": message,
            "retryable": code.spec.retryable,
            "request_id": request_id,
        }
    }
    if degraded:
        body["degraded"] = list(degraded)
    return body
