"""agent_service：旅行助手的 Python 服务（RAG、分类、行程生成，后续承载 Agent 编排）。

启动：cd agent_service && .venv-win\\Scripts\\python.exe app.py
"""
import asyncio
import json
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Literal, Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

from agent.orchestrator import PlanAgent
from agent.router import ChatRouter
from agent.schemas import ChatRequest, PlanRequest
from agent.tools import ToolExecutor
from classifier.labels import format_input
from classifier.service import ClassifierService
from config import get_settings
from core.amap import AmapClient
from core.errors import AppError, ErrorCode, error_body
from core.llm import LLMClient
from core.logging import get_logger, request_id_var, setup_logging
from rag.ingest import load_csv_records
from rag.pipeline import RagPipeline
from rag.retriever import Retriever
from rag.store import VectorStore
from memory.service import MemoryService
from memory.store import MemoryStore

settings = get_settings()
setup_logging(settings.logging.level, settings.logging.dir, settings.logging.retention_days)
log = get_logger("app")


class Services:
    llm: LLMClient
    store: VectorStore
    retriever: Retriever
    rag: RagPipeline
    amap: AmapClient
    agent: PlanAgent
    classifier: ClassifierService
    router: ChatRouter
    memory: Optional[MemoryService]


services = Services()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    log.info("starting agent_service", extra={"config": settings.public_view()})
    if not settings.llm_enabled:
        log.warning("LLM_API_KEY is not set; LLM features will fail with UPSTREAM_LLM_AUTH")
    services.llm = LLMClient(settings)
    services.store = VectorStore(settings.rag)
    services.retriever = Retriever(services.store, settings.retrieval)
    services.rag = RagPipeline(services.retriever, services.llm, settings.rag.llm_model)
    services.classifier = ClassifierService(services.llm, settings.classifier.models_dir,
                                            settings.classifier.style_model, settings.classifier.route_model,
                                            settings.classifier.route_confidence)
    services.amap = AmapClient(settings)
    executor = ToolExecutor(services.amap, services.retriever, settings.agent.notes_top_k)
    services.agent = PlanAgent(services.llm, services.amap, executor, settings.agent)
    services.memory = (MemoryService(MemoryStore(settings.memory.db_path), services.llm)
                       if settings.memory.enabled else None)
    services.router = ChatRouter(services.classifier, services.llm, services.rag, services.agent, executor,
                                 services.amap, services.memory)
    if not settings.amap_enabled:
        log.warning("AMAP_WEB_SERVICE_KEY is not set; /plan will fail")
    # 预加载 embedding 模型、向量库、BM25 索引和重排模型，避免首个请求因冷启动超时；
    # 失败不阻止启动，由 /readyz 反映，请求时会再次尝试加载
    try:
        count = await run_in_threadpool(services.store.count)
        log.info("vector store ready", extra={"count": count})
        await run_in_threadpool(services.retriever.load)
        await run_in_threadpool(services.classifier.load)
        if settings.retrieval.rerank == "cross_encoder":
            await run_in_threadpool(services.retriever.reranker.ensure_ready)
    except Exception:
        log.exception("warmup failed")
    yield
    await services.llm.aclose()
    await services.amap.aclose()
    if services.memory:
        services.memory.store.close()


app = FastAPI(title="WanderAI Agent Service", version="0.2.0", lifespan=lifespan)


# ---------------- 中间件：request-id + 访问日志 ----------------
@app.middleware("http")
async def request_context(request: Request, call_next):
    rid = request.headers.get("x-request-id") or uuid.uuid4().hex
    token = request_id_var.set(rid)
    started = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["X-Request-ID"] = rid
        return response
    finally:
        log.info(
            "access",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status": status,
                "latency_ms": round((time.perf_counter() - started) * 1000),
            },
        )
        request_id_var.reset(token)


# ---------------- 统一错误响应 ----------------
def _error_response(code: ErrorCode, message: Optional[str] = None) -> JSONResponse:
    return JSONResponse(
        status_code=code.spec.http_status,
        content=error_body(code, message or code.spec.message, request_id_var.get()),
        headers={"X-Request-ID": request_id_var.get()},
    )


@app.exception_handler(AppError)
async def handle_app_error(_request: Request, exc: AppError):
    level = log.error if exc.code == ErrorCode.INTERNAL else log.warning
    level("request failed", extra={"code": exc.code.value, "detail": str(exc.detail)[:500] if exc.detail else None})
    return _error_response(exc.code, exc.message)


@app.exception_handler(RequestValidationError)
async def handle_validation_error(_request: Request, exc: RequestValidationError):
    first = exc.errors()[0] if exc.errors() else {}
    field = ".".join(str(p) for p in first.get("loc", []) if p != "body")
    return _error_response(ErrorCode.BAD_REQUEST, f"参数错误：{field} {first.get('msg', '')}".strip())


@app.exception_handler(StarletteHTTPException)
async def handle_http_error(_request: Request, exc: StarletteHTTPException):
    code = ErrorCode.NOT_FOUND if exc.status_code == 404 else ErrorCode.BAD_REQUEST
    return _error_response(code, str(exc.detail) if exc.detail else None)


@app.exception_handler(Exception)
async def handle_unexpected(_request: Request, exc: Exception):
    log.exception("unhandled error")
    return _error_response(ErrorCode.INTERNAL)


# ---------------- 请求模型 ----------------
class IngestCsvRequest(BaseModel):
    csv_path: str = Field(..., description="CSV 路径，必须位于配置的 data 目录内")
    title_col: str = "title"
    text_col: str = "text"
    source: str = ""
    reset: bool = False
    chunk_size: int = Field(default=400, ge=100, le=2000)
    chunk_overlap: int = Field(default=80, ge=0, le=500)
    encoding: str = "utf-8-sig"


class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(default=5, ge=1, le=10)
    style_label: str = ""
    use_rerank: bool = True


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(default=5, ge=1, le=20)
    rerank_k: int = Field(default=10, ge=3, le=40)


class ClassifyRequest(BaseModel):
    """两种用法：直接给 text；或给结构化字段，由服务端拼成与训练数据一致的格式。"""

    text: str = Field("", max_length=1000)
    destination: str = Field("", max_length=50)
    preferences: List[str] = Field(default_factory=list, max_length=20)
    message: str = Field("", max_length=1000)


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(..., min_length=1, max_length=20000)


class ItineraryRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., min_length=1, max_length=40)
    temperature: float = Field(default=0.7, ge=0.0, le=1.5)


# ---------------- 探针 ----------------
@app.get("/healthz")
async def healthz():
    """存活探针：进程能响应即可。"""
    return {"status": "ok"}


@app.get("/readyz")
async def readyz():
    """就绪探针：逐项检查依赖。任一关键依赖不可用返回 503。"""
    checks = {}
    try:
        checks["vector_store"] = {"ok": True, "count": await run_in_threadpool(services.store.count)}
    except Exception as e:
        checks["vector_store"] = {"ok": False, "error": str(e)[:200]}
    checks["retriever"] = {
        "ok": services.retriever.ready,
        "mode": settings.retrieval.mode,
        "rerank": settings.retrieval.rerank,
        "reranker_loaded": services.retriever.reranker.ready,
        "notes_with_metadata": len(services.retriever.metadata),
    }
    breaker = services.llm.breaker.snapshot()
    checks["llm"] = {
        "ok": settings.llm_enabled and breaker["state"] != "open",
        "configured": settings.llm_enabled,
        "model": settings.llm.model,
        "breaker": breaker,
    }
    checks["classifier"] = {"ok": True, **services.classifier.status()}  # 模型缺失时退回 LLM，不影响就绪
    checks["amap"] = {
        "ok": settings.amap_enabled,
        "configured": settings.amap_enabled,
        "cache_entries": len(services.amap.cache._data),
        "cache_hits": services.amap.cache.hits,
    }
    ready = all(c["ok"] for c in checks.values())
    return JSONResponse(status_code=200 if ready else 503, content={"ready": ready, "checks": checks})


# ---------------- 业务接口 ----------------
@app.post("/ingest/csv")
async def ingest_csv(req: IngestCsvRequest):
    data_dir = settings.rag.data_dir
    csv_path = Path(req.csv_path)
    if not csv_path.is_absolute():
        csv_path = data_dir.parent / csv_path
    csv_path = csv_path.resolve()
    if data_dir not in csv_path.parents:
        raise AppError(ErrorCode.BAD_REQUEST, f"csv_path 必须位于 {data_dir} 内")
    if not csv_path.exists():
        raise AppError(ErrorCode.BAD_REQUEST, f"文件不存在：{csv_path.name}")

    try:
        records = await run_in_threadpool(load_csv_records, csv_path, req.title_col, req.text_col, req.encoding)
    except ValueError as e:
        raise AppError(ErrorCode.BAD_REQUEST, str(e)) from e
    if not records:
        raise AppError(ErrorCode.BAD_REQUEST, "没有可入库的文本行")

    result = await run_in_threadpool(
        services.store.ingest_records,
        records,
        req.source or csv_path.name,
        req.reset,
        req.chunk_size,
        req.chunk_overlap,
    )
    await run_in_threadpool(services.retriever.load)  # 重建 BM25 索引
    return {**result, "rows": len(records), "csv_path": str(csv_path)}


@app.post("/retrieve")
async def retrieve(req: RetrieveRequest):
    return await services.rag.retrieve_context(
        query=req.query.strip(), top_k=req.top_k, style_label=req.style_label, use_rerank=req.use_rerank
    )


@app.post("/search")
async def search(req: SearchRequest):
    return await services.rag.search(query=req.query.strip(), top_k=req.top_k, rerank_k=req.rerank_k)


@app.post("/classify")
async def classify(req: ClassifyRequest):
    # text 表示聊天原话（同时做路由）；结构化字段表示规划表单（只做风格识别）
    is_chat = bool(req.text.strip())
    text = req.text.strip() if is_chat else format_input(req.destination, req.preferences, req.message)
    if not text:
        raise AppError(ErrorCode.BAD_REQUEST, "text 或 destination/preferences/message 至少提供一项")
    return await services.classifier.classify(text, with_route=is_chat)


@app.post("/itinerary")
async def itinerary(req: ItineraryRequest):
    text = await services.llm.chat(
        [m.model_dump() for m in req.messages], temperature=req.temperature, purpose="itinerary"
    )
    return {"text": text, "model": services.llm.default_model}


async def _plan_with_memory(req: PlanRequest, emit=None) -> dict:
    """规划前注入该用户的长期记忆；同时从“补充需求”里学习新记忆，并把这轮记入会话。"""
    mem = services.memory
    learn_task = None
    if mem and req.user_id:
        req.memory = mem.prompt_lines(req.user_id)  # 覆盖客户端传入的值
        if req.message.strip():
            learn_task = asyncio.create_task(mem.learn(req.user_id, req.message, "plan_form"))
    else:
        req.memory = []
    result = (await services.agent.plan(req, emit) if emit else await services.agent.plan(req)).to_dict()
    if mem:
        mem.record(req.session_id, req.user_id, "user",
                   f"规划 {req.destination} {req.days or ''} 天。{req.message}".strip(), "plan")
        mem.record(req.session_id, req.user_id, "assistant", result["text"], "plan")
    if learn_task is not None:
        try:
            result["memory_updates"] = await learn_task
        except Exception:
            log.exception("memory learning failed")
            result["memory_updates"] = []
    result["memory_used"] = req.memory
    return result


def _require_memory(user_id: str) -> MemoryService:
    if services.memory is None:
        raise AppError(ErrorCode.NOT_FOUND, "记忆功能未开启")
    if not user_id or len(user_id) > 64 or not all(c.isalnum() or c in "_-" for c in user_id):
        raise AppError(ErrorCode.BAD_REQUEST, "缺少或非法的 user_id")
    return services.memory


@app.get("/memory")
async def list_memory(user_id: str):
    """查看该用户的长期记忆（active 与待确认 pending）。"""
    return {"memories": _require_memory(user_id).list(user_id)}


@app.delete("/memory/{memory_id}")
async def delete_memory(memory_id: int, user_id: str):
    deleted = _require_memory(user_id).store.delete(user_id, memory_id)
    if not deleted:
        raise AppError(ErrorCode.NOT_FOUND, "记忆不存在")
    return {"deleted": deleted}


@app.delete("/memory")
async def clear_memory(user_id: str):
    return {"deleted": _require_memory(user_id).store.delete(user_id)}


@app.post("/plan")
async def plan(req: PlanRequest):
    """Agent 规划（一次性返回）。"""
    return await _plan_with_memory(req)


def _sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _sse_response(job, op: str) -> StreamingResponse:
    """把一个接受 emit 回调的协程包装成 SSE：进度事件实时推送，最后 done 或 error；客户端断开时取消任务。"""
    queue: asyncio.Queue = asyncio.Queue()

    async def emit(event: str, data) -> None:
        await queue.put((event, data))

    async def run() -> None:
        try:
            await queue.put(("done", await job(emit)))
        except AppError as e:
            log.warning(f"{op} failed", extra={"code": e.code.value, "detail": str(e.detail)[:300] if e.detail else None})
            await queue.put(("error", error_body(e.code, e.message, request_id_var.get())))
        except Exception:
            log.exception(f"{op} crashed")
            await queue.put(("error", error_body(ErrorCode.INTERNAL, ErrorCode.INTERNAL.spec.message, request_id_var.get())))
        finally:
            await queue.put(None)

    task = asyncio.create_task(run())

    async def stream():
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield _sse(*item)
        finally:
            if not task.done():  # 客户端断开时停止处理，避免继续消耗 LLM 调用
                task.cancel()

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/plan/stream")
async def plan_stream(req: PlanRequest):
    """Agent 规划（SSE 流式）：status / tool_call / validation 进度事件，最后 done 或 error。"""

    async def job(emit):
        return await _plan_with_memory(req, emit)

    return _sse_response(job, "plan")


@app.post("/chat")
async def chat(req: ChatRequest):
    """聊天消息：先路由（plan / notes_qa / graph_qa / realtime / chitchat），再交给对应流程。"""
    return await services.router.handle(req)


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """同 /chat，SSE 推送 route / tool_call / status 等进度事件。"""
    return _sse_response(lambda emit: services.router.handle(req, emit), "chat")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host=settings.server.host, port=settings.server.port, reload=False, log_config=None)
