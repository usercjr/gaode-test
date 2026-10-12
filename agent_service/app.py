"""agent_service：旅行助手的 Python 服务（RAG、分类、行程生成，后续承载 Agent 编排）。

启动：cd agent_service && .venv-win\\Scripts\\python.exe app.py
"""
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Literal, Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

from classifier.llm_zero_shot import classify_style
from config import get_settings
from core.errors import AppError, ErrorCode, error_body
from core.llm import LLMClient
from core.logging import get_logger, request_id_var, setup_logging
from rag.ingest import load_csv_records
from rag.pipeline import RagPipeline
from rag.retriever import Retriever
from rag.store import VectorStore

settings = get_settings()
setup_logging(settings.logging.level, settings.logging.dir, settings.logging.retention_days)
log = get_logger("app")


class Services:
    llm: LLMClient
    store: VectorStore
    retriever: Retriever
    rag: RagPipeline


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
    # 预加载 embedding 模型、向量库、BM25 索引和重排模型，避免首个请求因冷启动超时；
    # 失败不阻止启动，由 /readyz 反映，请求时会再次尝试加载
    try:
        count = await run_in_threadpool(services.store.count)
        log.info("vector store ready", extra={"count": count})
        await run_in_threadpool(services.retriever.load)
        if settings.retrieval.rerank == "cross_encoder":
            await run_in_threadpool(services.retriever.reranker.ensure_ready)
    except Exception:
        log.exception("warmup failed")
    yield
    await services.llm.aclose()


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
    text: str = Field(..., min_length=1, max_length=1000)


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
    return await classify_style(services.llm, req.text.strip())


@app.post("/itinerary")
async def itinerary(req: ItineraryRequest):
    text = await services.llm.chat(
        [m.model_dump() for m in req.messages], temperature=req.temperature, purpose="itinerary"
    )
    return {"text": text, "model": services.llm.default_model}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host=settings.server.host, port=settings.server.port, reload=False, log_config=None)
