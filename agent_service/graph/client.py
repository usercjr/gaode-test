"""Neo4j 异步客户端：超时 + 熔断 + 统一错误码。图谱不可用时抛 GRAPH_UNAVAILABLE，由调用方降级。"""
import asyncio
import time
from typing import Any, Dict, List, Optional

from neo4j import AsyncGraphDatabase, Query
from neo4j.exceptions import Neo4jError, ServiceUnavailable

from config import Settings
from core.errors import AppError, ErrorCode
from core.logging import get_logger
from core.resilience import CircuitBreaker

log = get_logger("graph")


class GraphClient:
    def __init__(self, settings: Settings):
        self._cfg = settings.graph
        password = settings.neo4j_password.get_secret_value()
        self.enabled = self._cfg.enabled and bool(password)
        self._driver = (AsyncGraphDatabase.driver(self._cfg.uri, auth=(self._cfg.user, password),
                                                  connection_timeout=self._cfg.timeout_s,
                                                  # 驱动默认会在连不上时自己重试 30 秒，降级就失去意义了
                                                  max_transaction_retry_time=self._cfg.timeout_s)
                        if self.enabled else None)
        self.breaker = CircuitBreaker("graph", failure_threshold=3, recovery_timeout_s=30)

    async def close(self) -> None:
        if self._driver:
            await self._driver.close()

    async def run(self, query: str, params: Optional[Dict[str, Any]] = None, write: bool = False) -> List[Dict]:
        if not self.enabled:
            raise AppError(ErrorCode.GRAPH_UNAVAILABLE, detail="graph disabled or NEO4J_PASSWORD missing")
        if not self.breaker.allow():
            raise AppError(ErrorCode.GRAPH_UNAVAILABLE, detail="circuit open")
        started = time.perf_counter()
        try:
            # 注意：execute_query 的未知关键字参数会被当成 Cypher 参数，读超时要放在 Query 对象上
            q = query if write else Query(query, timeout=self._cfg.timeout_s)
            call = self._driver.execute_query(q, params or {}, routing_="w" if write else "r")
            # 读查询加整体时限（连接 + 驱动重试 + 执行）；写只在离线建图时用，不限时
            records, _, _ = await (call if write else asyncio.wait_for(call, self._cfg.timeout_s * 2))
        except (ServiceUnavailable, OSError, asyncio.TimeoutError) as e:
            self.breaker.record_failure()
            raise AppError(ErrorCode.GRAPH_UNAVAILABLE, detail=str(e)[:200]) from e
        except Neo4jError as e:  # 查询本身的错误（语法、约束），不代表服务挂了
            raise AppError(ErrorCode.INTERNAL, "图谱查询失败", detail=str(e)[:300]) from e
        self.breaker.record_success()
        log.info("graph query", extra={"latency_ms": round((time.perf_counter() - started) * 1000),
                                       "rows": len(records)})
        return [r.data() for r in records]

    async def ping(self) -> Dict[str, Any]:
        rows = await self.run("MATCH (n) RETURN count(n) AS nodes")
        return {"nodes": rows[0]["nodes"] if rows else 0}
