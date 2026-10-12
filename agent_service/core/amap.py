"""高德 Web 服务异步客户端：POI 检索、地理编码、路线（驾车/步行/公交）、天气。

- 网络错误/超时按 ErrorCode.UPSTREAM_AMAP_ERROR 重试；
- 业务错误（status != "1"，如 Key 无效、配额用尽）不重试，直接抛出带 infocode 的 AppError；
- 结果进 LRU + TTL 缓存：同一地点、同一路段在多次规划和校验中只查一次。
"""
import asyncio
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import httpx

from config import Settings
from core.errors import AppError, ErrorCode
from core.logging import get_logger
from core.resilience import retry_async

log = get_logger("amap")

# 高德 QPS 超限的 infocode：属于“稍后再试就能好”的错误，需要重试；配额用尽（10003/10044）等则不重试
QPS_INFOCODES = {"10019", "10020", "10021"}

# 直辖市的 adcode 前两位：区县 adcode 的前 4 位与市级 adcode 不同（如北京 110000 / 东城 110101）
MUNICIPALITY_PREFIXES = {"11", "12", "31", "50"}


class RateLimiter:
    """按固定间隔放行请求（每秒最多 qps 个），避免并行工具调用触发高德 QPS 限制。"""

    def __init__(self, qps: float):
        self.interval = 1.0 / qps if qps > 0 else 0.0
        self._lock = asyncio.Lock()
        self._next = 0.0

    async def wait(self) -> None:
        if not self.interval:
            return
        async with self._lock:
            now = time.monotonic()
            delay = self._next - now
            self._next = max(now, self._next) + self.interval
        if delay > 0:
            await asyncio.sleep(delay)


class TTLCache:
    def __init__(self, maxsize: int, ttl_s: int):
        self.maxsize = maxsize
        self.ttl_s = ttl_s
        self._data: "OrderedDict[Tuple, Tuple[float, Any]]" = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: Tuple) -> Optional[Any]:
        item = self._data.get(key)
        if item is None or (self.ttl_s and time.monotonic() - item[0] > self.ttl_s):
            self._data.pop(key, None)
            self.misses += 1
            return None
        self._data.move_to_end(key)
        self.hits += 1
        return item[1]

    def set(self, key: Tuple, value: Any) -> None:
        if self.maxsize <= 0:
            return
        self._data[key] = (time.monotonic(), value)
        self._data.move_to_end(key)
        while len(self._data) > self.maxsize:
            self._data.popitem(last=False)


def parse_location(text: str) -> Optional[Dict[str, float]]:
    try:
        lng, lat = (float(x) for x in str(text).split(","))
        return {"lng": lng, "lat": lat}
    except (TypeError, ValueError):
        return None


def region_prefix(adcode: str, level: str = "") -> str:
    """用于判断“是否同一城市”的 adcode 前缀：省级/直辖市取 2 位，其余取 4 位。"""
    adcode = str(adcode or "")
    if len(adcode) < 4:
        return adcode
    if level == "省" or adcode[:2] in MUNICIPALITY_PREFIXES:
        return adcode[:2]
    return adcode[:4]


def _str(value: Any) -> str:
    # 高德空字段常返回 []，统一成字符串
    return value if isinstance(value, str) else ""


class AmapClient:
    def __init__(self, settings: Settings):
        self._cfg = settings.amap
        self._key = settings.amap_web_service_key.get_secret_value().strip()
        self._client = httpx.AsyncClient(base_url=self._cfg.base_url, timeout=self._cfg.timeout_s)
        self.cache = TTLCache(self._cfg.cache_size, self._cfg.cache_ttl_s)
        self.limiter = RateLimiter(self._cfg.qps)
        self.calls = 0  # 实际发出的请求数（不含缓存命中），供评测统计

    @property
    def enabled(self) -> bool:
        return bool(self._key)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _get(self, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if not self.enabled:
            raise AppError(ErrorCode.UPSTREAM_AMAP_ERROR, "地图服务未配置")
        key = (path, tuple(sorted((k, str(v)) for k, v in params.items())))
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        async def once() -> Dict[str, Any]:
            await self.limiter.wait()
            started = time.perf_counter()
            try:
                resp = await self._client.get(path, params={**params, "key": self._key})
                self.calls += 1
                data = resp.json()
            except (httpx.HTTPError, ValueError) as e:
                log.warning("amap request failed", extra={"path": path, "error": str(e)[:200]})
                raise AppError(ErrorCode.UPSTREAM_AMAP_ERROR, detail=str(e)) from e
            latency = round((time.perf_counter() - started) * 1000)
            if str(data.get("status")) != "1":
                infocode = str(data.get("infocode"))
                info = f"{data.get('info')}({infocode})"
                log.warning("amap business error", extra={"path": path, "info": info, "latency_ms": latency})
                if infocode in QPS_INFOCODES:
                    raise AppError(ErrorCode.UPSTREAM_AMAP_ERROR, "地图服务繁忙", detail=info)  # 可重试
                # 其他业务错误（Key 无效、配额用尽、参数错误）重试也没用
                raise AppError(ErrorCode.BAD_REQUEST, f"地图服务返回错误：{info}", detail=info)
            log.info("amap call", extra={"path": path, "latency_ms": latency})
            return data

        data = await retry_async(once, attempts=self._cfg.retry_attempts, base_delay_s=0.5, op=f"amap{path}")
        self.cache.set(key, data)
        return data

    # ---------- 地点 ----------
    async def search_poi(self, keyword: str, city: str = "", limit: int = 5) -> List[Dict[str, Any]]:
        data = await self._get(
            "/v3/place/text",
            {"keywords": keyword, "city": city, "citylimit": "true" if city else "false",
             "offset": max(1, min(limit, 20)), "page": 1, "extensions": "all"},
        )
        pois = []
        for p in data.get("pois") or []:
            loc = parse_location(p.get("location"))
            if not loc:
                continue
            biz = p.get("biz_ext") or {}
            pois.append(
                {
                    "poi_id": p.get("id"),
                    "name": p.get("name"),
                    "type": _str(p.get("type")),
                    "address": _str(p.get("address")),
                    "city": _str(p.get("cityname")),
                    "district": _str(p.get("adname")),
                    "adcode": _str(p.get("adcode")),
                    "location": loc,
                    "rating": _str(biz.get("rating")),
                    "cost": _str(biz.get("cost")),
                }
            )
        return pois[:limit]

    async def geocode(self, address: str, city: str = "") -> Optional[Dict[str, Any]]:
        data = await self._get("/v3/geocode/geo", {"address": address, "city": city})
        geos = data.get("geocodes") or []
        if not geos:
            return None
        g = geos[0]
        loc = parse_location(g.get("location"))
        if not loc:
            return None
        return {
            "name": address,
            "formatted_address": _str(g.get("formatted_address")),
            "province": _str(g.get("province")),
            "city": _str(g.get("city")) or _str(g.get("province")),
            "adcode": _str(g.get("adcode")),
            "level": _str(g.get("level")),
            "location": loc,
        }

    # ---------- 路线 ----------
    async def route(self, origin: Dict[str, float], dest: Dict[str, float], mode: str = "driving",
                    city: str = "") -> Optional[Dict[str, Any]]:
        """返回 {mode, duration_min, distance_m}；无可行路线时返回 None。"""
        o = f"{origin['lng']:.6f},{origin['lat']:.6f}"
        d = f"{dest['lng']:.6f},{dest['lat']:.6f}"
        if mode == "transit":
            data = await self._get("/v3/direction/transit/integrated",
                                   {"origin": o, "destination": d, "city": city, "cityd": city})
            options = (data.get("route") or {}).get("transits") or []
        else:
            path = "/v3/direction/walking" if mode == "walking" else "/v3/direction/driving"
            data = await self._get(path, {"origin": o, "destination": d})
            options = (data.get("route") or {}).get("paths") or []
        if not options:
            return None
        best = options[0]
        try:
            return {
                "mode": mode,
                "duration_min": round(int(best.get("duration") or 0) / 60),
                "distance_m": int(float(best.get("distance") or 0)),
            }
        except (TypeError, ValueError):
            return None

    # ---------- 天气 ----------
    async def weather(self, adcode: str) -> List[Dict[str, str]]:
        data = await self._get("/v3/weather/weatherInfo", {"city": adcode, "extensions": "all"})
        forecasts = data.get("forecasts") or []
        casts = forecasts[0].get("casts", []) if forecasts else []
        return [
            {
                "date": c.get("date"),
                "day_weather": c.get("dayweather"),
                "night_weather": c.get("nightweather"),
                "temp": f"{c.get('nighttemp')}~{c.get('daytemp')}°C",
            }
            for c in casts
        ]
