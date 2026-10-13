"""线程安全的 LRU + TTL 缓存，带命中统计。高德请求与检索结果共用。"""
import threading
import time
from collections import OrderedDict
from typing import Any, Hashable, Optional, Tuple


class TTLCache:
    def __init__(self, maxsize: int, ttl_s: float):
        self.maxsize = maxsize
        self.ttl_s = ttl_s
        self._data: "OrderedDict[Hashable, Tuple[float, Any]]" = OrderedDict()
        self._lock = threading.Lock()  # 检索在线程池里执行，可能并发读写
        self.hits = 0
        self.misses = 0

    def get(self, key: Hashable) -> Optional[Any]:
        with self._lock:
            item = self._data.get(key)
            if item is None or (self.ttl_s and time.monotonic() - item[0] > self.ttl_s):
                self._data.pop(key, None)
                self.misses += 1
                return None
            self._data.move_to_end(key)
            self.hits += 1
            return item[1]

    def set(self, key: Hashable, value: Any) -> None:
        if self.maxsize <= 0:
            return
        with self._lock:
            self._data[key] = (time.monotonic(), value)
            self._data.move_to_end(key)
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {"size": len(self._data), "maxsize": self.maxsize, "hits": self.hits, "misses": self.misses,
                "hit_rate": round(self.hits / total, 3) if total else None}
