"""记忆存储（SQLite）：长期记忆 + 会话轮次 + 会话摘要。

长期记忆按固定键存储，写入规则确定、可测试：
- 单值键（companion / mobility / budget / pace / home_city）：新值覆盖旧值，旧值标记 superseded 保留历史
- 多值键（diet / likes / dislikes）：同键同值去重；“dislikes X”会覆盖已有的“likes X”，反之亦然
- 置信度低的记忆先进入 pending，同一条再次出现才转为 active；只有 active 会被注入提示词
"""
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

SINGLE_KEYS = {"companion", "mobility", "budget", "pace", "home_city"}
MULTI_KEYS = {"diet", "likes", "dislikes"}
ALL_KEYS = SINGLE_KEYS | MULTI_KEYS
OPPOSITE = {"likes": "dislikes", "dislikes": "likes"}
KEY_CN = {"companion": "同行人", "mobility": "体力/行动", "budget": "预算", "pace": "节奏", "home_city": "常住城市",
          "diet": "饮食", "likes": "喜欢", "dislikes": "不喜欢/避免"}
CONFIRM_THRESHOLD = 0.7

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL, key TEXT NOT NULL, value TEXT NOT NULL,
    confidence REAL NOT NULL, status TEXT NOT NULL,          -- active | pending | superseded | deleted
    evidence TEXT, source TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_mem_user ON memories(user_id, status);
CREATE TABLE IF NOT EXISTS turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL, user_id TEXT, role TEXT NOT NULL, content TEXT NOT NULL,
    route TEXT, created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_turn_session ON turns(session_id, id);
CREATE TABLE IF NOT EXISTS summaries (
    session_id TEXT PRIMARY KEY, summary TEXT NOT NULL, covered_until INTEGER NOT NULL, updated_at TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _norm(value: str) -> str:
    return "".join(value.split()).lower()


# 喜欢/不喜欢类记忆比较“对象”本身：LLM 常把动词也写进 value（“喜欢逛博物馆” vs “不喜欢博物馆”）
_SENTIMENT = ("不喜欢", "不太喜欢", "讨厌", "不想去", "不爱", "喜欢", "爱", "想去", "逛", "去", "看", "吃")


def _core(value: str) -> str:
    v = _norm(value)
    changed = True
    while changed:
        changed = False
        for w in _SENTIMENT:
            if v.startswith(w) and len(v) > len(w):
                v, changed = v[len(w):], True
    return v


def _same_object(a: str, b: str) -> bool:
    ca, cb = _core(a), _core(b)
    return ca == cb or (min(len(ca), len(cb)) >= 2 and (ca in cb or cb in ca))


class MemoryStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # ---------------- 长期记忆 ----------------
    def upsert(self, user_id: str, key: str, value: str, confidence: float, evidence: str = "",
               source: str = "") -> Optional[Dict]:
        """写入一条候选记忆，返回变化（{action, memory}）；无变化返回 None。"""
        if key not in ALL_KEYS or not value.strip():
            return None
        value = value.strip()[:40]
        with self._lock:
            cur = self._conn.cursor()
            live = cur.execute(
                "SELECT * FROM memories WHERE user_id=? AND key=? AND status IN ('active','pending')",
                (user_id, key)).fetchall()
            match = _same_object if key in OPPOSITE else (lambda x, y: _norm(x) == _norm(y))
            same = next((r for r in live if match(r["value"], value)), None)
            now = _now()

            if same is not None:
                # 再次出现：pending 转 active（重复提及即确认），active 刷新置信度与时间
                status = "active" if same["status"] == "pending" or confidence >= CONFIRM_THRESHOLD else same["status"]
                cur.execute("UPDATE memories SET status=?, confidence=MAX(confidence, ?), updated_at=? WHERE id=?",
                            (status, confidence, now, same["id"]))
                self._conn.commit()
                action = "confirmed" if same["status"] == "pending" and status == "active" else None
                return {"action": action, "memory": self._get(same["id"])} if action else None

            status = "active" if confidence >= CONFIRM_THRESHOLD else "pending"
            superseded = []
            if key in SINGLE_KEYS:
                # 低置信度的新说法不能推翻已确认的旧值，只替换同样未确认的
                superseded = [r["id"] for r in live if status == "active" or r["status"] == "pending"]
            elif key in OPPOSITE:  # “不喜欢 X”覆盖“喜欢 X”
                opposite = cur.execute(
                    "SELECT id, value FROM memories WHERE user_id=? AND key=? AND status IN ('active','pending')",
                    (user_id, OPPOSITE[key])).fetchall()
                superseded = [r["id"] for r in opposite if _same_object(r["value"], value)]
            for mid in superseded:
                cur.execute("UPDATE memories SET status='superseded', updated_at=? WHERE id=?", (now, mid))

            cur.execute(
                "INSERT INTO memories(user_id,key,value,confidence,status,evidence,source,created_at,updated_at) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (user_id, key, value, confidence, status, evidence[:200], source[:60], now, now))
            self._conn.commit()
            return {"action": "updated" if superseded else ("added" if status == "active" else "pending"),
                    "memory": self._get(cur.lastrowid), "superseded": superseded}

    def _get(self, mid: int) -> Dict:
        row = self._conn.execute("SELECT * FROM memories WHERE id=?", (mid,)).fetchone()
        return dict(row) if row else {}

    def list(self, user_id: str, statuses=("active", "pending")) -> List[Dict]:
        marks = ",".join("?" * len(statuses))
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM memories WHERE user_id=? AND status IN ({marks}) ORDER BY key, updated_at DESC",
                (user_id, *statuses)).fetchall()
        return [dict(r) for r in rows]

    def delete(self, user_id: str, memory_id: Optional[int] = None) -> int:
        """软删除（标记 deleted）。memory_id 为空时清空该用户全部记忆。"""
        with self._lock:
            if memory_id is None:
                cur = self._conn.execute(
                    "UPDATE memories SET status='deleted', updated_at=? WHERE user_id=? AND status!='deleted'",
                    (_now(), user_id))
            else:
                cur = self._conn.execute(
                    "UPDATE memories SET status='deleted', updated_at=? WHERE user_id=? AND id=? AND status!='deleted'",
                    (_now(), user_id, memory_id))
            self._conn.commit()
            return cur.rowcount

    def prompt_lines(self, user_id: str) -> List[str]:
        """注入提示词的记忆：只用 active。"""
        return [f"{KEY_CN[m['key']]}：{m['value']}" for m in self.list(user_id, ("active",))]

    # ---------------- 会话轮次与摘要 ----------------
    def add_turn(self, session_id: str, user_id: str, role: str, content: str, route: str = "") -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO turns(session_id,user_id,role,content,route,created_at) VALUES (?,?,?,?,?,?)",
                (session_id, user_id, role, content[:4000], route, _now()))
            self._conn.commit()
            return cur.lastrowid

    def turns(self, session_id: str, after_id: int = 0) -> List[Dict]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM turns WHERE session_id=? AND id>? ORDER BY id",
                                      (session_id, after_id)).fetchall()
        return [dict(r) for r in rows]

    def summary(self, session_id: str) -> Dict:
        with self._lock:
            row = self._conn.execute("SELECT * FROM summaries WHERE session_id=?", (session_id,)).fetchone()
        return dict(row) if row else {"summary": "", "covered_until": 0}

    def save_summary(self, session_id: str, summary: str, covered_until: int) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO summaries(session_id,summary,covered_until,updated_at) VALUES (?,?,?,?) "
                "ON CONFLICT(session_id) DO UPDATE SET summary=excluded.summary, "
                "covered_until=excluded.covered_until, updated_at=excluded.updated_at",
                (session_id, summary, covered_until, _now()))
            self._conn.commit()
