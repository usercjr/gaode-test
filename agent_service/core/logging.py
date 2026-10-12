"""结构化日志：每行一个 JSON，自动带上 request_id。"""
import contextvars
import json
import logging
import sys
import time
from datetime import datetime, timezone
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Optional

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")

# LogRecord 自带的属性，其余通过 extra= 传入的字段都会输出到 JSON
_RESERVED = set(vars(logging.LogRecord("", 0, "", 0, "", None, None))) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname.lower(),
            "logger": record.name,
            "request_id": request_id_var.get(),
            "msg": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


class SafeTimedRotatingFileHandler(TimedRotatingFileHandler):
    """Windows 上文件被其他进程打开时 rename 会失败（WinError 32）。
    标准实现会在之后每条日志都报错；这里改为本次跳过轮转、继续写当前文件，到下一个周期再试。"""

    def doRollover(self) -> None:
        try:
            super().doRollover()
        except PermissionError:
            if self.stream is None:
                self.stream = self._open()
            self.rolloverAt = self.computeRollover(int(time.time()))


def setup_logging(level: str = "INFO", log_dir: Optional[Path] = None, retention_days: int = 14,
                  name: str = "agent") -> None:
    """name 决定日志文件名（{name}.log / {name}.error.log）：服务与离线脚本各用各的文件，
    避免多个进程轮转同一个文件。"""
    formatter = JsonFormatter()
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level.upper())

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)

    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        for filename, file_level in ((f"{name}.log", logging.NOTSET), (f"{name}.error.log", logging.ERROR)):
            handler = SafeTimedRotatingFileHandler(
                log_dir / filename, when="midnight", backupCount=retention_days, encoding="utf-8"
            )
            handler.setLevel(file_level)
            handler.setFormatter(formatter)
            root.addHandler(handler)

    # 第三方库的访问日志由我们自己的中间件输出，避免重复
    logging.getLogger("uvicorn.access").disabled = True
    for noisy in ("httpx", "httpcore", "chromadb", "sentence_transformers", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
