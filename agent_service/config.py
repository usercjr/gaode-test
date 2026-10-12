"""配置外置化。

优先级：环境变量 > 根目录 .env > config.yaml > 代码默认值。
- 密钥只放 .env（LLM_API_KEY）。
- 非敏感配置放 config.yaml；需要临时覆盖时用双下划线环境变量，如 LLM__MODEL=deepseek-chat。
"""
from functools import lru_cache
from pathlib import Path
from typing import Literal, Tuple, Type

from pydantic import AliasChoices, BaseModel, Field, SecretStr, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

SERVICE_DIR = Path(__file__).resolve().parent
ROOT_DIR = SERVICE_DIR.parent


class LLMConfig(BaseModel):
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-chat"
    timeout_s: float = Field(60.0, gt=0)
    max_concurrency: int = Field(8, ge=1)
    retry_attempts: int = Field(3, ge=1, le=6)
    breaker_failure_threshold: int = Field(5, ge=1)
    breaker_recovery_s: float = Field(30.0, gt=0)

    @field_validator("base_url")
    @classmethod
    def _strip_slash(cls, v: str) -> str:
        return v.rstrip("/")


class RAGConfig(BaseModel):
    db_dir: Path = Path("agent_service/chroma_db")
    collection: str = "travel_notes"
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    llm_model: str = ""  # 为空时沿用 llm.model
    data_dir: Path = Path("data")  # 入库只允许读取该目录下的文件


class RetrievalConfig(BaseModel):
    mode: Literal["vector", "bm25", "hybrid"] = "hybrid"
    candidate_k: int = Field(30, ge=5, le=200)  # 每一路召回的候选数
    rerank_k: int = Field(20, ge=1, le=100)  # 送入交叉编码器的候选数（CPU 上约 140ms/条，决定重排延迟）
    rrf_k: int = Field(60, ge=1)
    rerank: Literal["cross_encoder", "llm", "none"] = "cross_encoder"
    reranker_model: str = "BAAI/bge-reranker-base"
    reranker_max_length: int = Field(256, ge=64, le=1024)
    city_filter: bool = True  # 查询中识别到城市/省份时，只在相关笔记中检索
    metadata_path: Path = Path("data/notes_metadata.jsonl")


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = Field(8002, ge=1, le=65535)


class LoggingConfig(BaseModel):
    level: str = "INFO"
    dir: Path = Path("logs")
    retention_days: int = Field(14, ge=1)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        yaml_file=ROOT_DIR / "config.yaml",
        yaml_file_encoding="utf-8",
        extra="ignore",
    )

    llm_api_key: SecretStr = Field(
        SecretStr(""), validation_alias=AliasChoices("LLM_API_KEY", "DASHSCOPE_API_KEY")
    )
    llm: LLMConfig = LLMConfig()
    rag: RAGConfig = RAGConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    server: ServerConfig = ServerConfig()
    logging: LoggingConfig = LoggingConfig()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls),
            file_secret_settings,
        )

    @model_validator(mode="after")
    def _resolve(self) -> "Settings":
        # 相对路径一律相对项目根目录，避免因启动目录不同而读写到别处
        for section, attr in (
            (self.rag, "db_dir"),
            (self.rag, "data_dir"),
            (self.retrieval, "metadata_path"),
            (self.logging, "dir"),
        ):
            path = getattr(section, attr)
            if not path.is_absolute():
                setattr(section, attr, (ROOT_DIR / path).resolve())
        if not self.rag.llm_model:
            self.rag.llm_model = self.llm.model
        return self

    @property
    def llm_enabled(self) -> bool:
        return bool(self.llm_api_key.get_secret_value().strip())

    def public_view(self) -> dict:
        """可安全输出到日志/接口的配置（不含密钥）。"""
        data = self.model_dump(mode="json", exclude={"llm_api_key"})
        data["llm_enabled"] = self.llm_enabled
        return data


@lru_cache
def get_settings() -> Settings:
    return Settings()
