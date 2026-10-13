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


class AmapConfig(BaseModel):
    base_url: str = "https://restapi.amap.com"
    timeout_s: float = Field(5.0, gt=0)
    retry_attempts: int = Field(3, ge=1, le=5)
    qps: float = Field(3.0, ge=0)  # 客户端限速；个人开发者 Key 的 QPS 配额较低，0 表示不限
    cache_size: int = Field(5000, ge=0)  # 进程内 LRU 条数；0 关闭
    cache_ttl_s: int = Field(86400, ge=0)


class AgentConfig(BaseModel):
    max_steps: int = Field(12, ge=2, le=40)  # 单次规划最多的 LLM 轮数（含工具调用轮）
    max_fix_rounds: int = Field(2, ge=0, le=5)  # 校验不通过后允许 LLM 修正的次数
    max_leg_minutes: int = Field(60, ge=5)  # 相邻两站交通时长上限
    max_day_minutes: int = Field(720, ge=60)  # 每日停留 + 交通总时长上限
    default_days: int = Field(3, ge=1, le=10)
    notes_top_k: int = Field(4, ge=1, le=10)
    temperature: float = Field(0.4, ge=0.0, le=1.5)


class GraphConfig(BaseModel):
    enabled: bool = True
    uri: str = "bolt://127.0.0.1:7687"
    user: str = "neo4j"
    timeout_s: float = Field(3.0, gt=0)
    near_radius_m: int = Field(1500, ge=100, le=10000)  # 建图时同城两点距离小于该值才连 NEAR 边


class MemoryConfig(BaseModel):
    db_path: Path = Path("data/memory.sqlite3")  # 长期记忆与会话记录（不入库）
    enabled: bool = True


class ClassifierConfig(BaseModel):
    models_dir: Path = Path("agent_service/models")
    style_model: str = "style_chinese-roberta-wwm-ext"  # 目录不存在时自动退回 LLM 零样本
    route_model: str = "route_chinese-roberta-wwm-ext"
    # 本地路由最高概率低于该值时交给 LLM。在人工编写的 dev 集上选定（eval/route_threshold_report.md）；
    # 设为 1.01 即全部交给 LLM，设为 0 即全部用本地模型
    route_confidence: float = Field(0.97, ge=0.0, le=1.01)


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
    amap_web_service_key: SecretStr = Field(SecretStr(""), validation_alias=AliasChoices("AMAP_WEB_SERVICE_KEY"))
    llm: LLMConfig = LLMConfig()
    rag: RAGConfig = RAGConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    amap: AmapConfig = AmapConfig()
    agent: AgentConfig = AgentConfig()
    classifier: ClassifierConfig = ClassifierConfig()
    memory: MemoryConfig = MemoryConfig()
    graph: GraphConfig = GraphConfig()
    neo4j_password: SecretStr = Field(SecretStr(""), validation_alias=AliasChoices("NEO4J_PASSWORD"))
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
            (self.classifier, "models_dir"),
            (self.memory, "db_path"),
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

    @property
    def amap_enabled(self) -> bool:
        return bool(self.amap_web_service_key.get_secret_value().strip())

    def public_view(self) -> dict:
        """可安全输出到日志/接口的配置（不含密钥）。"""
        data = self.model_dump(mode="json", exclude={"llm_api_key", "amap_web_service_key", "neo4j_password"})
        data["llm_enabled"] = self.llm_enabled
        data["amap_enabled"] = self.amap_enabled
        return data


@lru_cache
def get_settings() -> Settings:
    return Settings()
