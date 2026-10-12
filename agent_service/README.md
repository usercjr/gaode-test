# agent_service

旅行助手的 Python 服务（FastAPI, :8002）：RAG 检索问答、风格分类、行程生成。后续 Agent 编排、知识图谱、记忆模块都在这里扩展（见 `docs/upgrade_plan.md`）。

## 目录

```text
agent_service/
├─ app.py               # 入口：中间件（request-id/访问日志）、统一错误处理、探针、业务接口
├─ config.py            # 配置：config.yaml + .env + 环境变量，启动时校验
├─ ingest_csv.py        # 命令行入库
├─ core/
│  ├─ errors.py         # 错误码与统一错误响应
│  ├─ logging.py        # JSON 结构化日志（按天滚动，错误单独成文件）
│  ├─ resilience.py     # 熔断器、指数退避重试
│  └─ llm.py            # 异步 LLM 客户端：超时/重试/熔断/并发闸门/调用日志
├─ rag/
│  ├─ chunking.py       # 清洗与递归分块
│  ├─ store.py          # embedding + ChromaDB（同步，经线程池调用）
│  ├─ ingest.py         # CSV 读取
│  └─ pipeline.py       # 意图改写 → 召回 → LLM 重排 → 引用回答（逐级降级）
├─ classifier/
│  └─ llm_zero_shot.py  # 风格分类（LLM 零样本；本地微调模型上线后作兜底）
└─ tests/               # pytest 单元测试
```

## 1) 环境

Windows PowerShell（Python 3.10+）：

```powershell
cd D:\gaode_test\agent_service
py -3.13 -m venv .venv-win
.\.venv-win\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r requirements-dev.txt   # 跑测试时需要
```

> `agent_service/.venv` 是 Linux（WSL）环境，Windows 下不可用。

## 2) 配置

| 位置 | 放什么 |
|---|---|
| 项目根目录 `.env` | 密钥：`LLM_API_KEY` |
| 项目根目录 `config.yaml` | 非敏感配置：LLM 地址/模型/超时/重试/熔断、向量库、日志、端口 |
| 环境变量 | 临时覆盖，嵌套字段用双下划线，如 `LLM__MODEL=deepseek-chat` |

优先级：环境变量 > `.env` > `config.yaml` > 代码默认值。相对路径一律相对项目根目录。
配置非法（类型错误、端口越界等）时服务启动即失败并给出具体字段。

## 3) 入库

CSV 至少包含 `title`、`text` 两列；重复导入按稳定 ID 覆盖，不会产生重复；混入的重复表头行会被跳过。

```powershell
cd D:\gaode_test\agent_service
.\.venv-win\Scripts\python.exe ingest_csv.py --csv ..\data\travel_notes.csv --reset
```

## 4) 启动

```powershell
cd D:\gaode_test\agent_service
.\.venv-win\Scripts\python.exe app.py
```

启动时预加载 embedding 模型；日志写到项目根目录 `logs/agent.log`（错误另写 `agent.error.log`），每行一个 JSON，带 `request_id`。

## 5) 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/healthz` | 存活探针 |
| GET | `/readyz` | 就绪探针：向量库、LLM 配置与熔断状态；不就绪返回 503 |
| POST | `/retrieve` | 轻量检索 `{query, top_k, style_label, use_rerank}` |
| POST | `/search` | 完整 RAG 问答 `{query, top_k, rerank_k}` |
| POST | `/classify` | 风格分类 `{text}` → `{label, score, source}` |
| POST | `/itinerary` | 行程生成 `{messages, temperature}` → `{text, model}` |
| POST | `/ingest/csv` | 入库（仅本机，`csv_path` 必须位于 `data/` 内；不经网关暴露） |

成功响应中的 `degraded` 列出被跳过的步骤（如 `query_rewrite`、`llm_rerank`、`answer`、`classify`）。
失败响应统一为：

```json
{"error": {"code": "UPSTREAM_LLM_TIMEOUT", "message": "AI 生成超时，请稍后重试", "retryable": true, "request_id": "..."}}
```

错误码定义见 `core/errors.py`。

## 6) 测试

```powershell
cd D:\gaode_test\agent_service
.\.venv-win\Scripts\python.exe -m pytest -q tests
```
