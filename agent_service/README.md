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
│  ├─ llm.py            # 异步 LLM 客户端：超时/重试/熔断/并发闸门/调用日志，支持 tools
│  └─ amap.py           # 异步高德客户端：客户端限速（QPS）、QPS 超限重试、LRU+TTL 缓存
├─ rag/
│  ├─ chunking.py       # 清洗与递归分块
│  ├─ store.py          # embedding + ChromaDB（同步，经线程池调用）
│  ├─ ingest.py         # CSV 读取（内容哈希作为笔记 ID、去重、跳过混入的表头）
│  ├─ enrich.py         # LLM 抽取笔记元数据（城市/省份/景点/人群），缓存到 data/notes_metadata.jsonl
│  ├─ metadata.py       # 元数据读取与查询地名识别
│  ├─ bm25.py           # jieba + BM25 关键词检索、RRF 融合
│  ├─ rerank.py         # bge-reranker 交叉编码器
│  ├─ retriever.py      # 地名过滤 → 向量/BM25 → RRF → 重排
│  └─ pipeline.py       # 意图改写 → 混合检索 → 引用回答（逐级降级）
├─ agent/
│  ├─ orchestrator.py   # Function Calling 循环 + 校验自纠
│  ├─ router.py         # 动态路由：聊天消息按意图分发到规划 / RAG / 工具问答 / 闲聊
│  ├─ qa.py             # 轻量工具问答（周边、顺路、天气、交通时长），不产出行程
│  ├─ tools.py          # 工具定义与执行（笔记检索、POI、路线、天气、提交行程）
│  ├─ validator.py      # 规则校验：POI 真实性、城市一致、路段/每日时长，路线时长用高德实算覆盖
│  ├─ schemas.py        # 结构化行程（Pydantic）
│  ├─ prompts.py        # 系统提示词
│  └─ render.py         # 结构化行程 → 聊天文本
├─ memory/
│  ├─ store.py          # SQLite：长期记忆（固定键、覆盖/去重/待确认规则）、会话轮次、会话摘要
│  ├─ extractor.py      # LLM 提候选 + 确定性过滤（证据须出自原话、敏感信息丢弃）
│  ├─ session.py        # 最近几轮 + 滚动摘要；指代追问改写成独立问题
│  └─ service.py        # 给路由与规划用的门面
├─ classifier/
│  ├─ labels.py         # 标签体系 + 统一输入格式（训练/推理/评测共用）
│  ├─ model.py          # 编码器 + 线性头；Predictor 推理
│  ├─ service.py        # 在线分类：本地模型优先，缺失时退回 LLM；路由只对聊天原话
│  ├─ llm_zero_shot.py  # LLM 零样本分类（兜底）
│  ├─ data/             # 生成的训练/测试集、人工盲审表
│  ├─ results/          # 各方法评测结果 JSON
│  └─ train/            # gen_data / train / eval_llm / reeval / review / report
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

### 元数据（用于地名过滤与 BM25 词典）

增量执行：已抽取过的笔记会跳过；新增笔记后重跑即可。约每篇 1 次 LLM 调用。

```powershell
.\.venv-win\Scripts\python.exe -m rag.enrich --csv ..\data\travel_notes.csv
```

## 检索流程

```text
查询 ──► 地名识别（元数据词典最长匹配）──► 限定到相关笔记（结果不足时自动放宽）
     ──► 向量召回 top-30  ┐
     ──► BM25 召回 top-30 ┴► RRF 融合 ──► bge-reranker 重排前 20 ──► top-k
```

各环节都可在 `config.yaml` 的 `retrieval` 段开关；不同组合的效果与延迟见 `eval/rag_report.md`。
行程规划时 `ai-chat.html` 调 `/retrieve` 传 `use_rerank: false`，跳过重排以控制延迟。

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
| POST | `/plan` | Agent 规划 `{destination, days, preferences, style_prompt, message, previous_itinerary}` → 结构化行程 + 文本 + 校验统计 |
| POST | `/plan/stream` | 同上，SSE 推送 `status` / `tool_call` / `validation` 进度，最后 `done` 或 `error` |
| POST | `/chat` | 聊天消息：先路由（plan / notes_qa / graph_qa / realtime / chitchat）再处理 `{message, destination, previous_itinerary, ...}` |
| POST | `/chat/stream` | 同上，SSE 推送 `route` / `tool_call` / `status` 事件 |
| GET | `/memory?user_id=` | 查看长期记忆（生效 + 待确认） |
| DELETE | `/memory/{id}?user_id=` / `/memory?user_id=` | 删除一条 / 清空 |
| POST | `/itinerary` | 文本行程生成（旧流程，Agent 不可用时前端回退使用）`{messages, temperature}` → `{text, model}` |
| POST | `/ingest/csv` | 入库（仅本机，`csv_path` 必须位于 `data/` 内；不经网关暴露） |

成功响应中的 `degraded` 列出被跳过的步骤（如 `query_rewrite`、`llm_rerank`、`answer`、`classify`）。
失败响应统一为：

```json
{"error": {"code": "UPSTREAM_LLM_TIMEOUT", "message": "AI 生成超时，请稍后重试", "retryable": true, "request_id": "..."}}
```

错误码定义见 `core/errors.py`。

## Agent 规划

```text
用户需求 → 地理编码目的地
        → LLM（tools）循环：search_notes / poi_search / route_plan / get_weather（同一轮可并行）
        → submit_itinerary → 校验器
              ├─ 通过 → 用高德实算的路线时长回填 → 返回
              └─ 不通过 → 问题列表作为工具结果回填 → LLM 修正（最多 2 次，仍不通过则带 warnings 返回）
```

校验规则：结构合法；天数一致；每个地点的 `poi_id` 必须来自本次 `poi_search`（防止编造坐标）；
地点位于目的地（adcode 前缀，直辖市/省按 2 位）；不重复；相邻两站公交/地铁 ≤ 60 分钟；每日停留 + 交通 ≤ 12 小时。
多轮修改时把上一版行程作为 `previous_itinerary` 传入，其中的地点视为已查询过。

评测：`eval/run_agent_eval.py`，报告 `eval/agent_report.md`。

## 动态路由

| 路由 | 处理 | 工具 |
|---|---|---|
| plan | PlanAgent，在上一版行程上修改 | 规划工具集 |
| notes_qa | RAG 问答（混合检索 + 重排 + 引用回答），检索词自动补上行程城市 | — |
| graph_qa | 工具问答（最多 4 轮），优先查知识图谱 | graph_query / poi_search / nearby_search / route_plan / search_notes |
| realtime | 工具问答（最多 4 轮） | get_weather / poi_search / route_plan |
| chitchat | 直接简短回复 | — |

路由由本地 RoBERTa 路由模型判定，模型缺失时退回 LLM 零样本。评测：`eval/run_router_eval.py`，报告 `eval/router_report.md`。

## 记忆

| 层 | 内容 | 存储 |
|---|---|---|
| 长期记忆 | 同行人、行动限制、预算、节奏、常住城市（单值）；饮食、喜欢、不喜欢（多值） | SQLite `memories` |
| 会话摘要 | 未摘要轮数超过 10 时，把除最近 6 轮外的内容压缩成固定字段摘要（后台执行） | SQLite `summaries` |
| 最近对话 | 最近 6 轮原文 | SQLite `turns` |

写入规则：只记用户明确说出的、可复用的、稳定的信息；证据必须是原话子串；敏感信息丢弃；
置信度 < 0.7 先进“待确认”，再次提到才生效；单值键新值覆盖旧值（低置信度不能推翻已确认值）；“不喜欢 X”覆盖“喜欢 X”。
读取：规划与工具问答时注入生效的长期记忆；含指代或很短的追问先结合上下文改写成独立问题再路由。
记忆按前端生成的匿名 `user_id` 隔离（没有登录体系，仅适合演示）。评测：`eval/run_memory_eval.py`。

## 知识图谱（可选）

```powershell
cd D:\gaode_test
docker compose up -d neo4j                     # 需要 .env 里的 NEO4J_PASSWORD；只绑定 127.0.0.1:7474/7687
cd agent_service
.\.venv-win\Scripts\python.exe -m graph.align           # 地名 → 高德 POI 对齐（结果缓存在 data/graph/poi_alignment.jsonl）
.\.venv-win\Scripts\python.exe -m graph.build --reset   # 建图，统计写到 data/graph/build_stats.json
```

节点 POI / City / Tag / Note，边 LOCATED_IN / HAS_TAG / MENTIONS / CO_MENTIONED / NEAR。
`graph_query` 工具只能选 4 个参数化模板（nearby、co_visit、by_tag、evidence），不让 LLM 写 Cypher。
图谱不可用时：读超时（`graph.timeout_s`，整体不超过 2 倍）→ 连续 3 次失败熔断 30 秒 → 工具返回降级提示，模型改用 nearby_search；
`/readyz` 中 `graph.available=false`，但不影响就绪。评测：`eval/run_graph_eval.py` → `eval/graph_report.md`，对齐抽检见 `eval/graph_alignment_review.md`。

## 意图分类

```powershell
cd D:\gaode_test\agent_service
.\.venv-win\Scripts\python.exe -m classifier.train.gen_data                       # 生成数据（约 300 次 LLM 调用）
.\.venv-win\Scripts\python.exe -m classifier.train.train --task style --method finetune --base hfl/chinese-roberta-wwm-ext
.\.venv-win\Scripts\python.exe -m classifier.train.train --task route --method finetune --base hfl/chinese-roberta-wwm-ext
.\.venv-win\Scripts\python.exe -m classifier.train.eval_llm --task style          # LLM 零样本对照
.\.venv-win\Scripts\python.exe -m classifier.train.report                         # 汇总到 eval/classifier_report.md
```

模型写到 `agent_service/models/{task}_{模型名}/`，`config.yaml` 的 `classifier` 段指定线上使用哪个。

## 6) 测试

```powershell
cd D:\gaode_test\agent_service
.\.venv-win\Scripts\python.exe -m pytest -q tests
```
