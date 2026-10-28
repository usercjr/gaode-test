# 行界 WanderAI · 升级规划与详细设计文档

> 版本：v1.0　｜　日期：2026-10-12　｜　适用代码基线：`main` 工作区（DeepSeek + RAG 服务 + 高德代理版本）
>
> 本文档目标：把简历中描述的能力与后续需求（知识图谱 RAG、动态路由、记忆、容错/并发/日志/监控等）**全部真实落地**，并为简历里的【待填真实数值】给出可复现的测量方法。
> 当前实现状态见 `project_documentation.md`，本文档只写“要做什么、怎么做、怎么验收”。

---

## 0. 先对齐：简历描述 vs 代码现状

写进简历的每一句都要能在代码和评测数据里找到证据。下表是逐条核对的结果，**“现状”列是面试时被追问会暴露的地方**。

| 简历条目 | 代码现状 | 差距 | 对应章节 |
|---|---|---|---|
| ERNIE-3.5 / PaddleNLP | 现已切换为 DeepSeek（OpenAI 兼容），PaddleNLP 分类服务已删除 | 黑客松版本属实；当前版本需改写为“模型可切换” | §3.1、§6 |
| Function Calling / Agent 工具编排 | **未使用 Function Calling**。高德调用由地图页 JS 按固定流程执行，LLM 只输出文本 | 需要实现真正的 LLM 工具调用循环 | §4 |
| 工具结果回填上下文形成闭环 | 不存在：LLM 看不到地理编码/路线结果 | 同上 | §4 |
| 校验纠错 / 自纠 | 只有“三级地理编码降级”（POI 同城 → 全国 → geocode），没有把错误反馈给 LLM 重新生成 | 需要校验器 + 反馈重试 | §4.4 |
| 语义分块 | 实际是“按标点的递归字符分块”（400 字 / 80 重叠） | 要么实现语义分块，要么改措辞 | §5.2 |
| MD5 去重、增量入库 | MD5 生成稳定 ID + upsert，已具备；缺“文档级哈希跳过未变更文档” | 小改 | §5.2 |
| 召回相关性【待填】 | 没有评测集，无法给出数字 | 需要评测集 + 脚本 | §5.6 |
| 8 分类 SFT 微调，准确率【待填】 | 旧模型训练数据（游记长文）与推理输入（短表单串）分布不一致，效果差，已弃用 | 重新构建数据并微调 | §6 |
| 滑动窗口多轮上下文 | 已有（前端保留最近 30 条） | 无摘要、无长期记忆 | §9 |

问卷里的勾选也要对应（截图中的 RAG 题）：

- **混合检索**：当前**未实现**（纯向量召回）。完成 §5.3 之后再勾。
- **Reranker 重排**、**Query 改写**：当前**已实现**（基于 LLM 的重排与改写，`rag_core.py` 的 `llm_rerank` / `intent_understand`），可以勾选，措辞写“LLM 重排”。§5.4 会再加交叉编码器 Reranker。
- 训练深度题：目前只有 SFT（PaddleNLP）。§6.5 的 LoRA 是可选项，**真跑通并有结果后**再写。

---

## 1. 目标与范围

### 1.1 功能目标

1. **Agent**：LLM 通过 Function Calling 自主调用检索、高德地点/路线/天气等工具，输出结构化行程，经校验器检查后自动纠错。
2. **RAG 2.0**：混合检索（BM25 + 向量）+ 交叉编码器重排 + 元数据过滤 + 评测体系。
3. **知识图谱增强 RAG**：Neo4j 存储城市—景点—餐厅—标签—笔记关系，支持“附近”“顺路”“多跳”类问题。
4. **动态路由 RAG**：不同问题走不同知识库、不同检索策略（笔记 / 图谱 / 实时工具 / 直接回答）。
5. **意图微调**：重新构建数据，微调风格分类与路由分类模型，与 LLM 零样本做对比。
6. **Memory Agent**：判断什么值得记，长期用户画像 + 对话摘要 + 行程状态。
7. **工程化**：配置外置、结构化日志、错误分类与降级、容错、并发、缓存、监控页面与指标、Docker 部署。

### 1.2 不做的事（控制范围）

- 不做账号体系与支付；用户用匿名 `user_id`（浏览器生成并持久化）区分。
- 不做大规模分布式部署；“高并发”目标定义为单机下的异步化、限流、缓存与压测数据（见 §10.4）。
- 不爬取违反平台协议的数据（见 §5.1）。

---

## 2. 目标架构

### 2.1 总览

```text
┌──────────────────────────── 浏览器 ────────────────────────────┐
│ ai-chat.html（对话 + SSE 流式）  output-map.html  admin.html(监控) │
└───────────────┬────────────────────────────────────────────────┘
                │  HTTP / SSE（X-Request-ID）
┌───────────────▼──────────────── Node 网关（Express, :3000） ─────┐
│ 静态托管 · 限流 · request-id · 统一错误格式 · 高德代理 · /metrics │
└───────────────┬────────────────────────────────────────────────┘
                │  HTTP（内网）
┌───────────────▼──────────── agent_service（FastAPI, :8002） ─────┐
│  Orchestrator（Agent 循环 / Function Calling / 校验自纠）          │
│   ├─ Router（动态路由：notes / graph / tools / direct）            │
│   ├─ Retrieval：Hybrid(BM25+向量) → Reranker → 过滤               │
│   ├─ Graph：Neo4j 查询模板                                         │
│   ├─ Tools：高德 POI/地理编码/路线/天气、笔记检索、记忆读写       │
│   ├─ Classifier：风格 + 路由意图（本地微调模型，LLM 兜底）        │
│   ├─ Memory：画像 / 摘要 / 行程状态                               │
│   └─ Infra：配置、日志、错误码、重试、熔断、缓存、指标            │
└──────┬───────────┬──────────────┬──────────────┬───────────────┘
       │           │              │              │
   ChromaDB     Neo4j          Redis          SQLite
  (向量库)     (图谱)     (缓存/限流计数)   (记忆/会话/评测记录)
       外部：DeepSeek API（LLM） · 高德 Web 服务 API
```

### 2.2 关键决策

| 决策 | 选择 | 理由 |
|---|---|---|
| Agent 编排放在哪 | Python（`agent_service`），由现 `rag_service` 演进而来 | 检索、模型、图数据库驱动都在 Python 生态；Node 只做网关，职责清晰 |
| 是否保留 Node | 保留为网关/BFF | 保持“Python / Node.js 微服务”结构；静态托管、限流、密钥隔离放在边缘 |
| Agent 框架 | 自研轻量循环，不引入 LangChain/LangGraph | 代码量小（约 300 行）、可控、面试能讲清每一步；可在文档中说明与 LangGraph 的对应关系 |
| LLM 接入 | OpenAI 兼容接口，`LLM_*` 配置切换 | DeepSeek 支持 `tools` 参数；可切回 ERNIE/Qwen |
| 行程输出 | 结构化 JSON（Schema 校验）+ 文本渲染 | 去掉“文本 → 正则解析 → 地图”这条脆弱链路 |
| 训练框架 | PyTorch + HuggingFace，放弃 Paddle | 与 RAG 共用环境；生态与求职市场更通用 |

### 2.3 目录结构（目标）

```text
gaode_test/
├─ server.js                 # Node 网关
├─ web/                      # ai-chat.html, output-map.html, rag-search.html, admin.html, src/map.js
├─ agent_service/
│  ├─ app.py                 # FastAPI 入口、路由注册、lifespan
│  ├─ config.py              # pydantic-settings，读取 config.yaml + .env
│  ├─ core/
│  │  ├─ llm.py              # LLM 客户端（异步、重试、熔断、token 统计）
│  │  ├─ errors.py           # 错误码定义与异常类
│  │  ├─ logging.py          # 结构化日志、request-id 上下文
│  │  ├─ resilience.py       # 重试/熔断/并发闸门
│  │  └─ cache.py            # Redis/内存缓存
│  ├─ agent/
│  │  ├─ orchestrator.py     # Agent 循环
│  │  ├─ tools.py            # 工具定义（JSON Schema）与执行
│  │  ├─ validator.py        # 行程校验
│  │  └─ schemas.py          # Itinerary 等 Pydantic 模型
│  ├─ rag/
│  │  ├─ chunking.py  ingest.py  bm25.py  vector.py  hybrid.py  rerank.py
│  │  └─ router.py           # 动态路由
│  ├─ graph/
│  │  ├─ build_graph.py      # 抽取 + 实体对齐 + 入库
│  │  └─ queries.py          # Cypher 模板
│  ├─ memory/
│  │  ├─ extractor.py  store.py  summarizer.py
│  ├─ classifier/
│  │  ├─ infer.py            # 在线推理
│  │  └─ train/              # 数据生成、训练、评估脚本
│  └─ tests/
├─ eval/                     # 评测集与评测脚本、报告
├─ config.yaml               # 非敏感配置
├─ .env                      # 密钥（不提交）
└─ docker-compose.yml
```

---

## 3. 统一约定

### 3.1 配置外置化

- **分层**：`config.yaml`（非敏感、可提交）+ `.env`（密钥，不提交）+ 环境变量（部署时覆盖）。优先级：环境变量 > `.env` > `config.yaml` > 代码默认值。
- **Python**：`pydantic-settings` 定义 `Settings` 类，启动时校验（缺 Key、端口冲突、路径不存在直接报错退出，而不是运行时才失败）。
- **Node**：启动时校验必需变量，缺失则打印明确错误并退出。
- 配置示例：

```yaml
llm:
  base_url: https://api.deepseek.com
  model: deepseek-chat
  timeout_s: 60
  max_concurrency: 8
retrieval:
  top_k: 5
  hybrid_weights: {bm25: 0.4, vector: 0.6}   # 或使用 RRF
  reranker: BAAI/bge-reranker-base
  rerank_enabled: true
agent:
  max_steps: 8
  max_fix_rounds: 2
graph:
  enabled: true
  uri: bolt://localhost:7687
memory:
  summary_trigger_tokens: 3000
  keep_recent_turns: 6
```

### 3.2 错误码与统一响应

所有接口失败时返回同一结构，前端只根据 `code` 决定提示语和是否重试：

```json
{
  "error": {
    "code": "UPSTREAM_LLM_TIMEOUT",
    "message": "AI 生成超时，请稍后重试",
    "retryable": true,
    "request_id": "c0a8..."
  },
  "degraded": ["rag_notes"]
}
```

成功但发生降级时，`degraded` 字段列出被跳过的能力，前端据此提示（见 §10.3）。

| code | HTTP | 可重试 | 典型原因 |
|---|---|---|---|
| `BAD_REQUEST` | 400 | 否 | 参数缺失/非法 |
| `RATE_LIMITED` | 429 | 是 | 触发网关限流 |
| `UPSTREAM_LLM_TIMEOUT` | 504 | 是 | LLM 超时 |
| `UPSTREAM_LLM_QUOTA` | 502 | 否 | LLM 欠费/额度耗尽（402） |
| `UPSTREAM_LLM_RATE` | 503 | 是 | LLM 429 |
| `UPSTREAM_AMAP_ERROR` | 502 | 是 | 高德返回非 1 / 配额超限 |
| `RAG_UNAVAILABLE` | 503 | 是 | 向量库/模型不可用 |
| `GRAPH_UNAVAILABLE` | 503 | 是 | Neo4j 不可用 |
| `PLAN_INVALID` | 422 | 否 | 校验自纠后仍不合格 |
| `INTERNAL` | 500 | 否 | 未分类异常（必须记录堆栈） |

### 3.3 request-id 链路

Node 网关生成 `X-Request-ID`（或沿用前端传入的），转发到 Python；Python 用 `contextvars` 把它注入每条日志、每次 LLM/高德调用记录。排查问题时用一个 ID 串起整条链路。

---

## 4. 模块一：Agent 工具编排（Function Calling）

### 4.1 工具清单

| 工具名 | 参数 | 实现 | 说明 |
|---|---|---|---|
| `search_notes` | `query, city?, style?, top_k` | §5 混合检索 | 返回笔记片段与来源 ID |
| `graph_query` | `intent, entity, city, radius_km?` | §7 Cypher 模板 | 附近/顺路/多跳 |
| `poi_search` | `keyword, city` | 高德 `place/text` | 返回 poi_id、坐标、类型、评分 |
| `route_plan` | `origin, destination, mode` | 高德 `direction/*` | mode: driving/walking/transit |
| `get_weather` | `city` | 高德 `weather/weatherInfo` | 实况 + 预报 |
| `get_user_memory` | `user_id` | §9 | 画像与偏好 |

工具用 OpenAI `tools` 格式（JSON Schema）声明，DeepSeek `deepseek-chat` 支持该参数。

### 4.2 Agent 循环

```text
messages = [system(规划规则 + 记忆 + 风格提示), history..., user]
for step in range(max_steps):            # 默认 8
    resp = llm(messages, tools=TOOLS)
    if resp 有 tool_calls:
        并发执行这些工具（各自有超时/重试/缓存）
        把每个结果（成功或结构化错误）作为 role=tool 消息追加
        continue
    else:
        draft = 解析为 Itinerary JSON（response_format=json_object）
        break
issues = validator.check(draft)          # §4.4
for round in range(max_fix_rounds):      # 默认 2
    if not issues: break
    messages += [assistant(draft), user("以下问题请修正：" + issues)]
    draft = llm(...)；issues = validator.check(draft)
return draft, trace
```

要点：

- **工具失败也回填**：返回 `{"ok": false, "error": "POI_NOT_FOUND", "hint": "试试更标准的名称"}`，让 LLM 自己换关键词重试，这就是“工具调用失败自动纠错”。
- **步数与预算上限**：`max_steps`、单次请求 token 上限，防止死循环和费用失控。
- **trace**：记录每一步（工具名、参数、耗时、是否成功），既用于调试页展示，也用于统计 §4.6 的指标。
- **流式**：通过 SSE 向前端推送阶段事件（`classify → retrieve → tool_call(poi_search 外滩) → validate → done`），体验上不再“干等 30 秒”。

### 4.3 结构化行程 Schema

```json
{
  "title": "成都 3 日美食漫游",
  "city": "成都",
  "days": [
    {
      "day": 1,
      "theme": "老成都的烟火气",
      "stops": [
        {
          "name": "宽窄巷子",
          "poi_id": "B001C...",
          "location": {"lng": 104.05, "lat": 30.67},
          "arrive": "09:30", "stay_min": 90,
          "category": "attraction",
          "cost_per_person": 0,
          "tips": ["早上人少", "..."],
          "source_note_ids": ["row_12"],
          "transport_to_next": {"mode": "walking", "duration_min": 12}
        }
      ]
    }
  ],
  "warnings": []
}
```

- 地图页直接消费此 JSON，不再正则解析文本；文本攻略由前端按模板渲染（保留小红书风格的文案字段 `tips`）。
- `source_note_ids` 用于“引用溯源”，点开可看原始笔记片段。

### 4.4 校验器（Validator）

规则检查，不依赖 LLM，速度快且结果确定：

| 规则 | 检查方式 | 不通过时反馈给 LLM 的内容 |
|---|---|---|
| Schema 合法 | Pydantic 校验 | 缺失字段列表 |
| 天数一致 | `len(days)` 等于用户要求 | “要求 3 天，实际 2 天” |
| 地点存在 | 每个 stop 都有 `poi_id`/坐标 | “XX 未找到，请调用 poi_search 换名称” |
| 城市一致 | POI 的 adcode/city 属于目标城市 | “XX 位于上海，不在成都” |
| 交通合理 | 相邻点路线时长 ≤ 阈值（如 60 分钟） | “第 2 天 A→B 驾车 85 分钟，建议调整顺序” |
| 日程容量 | 每日停留 + 交通 ≤ 12 小时 | “第 1 天总时长 14 小时” |
| 记忆约束 | 不违反画像（如“老人同行”→ 避免高强度徒步） | “用户有老人同行，XX 需爬 2 小时山” |

### 4.5 降级

- LLM 不支持/调用失败 → 回退到现有“文本生成 + 前端解析”流程，并在 `degraded` 标记 `agent`。
- 高德超限 → 使用 `data/geocode-cache.json` 与 Redis 缓存；仍失败则跳过路线，地图只画点。

### 4.6 指标与测量（填简历数字）

评测集 `eval/plan_cases.jsonl`：50 条规划请求，覆盖 10+ 城市、不同天数/人群/风格。脚本 `eval/run_agent_eval.py` 批量跑并统计：

| 指标 | 定义 | 对应简历 |
|---|---|---|
| 工具调用成功率 | 成功的工具调用数 / 全部工具调用数（含重试后成功） | “工具调用成功率” |
| 首次通过率 / 自纠后通过率 | 首次校验无问题的比例；经自纠后无问题的比例 | “校验纠错”的效果 |
| 地理匹配率 | 有效坐标的景点数 / 全部景点数，对比旧流程（文本解析 + 三级降级）与新流程 | “地理匹配率提升” |
| 端到端耗时 P50/P95 | 从请求到 done | 性能 |

---

## 5. 模块二：RAG 2.0

### 5.1 数据扩充

217 条笔记太少，检索和图谱都撑不起来。目标 1,000–2,000 条，来源：

- 合法公开数据集（如开源中文游记/点评数据集，注意许可证）。
- 景点结构化资料：高德 POI 详情（名称、类型、地址、评分）、各地文旅官网公开信息（开放时间、门票）。
- 自己撰写或 LLM 辅助生成的攻略（需标注 `source=synthetic`，评测时与真实数据分开统计）。

注意：小红书等平台的用户协议通常禁止爬取，不建议批量抓取，也不要写进简历。

### 5.2 入库与分块

- **元数据抽取**：入库时用 LLM 为每篇笔记抽取 `city`、`pois[]`、`styles[]`、`season`、`budget_level`，存入 Chroma metadata，用于过滤与图谱构建。
- **分块**：两种可选——
  - 现有“按标点递归分块”（简历措辞改为“递归分块”）；
  - 语义分块：按句切分 → 计算相邻句向量相似度 → 在相似度骤降处断开（阈值取第 90 百分位落差），再按最大长度合并。实现后简历可写“语义分块”，并在评测中对比两种方式的 Recall。
- **增量**：文档级 `content_hash`，未变化的文档直接跳过；变化的文档先删除旧 chunk 再写入（chunk ID 仍用 MD5）。

### 5.3 混合检索

- **BM25**：`jieba` 分词（加载自定义词典：城市、景点名）+ `rank_bm25`；索引持久化到磁盘，入库时增量更新。
- **向量**：现有 `bge-small-zh-v1.5` + Chroma。可评估升级为 `bge-base-zh-v1.5` 或 `bge-m3`。
- **融合**：RRF（Reciprocal Rank Fusion，`score = Σ 1/(60 + rank)`），无需调权重；另实现加权融合作对比。
- **过滤**：识别出目标城市时，先按 `city` 元数据过滤，再检索。

### 5.4 重排

- 交叉编码器 `BAAI/bge-reranker-base`（sentence-transformers `CrossEncoder`），对 Top-30 候选重排取 Top-5，CPU 下约 200–500ms。
- 现有 LLM 重排保留为可选项，用于对比效果与成本。

### 5.5 Query 改写

保留现有 LLM 改写；新增两点：
- 改写结果中抽取 `city`，用于 §5.3 的过滤；
- 多轮对话中结合记忆与上一轮行程补全指代（“那第二天呢” → “成都第 2 天行程”）。

### 5.6 评测（填“召回相关性”）

- **评测集** `eval/rag_queries.jsonl`：150 条查询，每条人工标注相关笔记 ID（先用 LLM 预标注候选，再人工确认，约 2–3 小时工作量）。
- **指标**：Recall@5、MRR@10、nDCG@5。
- **对比实验**（消融表，可直接放进简历/面试）：

| 配置 | Recall@5 | MRR@10 | nDCG@5 | P95 延迟 |
|---|---|---|---|---|
| 向量 |  |  |  |  |
| BM25 |  |  |  |  |
| 混合（RRF） |  |  |  |  |
| 混合 + bge-reranker |  |  |  |  |
| 混合 + reranker + Query 改写 |  |  |  |  |

---

## 6. 模块三：意图微调（风格分类 + 路由分类）

### 6.1 问题回顾

旧模型用游记长文训练，推理时输入的却是表单短串，分布不一致；每类只有 19–45 条；`Vlog` 与 `拍照` 难区分，`智能综合` 是兜底类。

### 6.2 任务重新定义

| 模型 | 输入 | 输出 |
|---|---|---|
| 风格分类（多标签） | 用户需求原话 + 表单偏好拼接，如“带爸妈去成都，想吃好吃的别太累” | `food, relax, culture, citywalk, photo_vlog, rush` 中的若干个（sigmoid，多标签） |
| 路由分类（单标签） | 用户单句消息 | `plan`（规划/修改行程）、`notes_qa`（经验类问答）、`graph_qa`（附近/顺路/多跳）、`realtime`（天气/路况/营业）、`chitchat` |

标签调整：合并 `Vlog_Creator_Trip` 与 `Photo_Experience` 为 `photo_vlog`；去掉 `Customized_AI_Tour`，改为“所有标签概率都低于阈值时视为综合”。

### 6.3 数据构造

1. **种子反推**：对每篇笔记，让 LLM 生成 3 条“写这篇笔记的人当初可能会提的需求”，标签继承笔记的 `final_label`（经过映射）。
2. **定向生成**：每个标签、每种表达风格（口语/简短/带错别字/多意图混合/否定表达如“不想排队”）各生成若干条。
3. **去重**：用 bge 向量做近重复过滤（余弦 > 0.92 视为重复）。
4. **人工抽检**：抽 10% 校对，记录错误率；**测试集 100–150 条由人工撰写**（或找同学写），不参与训练，避免“LLM 生成、LLM 评测”的自我循环。
5. 规模目标：风格每类 200–300 条，路由每类 150–200 条。

### 6.4 训练方案（逐级）

| 级别 | 方案 | 成本 | 说明 |
|---|---|---|---|
| 基线 | bge-small-zh 句向量 + 逻辑回归（scikit-learn） | CPU 数分钟 | 先确认数据是否可学 |
| 主方案 | 全参微调 `hfl/chinese-roberta-wwm-ext` 或 `bge-small-zh`（HF `Trainer`） | CPU 可跑，GPU 更快 | lr 2e-5、3–5 epoch、max_len 64、early stopping、类别不平衡用加权 loss |
| 部署优化 | 导出 ONNX + 动态量化 | — | 推理 < 10ms |

**评估**：同一测试集上对比 *DeepSeek 零样本* / *基线* / *微调模型* 的 macro-F1、每类 F1、混淆矩阵、单次推理延迟与成本。只有本地模型在准确率接近 LLM（例如差距 < 3 个百分点）且延迟/成本明显更优时，才作为默认路由；否则作为 LLM 的前置快速通道（高置信直接用，低置信交给 LLM）。

### 6.5 可选：LoRA 微调小参数 LLM（真跑通再写进简历）

- 任务：把用户需求解析为结构化槽位 JSON（目的地、天数、预算、同行人、风格、禁忌），供 Agent 使用。
- 模型：Qwen 系列 1.5B 左右的指令模型；框架 `transformers` + `peft`（LoRA r=16）；数据由 §6.3 扩展得到 2–3k 条；需要 GPU（AutoDL/Colab，单卡数小时）。
- 评估：槽位级 F1、JSON 合法率，与 DeepSeek 对比。
- DPO/RLHF 不建议在本项目做：缺少偏好数据与资源，投入产出比低。

### 6.6 上线

`agent_service/classifier/infer.py` 启动时加载模型；`/classify` 返回 `{labels:[{label,score}], route, source:"local"|"llm"}`。本地模型异常时自动退回 LLM 零样本（已有实现）。

---

## 7. 模块四：知识图谱增强 RAG（Neo4j）

### 7.1 图模型

```text
(:City {name, adcode})
(:POI {poi_id, name, type, lng, lat, rating})          # 景点/餐厅/街区，用高德 poi_id 做主键
(:Tag {name})                                          # 美食、拍照、夜景、亲子…
(:Note {note_id, title})

(:POI)-[:LOCATED_IN]->(:City)
(:POI)-[:NEAR {distance_m}]->(:POI)                    # 坐标计算，阈值如 1.5km
(:POI)-[:HAS_TAG]->(:Tag)
(:Note)-[:MENTIONS {sentiment, tip}]->(:POI)
(:POI)-[:CO_VISITED {count}]->(:POI)                   # 同一篇笔记/同一天出现
```

### 7.2 构建流程（`graph/build_graph.py`）

1. LLM 从每篇笔记抽取 `{pois:[{name, type, tip, sentiment}], tags:[]}`（JSON Schema 约束输出）。
2. **实体对齐**：用 `poi_search` 把抽取出的名称映射到高德 `poi_id`（“宽窄巷子”“宽窄”“宽窄巷”合并为同一节点）；对不上的丢弃或标记待审。
3. 计算 `NEAR`（同城 POI 两两算球面距离，只存阈值内的边）与 `CO_VISITED`。
4. 批量 `MERGE` 写入 Neo4j；记录抽取成功率、对齐成功率作为构建质量指标。

### 7.3 查询方式

优先用**参数化 Cypher 模板**（安全、稳定），LLM 只负责选模板和填参数：

| 模板 | 示例问题 | Cypher 思路 |
|---|---|---|
| `nearby` | “宽窄巷子附近有什么好吃的” | `MATCH (p:POI {name:$e})-[r:NEAR]-(q:POI)-[:HAS_TAG]->(:Tag {name:'美食'}) RETURN q ORDER BY r.distance_m` |
| `co_visit` | “去了大熊猫基地还能顺路去哪” | 按 `CO_VISITED.count` 排序 |
| `by_tag` | “成都适合拍照的地方” | `City ← LOCATED_IN − POI − HAS_TAG → Tag` |
| `evidence` | “大家怎么评价锦里” | `Note −MENTIONS→ POI`，返回 tip 与来源笔记 |

可选：text2cypher 作为兜底（只读账号 + 结果行数上限 + 超时）。

### 7.4 在规划中的作用

- Agent 用 `NEAR`/`CO_VISITED` 做**每日景点聚类**，减少折返（可量化：平均每日交通时长对比无图谱版本）。
- 图谱查询结果与笔记片段一起作为上下文，答案中给出 POI 与来源笔记。

### 7.5 部署与降级

Neo4j Community 版用 Docker 运行；不可用时路由层跳过 `graph` 路线，改走向量检索，并标记 `degraded: ["graph"]`。

---

## 8. 模块五：动态路由 RAG

### 8.1 路由表

| 路由 | 触发示例 | 知识源 | 检索策略 |
|---|---|---|---|
| `plan` | “帮我排成都 3 天行程” | 全部 | 进入 Agent 循环（§4） |
| `notes_qa` | “带老人去北京要注意什么” | 笔记库 | 混合检索 + 重排 + 引用回答 |
| `graph_qa` | “XX 附近有什么吃的” | 图谱 + 笔记 | Cypher 模板 → 用证据笔记补充 |
| `realtime` | “明天成都下雨吗”“去 XX 要多久” | 高德工具 | 直接调用工具，不检索知识库 |
| `chitchat` | “谢谢”“你是谁” | 无 | 直接回答 |

### 8.2 路由实现

两级：
1. 本地路由分类模型（§6），置信度 ≥ 0.8 直接采用；
2. 否则交给 LLM 结构化输出 `{route, city, entities}`。

记录每次路由的来源与结果，定期抽样人工检查，作为再训练数据。

### 8.3 多知识库

Chroma 中按用途拆分 collection：`notes`（游记经验）、`poi_profiles`（景点介绍、开放时间、门票）、`faq`（出行常识、预约规则）。路由决定查哪些 collection，避免“问门票却召回一堆情侣 vlog”。

### 8.4 评估

路由准确率（路由测试集）、各路由下的答案质量（LLM-as-judge 按 1–5 分评“正确性、相关性、可执行性”，并人工抽检 20% 校准）。

---

## 9. 模块六：Memory Agent

### 9.1 记忆分层

| 层 | 内容 | 存储 | 生命周期 |
|---|---|---|---|
| 工作记忆 | 最近 k 轮原文（默认 6 轮） | 请求内 | 单次请求 |
| 会话摘要 | 更早对话的滚动摘要 | SQLite `session_summary` | 会话期间 |
| 行程状态 | 当前行程 JSON + 版本号 + 用户已确认/否决的地点 | SQLite `trip_state` | 会话/行程期间 |
| 长期画像 | 稳定事实与偏好 | SQLite `user_memory` + 向量索引 | 长期，可删除 |

### 9.2 什么值得记（写入策略）

每轮对话结束后，异步调用记忆抽取（不阻塞回复），候选记忆必须满足：

1. **用户明确表达**，不是模型推测；
2. **可复用**：对以后的规划有影响（同行人、饮食禁忌、体力限制、预算区间、喜欢/讨厌的类型、常住城市）；
3. **稳定**：不是一次性信息（“这次想吃火锅” → 只进行程状态，不进长期画像）。

不记：闲聊、情绪性表达、敏感信息（身份证号、手机号、精确住址、健康细节等直接丢弃）。

记忆结构：

```json
{"user_id":"u_x","type":"constraint","key":"companion","value":"带 70 岁老人",
 "confidence":0.9,"source":"session_12/turn_3","updated_at":"2026-10-12"}
```

**冲突处理**：同一 `key` 出现新值时覆盖并保留历史版本；低置信度（< 0.7）的记忆先进入“待确认”，在下一次相关场景中向用户确认（“你上次说不吃辣，这次还是避开辣菜吗？”）。

### 9.3 读取策略

- 规划前读取全部 `constraint` 与 `preference` 类长期记忆注入 system；
- 其他记忆按与当前问题的向量相似度取 Top-3；
- 校验器（§4.4）把 `constraint` 当作硬约束检查。

### 9.4 摘要

工作记忆 + 摘要的 token 数超过阈值（默认 3000）时，把最早的若干轮压缩进摘要；摘要模板固定字段：目的地、日期、已确定内容、待定事项、用户否决过的方案。

### 9.5 用户可控

前端“我的偏好”面板展示长期记忆，支持删除与清空；这既是隐私要求，也便于演示。

### 9.6 评估

构造 20 组多轮对话脚本（含偏好表达、修改、冲突），检查：记忆抽取准确率/召回率、规划结果是否遵守约束（约束遵守率）、摘要后关键信息保留率。

---

## 10. 模块七：工程化

### 10.1 日志

- Python：`structlog`（或 `loguru`）输出 JSON 行日志；Node：`pino`。
- 每条日志带 `request_id、user_id(脱敏)、route、latency_ms、status`。
- LLM 调用单独记录：模型、prompt/completion token、耗时、是否重试、错误码（不记录完整密钥，prompt 全文仅在 DEBUG 级别记录）。
- 按天滚动，保留 14 天；错误日志单独一个文件。
- 去掉前端写死的 `DEBUG = true`，改为通过 URL 参数或配置开启。

### 10.2 容错

| 手段 | 实现 | 应用位置 |
|---|---|---|
| 超时 | `httpx` 按依赖分别设置（LLM 60s、高德 5s、Neo4j 3s、Reranker 2s） | 所有外部调用 |
| 重试 | `tenacity`，指数退避 + 抖动，仅对可重试错误（超时、429、5xx），最多 2–3 次 | LLM、高德 |
| 熔断 | 连续失败 N 次（如 5 次）后断开 30 秒，期间直接走降级（`pybreaker` 或自研） | LLM、Neo4j、高德 |
| 并发闸门 | `asyncio.Semaphore` 限制同时进行的 LLM 调用数（如 8） | LLM |
| 幂等 | 入库按 content_hash；规划请求带 `client_request_id` 防重复提交 | 入库、规划 |

### 10.3 降级矩阵与用户提示

| 故障 | 降级行为 | 前端提示 |
|---|---|---|
| LLM 超时/限流 | 重试 → 失败后返回示例行程 | “AI 繁忙，先给你一份参考行程，稍后可重新生成” |
| LLM 欠费/额度耗尽 | 不重试，直接示例行程 + 告警 | “AI 服务暂不可用（已通知管理员）” |
| 向量库/模型不可用 | 规划不带笔记参考 | “本次未参考真实游记” |
| Neo4j 不可用 | 走纯向量检索 | 不提示（对用户无感） |
| 本地分类模型异常 | LLM 零样本 → 默认“综合” | 不提示 |
| 高德超限/失败 | 读缓存 → 只画点不画路线 | “部分路线暂时无法计算” |
| Reranker 超时 | 使用 RRF 原始排序 | 不提示 |

### 10.4 并发与性能

- **全链路异步**：现 `rag_core.py` 使用同步 `requests`，FastAPI 同步接口跑在线程池里，改为 `httpx.AsyncClient` + `async def`；CPU 密集的 embedding/rerank 用 `run_in_threadpool`，并对同一批请求做微批处理。
- **流式输出**：LLM 用 `stream=true`，经 SSE 推给前端，首字时间从十几秒降到 1–2 秒。
- **缓存**：Redis（无 Redis 时退回进程内 LRU）缓存：地理编码/POI（7 天）、路线（1 天）、天气（30 分钟）、query embedding（1 天）、相同参数的检索结果（10 分钟）。
- **限流**：Node 网关 `express-rate-limit`，按 IP 每分钟 N 次规划请求；超限返回 `RATE_LIMITED`。
- **多进程**：`uvicorn --workers N`（注意每个 worker 都会加载模型，内存按 N 倍估算）。
- **压测**：`locust` 或 `k6`，分别测 `/retrieve`（无 LLM，测本地吞吐）与 `/chat`（有 LLM，用 mock LLM 排除外部限速），记录 QPS、P50/P95/P99、错误率，压测报告放 `eval/load_test.md`。真实瓶颈通常是 LLM 供应商的速率限制，报告里要写清楚。

### 10.5 监控

- **探针**：`/healthz`（进程存活）、`/readyz`（逐项检查 LLM 配置、Chroma、Neo4j、Redis、模型是否加载）。
- **指标**：Python `prometheus-fastapi-instrumentator` + 自定义指标；Node `prom-client`。关键指标：
  - `http_requests_total{route,status}`、`http_request_duration_seconds`
  - `llm_calls_total{model,status}`、`llm_latency_seconds`、`llm_tokens_total`
  - `tool_calls_total{tool,status}`
  - `rag_retrieve_latency_seconds`、`degrade_total{component}`
  - `circuit_breaker_state{dependency}`
- **监控页面**：
  - 轻量版：`admin.html` 调用 `/api/admin/stats`，展示依赖健康状态、最近 1 小时请求量/错误率/P95、LLM token 消耗、降级次数、最近 50 条错误（带 request_id，可点击查看链路日志）。
  - 完整版：docker-compose 增加 Prometheus + Grafana，导入预置看板。
- **告警**（可选）：错误率 > 5% 或 LLM 额度错误出现时，推送到邮件/企业微信 webhook。

### 10.6 测试与部署

- `pytest`：分块、BM25、RRF、校验器、路由、记忆抽取规则的单元测试；用 mock LLM 做 Agent 循环测试。
- 前端解析与地图逻辑合并为一份（目前 `output-map.html`、`src/parser.js`、`example/` 三处重复）。
- `docker-compose.yml`：`gateway`（Node）、`agent`（FastAPI）、`neo4j`、`redis`、可选 `prometheus`/`grafana`。
- GitHub Actions：lint + 单元测试；评测脚本手动触发。

---

## 11. 实施计划

按“先打地基、先出数字”的顺序排期（按每周 15–20 小时估算）：

| 阶段 | 内容 | 预计 | 产出 / 验收 |
|---|---|---|---|
| P0 ✅ | 修复 RAG 环境、高德 Key 后移、切换 DeepSeek（已完成） | — | 当前版本可运行 |
| P1 ✅ | 工程地基：`rag_service` → `agent_service`、配置外置、结构化日志、错误码、request-id、异步 LLM 客户端、重试/熔断 | 1 周 | 统一错误格式；日志可按 request_id 追踪 |
| P2 ✅（数据扩充未做） | RAG 2.0：数据扩充、元数据抽取、BM25 + RRF、bge-reranker、评测集与消融表 | 1.5 周 | `eval/rag_report.md`，填入 Recall@5 / MRR |
| P3 ✅ | Agent：Function Calling 循环、6 个工具、结构化行程、校验器、自纠、SSE、地图页改为读 JSON | 2 周 | `eval/agent_report.md`，填入工具成功率、地理匹配率 |
| P4 ✅（待人工核验） | 意图模型：数据构造、基线、微调、与 LLM 对比、路由上线 | 1.5 周 | `eval/classifier_report.md`，填入 macro-F1 |
| P5 ✅ | 动态路由 + 多知识库 | 0.5 周 | 路由准确率 |
| P6 ✅ | Memory：抽取/存储/摘要/约束校验/偏好面板 | 1 周 | 约束遵守率 |
| P7 ✅（规划聚类未做） | 知识图谱：抽取、对齐、Neo4j、Cypher 模板、规划聚类 | 1.5 周 | 图谱规模、对齐率、交通时长对比 |
| P8 ✅ | 性能与运维：缓存、限流、压测、监控页面、Docker | 1 周 | `eval/load_test.md`、`admin.html` |
| 可选 | LoRA 槽位解析模型 | 1–2 周 | 槽位 F1 |

总计约 10–11 周。时间不够时的优先级：**P1 → P2 → P3 → P4**（覆盖简历四条主线并产出全部【待填】数字）> P6 > P5 > P8 > P7。

---

## 12. 指标汇总（简历数字来源）

| 简历位置 | 指标 | 来源脚本 / 报告 |
|---|---|---|
| Agent 工具编排 | 工具调用成功率 | `eval/run_agent_eval.py` → `eval/agent_report.md` |
| 结果校验与自纠 | 首次通过率 → 自纠后通过率；地理匹配率（旧流程 vs 新流程） | 同上 |
| RAG | Recall@5、MRR@10（各配置消融） | `eval/run_rag_eval.py` → `eval/rag_report.md` |
| 意图微调 | macro-F1 / 准确率，对比 LLM 零样本 | `classifier/train/evaluate.py` → `eval/classifier_report.md` |
| 动态路由 | 路由准确率 | `eval/run_router_eval.py` |
| Memory | 约束遵守率、记忆抽取 F1 | `eval/run_memory_eval.py` |
| 工程化 | P95 延迟、压测 QPS、降级成功率 | `eval/load_test.md` |

所有报告需包含：评测集规模、构造方式、运行日期、模型与配置版本，保证面试时可复现、可解释。

---

## 13. 简历改写建议（全部完成后的参考写法）

> 数值处保留占位，以评测报告为准；未完成的模块不要写。

- **Agent 工具编排**：基于 Function Calling 自研 Agent 循环，编排笔记检索、知识图谱、高德 POI/路线/天气等 6 类工具，工具结果回填上下文；规则校验器检查城市一致性、交通时长与用户约束，并把问题反馈给 LLM 自动修正，工具调用成功率 X%，行程首次校验通过率 A% → 自纠后 B%。
- **RAG**：BM25 + bge 向量的 RRF 混合检索 + bge-reranker 重排 + 元数据过滤，在 150 条人工标注查询上 Recall@5 从 X% 提升至 Y%。
- **知识图谱 + 动态路由**：LLM 抽取游记中的景点关系并通过高德 POI 做实体对齐，构建 Neo4j 图谱（N 个节点）；路由模型把问题分发到笔记库/图谱/实时工具，路由准确率 Z%。
- **意图微调**：针对旧模型训练与推理分布不一致的问题重新构建数据，微调 RoBERTa 多标签风格分类，macro-F1 达到 X%（LLM 零样本为 Y%），单次推理 < 10ms。
- **记忆与工程化**：分层记忆（画像/摘要/行程状态）与约束校验；结构化日志 + 错误码 + 重试/熔断/降级 + Prometheus 指标与监控页，压测 P95 为 X ms。
