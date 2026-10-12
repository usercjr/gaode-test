# WanderAI · 行界智能旅行助手 — 技术项目文档

> **项目根目录：** `D:\gaode_test`
> **文档更新时间：** 2026-10-13（P5 动态路由完成后）
> **覆盖范围：** 前端（`ai-chat.html` / `output-map.html` / `rag-search.html`）、Node 网关（`server.js`）、Python 服务（`agent_service`）
> **升级规划：** 见 `docs/upgrade_plan.md`
> **面试准备笔记（各阶段做法、决策、踩坑、简历可用数据）：** 见 `docs/interview_notes.md`

---

## 一、项目定位与现状

两条能力链路：

1. **旅行规划（Agent）**：本地微调模型识别风格（多标签，缺模型时退回 LLM）→ Agent 通过 Function Calling 调用笔记检索、高德 POI / 路线 / 天气 → 提交结构化行程 → 规则校验（地点真实、城市一致、交通时长）并回填问题让 LLM 修正 → 地图页直接使用已校验坐标。Agent 不可用时回退到旧的“检索笔记 + 文本生成 + 前端解析”流程。
2. **聊天（动态路由）**：聊天框消息先由路由模型判断类型——修改行程才交给 Agent，经验问答走 RAG，“附近 / 顺不顺路”走周边搜索与路线工具，天气与交通时长走实时工具，闲聊直接回复；只有“修改行程”会改动行程。
3. **RAG 问答**：LLM 意图改写 → 地名过滤 + 混合检索（向量 + BM25，RRF 融合）→ bge-reranker 重排 → 基于片段的引用回答。

所有 LLM 调用集中在 `agent_service`，统一经过超时、重试、熔断、并发控制与日志；Node 网关不持有 LLM 密钥。

---

## 二、技术栈

| 层级 | 技术 | 用途 |
|---|---|---|
| 前端 | HTML + CSS + Vanilla JS | 聊天行程页、地图页、RAG 搜索页 |
| 地图 | 高德 JS API 2.0 + Web 服务 API | POI 检索、地理编码、路线、地图展示 |
| 网关 | Express 5 + node-fetch + dotenv | 页面托管、request-id、统一错误格式、高德代理、转发 |
| Python 服务 | FastAPI + httpx + pydantic-settings | LLM 客户端、RAG、分类、行程生成 |
| RAG | ChromaDB + bge-small-zh-v1.5（向量）、jieba + rank_bm25（关键词）、bge-reranker-base（重排） | 入库、混合召回、重排 |
| LLM | DeepSeek（`deepseek-chat`，OpenAI 兼容接口，可在 `config.yaml` 切换） | 分类、行程、RAG 改写/重排/回答 |

---

## 三、架构

```text
浏览器
  ├─ ai-chat.html ── /api/plan/stream (SSE) ─┐
  │                ── /api/classify ─────┤
  │                ── /api/rag/retrieve ─┤
  │                ── /api/itinerary ────┤
  ├─ rag-search.html ─ /api/rag/search ──┤
  └─ output-map.html ─ /api/amap/* ──┐   │
                                     ▼   ▼
                    Node 网关 server.js (:3000)
                     · 只托管页面与 src/ 脚本
                     · X-Request-ID 生成与透传、JSON 访问日志
                     · 统一错误格式、下游超时
                     │                      │
          高德 REST API（服务端加 Key）   agent_service (:8002)
                                           · /classify /retrieve /search /itinerary
                                           · /healthz /readyz /ingest/csv(仅本机)
                                           · LLMClient：超时/重试/熔断/并发闸门
                                           · ChromaDB 向量库
                                               │
                                           DeepSeek API
```

---

## 四、模块说明

### 4.1 Node 网关（`server.js`）

- **静态托管**：只开放 `ai-chat.html`、`output-map.html`、`rag-search.html` 和 `src/`；配置、数据、日志、Python 源码不对外。
- **request-id**：沿用前端/调用方传入的 `X-Request-ID`，否则生成；透传给 `agent_service` 并写入响应头。
- **转发**：`/api/chat`、`/api/plan`、`/api/classify`、`/api/itinerary`、`/api/rag/search`、`/api/rag/retrieve` → `agent_service`；`/api/plan/stream`、`/api/chat/stream` 逐块透传 SSE，浏览器断开时中止上游，透传状态码与响应体；下游不可达返回 `AGENT_UNAVAILABLE`，超时返回 `GATEWAY_TIMEOUT`。
- **高德代理**：`GET /api/amap/{place-text|geocode|driving|walking}`，白名单接口，服务端追加 Key；高德业务错误（`status:"0"`）会记录日志。
- **就绪检查**：`GET /api/readyz` 汇总网关与 `agent_service` 的依赖状态。
- **启动校验**：端口、`AGENT_BASE` 非法时直接退出；未配置高德 Key 时告警。

### 4.2 `agent_service`（Python）

详见 `agent_service/README.md`。要点：

- **配置外置**：`config.yaml`（非敏感）+ `.env`（密钥）+ 环境变量覆盖，`pydantic-settings` 启动时校验。
- **错误码**：`core/errors.py` 统一定义 code、HTTP 状态、是否可重试、用户提示语；全局异常处理器统一输出。
- **LLM 客户端**（`core/llm.py`）：
  - 按错误类型重试：超时、429、5xx 指数退避重试；额度不足、鉴权失败不重试。
  - 熔断：连续 5 次服务端类失败后断开 30 秒，期间快速失败。
  - 并发闸门：同时最多 8 个 LLM 调用。
  - 每次调用记录用途、模型、耗时、token 用量、错误码。
- **混合检索**（`rag/retriever.py`）：查询中识别到城市/省份时先限定到相关笔记（元数据由 `rag/enrich.py` 用 LLM 抽取），再做向量 + BM25 召回、RRF 融合、交叉编码器重排。评测见 `eval/rag_report.md`：nDCG@5 从纯向量 0.775 提升到 0.859，Recall@5 从 0.845 到 0.896（LLM 判分，待人工核验）。
- **RAG 降级**：改写、回答任一步 LLM 失败或重排模型出错都不让请求失败，跳过的步骤写入响应的 `degraded`。
- **意图分类**（`classifier/`）：微调 chinese-roberta-wwm-ext 做风格 6 类多标签与路由 5 类，CPU 单条 23~43ms；模型目录缺失时退回 LLM 零样本，LLM 也不可用时返回默认标签。训练与评测见 `eval/classifier_report.md`，模型产物在 `agent_service/models/`（不入库，用 `python -m classifier.train.train` 重新训练）。
- **日志**：JSON 行格式，带 `request_id`，按天滚动保留 14 天，错误单独写 `logs/agent.error.log`。

### 4.3 `ai-chat.html`

0. 表单新增“天数”；规划优先走 Agent（`/api/plan/stream`），进度在一个气泡里实时更新；多轮修改把上一版结构化行程回传给 Agent。以下 2~4 步是 Agent 失败时的回退流程。
1. 用户手动选了风格标签则直接用，否则调用 `/api/classify`（10 秒超时）。
2. 用识别出的标签调用 `/api/rag/retrieve`（仅向量召回，15 秒超时）；失败时在对话中提示“本次未参考真实游记”。
3. 风格提示 + 笔记片段组成 system 消息，连同最近 30 条对话发给 `/api/itinerary`。
4. 行程写入 localStorage，供地图页读取；LLM 失败时回退示例行程，提示语带错误原因与请求 ID 前 8 位。

调试日志默认关闭，访问 `ai-chat.html?debug` 开启。

### 4.4 `output-map.html`

解析行程文本 → 景点多级匹配（同城 POI → 全国 POI → 地理编码）→ 驾车/步行路线 → 地图渲染（`src/map.js`）。高德 REST 请求全部经 `/api/amap/*`。

### 4.5 `rag-search.html`

RAG 调试页：展示意图、改写查询、召回数、答案（含降级说明）、召回片段。

### 4.6 数据与其他目录

- `data/travel_notes.csv`：217 条游记（`id, title, text, suggested_label, final_label`）。
- `data/*-cache.json`、`example/`、`src/parser.js`、`src/toAmapJson.js`：早期离线流程（`npm run generate:amap`）。

---

## 五、配置

**`.env`（密钥，不提交）**

```env
LLM_API_KEY=your_deepseek_key_here
AMAP_WEB_SERVICE_KEY=your_amap_web_service_key_here
PORT=3000
AGENT_BASE=http://127.0.0.1:8002
```

**`config.yaml`（非敏感，可提交）**：LLM 地址/模型/超时/并发/重试/熔断参数、向量库路径与 embedding 模型、服务端口、日志级别与目录。切换 LLM 服务商只需改 `llm.base_url`、`llm.model` 和 `.env` 里的 `LLM_API_KEY`。

---

## 六、启动

```powershell
# 1) Python 服务（:8002）
cd D:\gaode_test\agent_service
.\.venv-win\Scripts\python.exe app.py

# 2) Node 网关（:3000）
cd D:\gaode_test
npm start
```

首次使用需建环境并入库，见 `agent_service/README.md`。

验证：

```powershell
Invoke-RestMethod http://127.0.0.1:3000/api/readyz
```

浏览器打开 `http://localhost:3000/`。

---

## 七、已知问题

1. 高德 JS API 2.0 新 Key 需配置 `securityJsCode`，当前未配置。
2. Agent 规划单次约 30~50 秒、约 5 万 token；高德个人 Key 有 QPS 与日配额限制（客户端限速 3 QPS）。
3. 行程解析逻辑在 `output-map.html`、`src/parser.js`、`example/runAmapExample.js` 中各有一份。
4. 检索评测的相关性标注由 LLM 完成，需人工抽检 `eval/data/rag_review.csv` 后再引用数字。
5. 交叉编码器在 CPU 上重排 20 条约 3 秒，只用于 RAG 问答页；行程规划走不重排的路径（约 30ms）。
6. `data/travel_notes.csv` 由两个文件合并，`id` 列有重复；系统改用内容哈希作为笔记 ID，原 `id` 保留为 `csv_id`。

---

## 八、评测

```powershell
# 构建评测集（生成查询 → 候选池 → LLM 判分；已完成的步骤会跳过）
agent_service\.venv-win\Scripts\python.exe eval\build_rag_eval.py
# 消融评测，输出 eval/rag_report.md
agent_service\.venv-win\Scripts\python.exe eval\run_rag_eval.py
```
