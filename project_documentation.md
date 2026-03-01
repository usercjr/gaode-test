# WanderAI · 行界 智能旅行助手 — 技术项目文档

> **项目根目录：** `D:\gaode_test`  
> **文档生成时间：** 2026-02-28  
> **分析范围：** 所有核心源代码文件（server.js / app.py / AI_Travel_Assistant*.py / ai-chat.html / best_model/）

---

## 一、技术栈概览

| 层次 | 技术 / 框架 | 用途 |
|------|------------|------|
| **前端** | 原生 HTML5 + CSS3 + Vanilla JS | AI 聊天界面（`ai-chat.html`）、地图输出页（`output-map.html`） |
| **前端地图** | 高德地图 JS API（AMap） | 行程景点地图可视化 |
| **Node.js 中间层** | Express 5 + node-fetch + CORS | 静态文件服务、API 反向代理（`server.js`）|
| **Python 推理服务** | FastAPI + Uvicorn | 本地分类模型 REST 服务（`python_service/app.py`）|
| **本地 AI 模型** | PaddlePaddle + PaddleNLP | 旅行偏好文本分类（ERNIE 序列分类微调模型）|
| **远程 LLM** | 百度文心大模型（Ernie-3.5-8k）/ 百度星河 AI Studio OpenAI 兼容 API | 行程文本生成 |
| **远程 LLM SDK** | OpenAI Python SDK（兼容百度 OpenAI 接口） | Python 脚本调用 LLM |
| **模型基座** | `ernie-3.0-base-zh`（vocab_size=40000, hidden_size=768, 12层） | 微调后保存至 `best_model/` |
| **包管理** | npm（Node.js）、pip（Python） | 依赖管理 |

### 架构模式

```
┌──────────────────────┐   HTTP /api/classify  ┌───────────────────────────────┐
│   ai-chat.html       │ ──────────────────────▶│  python_service/app.py        │
│  (前端 SPA)          │                        │  FastAPI  :8001               │
│                      │ ◀──────────────────────│  本地 ERNIE + PaddleNLP 分类  │
│                      │   { label, score }     └───────────────────────────────┘
│                      │
│                      │   HTTP /api/itinerary  ┌───────────────────────────────┐
│                      │ ──────────────────────▶│  server.js (Express :3000)    │
│                      │                        │  反向代理 ──────────────────────▶ 百度 AI Studio
│                      │ ◀──────────────────────│                               │  (文心 LLM API)
└──────────────────────┘   LLM 响应文本          └───────────────────────────────┘
```

---

## 二、代码逐模块分析

### 2.1 `server.js` — Node.js 中间层（端口 3000）

**职责：** 静态文件托管 + 两路 API 代理

| 路由 | 方法 | 说明 |
|------|------|------|
| `/*` | 静态 | 直接服务 `__dirname` 下的 HTML/JS 文件 |
| `/api/classify` | POST | 将请求透传到 `http://127.0.0.1:8001/classify`（Python 分类服务）|
| `/api/itinerary` | POST | 将请求透传到百度 AI Studio `chat/completions`，附加 `BAIDU_API_KEY` |

**关键设计：**
- `API_KEY` 仅保存在服务器环境变量中，前端不持有密钥 ✅
- `PY_BASE`、`BAIDU_BASE_URL`、`BAIDU_MODEL` 均可通过环境变量覆盖 ✅
- `/api/classify` 代理层无超时设置（前端已有 5s 超时兜底）⚠️

**数据流：**
```
前端 POST { text } → /api/classify → fetch(PY_BASE/classify) → Python FastAPI → 返回 { label, score }
前端 POST { model, messages } → /api/itinerary → fetch(BAIDU_ENDPOINT) → 百度 LLM → 返回 OpenAI 格式响应
```

---

### 2.2 `python_service/app.py` — 本地分类推理服务（端口 8001）

**职责：** 用 FastAPI 暴露本地 ERNIE 文本分类模型

**关键函数：**

| 函数/端点 | 行号 | 说明 |
|-----------|------|------|
| `_maybe_load_model()` | 55–77 | FastAPI `startup` 事件，懒加载 `best_model/`; 目录不存在时以 503 占位启动 |
| `_load_label_map()` | 24–45 | 从 `label_map.json` 读取标签映射；文件不存在时回退到5类硬编码默认值 |
| `_softmax(x)` | 48–52 | NumPy 实现 softmax，用于将 logits 转为概率 |
| `POST /classify` | 94–119 | **真正的推理调用**：tokenize → `model()` → softmax → 返回 `{ label, score }` |
| `GET /healthz` | 84–91 | 健康检查，返回 `MODEL_READY` 状态 |

**数据流：**
```
请求 { text } → tokenizer(text, max_seq_len=256) → model(input_ids, token_type_ids) → logits
→ softmax → argmax → ID2LABEL[pred_id] → 返回 { label, score }
```

---

### 2.3 `python_service/fine_tune/AI_Travel_Assistant.py` — 独立演示脚本

**职责：** 本地 CLI 演示，将 PaddleNLP 分类 + 文心 LLM 串联

**关键函数：**

| 函数 | 行号 | 说明 |
|------|------|------|
| `classify_preference(text)` | 105–123 | 直接调用本地 ERNIE 模型进行分类 |
| `generate_itinerary(...)` | 45–95 | 通过 OpenAI SDK 调用百度星河 `chat/completions` |
| `_extract_text(obj)` | 66–90 | 多层字段兼容解析，处理标准 OpenAI 与百度私有格式 |

**注意：** 该脚本在模块顶层（第101行）直接 `AutoModelForSequenceClassification.from_pretrained(BEST_MODEL_DIR)`，**进程启动时即加载模型**，无懒加载。

---

### 2.4 `python_service/fine_tune/AI_Travel_Assistant_cli.py` — 多轮对话 CLI

**职责：** 命令行多轮旅行助理（含偏好分类 + LLM 调用），用于终端交互测试

**关键设计：**
- `classify_text()` (行61–71)：每轮用户输入都先过本地分类模型，获取偏好标签
- `LABEL_PROMPTS` (行74–148)：8 类偏好标签各有专属系统 Prompt，动态注入风格指令
- `chat_loop()` (行175–230)：维护 `messages` 历史；当超过 30 条时滑窗保留最近 20 条

**数据流：**
```
用户输入 → classify_text() [本地模型] → label → build_system_prompt(label)
→ messages.append(system_label_msg + user_msg)
→ client.chat.completions.create() [远程文心 LLM]
→ 打印 assistant 回复
```

---

### 2.5 `ai-chat.html` — 前端 SPA（WanderAI 界面）

**职责：** 用户交互入口，采集行程信息、调用两路后端 API、渲染结果

**关键函数：**

| 函数 | 说明 |
|------|------|
| `callLocalClassifier(text)` | POST 到 `/api/classify`，带 5s AbortController 超时 |
| `callLLMItinerary({...})` | POST 到 `/api/itinerary`，构造 OpenAI 格式 body |
| `buildMockPlan(...)` | 本地离线 Fallback，生成硬编码的"上海三日游"示例数据 |
| `persistState(plan)` | 将行程数据写入 `localStorage` |
| `renderPreferenceOptions()` | 动态渲染偏好标签选择 UI |

**LLM 配置（第 630–636 行）：**
```javascript
const LLM_CONFIG = {
  enabled: true,
  endpoint: '/api/itinerary',   // 通过 Node.js 代理
  model: 'ernie-3.5-8k',
  key: ''                        // 不在前端持有密钥
};
```

---

### 2.6 `best_model/` — 本地微调模型文件

| 文件 | 说明 |
|------|------|
| `config.json` | ERNIE 架构配置；`model_type: "ernie"`; 8类标签（`LABEL_0`~`LABEL_7`）；12层 Transformer；`hidden_size=768` |
| `model_state.pdparams` | PaddlePaddle 格式权重文件（**约 450 MB**） |
| `tokenizer_config.json` | 基于 `ernie-3.0-base-zh`，`do_lower_case=true` |
| `vocab.txt` | 40000 字符词表 |

> ⚠️ `best_model/config.json` 中的 `id2label` 为 `{"0":"LABEL_0","1":"LABEL_1",...}` 通用占位符，**并非真实类别名称**。真实标签映射需要单独的 `label_map.json` 文件（目前该文件不存在于 `best_model/` 目录）。

---

## 三、模型调用链路分析

### ✅ 是否真正调用了本地训练好的模型进行推理？

**结论：是的，代码中确实存在本地模型的推理调用，但存在一个关键配置缺失导致功能降级。**

#### 调用路径 A：生产流程（前端 → Node.js → FastAPI → 本地模型）

```
ai-chat.html L649   callLocalClassifier(text)
→ POST /api/classify
  → server.js L62   fetch(`${PY_BASE}/classify`, ...)
    → python_service/app.py L106-119  (FastAPI /classify 端点)
      → tokenizer(req.text, ...)               ← 真实 Tokenization
      → model(enc["input_ids"], enc["token_type_ids"])  ← 🔴 真实推理
      → softmax → argmax → ID2LABEL[pred_id]
      → return { label, score }
```

**关键代码行（`python_service/app.py`）：**
```python
# 第 106-118 行：
enc = tokenizer(req.text, max_seq_len=256, pad_to_max_seq_len=True, return_tensors="pd")
with paddle.no_grad():
    logits = model(enc["input_ids"], enc["token_type_ids"])   # ← 真实本地推理
logits_np = np.array(logits)
probs = _softmax(logits_np)[0]
pred_id = int(np.argmax(probs))
label = ID2LABEL.get(pred_id, str(pred_id))
```

#### 调用路径 B：CLI 演示（`AI_Travel_Assistant_cli.py`）

```python
# 第 27-28 行（模块加载时）：
tokenizer = AutoTokenizer.from_pretrained(BEST_MODEL_DIR)     # 加载 tokenizer
clf_model = AutoModelForSequenceClassification.from_pretrained(BEST_MODEL_DIR)  # 加载本地模型权重

# 第 68-69 行（classify_text 函数，每次推理）：
with paddle.no_grad():
    logits = clf_model(enc["input_ids"], enc["token_type_ids"])  # ← 真实推理
```

---

### ⚠️ 已发现的问题与未闭环点

#### 问题 1：标签映射文件缺失（严重）

- `best_model/config.json` 中 `id2label` 为占位符（`LABEL_0` ~ `LABEL_7`），不是真实类别名
- `best_model/label_map.json` 文件**不存在**
- `app.py` 第 38-45 行会回退到**5类**硬编码默认列表（`Rush_Trip`, `Deep_Cultural_Trip`, `Food_Adventure`, `Family_Friendly`, `Nightlife`）
- `AI_Travel_Assistant_cli.py` 第 47-56 行回退到**8类**硬编码默认列表

**后果：** 分类模型虽然真实运行，但返回的标签名称来自硬编码默认值，不一定与训练时的真实标签对齐，**分类结果不可信**。

**修复方法：** 在训练完成后将真实 `id2label` 写入 `best_model/label_map.json`：
```json
{ "id2label": {"0":"Rush_Trip","1":"Deep_Cultural_Trip","2":"Food_Adventure","3":"Nature_Relaxation","4":"City_Leisure_Tour","5":"Customized_AI_Tour","6":"Vlog_Creator_Trip","7":"Photo_Experience"} }
```

#### 问题 2：app.py 的默认标签数（5）与模型实际输出维度（8）不匹配

- `config.json` 中 `id2label` 有 8 个类别（0~7）
- `app.py` 回退标签只有 5 个类别（0~4）
- 若模型输出 pred_id ∈ {5,6,7}，会触发 `ID2LABEL.get(pred_id, str(pred_id))` 返回数字字符串

#### 问题 3：行程文本生成使用远程 LLM，不是本地模型

- 行程文本（itinerary）由**百度星河 文心大模型**（云端 API）生成
- **本地模型（ERNIE 微调）仅用于"旅行偏好分类"**，并不生成回复文本
- 如果未设置 `BAIDU_API_KEY`，`/api/itinerary` 会返回 500 错误，前端会 fallback 到硬编码的上海示例

---

## 四、未完成部分与技术债务

| 类型 | 位置 | 描述 |
|------|------|------|
| 🔴 **功能缺失** | `best_model/` | `label_map.json` 不存在，标签映射依赖硬编码占位 |
| 🔴 **类别数不一致** | `app.py` L38-45 | 回退标签 5 类 vs 模型输出 8 类 |
| 🟡 **硬编码** | `ai-chat.html` L728-796 `buildMockPlan()` | 上海三日行程硬编码为 Fallback，实际需要 LLM 才有真实结果 |
| 🟡 **硬编码** | `ai-chat.html` L634 | `model: 'ernie-3.5-8k'` 写死，不支持切换 |
| 🟡 **硬编码** | `AI_Travel_Assistant.py` L128 | `user_desc = "目的地：上海；偏好：独自出行、经典必去"` 测试入参写死 |
| 🟡 **无超时** | `server.js` L62-68 | `/api/classify` 代理无服务端超时（前端有，但服务器侧可能资源泄露）|
| 🟡 **异常处理不完善** | `app.py` L112-113 | `model(enc["input_ids"], enc["token_type_ids"])` 若 CUDA OOM 无降级处理 |
| 🟡 **多轮历史滑窗简单粗暴** | `AI_Travel_Assistant_cli.py` L228-230 | 超过30条消息直接截断，可能截断关键上下文 |
| 🟢 **信息性** | `fine_tune/README.md` L52 | 末行残留 `*** End Patch``` } />` 格式噪声 |
| 🟢 **未使用依赖** | `package.json` | `dotenv` 已安装但 `server.js` 未调用（改用了直接 `process.env`）|

---

## 五、环境配置要求

### Node.js 服务（端口 3000）

```powershell
# 安装依赖
npm install

# 启动（必须设置环境变量）
$env:BAIDU_API_KEY = "your_key_here"
node server.js
```

**可选环境变量：**
| 变量 | 默认值 | 说明 |
|------|--------|------|
| `BAIDU_API_KEY` | —（必填）| 百度 AI Studio API Key |
| `BAIDU_BASE_URL` | `https://aistudio.baidu.com/llm/lmapi` | LLM 接口基址 |
| `BAIDU_API_VERSION` | `v1` | API 版本（v1 或 v3）|
| `BAIDU_MODEL` | — | 默认模型名（优先前端传入）|
| `PY_CLASSIFY_BASE` | `http://127.0.0.1:8001` | Python 分类服务地址 |
| `PORT` | `3000` | Node 服务端口 |

---

### Python FastAPI 服务（端口 8001）

```powershell
# 创建 Python 3.10 虚拟环境
cd D:\gaode_test\python_service
py -3.10 -m venv .venv310
.\.venv310\Scripts\Activate.ps1

# 安装依赖
pip install -r requirements.txt   # fastapi, uvicorn, paddlenlp, paddlepaddle

# 启动
uvicorn app:app --host 0.0.0.0 --port 8001

# 可选环境变量
$env:BEST_MODEL_DIR = "D:\gaode_test\best_model"
```

**Python 环境要求：**
- Python 3.10（PaddlePaddle 对 3.11+ 支持有限）
- PaddlePaddle CPU 版即可运行（GPU 版对 CUDA 版本有严格要求）

---

## 六、潜在性能瓶颈与优化建议

### 性能瓶颈

| 问题 | 描述 | 影响 |
|------|------|------|
| **模型每次推理无批处理** | `classify` 端点每次只处理单条文本 | 高并发时吞吐量低 |
| **模型常驻内存** | 450MB 权重文件常驻进程内存 | 内存占用高（CPU 模式约 1.5GB）|
| **无推理结果缓存** | 相同文本每次都重新推理 | 重复请求浪费资源 |
| **LLM 响应延迟** | 百度云端 API 网络 RTT 不稳定 | 前端等待时间长，无流式输出 |
| **前端无 SSE/WebSocket** | 行程生成为一次性 JSON，不支持流式显示 | 用户体验差（等待感强）|

### 优化建议

1. **补充 `label_map.json`**（首要）：将训练时的真实标签写入文件，使分类结果可读且可靠。

2. **开启 LLM 流式输出（SSE）**：百度 API 支持 `stream: true`，前端配合 `EventSource` 可实现逐字输出，大幅提升体验。

3. **分类结果本地缓存**：前端或 Node 层按文本哈希缓存 `classify` 结果，减少重复调用。

4. **CUDA 内存管理**（若使用 GPU）：加入 `paddle.device.cuda.empty_cache()` 在每次推理后释放临时显存。

5. **统一回退标签数量**：`app.py` 的默认回退改为与 `AI_Travel_Assistant_cli.py` 一致的 8 类，同时确保类别顺序与训练时一致。

6. **健康检查集成**：前端启动时调用 `GET /api/classify/healthz`（需对外暴露），展示本地模型状态 badge，提升开发调试体验。

---

## 七、总结：项目整体评价

### ✅ 优点

- 架构职责清晰（前端 / Node 代理 / Python 推理）
- API 密钥保存在服务端，安全设计合理
- 分类服务支持懒加载（无模型文件也能启动）
- CLI 多轮对话脚本设计完整，便于独立测试

### ⚠️ 核心待解决问题

> [!CAUTION]
> **本地模型确实在运行，但标签语义已断链。** 模型输出的 pred_id (0~7) 无法映射到有意义的类别名称，因为 `best_model/label_map.json` 文件缺失。这是当前最关键的未闭环问题，修复后整条链路才真正完整。

> [!IMPORTANT]
> **行程文本由远程文心大模型（云端）生成，本地模型仅负责偏好分类。** 本地微调的 ERNIE 模型是分类器，不是文本生成器。如果希望本地模型承担生成任务，需要换用支持文本生成的模型架构（如 ERNIE-Bot fine-tune 或 ChatGLM 本地部署）。
