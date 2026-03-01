// 简易静态服务 + 本地分类代理 + LLM 代理（OpenAI 兼容 /chat/completions）
// 运行：
//   推荐：在项目根目录创建 .env 后直接 npm start / node server.js

import 'dotenv/config';
import express from 'express';
import fetch from 'node-fetch';
import cors from 'cors';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();

// 开发期：同源访问无需 CORS，但保留以便你从其它端口打开
app.use(cors());
app.use(express.json({ limit: '1mb' }));

// 轻量日志，便于你在控制台定位
app.use((req, _res, next) => {
  if (req.path.startsWith('/api/')) {
    console.log(`[api] ${req.method} ${req.path}`);
  }
  next();
});

// 静态资源服务（直接访问 http://127.0.0.1:3000/ai-chat.html）
app.use(express.static(__dirname));

// 你的百度 API Key 仅保存在服务器环境变量中
// 清洗可能的换行/引号，避免 "Invalid character in header content"
const API_KEY = (process.env.BAIDU_API_KEY || '')
  .replace(/[\r\n]/g, '')
  .trim()
  .replace(/^['"]|['"]$/g, '');
// OpenAI 兼容入口（百度 AI Studio 的 Chat Completions 路径）
// 可用环境变量覆盖：
//   BAIDU_BASE_URL    默认 https://aistudio.baidu.com/llm/lmapi
//   BAIDU_API_VERSION 默认 v1（如需 v3 可设为 v3）
const BAIDU_BASE_URL = (process.env.BAIDU_BASE_URL || 'https://aistudio.baidu.com/llm/lmapi').replace(/\/+$/,'');
const BAIDU_API_VERSION = (process.env.BAIDU_API_VERSION || 'v1').trim();
const REMOTE_ENDPOINT = `${BAIDU_BASE_URL}/${BAIDU_API_VERSION}/chat/completions`;

// 默认模型（可选），如果前端未传 model，则使用该值
const DEFAULT_MODEL = (process.env.BAIDU_MODEL || '').trim();

// 本地 Python 分类服务基址（FastAPI），可通过环境变量覆盖
const PY_BASE = process.env.PY_CLASSIFY_BASE || 'http://127.0.0.1:8001';

// 处理可能的预检请求
app.options('/api/classify', (_req, res) => res.sendStatus(204));
app.options('/api/itinerary', (_req, res) => res.sendStatus(204));

// 将前端的分类请求代理到本地 Python 服务
app.post('/api/classify', async (req, res) => {
  try {
    console.log('[api] /api/classify body:', req.body);
    const r = await fetch(`${PY_BASE}/classify`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req.body || {})
    });
    const text = await r.text();
    res.status(r.status).type('application/json').send(text);
  } catch (e) {
    console.error('Classify proxy error:', e);
    res.status(500).json({ error: 'classify_proxy_failed', message: String(e) });
  }
});

app.post('/api/itinerary', async (req, res) => {
  if (!API_KEY) {
    return res.status(500).json({ error: 'missing_api_key', message: '请设置环境变量 BAIDU_API_KEY' });
  }
  try {
    const body = req.body || {};
    if (!body.model && DEFAULT_MODEL) body.model = DEFAULT_MODEL;
    console.log('[api] /api/itinerary endpoint:', REMOTE_ENDPOINT);
    console.log('[api] /api/itinerary body keys:', Object.keys(body || {}));
    const resp = await fetch(REMOTE_ENDPOINT, {
      method: 'POST',
      headers: {
        'Authorization': 'Bearer ' + API_KEY,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(body)
    });
    const text = await resp.text();
    res.status(resp.status).type('application/json').send(text);
  } catch (e) {
    console.error('Proxy error:', e);
    res.status(500).json({ error: 'proxy_failed', message: String(e) });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Static + proxy server running at http://127.0.0.1:${PORT}`);
});


