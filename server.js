// Node 网关：静态页面托管 + request-id + 统一错误格式 + 限流与过载保护 + 指标 + 高德代理 + agent_service 转发
// 运行：在项目根目录配置 .env 后 npm start
// 所有 LLM 调用都在 agent_service（Python）中完成，网关不持有 LLM 密钥。

import 'dotenv/config';
import express from 'express';
import fetch from 'node-fetch';
import cors from 'cors';
import crypto from 'crypto';
import path from 'path';
import { fileURLToPath } from 'url';
import { ConcurrencyGate, GatewayMetrics, RateLimiter, isLoopback } from './gateway/ops.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// ---------------- 配置（启动时校验） ----------------
const config = {
  port: Number(process.env.PORT || 3000),
  agentBase: (process.env.AGENT_BASE || process.env.RAG_BASE || 'http://127.0.0.1:8002').replace(/\/+$/, ''),
  amapKey: (process.env.AMAP_WEB_SERVICE_KEY || '').trim(),
  // 地图页用的 JS API Key 与安全密钥：由 /api/config 下发，不写死在页面里
  amapJsKey: (process.env.AMAP_JS_KEY || '').trim(),
  amapJsSecurityCode: (process.env.AMAP_JS_SECURITY_CODE || '').trim(),
  // 监控接口与页面：设置了 ADMIN_TOKEN 时凭 token 访问，否则只允许本机访问
  adminToken: (process.env.ADMIN_TOKEN || '').trim(),
  // 限流（每个 IP 每分钟）与过载保护（全局同时进行的重请求数）
  rateLimit: {
    apiPerMin: Number(process.env.RATE_LIMIT_API_PER_MIN || 120),
    heavyPerMin: Number(process.env.RATE_LIMIT_HEAVY_PER_MIN || 10),
    heavyMaxInFlight: Number(process.env.HEAVY_MAX_IN_FLIGHT || 8)
  },
  // 各下游超时（毫秒）。行程生成包含 LLM 重试，给得最长
  timeouts: { agentDefault: 30000, agentLong: 180000, agentPlan: 300000, amap: 8000 }
};

function validateConfig() {
  const problems = [];
  if (!Number.isInteger(config.port) || config.port <= 0 || config.port > 65535) problems.push(`PORT 非法：${process.env.PORT}`);
  try { new URL(config.agentBase); } catch { problems.push(`AGENT_BASE 不是合法 URL：${config.agentBase}`); }
  for (const [k, v] of Object.entries(config.rateLimit)) {
    if (!Number.isFinite(v) || v < 0) problems.push(`限流配置 ${k} 非法：${v}`);
  }
  if (problems.length) {
    problems.forEach(p => log('error', 'config invalid', { problem: p }));
    process.exit(1);
  }
  if (!config.amapKey) log('warn', 'AMAP_WEB_SERVICE_KEY 未设置，地图 POI/路线功能将不可用');
}

// ---------------- 结构化日志（JSON 行） ----------------
function log(level, msg, fields = {}) {
  const line = JSON.stringify({ ts: new Date().toISOString(), level, logger: 'gateway', msg, ...fields });
  (level === 'error' || level === 'warn' ? console.error : console.log)(line);
}

// ---------------- 统一错误格式（与 agent_service/core/errors.py 保持一致） ----------------
const ERRORS = {
  BAD_REQUEST: [400, false, '请求参数有误'],
  NOT_FOUND: [404, false, '资源不存在'],
  FORBIDDEN: [403, false, '没有访问权限'],
  RATE_LIMITED: [429, true, '请求太频繁，请稍后再试'],
  SERVER_BUSY: [503, true, '当前使用人数较多，请稍后再试'],
  UPSTREAM_AMAP_ERROR: [502, true, '地图服务暂时不可用'],
  AGENT_UNAVAILABLE: [503, true, '智能服务未启动或暂时不可用'],
  GATEWAY_TIMEOUT: [504, true, '请求处理超时，请稍后重试'],
  INTERNAL: [500, false, '服务内部错误']
};

function sendError(res, code, message) {
  const [status, retryable, defaultMessage] = ERRORS[code] || ERRORS.INTERNAL;
  res.locals.errorCode = code;
  res.status(status).json({
    error: { code, message: message || defaultMessage, retryable, request_id: res.locals.requestId }
  });
}

// ---------------- 限流、过载保护、指标 ----------------
const apiLimiter = new RateLimiter(config.rateLimit.apiPerMin);
const heavyLimiter = new RateLimiter(config.rateLimit.heavyPerMin);
const heavyGate = new ConcurrencyGate(config.rateLimit.heavyMaxInFlight);
const metrics = new GatewayMetrics();
setInterval(() => { apiLimiter.sweep(); heavyLimiter.sweep(); }, 60000).unref();

// 一次请求要跑几十秒、调用多次 LLM 的接口
const HEAVY_PATHS = ['/api/plan', '/api/chat', '/api/itinerary', '/api/rag/search'];

function rateLimit(limiter, tier) {
  return (req, res, next) => {
    const { ok, retryAfterS } = limiter.take(`${tier}:${req.ip}`);
    if (ok) return next();
    res.set('Retry-After', String(retryAfterS));
    log('warn', 'rate limited', { request_id: res.locals.requestId, tier, ip: req.ip, path: req.path });
    sendError(res, 'RATE_LIMITED', `请求太频繁，请 ${retryAfterS} 秒后再试`);
  };
}

function heavyConcurrency(req, res, next) {
  const release = heavyGate.tryEnter();
  if (!release) {
    log('warn', 'overloaded', { request_id: res.locals.requestId, path: req.path, in_flight: heavyGate.inFlight });
    return sendError(res, 'SERVER_BUSY');
  }
  res.on('finish', release);
  res.on('close', release); // 浏览器中途断开（如关闭 SSE）也要释放
  next();
}

// 监控接口与页面：有 ADMIN_TOKEN 时凭 token（Authorization: Bearer 或 ?token=），否则只允许本机访问
function adminOnly(req, res, next) {
  if (config.adminToken) {
    const given = (req.get('authorization') || '').replace(/^Bearer\s+/i, '') || String(req.query.token || '');
    const a = Buffer.from(given);
    const b = Buffer.from(config.adminToken);
    if (a.length === b.length && crypto.timingSafeEqual(a, b)) return next();
  } else if (isLoopback(req.socket.remoteAddress)) {
    return next();
  }
  sendError(res, 'FORBIDDEN', config.adminToken ? '监控接口需要有效的 token' : '监控接口只允许本机访问；远程访问请设置 ADMIN_TOKEN');
}

// ---------------- app ----------------
const app = express();
app.disable('x-powered-by');
app.use(cors());

// request-id + 访问日志 + 指标（放在 body 解析之前，解析失败的请求也能带上 request-id）
app.use((req, res, next) => {
  const incoming = String(req.get('x-request-id') || '');
  const requestId = /^[A-Za-z0-9-]{8,64}$/.test(incoming) ? incoming : crypto.randomUUID().replace(/-/g, '');
  res.locals.requestId = requestId;
  res.set('X-Request-ID', requestId);
  if (!req.path.startsWith('/api/')) return next();
  const started = process.hrtime.bigint();
  metrics.started();
  res.on('finish', () => {
    const latencyMs = Number((process.hrtime.bigint() - started) / 1000000n);
    // 按路由模板统计（/api/memory/:id 而不是每个 id 一行）
    const route = req.route ? req.baseUrl + req.route.path : (res.locals.errorCode === 'NOT_FOUND' ? 'unmatched' : req.path);
    metrics.finished(req.method, route, res.statusCode, latencyMs, res.locals.errorCode);
    log(res.statusCode >= 500 ? 'warn' : 'info', 'access', {
      request_id: requestId,
      method: req.method,
      path: req.path,
      status: res.statusCode,
      latency_ms: latencyMs
    });
  });
  next();
});
app.use('/api', rateLimit(apiLimiter, 'api'));
app.use(HEAVY_PATHS, rateLimit(heavyLimiter, 'heavy'), heavyConcurrency);
app.use(express.json({ limit: '1mb' }));

// 静态资源：只开放页面与前端脚本，不能把整个项目目录（配置、数据、日志、源码）暴露出去
const PAGES = ['ai-chat.html', 'output-map.html', 'rag-search.html'];
app.use('/src', express.static(path.join(__dirname, 'src')));
app.get('/', (_req, res) => res.redirect('/ai-chat.html'));
for (const page of PAGES) {
  app.get(`/${page}`, (_req, res) => res.sendFile(path.join(__dirname, page)));
}
app.get('/admin.html', adminOnly, (_req, res) => res.sendFile(path.join(__dirname, 'admin.html')));

// 前端运行配置：地图 JS API Key 本来就要下发到浏览器，放在这里是为了不写死在页面和仓库里
app.get('/api/config', (_req, res) => {
  res.json({ amapJsKey: config.amapJsKey, amapJsSecurityCode: config.amapJsSecurityCode });
});

// fetch + 超时
async function fetchWithTimeout(url, options, timeoutMs) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } finally {
    clearTimeout(timer);
  }
}

// 转发到 agent_service：透传状态码与响应体（其错误体已是统一格式），只处理网络层失败
function proxyToAgent(agentPath, timeoutMs = config.timeouts.agentDefault) {
  return async (req, res) => {
    try {
      const r = await fetchWithTimeout(`${config.agentBase}${agentPath}`, {
        method: req.method,
        headers: { 'Content-Type': 'application/json', 'X-Request-ID': res.locals.requestId },
        body: req.method === 'GET' ? undefined : JSON.stringify(req.body || {})
      }, timeoutMs);
      res.status(r.status).type('application/json').send(await r.text());
    } catch (e) {
      if (e.name === 'AbortError') {
        log('warn', 'agent timeout', { request_id: res.locals.requestId, path: agentPath, timeout_ms: timeoutMs });
        return sendError(res, 'GATEWAY_TIMEOUT');
      }
      log('error', 'agent unreachable', { request_id: res.locals.requestId, path: agentPath, error: String(e) });
      sendError(res, 'AGENT_UNAVAILABLE');
    }
  };
}

// SSE 流式转发：逐块透传 agent_service 的事件；浏览器断开时中止上游，避免继续消耗 LLM 调用
function streamFromAgent(agentPath, timeoutMs) {
  return async (req, res) => {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    res.on('close', () => controller.abort());
    let upstream;
    try {
      upstream = await fetch(`${config.agentBase}${agentPath}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-Request-ID': res.locals.requestId },
        body: JSON.stringify(req.body || {}),
        signal: controller.signal
      });
    } catch (e) {
      clearTimeout(timer);
      log('error', 'agent unreachable', { request_id: res.locals.requestId, path: agentPath, error: String(e) });
      return sendError(res, e.name === 'AbortError' ? 'GATEWAY_TIMEOUT' : 'AGENT_UNAVAILABLE');
    }
    if (!upstream.ok) {
      clearTimeout(timer);
      return res.status(upstream.status).type('application/json').send(await upstream.text());
    }
    res.status(200).set({ 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache', Connection: 'keep-alive' });
    res.flushHeaders();
    upstream.body.on('data', chunk => res.write(chunk));
    upstream.body.on('end', () => { clearTimeout(timer); res.end(); });
    upstream.body.on('error', e => {
      clearTimeout(timer);
      if (e.name !== 'AbortError') log('warn', 'agent stream error', { request_id: res.locals.requestId, error: String(e) });
      res.end();
    });
  };
}

app.post('/api/plan', proxyToAgent('/plan', config.timeouts.agentPlan));
app.post('/api/plan/stream', streamFromAgent('/plan/stream', config.timeouts.agentPlan));
app.post('/api/chat', proxyToAgent('/chat', config.timeouts.agentPlan));
app.post('/api/chat/stream', streamFromAgent('/chat/stream', config.timeouts.agentPlan));
// 长期记忆的查看与删除：转发 user_id 查询参数与记忆 id
app.get('/api/memory', (req, res) => proxyToAgent(`/memory?${new URLSearchParams({ user_id: String(req.query.user_id || '') })}`)(req, res));
app.delete('/api/memory', (req, res) => proxyToAgent(`/memory?${new URLSearchParams({ user_id: String(req.query.user_id || '') })}`)(req, res));
app.delete('/api/memory/:id', (req, res) => {
  if (!/^\d+$/.test(req.params.id)) return sendError(res, 'BAD_REQUEST', '记忆 id 不合法');
  proxyToAgent(`/memory/${req.params.id}?${new URLSearchParams({ user_id: String(req.query.user_id || '') })}`)(req, res);
});
app.post('/api/classify', proxyToAgent('/classify'));
app.post('/api/itinerary', proxyToAgent('/itinerary', config.timeouts.agentLong));
app.post('/api/rag/search', proxyToAgent('/search', config.timeouts.agentLong));
app.post('/api/rag/retrieve', proxyToAgent('/retrieve'));
// 注：RAG 入库（/ingest/csv）不经网关暴露，请在本机用 agent_service/ingest_csv.py

// 就绪检查：网关自身 + agent_service 依赖状态
app.get('/api/readyz', async (_req, res) => {
  const gateway = { ok: true, amap_configured: Boolean(config.amapKey) };
  try {
    const r = await fetchWithTimeout(`${config.agentBase}/readyz`, {}, 5000);
    const agent = await r.json();
    res.status(r.ok ? 200 : 503).json({ ready: r.ok, gateway, agent });
  } catch (e) {
    res.status(503).json({ ready: false, gateway, agent: { ready: false, error: String(e) } });
  }
});

// 监控：网关自身指标 + agent_service 指标。format=prometheus 时转发 agent_service 的 Prometheus 文本
app.get('/api/metrics', adminOnly, async (req, res) => {
  if (req.query.format === 'prometheus') {
    try {
      const r = await fetchWithTimeout(`${config.agentBase}/metrics?format=prometheus`, {}, 5000);
      return res.status(r.status).type('text/plain; version=0.0.4').send(await r.text());
    } catch (e) {
      return sendError(res, 'AGENT_UNAVAILABLE');
    }
  }
  const gateway = metrics.snapshot({
    rate_limit: {
      api_per_min: config.rateLimit.apiPerMin,
      heavy_per_min: config.rateLimit.heavyPerMin,
      rejected_api: apiLimiter.rejected,
      rejected_heavy: heavyLimiter.rejected,
      tracked_clients: apiLimiter.buckets.size
    },
    heavy_gate: { limit: heavyGate.limit, in_flight: heavyGate.inFlight, rejected: heavyGate.rejected }
  });
  let agent;
  try {
    const r = await fetchWithTimeout(`${config.agentBase}/metrics`, {}, 5000);
    agent = await r.json();
  } catch (e) {
    agent = { error: 'agent_service 不可达', detail: String(e).slice(0, 200) };
  }
  res.json({ gateway, agent });
});

// 高德 Web 服务代理：只放行地图页用到的接口，Key 由服务端追加
const AMAP_ENDPOINTS = {
  'place-text': 'https://restapi.amap.com/v3/place/text',
  'geocode': 'https://restapi.amap.com/v3/geocode/geo',
  'driving': 'https://restapi.amap.com/v3/direction/driving',
  'walking': 'https://restapi.amap.com/v3/direction/walking'
};

app.get('/api/amap/:service', async (req, res) => {
  const base = AMAP_ENDPOINTS[req.params.service];
  if (!base) return sendError(res, 'NOT_FOUND', '未知的地图接口');
  if (!config.amapKey) return sendError(res, 'UPSTREAM_AMAP_ERROR', '地图服务未配置');
  const params = new URLSearchParams();
  for (const [k, v] of Object.entries(req.query)) {
    if (k !== 'key' && typeof v === 'string') params.set(k, v);
  }
  params.set('key', config.amapKey);
  try {
    const r = await fetchWithTimeout(`${base}?${params.toString()}`, {}, config.timeouts.amap);
    const text = await r.text();
    // 高德业务错误以 HTTP 200 + status:"0" 返回，记录下来便于排查配额问题
    if (text.includes('"status":"0"')) {
      log('warn', 'amap business error', { request_id: res.locals.requestId, service: req.params.service, body: text.slice(0, 200) });
    }
    res.status(r.status).type('application/json').send(text);
  } catch (e) {
    log('error', 'amap request failed', { request_id: res.locals.requestId, service: req.params.service, error: String(e) });
    sendError(res, e.name === 'AbortError' ? 'GATEWAY_TIMEOUT' : 'UPSTREAM_AMAP_ERROR');
  }
});

// 未匹配的 /api 路由
app.use('/api', (_req, res) => sendError(res, 'NOT_FOUND'));

// 兜底错误处理（如请求体 JSON 解析失败）
app.use((err, _req, res, _next) => {
  if (err.type === 'entity.parse.failed') return sendError(res, 'BAD_REQUEST', '请求体不是合法 JSON');
  log('error', 'unhandled error', { request_id: res.locals.requestId, error: String(err), stack: err.stack });
  sendError(res, 'INTERNAL');
});

validateConfig();
app.listen(config.port, () => {
  log('info', 'gateway started', { url: `http://127.0.0.1:${config.port}`, agent_base: config.agentBase });
});
