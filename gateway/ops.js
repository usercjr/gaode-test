// 网关的运维组件：按 IP 的令牌桶限流、重请求并发闸门、进程内指标。
// 单进程网关，状态放内存即可；多实例部署时限流需要换成 Redis 等共享存储。

// ---------------- 令牌桶限流 ----------------
// 每个 key（IP + 档位）一个桶：容量 = 每分钟配额，按时间匀速补充。允许短时突发，长期速率受限。
export class RateLimiter {
  constructor(perMinute, { now = () => Date.now() } = {}) {
    this.capacity = perMinute;
    this.refillPerMs = perMinute / 60000;
    this.now = now;
    this.buckets = new Map();
    this.rejected = 0;
  }

  // 返回 { ok, retryAfterS }
  take(key) {
    if (this.capacity <= 0) return { ok: true, retryAfterS: 0 }; // 0 表示不限流
    const t = this.now();
    let b = this.buckets.get(key);
    if (!b) {
      b = { tokens: this.capacity, at: t };
      this.buckets.set(key, b);
    }
    b.tokens = Math.min(this.capacity, b.tokens + (t - b.at) * this.refillPerMs);
    b.at = t;
    if (b.tokens >= 1) {
      b.tokens -= 1;
      return { ok: true, retryAfterS: 0 };
    }
    this.rejected += 1;
    return { ok: false, retryAfterS: Math.ceil((1 - b.tokens) / this.refillPerMs / 1000) };
  }

  // 清理已经补满的桶，避免 IP 越来越多导致内存增长
  sweep() {
    const t = this.now();
    for (const [key, b] of this.buckets) {
      if (b.tokens + (t - b.at) * this.refillPerMs >= this.capacity) this.buckets.delete(key);
    }
  }
}

// ---------------- 并发闸门（过载保护） ----------------
// 规划 / 聊天一次要几十秒、多次 LLM 调用。超过上限直接返回“繁忙”，而不是让所有请求一起变慢直到超时。
export class ConcurrencyGate {
  constructor(limit) {
    this.limit = limit;
    this.inFlight = 0;
    this.rejected = 0;
  }

  tryEnter() {
    if (this.limit > 0 && this.inFlight >= this.limit) {
      this.rejected += 1;
      return null;
    }
    this.inFlight += 1;
    let released = false;
    return () => { // 幂等释放：finish 和 close 都可能触发
      if (!released) {
        released = true;
        this.inFlight -= 1;
      }
    };
  }
}

// ---------------- 指标 ----------------
const WINDOW = 2000;

export function percentile(values, p) {
  if (!values.length) return null;
  const s = [...values].sort((a, b) => a - b);
  const k = Math.min(s.length - 1, Math.max(0, Math.round((p / 100) * (s.length - 1))));
  return s[k];
}

export class GatewayMetrics {
  constructor() {
    this.startedAt = Date.now();
    this.inFlight = 0;
    this.routes = new Map(); // "METHOD route" -> { count, errors4xx, errors5xx, latencies[] }
    this.errorCodes = new Map();
  }

  started() {
    this.inFlight += 1;
  }

  finished(method, route, status, latencyMs, errorCode) {
    this.inFlight -= 1;
    const key = `${method} ${route}`;
    let r = this.routes.get(key);
    if (!r) {
      r = { count: 0, errors4xx: 0, errors5xx: 0, latencies: [] };
      this.routes.set(key, r);
    }
    r.count += 1;
    if (status >= 500) r.errors5xx += 1;
    else if (status >= 400) r.errors4xx += 1;
    r.latencies.push(latencyMs);
    if (r.latencies.length > WINDOW) r.latencies.shift();
    if (errorCode) this.errorCodes.set(errorCode, (this.errorCodes.get(errorCode) || 0) + 1);
  }

  snapshot(extra = {}) {
    const routes = {};
    let total = 0;
    let errors5xx = 0;
    for (const [key, r] of [...this.routes].sort()) {
      total += r.count;
      errors5xx += r.errors5xx;
      routes[key] = {
        count: r.count,
        errors_4xx: r.errors4xx,
        errors_5xx: r.errors5xx,
        latency_ms: {
          p50: percentile(r.latencies, 50),
          p95: percentile(r.latencies, 95),
          p99: percentile(r.latencies, 99),
          max: r.latencies.length ? Math.max(...r.latencies) : null
        }
      };
    }
    return {
      uptime_s: Math.round((Date.now() - this.startedAt) / 1000),
      in_flight: this.inFlight,
      requests_total: total,
      errors_5xx_total: errors5xx,
      routes,
      error_codes: Object.fromEntries([...this.errorCodes].sort((a, b) => b[1] - a[1])),
      ...extra
    };
  }
}

// 只认本机回环地址（含 IPv4 映射到 IPv6 的形式）
export function isLoopback(ip = '') {
  return ip === '127.0.0.1' || ip === '::1' || ip === '::ffff:127.0.0.1';
}
