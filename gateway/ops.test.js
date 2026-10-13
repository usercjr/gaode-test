// 网关运维组件的单元测试：npm test
import assert from 'node:assert/strict';
import test from 'node:test';
import { ConcurrencyGate, GatewayMetrics, RateLimiter, isLoopback, percentile } from './ops.js';

test('令牌桶：允许突发到容量，之后按速率补充，并给出 Retry-After', () => {
  let now = 0;
  const rl = new RateLimiter(10, { now: () => now });
  for (let i = 0; i < 10; i++) assert.equal(rl.take('ip').ok, true);
  const denied = rl.take('ip');
  assert.equal(denied.ok, false);
  assert.equal(denied.retryAfterS, 6); // 每分钟 10 个 → 6 秒补 1 个
  assert.equal(rl.take('other-ip').ok, true); // 不同客户端互不影响
  now += 6000;
  assert.equal(rl.take('ip').ok, true);
  assert.equal(rl.rejected, 1);
});

test('令牌桶：配额为 0 表示不限流；sweep 清理已补满的桶', () => {
  assert.equal(new RateLimiter(0).take('x').ok, true);
  let now = 0;
  const rl = new RateLimiter(60, { now: () => now });
  rl.take('a');
  now += 60000;
  rl.sweep();
  assert.equal(rl.buckets.size, 0);
});

test('并发闸门：超过上限拒绝，释放可重复调用但只生效一次', () => {
  const gate = new ConcurrencyGate(2);
  const r1 = gate.tryEnter();
  const r2 = gate.tryEnter();
  assert.equal(gate.tryEnter(), null);
  assert.equal(gate.rejected, 1);
  r1();
  r1(); // finish 和 close 都会触发释放
  assert.equal(gate.inFlight, 1);
  assert.ok(gate.tryEnter());
  r2();
});

test('指标：按路由统计次数、错误与分位数', () => {
  const m = new GatewayMetrics();
  for (const ms of [10, 20, 30, 40]) {
    m.started();
    m.finished('POST', '/api/chat', 200, ms);
  }
  m.started();
  m.finished('POST', '/api/chat', 503, 1, 'SERVER_BUSY');
  const s = m.snapshot({ extra: 1 });
  assert.equal(s.requests_total, 5);
  assert.equal(s.errors_5xx_total, 1);
  assert.equal(s.in_flight, 0);
  assert.equal(s.routes['POST /api/chat'].latency_ms.max, 40);
  assert.deepEqual(s.error_codes, { SERVER_BUSY: 1 });
  assert.equal(s.extra, 1);
  assert.equal(percentile([], 50), null);
});

test('isLoopback 只认本机地址', () => {
  assert.ok(isLoopback('127.0.0.1') && isLoopback('::1') && isLoopback('::ffff:127.0.0.1'));
  assert.ok(!isLoopback('172.18.0.1') && !isLoopback('192.168.1.5'));
});
