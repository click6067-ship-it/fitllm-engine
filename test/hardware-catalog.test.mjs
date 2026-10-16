// 하드웨어 카탈로그 게이트 — 출처 없는 숫자는 머지 불가(AGENTS.md 규칙 1).
// v2(src/lib/engine.gpu.test.js)에 있던 무결성 검사를 공개 엔진에도 둔다: 하드웨어 숫자는 여기서 먼저 들어온다.
import test from 'node:test';
import assert from 'node:assert/strict';
import { GPUS, resolveGpuByName } from '../engine.js';

const host = (u) => new URL(u).hostname.replace(/^www\./, '');

test('GPUS 전 행: 수치 > 0 · 출처 ≥2 독립 도메인 · VERIFIED · verifiedAt', () => {
  for (const g of GPUS) {
    assert.ok(g.vramGB > 0 && g.bandwidthGBs > 0, g.name);
    assert.equal(g.status, 'VERIFIED', g.name);
    assert.match(g.verifiedAt, /^\d{4}-\d{2}-\d{2}$/, g.name);
    for (const field of ['vramGB', 'bandwidthGBs']) {
      const srcs = g.sources?.[field] || [];
      assert.ok(new Set(srcs.map(host)).size >= 2, `${g.name} ${field}: needs 2 independent domains`);
    }
  }
});

// 2026-10-16 추가분 — 두 트랙(Claude hw-crawler ∥ Codex) 블라인드 리콘실 값. 바뀌면 출처부터 다시 본다.
const ADDED_2026_10 = {
  'RTX 5060 Ti 8GB': [8, 448],
  'RTX 5060': [8, 448],
  'RTX 5050': [8, 320],
  'RTX PRO 5000 Blackwell 48GB': [48, 1344],
  'RTX PRO 5000 Blackwell 72GB': [72, 1344],
  'RTX PRO 4500 Blackwell': [32, 896],
  'RTX PRO 4000 Blackwell': [24, 672],
  'RTX PRO 2000 Blackwell': [16, 288],
  'DGX Spark 128GB': [128, 273],
};

test('2026-10 추가 하드웨어: VRAM·대역폭 앵커', () => {
  for (const [name, [vram, bw]] of Object.entries(ADDED_2026_10)) {
    const g = GPUS.find((x) => x.name === name);
    assert.ok(g, name);
    assert.equal(g.vramGB, vram, `${name} vram`);
    assert.equal(g.bandwidthGBs, bw, `${name} bandwidth`);
    assert.equal(g.verifiedAt, '2026-10-16', name);
  }
  assert.equal(GPUS.length, 45);
});

test('출시 전·루머 제품은 카탈로그에 없다 (RTX 50 SUPER · PRO 5500 · DGX Spark 64GB)', () => {
  for (const q of ['RTX 5080 SUPER', 'RTX 5070 Ti SUPER', 'RTX 5070 SUPER', 'RTX PRO 5500', 'DGX Spark 64GB']) {
    assert.notEqual(resolveGpuByName(q).status, 'resolved', q);
  }
});

test('용량만 다른 신규 SKU는 모호로 남고, 기반 이름은 그 제품으로 해석된다', () => {
  const ambiguous = {
    '5060 ti': ['RTX 5060 Ti 16GB', 'RTX 5060 Ti 8GB'],
    'pro 5000': ['RTX PRO 5000 Blackwell 48GB', 'RTX PRO 5000 Blackwell 72GB'],
  };
  for (const [q, names] of Object.entries(ambiguous)) {
    const r = resolveGpuByName(q);
    assert.equal(r.status, 'ambiguous', q);
    assert.deepEqual(r.candidates.map((c) => c.name).sort(), names, q);
  }
  // 'RTX 5060'은 Ti 행들이 확장하는 최소 이름 → 그 카드(예전에는 유일 후보였던 5060 Ti 16GB로 해석됐다)
  assert.equal(resolveGpuByName('5060').canonicalName, 'RTX 5060');
  assert.equal(resolveGpuByName('rtx5050').canonicalName, 'RTX 5050');
});

test('같은 VRAM 에디션은 별칭으로 정본 행에 해석된다', () => {
  const cases = {
    'RTX PRO 6000 Blackwell Max-Q': 'RTX PRO 6000 Blackwell',
    'rtx pro 6000 server edition': 'RTX PRO 6000 Blackwell',
    'RTX PRO 6000 Blackwell Workstation Edition': 'RTX PRO 6000 Blackwell',
    'RTX PRO 4500 Blackwell Server Edition': 'RTX PRO 4500 Blackwell',
    'RTX PRO 4000 SFF': 'RTX PRO 4000 Blackwell',
    'rtx pro 6000 max-q': 'RTX PRO 6000 Blackwell',
    'RTX PRO 4500 Server Edition': 'RTX PRO 4500 Blackwell',
  };
  for (const [q, want] of Object.entries(cases)) {
    const r = resolveGpuByName(q);
    assert.equal(r.status, 'resolved', q);
    assert.equal(r.canonicalName, want, q);
  }
  // 별칭이 있어도 기존 모호성은 그대로다
  assert.equal(resolveGpuByName('6000').status, 'ambiguous');
  assert.equal(resolveGpuByName('a100').status, 'ambiguous');
});

// Codex 2차 리뷰(2026-10-16) 회귀: 이름 일부만으로 다른 세대·다른 기기를 확정하면 거짓 FITS가 난다.
test('세대 식별자 없는 번호·용량만·제조사 계열명만으로는 확정하지 않는다', () => {
  // 'RTX 4500/4000/2000/5000'은 Ada·Quadro 카드 이름이다(24/20·8/16/32GB) — PRO Blackwell로 끌려오면 안 된다
  // 용량 토큰만('48gb'·'128gb'·'141gb')·'dgx'만으로도 고르지 않는다
  for (const q of ['RTX 4500', 'rtx 4000', 'rtx 2000', 'rtx 5000', '4500', '48gb', '72gb', '128gb', '141gb', '40gb', 'dgx', 'dgx 128gb', '5060 8gb',
    // Codex 3차: 확정 자격(matchRequires)이 후보를 지우면 남은 한 행이 확정된다 — 계열명·용량만으로는 모호/unknown이어야 한다
    'blackwell', 'rtx blackwell', 'pro blackwell', 'rtx pro', 'pro 48gb', 'rtx pro 48gb', 'pro blackwell 72gb']) {
    assert.notEqual(resolveGpuByName(q).status, 'resolved', q);
  }
  assert.equal(resolveGpuByName('blackwell').status, 'ambiguous');
  assert.ok(resolveGpuByName('blackwell').total >= 6);
  // 세대를 밝히면 해석된다
  const ok = {
    'pro 4500': 'RTX PRO 4500 Blackwell',
    'rtx pro 4000': 'RTX PRO 4000 Blackwell',
    'pro 2000 blackwell': 'RTX PRO 2000 Blackwell',
    'pro 5000 72gb': 'RTX PRO 5000 Blackwell 72GB',
    'dgx spark': 'DGX Spark 128GB',
    'RTX 5060 8GB': 'RTX 5060', // 판매점 표기 — Ti가 아니다
    '5060 ti 8gb': 'RTX 5060 Ti 8GB',
    'h200': 'H200 141GB',
  };
  for (const [q, want] of Object.entries(ok)) assert.equal(resolveGpuByName(q).canonicalName, want, q);
});
