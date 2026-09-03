// 이름 해석 계약 — 계획서 AC-1/AC-2. CLI·REST·MCP·badge·receipt가 모두 이 정본만 쓴다.
// v2 미러(src/lib/model-resolution.test.js)와 같은 계약.
//
// 계기(2026-09-03 실측): 표면마다 `.includes()` 첫-일치 matcher가 따로 있었고, 카탈로그 배열
// 순서가 답을 결정했다. 'llama'는 후보 3개 중 3B, 'gemma'는 후보 6개 중 가장 작은 e2b가 뽑혔다.
// 작은 모델은 메모리를 덜 먹으니 판정이 fits 쪽으로 기운다 — 거짓 FITS와 같은 방향의 실패다.
import test from 'node:test';
import assert from 'node:assert/strict';
import {
  GPUS, LOCAL_MODELS, normalizeNameTokens, resolveByName, resolveGpuByName, resolveLocalModel,
} from '../engine.js';

test('정확한 이름은 표기 차이와 무관하게 해석된다', () => {
  for (const q of ['GLM-4.7-Flash', 'glm-4.7-flash', 'GLM 4.7 Flash', '  glm_4_7_flash  ']) {
    const r = resolveLocalModel(q);
    assert.equal(r.status, 'resolved');
    assert.equal(r.canonicalName, 'GLM-4.7-Flash');
  }
});

test('질의 토큰 전체를 품는 후보가 유일하면 해석된다', () => {
  assert.equal(resolveLocalModel('gemma 31b').canonicalName, 'Gemma 4 31b');
  assert.equal(resolveLocalModel('qwen 3.6 35b').canonicalName, 'Qwen 3.6 35B-A3B');
});

test('모호한 질의는 첫 항목을 고르지 않고 후보를 돌려준다', () => {
  for (const q of ['llama', 'gemma', 'qwen', 'glm']) {
    const r = resolveLocalModel(q);
    assert.equal(r.status, 'ambiguous', `'${q}'가 해석되면 안 된다`);
    assert.ok(r.total > 1);
    assert.equal(r.match, undefined, '모호할 때 match를 노출하면 호출자가 임의 선택을 하게 된다');
  }
});

test('용량만 다른 동명 GPU는 모호로 남는다 — 잘못 고르면 판정이 뒤집힌다', () => {
  const r = resolveGpuByName('a100');
  assert.equal(r.status, 'ambiguous');
  assert.deepEqual(r.candidates.map((c) => c.name).sort(), ['A100 40GB', 'A100 80GB']);
});

test('리그 항목이 있어도 광고된 단일 카드 질의는 유지된다 (base-name 규칙)', () => {
  for (const [q, want] of [['4090', 'RTX 4090'], ['3090', 'RTX 3090'], ['5090', 'RTX 5090'], ['7900 xtx', 'RX 7900 XTX']]) {
    assert.equal(resolveGpuByName(q).canonicalName, want, `gpu=${q}`);
  }
});

test('아무것도 못 맞추면 unknown이고 임의 선택은 없다', () => {
  assert.equal(resolveLocalModel('nonexistent-model-xyz').status, 'unknown');
  for (const empty of ['', null, undefined, '   ', '!!!']) {
    const r = resolveLocalModel(empty);
    assert.equal(r.status, 'unknown');
    assert.equal(r.total, 0);
  }
});

test('normalizeNameTokens는 NFKC·소문자·영숫자 토큰만 남긴다', () => {
  assert.deepEqual(normalizeNameTokens('Llama-3.1-8B-Instruct'), ['llama', '3', '1', '8b', 'instruct']);
  assert.deepEqual(normalizeNameTokens('2× RTX 4090'), ['2', 'rtx', '4090']);
  assert.ok(normalizeNameTokens('qwen chat').includes('chat'));
});

test('카탈로그 전 항목이 자기 이름으로 정확히 해석된다 (자기동일성)', () => {
  for (const m of LOCAL_MODELS) {
    assert.equal(resolveLocalModel(m.name).canonicalName, m.name, `${m.name} 자기 해석 실패`);
  }
  for (const g of GPUS) {
    assert.equal(resolveGpuByName(g.name).canonicalName, g.name, `${g.name} 자기 해석 실패`);
  }
});

test('resolveByName은 limit을 지킨다', () => {
  const r = resolveByName(GPUS, 'rtx', { limit: 2 });
  assert.equal(r.status, 'ambiguous');
  assert.equal(r.candidates.length, 2);
  assert.ok(r.total > 2);
});
