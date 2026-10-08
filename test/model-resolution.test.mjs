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

// 2026-10-08 감사: HF에서 보이는 붙여 쓴 이름(`Qwen3.8-27B`)이 unknown이었다 — 'fuzzy names' 계약 위반.
// 고친 범위는 "영문 2자 이상 계열명 + 숫자"(qwen3·gemma4·glm5)뿐이다. 단일문자 접두(e2b·a3b·a100)와
// `org/model`은 가르지 않고, 1차(현행) 해석이 unknown일 때만 2차로 본다.
test('붙여 쓴 계열명+버전은 띄어 쓴 이름과 같은 모델로 해석된다', () => {
  const cases = [
    ['Qwen3.8-27B', 'Qwen 3.8 27B'],
    ['qwen3.8-27b', 'Qwen 3.8 27B'],
    ['qwen3.8 27b', 'Qwen 3.8 27B'],
    ['Qwen3.8-2.4T-A95B', 'Qwen 3.8 2.4T-A95B'],
    ['qwen3.6-27b', 'Qwen 3.6 27B'],
    ['QWEN3.6_27B', 'Qwen 3.6 27B'],
    ['Qwen3.6-35B-A3B', 'Qwen 3.6 35B-A3B'],
    ['gemma4-31b', 'Gemma 4 31b'],
    ['glm5.3', 'GLM-5.3'],
  ];
  for (const [glued, canonical] of cases) {
    const spaced = resolveLocalModel(canonical);
    const r = resolveLocalModel(glued);
    assert.equal(r.status, 'resolved', `${glued} 해석 실패`);
    assert.equal(r.canonicalName, canonical, glued);
    assert.equal(r.match, spaced.match, `${glued}는 띄어 쓴 이름과 같은 카탈로그 객체여야 한다`);
  }
});

test('붙여 쓴 이름도 모호·버전·단일문자 접두를 지킨다', () => {
  // 2차 패스는 정확한 전체 이름만 해석한다 — 계열·버전만 준 질의는 고르지 않는다
  assert.notEqual(resolveLocalModel('qwen3.6').status, 'resolved');
  assert.notEqual(resolveLocalModel('gemma4').status, 'resolved');
  // 1차 해석이 이미 ambiguous면 2차로 넘어가지 않는다 — Qwen3 소형 두 개 사이에서 고르지 않는다
  const amb3 = resolveLocalModel('qwen3');
  assert.equal(amb3.status, 'ambiguous');
  assert.deepEqual(amb3.candidates.map((c) => c.name).sort(), ['Qwen3-0.6B', 'Qwen3-1.7B']);
  // 소수점 버전과 붙은 숫자를 섞지 않는다: 38 ≠ 3.8, 3.8 ≠ 3.6
  assert.equal(resolveLocalModel('qwen38-27b').status, 'unknown');
  // Codex 리뷰(2026-10-08) 실측 반례 — 부분 일치를 허용하면 다른 버전·크기의 내장 모델로 둔갑했다
  for (const q of ['Qwen3-8.27B', 'MiniCPM5.5-1B', 'GLM5.2.2', 'Qwen3-35B-A3B', 'Qwen3-2.4T-A95B', 'qwen3.6 35b', 'Qwen3.8-.27B', 'Gemma4-.12b', 'qwen3.8.-27b']) {
    assert.notEqual(resolveLocalModel(q).status, 'resolved', q);
  }
  assert.notEqual(resolveLocalModel('qwen3.8-27b').canonicalName, 'Qwen 3.6 27B');
  // 단일문자 접두는 가르지 않는다 — Gemma 2 2B를 Gemma 4 E2B로 바꿔 답하면 안 된다
  assert.equal(resolveLocalModel('gemma 2b').status, 'unknown');
  assert.equal(resolveLocalModel('gemma 4b').status, 'unknown');
  // 정확 일치 우선은 그대로다
  assert.equal(resolveLocalModel('Qwen3-0.6B').canonicalName, 'Qwen3-0.6B');
  assert.equal(resolveLocalModel('Qwen3-0.6B').matchedBy, 'exact');
});

test('org/model과 파생 이름은 붙여 쓰기 해석으로 카탈로그에 끌려오지 않는다', () => {
  // CLI는 카탈로그 unknown일 때만 HF config 경로로 간다 — 이 경로를 가리면 안 된다
  for (const q of ['Qwen/Qwen3.8-27B', 'Qwen/Qwen3.6-27B', 'Qwen/Qwen3.8-Flash-Next-NVFP4']) {
    assert.equal(resolveLocalModel(q).status, 'unknown', q);
  }
  // 미지원 아키텍처·양자화 파생은 내장 모델로 둔갑하지 않는다
  for (const q of ['Qwen3.8-Flash-Next', 'Qwen3.6-27B-GGUF', 'Qwen3.8-27B-FP8', 'Qwen3.8-27B-DFlash2']) {
    assert.notEqual(resolveLocalModel(q).status, 'resolved', q);
  }
});

test('GPU 해석은 붙여 쓰기 2차 패스 뒤에도 그대로다', () => {
  const a100 = resolveGpuByName('a100');
  assert.equal(a100.status, 'ambiguous');
  assert.deepEqual(a100.candidates.map((c) => c.name).sort(), ['A100 40GB', 'A100 80GB']);
  assert.equal(resolveGpuByName('4090').canonicalName, 'RTX 4090');
  assert.equal(resolveGpuByName('h100').canonicalName, 'H100 80GB');
  assert.equal(resolveGpuByName('rtx4090').canonicalName, 'RTX 4090');
});

test('카탈로그 전 항목의 붙여 쓴 표기도 자기 자신으로만 해석된다 (2차 키 충돌 없음)', () => {
  // 'Qwen 3.6 27B' → 'Qwen3.6 27B', 'GLM-5.2' → 'GLM5.2', 'RTX 4090' → 'RTX4090' …
  const glue = (name) => name.replace(/([A-Za-z]{2,})[\s_-]+(?=\d)/g, '$1');
  let changed = 0;
  for (const [list, resolve] of [[LOCAL_MODELS, resolveLocalModel], [GPUS, resolveGpuByName]]) {
    for (const item of list) {
      const glued = glue(item.name);
      if (glued !== item.name) changed += 1;
      const r = resolve(glued);
      assert.equal(r.status, 'resolved', `${glued} (${item.name}) 해석 실패`);
      assert.equal(r.canonicalName, item.name, glued);
    }
  }
  assert.ok(changed >= 30, `붙여 쓴 표기가 실제로 만들어져야 의미가 있다 (changed=${changed})`);
});
