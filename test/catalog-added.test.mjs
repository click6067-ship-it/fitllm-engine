// 카탈로그 행의 added(처음 들어온 날) — 계산에 안 쓰는 메타데이터지만 fitllm.run '새 모델' 띠가 이 값 하나로 최근 행을 뽑는다.
// 새 행이 날짜 없이 들어오면 띠에서 조용히 빠지므로 여기서 막는다.
import test from 'node:test';
import assert from 'node:assert/strict';
import { MODELS, simulate, LOCAL_MODELS } from '../engine.js';

const ISO_DAY = /^\d{4}-\d{2}-\d{2}$/;

test('every catalog row carries a real calendar added date', () => {
  for (const m of MODELS) {
    assert.match(m.added ?? '', ISO_DAY, `${m.name}: added must be YYYY-MM-DD`);
    const d = new Date(`${m.added}T00:00:00Z`);
    assert.equal(d.toISOString().slice(0, 10), m.added, `${m.name}: ${m.added} is not a calendar date`);
    assert.ok(m.added >= '2026-05-25', `${m.name}: before the v2 catalog existed`);
  }
});

test('rows appended after the 2026-07-09 reorder never go back in time', () => {
  // ?m= 링크 때문에 배열은 append-only — 2026-07-09 재정렬 이후 붙은 행은 앞 행보다 이른 날짜일 수 없다.
  const tail = MODELS.slice(MODELS.findIndex((m) => m.name === 'Hy3'));
  for (let i = 1; i < tail.length; i++) {
    assert.ok(tail[i].added >= tail[i - 1].added, `${tail[i].name} (${tail[i].added}) < ${tail[i - 1].name} (${tail[i - 1].added})`);
  }
});

test('added is metadata only — the verdict ignores it', () => {
  for (const m of LOCAL_MODELS) {
    const { added, ...rest } = m;
    const verdict = (row) => JSON.stringify({ ...simulate(row, 64, 32768, 4), model: undefined });
    assert.equal(verdict(m), verdict(rest), m.name);
  }
});

// officialParams(2.21.3) — 모델 카드가 공표한 총 파라미터(B). '새 모델' 띠의 크기 표기용이고 계산은 totalParams를 쓴다.
test('officialParams is a card number or an explicit null, never a computed input', () => {
  for (const m of MODELS) {
    if (!('officialParams' in m)) continue;
    const v = m.officialParams;
    assert.ok(v === null || (Number.isFinite(v) && v > 0), `${m.name}: officialParams must be null or a positive number`);
  }
  const card = Object.fromEntries(MODELS.filter((m) => 'officialParams' in m).map((m) => [m.name, m.officialParams]));
  assert.deepEqual(card, {
    'GLM-5.3': 744, 'MiniCPM5-2B': 2.517, 'Nex-N2.5-mini': null, 'Nex-N2.5-Pro': null,
    'DeepSeek-V4.1-Flash-1017': 552, 'DeepSeek-V4-Flash': 284, 'DeepSeek-V4-Pro': 1600, 'Muse Spark Open': 816,
  });
  for (const m of LOCAL_MODELS) {
    const { officialParams, ...rest } = m;
    const verdict = (row) => JSON.stringify({ ...simulate(row, 512, 131072, 4), model: undefined });
    assert.equal(verdict(m), verdict(rest), m.name);
  }
});
