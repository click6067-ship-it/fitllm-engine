// 표면 정합 불변식 게이트 — "한 진실, 두 투영"의 결정론 강제 장치.
// 1) 엔진 방정식: used == param + kv + rtDyn + reserve (모든 플랫폼·아키텍처)
// 2) CLI --json breakdown 합 == usedGB (Apple 고정 2GB 이중표시 회귀 방지)
// 3) fixtures/measured.json 스키마 준수 (measurementKind 필수 — 타입 없는 실측 금지)
// 4) census: predicted_total_to_run_gb == used_gb, 실측 붙은 행은 measurement_kind 필수
import { test } from 'node:test';
import { strict as assert } from 'node:assert';
import { execFileSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { LOCAL_MODELS, GPUS, simulate, gpuDevice } from '../engine.js';

const BIN = new URL('../bin/fitllm.mjs', import.meta.url).pathname;

test('simulate: used == param + kv + rtDyn + reserve on every model × platform', () => {
  const gpu = GPUS.find((g) => g.name === 'RTX 4090');
  for (const m of LOCAL_MODELS) {
    for (const [label, dev, quant] of [
      ['mac128-4bit', 128, { weightBpw: 4, kvBits: 16 }],
      ['mac36-8bit', 36, { weightBpw: 8, kvBits: 16 }],
      ['4090-q4', gpuDevice(gpu), { weightBpw: 4.85, kvBits: 16 }],
    ]) {
      const s = simulate(m, dev, Math.min(8192, m.maxContext), quant);
      const sum = s.param + s.kv + s.rtDyn + s.reserve;
      assert.ok(Math.abs(sum - s.used) < 1e-9, `${m.name} @ ${label}: breakdown sum ${sum} != used ${s.used}`);
    }
  }
});

test('negative/NaN ctx cannot flip verdict via negative KV (public-input guard)', () => {
  const m = LOCAL_MODELS.find((x) => x.name === 'gpt-oss-120b');
  const gpu = gpuDevice(GPUS.find((g) => g.name === 'RTX 4090'));
  const bad = simulate(m, gpu, -1e9, { weightBpw: 4.85, kvBits: 16 });
  assert.ok(bad.used > 0, `used ${bad.used} — 음수 ctx가 통과함`);
  assert.ok(bad.kv >= 0);
  assert.equal(bad.verdict, simulate(m, gpu, 1, { weightBpw: 4.85, kvBits: 16 }).verdict);
  const nan = simulate(m, gpu, 'abc', { weightBpw: 4.85, kvBits: 16 });
  assert.ok(nan.used > 0 && nan.kv >= 0);
});

test('CLI --json: breakdown fields sum to usedGB (Apple path — the 2GB double-display regression)', () => {
  const out = execFileSync(process.execPath, [BIN, 'Qwen 3.6 27B', '--mac', '128', '--quant', '4', '--ctx', '32768', '--json'], { encoding: 'utf8' });
  const j = JSON.parse(out);
  const sum = j.breakdown.paramGB + j.breakdown.kvGB + j.breakdown.overheadGB + j.breakdown.reserveGB;
  assert.ok(Math.abs(sum - j.usedGB) < 0.05, `CLI breakdown sum ${sum} != usedGB ${j.usedGB}`); // 반올림 오차만 허용
});

test('fixtures/measured.json: every entry typed and schema-conformant', () => {
  const KINDS = ['idle_resident', 'load_peak', 'generation_peak', 'system_total_peak', 'unknown'];
  const data = JSON.parse(readFileSync(new URL('../fixtures/measured.json', import.meta.url), 'utf8'));
  assert.ok(data.length > 0);
  for (const e of data) {
    for (const k of ['model', 'device', 'quant', 'ctx', 'kvBits', 'measuredPeakGB', 'measurementKind', 'runtime', 'source', 'date']) {
      assert.ok(k in e, `${e.model || '?'}: missing required field ${k}`);
    }
    assert.ok(KINDS.includes(e.measurementKind), `${e.model}: bad measurementKind ${e.measurementKind}`);
    if (e.unit != null) assert.ok(['GiB', 'GB'].includes(e.unit), `${e.model}: bad unit ${e.unit}`);
    if (e.evidenceLevel != null) assert.ok(['maintainer_verified', 'community_unverified'].includes(e.evidenceLevel), `${e.model}: bad evidenceLevel`);
    assert.ok(e.measuredPeakGB > 0);
    assert.match(e.date, /^\d{4}-\d{2}-\d{2}$/);
    assert.match(e.source, /^https:\/\//);
  }
});

test('census: predicted_total_to_run_gb == used_gb, measured rows carry measurement_kind', () => {
  const census = JSON.parse(readFileSync(new URL('./census-v1.json', import.meta.url.replace('/test/', '/census/')), 'utf8'));
  assert.ok(census.definitions, 'census header must ship column definitions');
  for (const r of census.data) {
    assert.equal(r.predicted_total_to_run_gb, r.used_gb, `${r.model}|${r.device}: total alias mismatch`);
    if (r.measured_peak_gb != null) {
      assert.ok(r.measurement_kind, `${r.model}|${r.device}: measured value without measurement_kind`);
      assert.ok(r.measurement_match, `${r.model}|${r.device}: measured value without measurement_match`);
    }
  }
});
