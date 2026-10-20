// GB10 한 대 — 공개 실측(fixtures/measured.json)과 계산의 대조가 회귀 기준이다 (2.21.4 보정).
// 실측 값은 런타임이 직접 찍은 할당(llama.cpp 버퍼 로그 합 · Ollama 할당 요약 · nvidia-smi 프로세스)이라
// OS 몫이 없다. 계산 쪽은 같은 범위로 맞춘다: 사용량 − OS 몫 = 가중치 + KV + 상태 + 런타임 동적 몫 + reserve의 런타임 2 GiB.
// ① 계산이 실측 아래로 내려가지 않는다(거짓 '들어감' 방지) ② 계산이 실측보다 7% 넘게 크지 않다
//    (2.21.3까지는 가중치 12% 몫 때문에 +11~18%였다 — 이 시험은 옛 식에서 네 건 모두 실패한다).
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { GPUS, LOCAL_MODELS, GPU_QUANTS, gpuDevice, appleDevice, simulate, calcRuntimeOverhead } from '../engine.js';

const spark = GPUS.find((g) => g.name === 'DGX Spark 128GB');
const device = gpuDevice(spark);
const measured = JSON.parse(readFileSync(new URL('../fixtures/measured.json', import.meta.url), 'utf8'));
const gb10 = measured.filter((m) => m.device === spark.name);
const bpwOf = (quant) => GPU_QUANTS.find((q) => q.tier === quant).bpw;
const OS_SHARE = device.reserveGB - spark.unifiedMemory.runtimeFixedGB; // 128 − 121.69 + 5.0 = 11.31

function predictedProcess(m) {
  const model = LOCAL_MODELS.find((x) => x.name === m.model);
  assert.ok(model, `unknown model ${m.model}`);
  const s = simulate(model, device, m.ctx, { weightBpw: bpwOf(m.quant), kvBits: m.kvBits });
  return { s, value: s.used - OS_SHARE };
}

test('원장: GB10 실측 4건 — 타입·단위·검증 등급·원본 해시', () => {
  assert.equal(gb10.length, 4);
  for (const m of gb10) {
    assert.equal(m.measurementKind, 'generation_peak');
    assert.equal(m.unit, 'GiB');
    assert.equal(m.includesOsReserve, false);
    assert.ok(m.caseId && m.modelRevision && m.verifiedBy && m.reporter && m.reporter !== m.verifiedBy, m.caseId);
  }
  assert.equal(gb10.filter((m) => m.evidenceLevel === 'maintainer_verified').length, 3);
  // 로그 첨부가 있는 세 건은 원본 sha256을 단다(포럼 글은 붙여 넣은 nvidia-smi 출력이 원본)
  assert.equal(gb10.filter((m) => /^[0-9a-f]{64}$/.test(m.rawOutputSha256 || '')).length, 3);
});

test('계산은 실측 아래로 내려가지 않고, 7% 넘게 크지도 않다', () => {
  for (const m of gb10) {
    const { value } = predictedProcess(m);
    assert.ok(value >= m.measuredPeakGB, `${m.caseId}: predicted ${value.toFixed(2)} < measured ${m.measuredPeakGB}`);
    assert.ok(value <= m.measuredPeakGB * 1.07, `${m.caseId}: predicted ${value.toFixed(2)} > measured ${m.measuredPeakGB} + 7%`);
  }
});

test('손계산: GX10 · Qwen3.6-35B-A3B BF16 · 65,536 · KV 8비트', () => {
  const m = gb10.find((x) => x.caseId === 'gb10-qwen36-35b-a3b-bf16-ctx65536-kvq8');
  const { s, value } = predictedProcess(m);
  // 가중치 35.0B(카탈로그) × 16.0005/8 = 65.195 GiB · KV 10층 × 2 × 2헤드 × 256 × 1 B × 65,536 = 0.625 GiB · 상태 0.060
  // 런타임 = 0.15 × 0.625 + 65,536 × 0.00003 = 0.094 + 1.966 = 2.060 (가중치 몫 0) → 67.94 + 2.0 = 69.94 GiB
  assert.equal(+s.param.toFixed(3), 65.195);
  assert.equal(+s.kv.toFixed(3), 0.625);
  assert.equal(+s.rtDyn.toFixed(3), 2.06);
  assert.equal(+value.toFixed(2), 69.94);
  assert.equal(+((value / m.measuredPeakGB - 1) * 100).toFixed(1), 4.6); // 2.21.3: 77.76 (+16.3%)
});

test('가중치 12% 몫: GB10 한 대만 0, 클러스터·디스크리트·맥은 그대로', () => {
  const qwen = LOCAL_MODELS.find((x) => x.name === 'Qwen 3.6 35B-A3B');
  const q = { weightBpw: 8.5008, kvBits: 16 };
  const one = calcRuntimeOverhead(qwen, 8192, q, device);
  assert.equal(one.paramOverheadGB, 0);
  for (const name of ['2× DGX Spark 128GB', '4× DGX Spark 128GB', 'RTX PRO 6000 Blackwell', 'RTX 4090']) {
    const d = gpuDevice(GPUS.find((g) => g.name === name));
    const ov = calcRuntimeOverhead(qwen, 8192, q, d);
    const s = simulate(qwen, d, 8192, q);
    assert.ok(Math.abs(ov.paramOverheadGB - s.param * 0.12) < 1e-12, name);
  }
  const mac = calcRuntimeOverhead(qwen, 8192, { weightBpw: 8, kvBits: 16 }, appleDevice(64));
  assert.ok(mac.paramOverheadGB > 0);
});
