// GB10 한 대 — 공개 로그의 할당 값(fixtures/measured.json, allocation_floor)이 회귀 기준이다 (2.21.4 보정).
// 로그·nvidia-smi 값은 '바닥'이다: CUDA 컨텍스트·호스트 메모리(CPU 비전 인코더, 프롬프트 캐시, 체크포인트)처럼
// 출처가 안 보여 주는 몫과 샘플링하지 않은 최대치가 빠져 있다. 그래서 두 가지만 고정한다.
// ① 바닥: 계산(사용량 − OS 몫 = 가중치 + KV + 상태 + 런타임 동적 몫 + 런타임 2 GiB)이 할당 값 아래로 내려가지 않는다.
// ② 가중치: 로그가 모델 버퍼를 다 보여 주는 행에서, 계산한 상주 가중치가 로그의 모델 버퍼와 ±3% 안이다.
//    2.21.3까지는 가중치에 12%를 더 얹어 +11.3%였다 — 이 시험은 옛 식에서 실패한다.
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

function predict(m) {
  const model = LOCAL_MODELS.find((x) => x.name === m.model);
  assert.ok(model, `unknown model ${m.model}`);
  const q = { weightBpw: bpwOf(m.quant), kvBits: m.kvBits };
  const s = simulate(model, device, m.ctx, q);
  const ov = calcRuntimeOverhead(model, s.ctx, q, device);
  return { s, process: s.used - OS_SHARE, residentWeights: s.param + ov.paramOverheadGB };
}

test('원장: GB10 4건은 allocation_floor — 범위·단위·검증 등급·원본 해시', () => {
  assert.equal(gb10.length, 4);
  for (const m of gb10) {
    assert.equal(m.measurementKind, 'allocation_floor', m.caseId);
    assert.ok(m.scope && /floor/.test(m.scope), `${m.caseId}: scope must say it is a floor`);
    assert.equal(m.unit, 'GiB');
    assert.equal(m.includesOsReserve, false);
    assert.ok(m.caseId && m.modelRevision && m.verifiedBy && m.reporter && m.reporter !== m.verifiedBy, m.caseId);
    if (m.allocationsMiB) { // 로그 줄의 합이 기록한 값과 같다
      const sum = Object.values(m.allocationsMiB).reduce((a, b) => a + b, 0) / 1024;
      assert.equal(+sum.toFixed(2), m.measuredPeakGB, m.caseId);
    }
  }
  assert.equal(gb10.filter((m) => m.evidenceLevel === 'maintainer_verified').length, 3);
  assert.equal(gb10.filter((m) => /^[0-9a-f]{64}$/.test(m.rawOutputSha256 || '')).length, 3); // 포럼 글은 붙여 넣은 출력이 원본
});

test('바닥: 계산은 어느 할당 값보다도 작지 않다', () => {
  for (const m of gb10) {
    const { process } = predict(m);
    assert.ok(process >= m.measuredPeakGB, `${m.caseId}: predicted ${process.toFixed(2)} < logged ${m.measuredPeakGB}`);
  }
});

test('가중치: 로그의 모델 버퍼(= GGUF 파일 크기)와 ±3% 안 — 가중치를 두 번 세지 않는다', () => {
  const rows = gb10.filter((m) => m.allocationsMiB?.model && m.evidenceLevel === 'maintainer_verified');
  assert.equal(rows.length, 2);
  for (const m of rows) {
    const logged = m.allocationsMiB.model / 1024; // 66,221.98 + 970.00 MiB = 65.62 GiB
    const { residentWeights } = predict(m);
    const err = residentWeights / logged - 1;
    assert.ok(Math.abs(err) <= 0.03, `${m.caseId}: weights ${residentWeights.toFixed(2)} vs logged ${logged.toFixed(2)} (${(err * 100).toFixed(1)}%)`);
  }
});

test('손계산: GX10 · Qwen3.6-35B-A3B BF16 · 65,536 · KV 8비트', () => {
  const m = gb10.find((x) => x.caseId === 'gb10-qwen36-35b-a3b-bf16-ctx65536-kvq8');
  const { s, process } = predict(m);
  // 가중치 = 공개 BF16 GGUF 하한 bartowski/Qwen_Qwen3.6-35B-A3B-GGUF@5c2410d 71,065,941,536 B = 66.185 GiB(MTP 층 포함 — 로그 파일 65.62 GiB보다
  //   0.9% 크다; 공식 35.0B × 16.0005/8 = 65.195는 그 파일보다 1.5% 작다) · KV 10층 × 2 × 2헤드 × 256 × 1 B × 65,536 = 0.625 GiB · 상태 0.060
  // 런타임 = 0.15 × 0.625 + 65,536 × 0.00003 = 0.094 + 1.966 = 2.060 (가중치 몫 0) → 68.93 + 런타임 2.0 = 70.93 GiB
  assert.equal(+s.param.toFixed(3), 66.185);
  assert.equal(+s.kv.toFixed(3), 0.625);
  assert.equal(+s.rtDyn.toFixed(3), 2.06);
  assert.equal(+process.toFixed(2), 70.93); // 2.21.3: 77.76 · 2.21.4: 69.94
});

test('가중치 12% 몫: GB10 한 대만 0, 클러스터·디스크리트·맥은 그대로', () => {
  const qwen = LOCAL_MODELS.find((x) => x.name === 'Qwen 3.6 35B-A3B');
  const q = { weightBpw: 8.5008, kvBits: 16 };
  assert.equal(calcRuntimeOverhead(qwen, 8192, q, device).paramOverheadGB, 0);
  for (const name of ['2× DGX Spark 128GB', '4× DGX Spark 128GB', 'RTX PRO 6000 Blackwell', 'RTX 4090']) {
    const d = gpuDevice(GPUS.find((g) => g.name === name));
    const ov = calcRuntimeOverhead(qwen, 8192, q, d);
    const s = simulate(qwen, d, 8192, q);
    assert.ok(Math.abs(ov.paramOverheadGB - s.param * 0.12) < 1e-12, name);
  }
  assert.ok(calcRuntimeOverhead(qwen, 8192, { weightBpw: 8, kvBits: 16 }, appleDevice(64)).paramOverheadGB > 0);
});

test('전제 문구: 한 대 GB10 판정은 flash attention·llama.cpp 계열 기준이고, 끄거나 vLLM이면 더 든다고 밝힌다', () => {
  const qwen = LOCAL_MODELS.find((x) => x.name === 'Qwen 3.6 35B-A3B');
  const s = simulate(qwen, device, 8192, { weightBpw: 4.8944, kvBits: 16 });
  const unified = s.structuralAssumptions.find((a) => a.id === 'gb10-unified-memory-reserve');
  assert.ok(unified);
  for (const phrase of ['flash attention', '--flash-attn auto', 'vLLM', '--cache-ram']) {
    assert.ok(unified.statement.includes(phrase), `premise must mention ${phrase}`);
  }
});
