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

// ── PLE(Per-Layer Embeddings) 상주 분리 — 이슈 #7 ──────────────────────────
// 기대값은 전부 손계산: PLE = vocab_size_per_layer_input 262144 × hidden_size_per_layer_input 256 × L
// (e2b 35L = 2,348,810,240 → 2.349B / e4b 42L = 2,818,572,288 → 2.819B, config.json 필드 산술)
test('PLE: GPU weights = (totalParams - pleParams) × bpw — hand-derived literals', async () => {
  const { calcParamMemory } = await import('../engine.js');
  const e2b = LOCAL_MODELS.find((m) => m.name === 'Gemma 4 e2b');
  const e4b = LOCAL_MODELS.find((m) => m.name === 'Gemma 4 e4b');
  const gpu = { type: 'gpu' };
  // e2b Q4_K_M: (5.1-2.349)e9 × 4.8944/8 ÷ 1024³ = 1,683,181,800/1024³... 손계산 1.5675 GB
  assert.ok(Math.abs(calcParamMemory(e2b, 4.8944, gpu).totalGB - 1.5675) < 5e-4, 'e2b GPU Q4');
  // e2b FP16: 2.751e9 × 2 ÷ 1024³ = 5.1241 GB (구값 9.499 — 5.1e9×2×1.0 — 대비 4.37 GB 감소)
  assert.ok(Math.abs(calcParamMemory(e2b, 16, gpu).totalGB - 5.1241) < 5e-4, 'e2b GPU FP16');
  // e4b Q4_K_M: (8-2.819)e9 × 0.6118 ÷ 1024³ = 2.9520 GB
  assert.ok(Math.abs(calcParamMemory(e4b, 4.8944, gpu).totalGB - 2.9520) < 5e-4, 'e4b GPU Q4');
});

test('PLE: Apple unified memory path unchanged (same pool — totalParams stays)', async () => {
  const { calcParamMemory } = await import('../engine.js');
  const e2b = LOCAL_MODELS.find((m) => m.name === 'Gemma 4 e2b');
  // 회귀 고정: 5.1e9 × 0.5 × quantAdjust 1.39 ÷ 1024³ = 3.3011 GB (변경 전과 동일해야 함)
  assert.ok(Math.abs(calcParamMemory(e2b, 4).totalGB - 3.3011) < 5e-4, 'e2b Mac 4bit param drifted');
  const s = simulate(e2b, 16, 8192, { weightBpw: 4, kvBits: 16 });
  assert.ok(Math.abs(s.used - 13.0216) < 5e-3, `e2b mac16 used drifted: ${s.used}`);
  assert.equal(s.pleOffloadGB, 0, 'pleOffloadGB must be 0 on Apple');
});

test('PLE: verdict flip — e2b FP16 on RTX 3060 8GB (linux-headless) no→yes', () => {
  const g = GPUS.find((x) => x.name === 'RTX 3060 8GB');
  const s = simulate(LOCAL_MODELS.find((m) => m.name === 'Gemma 4 e2b'), gpuDevice(g, 'linux-headless'), 8192, { weightBpw: 16, kvBits: 16 });
  assert.equal(s.verdict, 'yes', `expected yes, got ${s.verdict} (used ${s.used})`);
  // pleOffloadGB 정보값: 2.349e9 × 2 ÷ 1024³ = 4.3754 GB가 시스템 RAM으로
  assert.ok(Math.abs(s.pleOffloadGB - 4.3754) < 5e-4);
  // 비-PLE 모델은 GPU에서도 0
  const s2 = simulate(LOCAL_MODELS.find((m) => m.name === 'Gemma 4 12b'), gpuDevice(g, 'linux-headless'), 8192, { weightBpw: 4.8944, kvBits: 16 });
  assert.equal(s2.pleOffloadGB, 0);
});

test('PLE: parseHfConfig detects vocab_size_per_layer_input × hidden_size_per_layer_input × L', async () => {
  const { parseHfConfig } = await import('../engine.js');
  const cfg = { text_config: { num_hidden_layers: 35, num_attention_heads: 8, num_key_value_heads: 1, head_dim: 256, hidden_size: 1536, vocab_size_per_layer_input: 262144, hidden_size_per_layer_input: 256, max_position_embeddings: 131072, sliding_window: 512, layer_types: Array.from({ length: 35 }, (_, i) => ((i + 1) % 5 === 0 ? 'full_attention' : 'sliding_attention')) } };
  const m = parseHfConfig('google/gemma-4-E2B-it', cfg, 10246356102); // bf16 2B × 5,123,178,051
  assert.equal(m.pleParams, 2.349); // 262144×256×35/1e9 = 2.34881 → toFixed(3)
  const plain = parseHfConfig('meta-llama/Llama-3.1-8B', { num_hidden_layers: 32, num_attention_heads: 32, hidden_size: 4096 }, 16060522496);
  assert.equal(plain.pleParams, undefined);
});

test('PLE guard: pleParams >= totalParams (broken/hostile config) cannot produce negative resident weights', async () => {
  const { calcParamMemory } = await import('../engine.js');
  const evil = { name: 'evil', totalParams: 5, pleParams: 99, layerCount: 10, kvHeads: 1, kvHeadDim: 64, attnHeads: 8, hiddenSize: 512, maxContext: 8192 };
  const gb = calcParamMemory(evil, 16, { type: 'gpu' }).totalGB;
  // 가드 작동 = pleParams 무시 → 전체 5B × 2B = 9.3132 GB (음수/0 아님)
  assert.ok(Math.abs(gb - 9.3132) < 5e-4, `expected full-weight fallback, got ${gb}`);
  // 표시값도 같은 가드: 가드된 판정 + 비가드 pleOffloadGB 모순 금지 (Codex 리뷰 minor)
  const s = simulate(evil, gpuDevice(GPUS.find((g) => g.name === 'RTX 3060 8GB'), 'linux-headless'), 4096, { weightBpw: 16, kvBits: 16 });
  assert.equal(s.pleOffloadGB, 0, 'guarded model must not report PLE offload');
});

// ── Hy3 (Day-0 2026-07-13) — 기대값 전부 손계산 ──
test('Hy3: standard-GQA KV = 2×8kvh×128d×2B×80L×262144 = 85,899,345,920 B exactly', async () => {
  const { calcKVCache } = await import('../engine.js');
  const hy3 = LOCAL_MODELS.find((m) => m.name === 'Hy3');
  assert.ok(hy3, 'Hy3 in catalog');
  assert.equal(calcKVCache(hy3, 262144, 16).totalBytes, 85899345920);
  // 토큰당 한계비용: 2×8×128×2×80 = 327,680 B
  assert.equal(calcKVCache(hy3, 1, 16).kvPerToken, 327680);
});
test('Hy3: verdicts — 512GB Mac 4bit fits, RTX 4090 Q4 does not', () => {
  const hy3 = LOCAL_MODELS.find((m) => m.name === 'Hy3');
  const mac = simulate(hy3, 512, 8192, { weightBpw: 4, kvBits: 16 });
  // 손계산: weights 298.8e9×0.5 = 149.4e9 B = 139.14 GB (quantAdjust 미등재 → ×1.0, MTP 포함 디스크 기준)
  assert.ok(Math.abs(mac.param - 139.14) < 0.01, `param ${mac.param}`);
  assert.notEqual(mac.verdict, 'no', '512GB Mac @4bit must fit');
  const gpu = simulate(hy3, gpuDevice(GPUS.find((g) => g.name === 'RTX 4090')), 8192, { weightBpw: 4.8944, kvBits: 16 });
  assert.equal(gpu.verdict, 'no', '24GB card cannot hold 295B MoE');
});
test('MiniCPM5-1B: standard-GQA KV = 2×2kvh×128d×2B×24L×131072 = 3,221,225,472 B exactly', async () => {
  const { calcKVCache } = await import('../engine.js');
  const m = LOCAL_MODELS.find((x) => x.name === 'MiniCPM5-1B');
  assert.ok(m, 'MiniCPM5-1B in catalog');
  assert.equal(calcKVCache(m, 131072, 16).totalBytes, 3221225472);
  assert.equal(calcKVCache(m, 1, 16).kvPerToken, 24576); // 2×2kvh×128d×2B×24L
});
