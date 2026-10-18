// Muse Spark Open (Meta, 2026-10-15) — 2.21.0. 표준 GQA + 슬라이딩/전체 layer_types라 기존 슬라이딩 경로로 계산한다(신규 수학 없음).
// 원본 meta-models/Muse-Spark-Open-816B-A42B는 gated라 config를 직접 못 받는다. 공개 미러 unsloth/…@49cade7의 config.json이
// HF API가 보고하는 원본 config의 git blob과 같다는 것을 바이트로 고정하고, 그 config로 카탈로그 행을 검증한다. 네트워크 없음.
import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { GPUS, LOCAL_MODELS, calcKVCache, gpuDevice, parseHfConfig, resolveLocalModel, simulate, structuralAssumptions } from '../engine.js';

const BIN = new URL('../bin/fitllm.mjs', import.meta.url).pathname;
const RAW = readFileSync(new URL('./fixtures/unsloth__Muse-Spark-Open-816B-A42B.config.json', import.meta.url));
const CONFIG = JSON.parse(RAW);
const EVIDENCE = JSON.parse(readFileSync(new URL('./fixtures/hf-api-evidence-2026-10-18.json', import.meta.url), 'utf8'));
const MIRROR = EVIDENCE['unsloth/Muse-Spark-Open-816B-A42B'];
const ORIGINAL = EVIDENCE['meta-models/Muse-Spark-Open-816B-A42B'];
const muse = () => LOCAL_MODELS.find((m) => m.name === 'Muse Spark Open');
const parse = (raw = CONFIG) => parseHfConfig('unsloth/Muse-Spark-Open-816B-A42B', raw, MIRROR.checkpointBytes, {
  revision: MIRROR.sha, safetensorsParameters: MIRROR.safetensors.parameters, safetensorsTotal: MIRROR.safetensors.total,
});

test('the mirror config is byte-identical to the gated original (same git blob) and the weights match too', () => {
  const blob = createHash('sha1').update(`blob ${RAW.length}\0`).update(RAW).digest('hex');
  assert.equal(blob, 'e6df3a846ebfcb0d55c350c1e79e7886c65b7399');
  assert.equal(ORIGINAL.configBlob, blob);
  assert.equal(MIRROR.configBlob, blob);
  assert.equal(ORIGINAL.gated, 'manual');
  assert.deepEqual(MIRROR.safetensors, ORIGINAL.safetensors);
});

test('catalog row == parse of the pinned config; KV follows the 16 full + 46 sliding(2048) layout by hand', () => {
  const row = muse();
  const parsed = parse();
  for (const key of ['totalParams', 'layerCount', 'kvHeads', 'kvHeadDim', 'attnHeads', 'hiddenSize', 'numExperts', 'expertsPerToken',
    'maxContext', 'slidingWindow', 'slidingPattern', 'globalAttnLayers', 'fullAttnLayers', 'compressedKv', 'mlaKvLoraRank', 'linearAttn', 'mtpLayerCount']) {
    assert.deepEqual(parsed[key], row[key], key);
  }
  const t = CONFIG.text_config;
  const full = t.layer_types.filter((x) => x === 'full_attention').length;
  const sliding = t.layer_types.filter((x) => x === 'sliding_attention').length;
  assert.deepEqual([full, sliding, t.sliding_window], [16, 46, 2048]);
  const perLayerToken = 2 * t.num_key_value_heads * t.head_dim * 2; // K+V × 16 × 64 × bf16
  for (const ctx of [1024, 8192, 1048576]) {
    assert.equal(calcKVCache(row, ctx, 16).totalBytes, perLayerToken * (full * ctx + sliding * Math.min(ctx, 2048)), `ctx ${ctx}`);
  }
  for (const d of [...GPUS.map((g) => gpuDevice(g)), 128, 256, 512]) {
    const a = simulate(row, d, 131072, { weightBpw: 4.8944, kvBits: 16 });
    const b = simulate(parsed, d, 131072, { weightBpw: 4.8944, kvBits: 16 });
    assert.equal(a.used, b.used);
    assert.equal(a.verdict, b.verdict);
  }
  assert.equal(row.totalParams, 822); // 821,906,490,368 BF16 + 130,039,296 F32 (비전·오디오 포함)
  assert.deepEqual(structuralAssumptions(row, 512), []); // 표준 GQA — 전제 없음
});

test('family keys are allowed only for muse_spark, and only in the shape the config uses', () => {
  const t = CONFIG.text_config;
  const withText = (patch) => ({ ...CONFIG, text_config: { ...t, ...patch } });
  assert.throws(() => parse(withText({ layer_rope_theta: t.layer_rope_theta.slice(1) })), /layer_rope_theta/);
  assert.throws(() => parse(withText({ mlp_layer_types: t.mlp_layer_types.map(() => 'hybrid') })), /mlp_layer_types/);
  // 같은 키를 다른 계열이 쓰면 의미가 같다는 보장이 없다 — 미지 구조 키로 거부
  assert.throws(() => parse({ ...withText({ model_type: 'llama' }), model_type: 'llama' }), /구조를 알 수 없는 config 필드/);
});

test('layer_types is authoritative: an all-full_attention schedule is not treated as sliding (Codex review 2026-10-18)', () => {
  const t = CONFIG.text_config;
  const allFull = parse({ ...CONFIG, text_config: { ...t, layer_types: t.layer_types.map(() => 'full_attention') } });
  assert.equal(allFull.slidingWindow, undefined);
  assert.equal(allFull.globalAttnLayers, undefined);
  // 62층 × 2 × 16 × 64 × 2 B × 1M = 248 GiB — 종래 코드는 sliding_window만 보고 5:1로 나눠 40.4 GiB를 냈다
  assert.equal(calcKVCache(allFull, 1048576, 16).totalGB, 248);
  // 같은 규칙은 다른 계열에도 적용된다(계열 무관 경로): llama 형상 + 전 층 full + sliding_window
  const llama = parseHfConfig('x/llama', {
    model_type: 'llama', num_hidden_layers: 4, num_attention_heads: 8, num_key_value_heads: 2, head_dim: 64, hidden_size: 512,
    sliding_window: 4096, max_position_embeddings: 131072, layer_types: Array(4).fill('full_attention'), torch_dtype: 'bfloat16',
  }, 2e9);
  assert.equal(llama.slidingWindow, undefined);
  assert.equal(calcKVCache(llama, 131072, 16).totalBytes, 4 * 2 * 2 * 64 * 2 * 131072);
});

test('names: the gated HF ID and the repo name resolve to the catalog row; "spark" alone stays ambiguous', () => {
  for (const q of ['Muse Spark Open', 'muse spark', 'Muse-Spark-Open-816B-A42B', 'meta-models/Muse-Spark-Open-816B-A42B']) {
    const r = resolveLocalModel(q);
    assert.equal(r.status, 'resolved', q);
    assert.equal(r.canonicalName, 'Muse Spark Open', q);
  }
  assert.equal(resolveLocalModel('spark').status, 'ambiguous');
  assert.equal(resolveLocalModel('unsloth/Muse-Spark-Open-816B-A42B').status, 'unknown'); // 미러는 HF 경로(같은 숫자 — 위 parity)
});

test('CLI: the gated original ID answers without network (alias → catalog); 4× DGX Spark is short, M3 Ultra 512GB is tight', () => {
  const run = (...args) => spawnSync(process.execPath, [BIN, ...args], { encoding: 'utf8' });
  const spark = run('meta-models/Muse-Spark-Open-816B-A42B', '--gpu', 'DGX Spark', '--count', '4', '--json');
  assert.equal(spark.status, 1, spark.stderr);
  const j = JSON.parse(spark.stdout);
  assert.equal(j.model, 'Muse Spark Open');
  assert.equal(j.verdict, 'no');
  assert.equal(j.modelSource, undefined); // 카탈로그 행 — HF 요청을 하지 않았다
  const mac = run('Muse Spark Open', '--mac', '512', '--quant', '4', '--json');
  assert.equal(mac.status, 0);
  assert.equal(JSON.parse(mac.stdout).verdict, 'tight');
});
