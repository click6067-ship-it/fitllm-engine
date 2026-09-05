// Spark-X2.5-4B (XHToken, model_type spark2_5) — 2026-09-05 Day-0 (#103). 기대값은 전부 손계산 리터럴.
// 1차 출처(전부 pinned revision 5e10fcc0286756aebf7c41dc52c1e42d95c70281):
//   config.json  https://huggingface.co/XHToken/Spark-X2.5-4B/blob/5e10fcc0286756aebf7c41dc52c1e42d95c70281/config.json
//   modeling     https://huggingface.co/XHToken/Spark-X2.5-4B/blob/5e10fcc0286756aebf7c41dc52c1e42d95c70281/modeling_spark.py
//   index        https://huggingface.co/XHToken/Spark-X2.5-4B/blob/5e10fcc0286756aebf7c41dc52c1e42d95c70281/model.safetensors.index.json
//   HF API       https://huggingface.co/api/models/XHToken/Spark-X2.5-4B/revision/5e10fcc0286756aebf7c41dc52c1e42d95c70281
// 네트워크 의존 없음 — config는 fixtures/day0에 pinned 사본, 파라미터 증거는 API 응답 리터럴.
import { test } from 'node:test';
import { strict as assert } from 'node:assert';
import { readFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import {
  MODELS, LOCAL_MODELS, MODEL_GROUP_ORDER, GPUS, ENGINE_VERSION, gpuDevice,
  parseHfConfig, calcKVCache, naiveKVCache, structuralAssumptions, resolveLocalModel,
} from '../engine.js';

const BIN = new URL('../bin/fitllm.mjs', import.meta.url).pathname;
const run = (args) => {
  try { return { out: execFileSync(process.execPath, [BIN, ...args], { encoding: 'utf8' }), code: 0 }; }
  catch (e) { return { out: (e.stdout || '') + (e.stderr || ''), code: e.status }; }
};
const readJson = (rel) => JSON.parse(readFileSync(new URL(rel, import.meta.url), 'utf8'));

const REVISION = '5e10fcc0286756aebf7c41dc52c1e42d95c70281';
const CONFIG = readJson('./fixtures/day0/spark-x2.5-4b.config.json');
// HF API tree(pinned revision)의 5개 shard 크기. 합계 8,224,192,408 B = 텐서 8,224,158,720 B(= 4,112,079,360 × 2) + 헤더 33,688 B.
const SHARD_BYTES = [1982449560, 1993132424, 1993225080, 1993132448, 262252896];
const CHECKPOINT_BYTES = SHARD_BYTES.reduce((a, b) => a + b, 0);
// HF API safetensors: { parameters: { BF16: 4112079360 }, total: 4112079360 } == index metadata.total_parameters
const EVIDENCE = { revision: REVISION, safetensorsParameters: { BF16: 4112079360 }, safetensorsTotal: 4112079360 };
const TOTAL_PARAMS = 4112079360;
const PER_LAYER_TOKEN = 2 * 4 * 256 * 2; // K+V × 4 kv-heads × head_dim 256 × 2 B(F16) = 4,096 B/layer/token
const GATE_KEYS = { gate_attn_act_mode: 'sigmoid', headwise_attn_output_gate: true };
const spark = () => LOCAL_MODELS.find((m) => m.name === 'Spark-X2.5-4B');
const without = (obj, ...keys) => Object.fromEntries(Object.entries(obj).filter(([k]) => !keys.includes(k)));
const gpu4090 = () => gpuDevice(GPUS.find((g) => g.name === 'RTX 4090'));

test('test_pinned_config_reproduces_contract_claims', () => {
  // 킬 조건 — pinned 아티팩트가 계약 수치를 정확히 재현하지 않으면 모델을 넣지 않는다.
  assert.equal(CONFIG.model_type, 'spark2_5');
  assert.equal(CONFIG.num_hidden_layers, 36);
  assert.equal(CONFIG.hidden_size, 2560);
  assert.equal(CONFIG.num_attention_heads, 16);
  assert.equal(CONFIG.num_key_value_heads, 4);
  assert.equal(CONFIG.head_dim, 256);
  assert.equal(CONFIG.intermediate_size, 10240);
  assert.equal(CONFIG.vocab_size, 131072);
  assert.equal(CONFIG.tie_word_embeddings, true);
  assert.equal(CONFIG.max_position_embeddings, 1048576);
  assert.equal(CONFIG.sliding_window, 512);
  assert.equal(CONFIG.layer_types.length, 36);
  assert.equal(CONFIG.layer_types.filter((t) => t === 'sliding_attention').length, 27);
  assert.equal(CONFIG.layer_types.filter((t) => t === 'full_attention').length, 9);
  assert.deepEqual(CONFIG.layer_types.map((t, i) => (t === 'full_attention' ? i : -1)).filter((i) => i >= 0), [3, 7, 11, 15, 19, 23, 27, 31, 35]);
  // 파라미터 손계산(safetensors 텐서명 기준: q_k_v_proj·out_proj·g_proj·MLP 3종·norm 2종 / 임베딩 tied / 최종 norm)
  const h = 2560, H = 16, K = 4, d = 256, I = 10240, V = 131072, L = 36;
  const qkv = h * (H + 2 * K) * d;        // 2560 × 6144 = 15,728,640
  const out = H * d * h;                  // 4096 × 2560 = 10,485,760
  const gProj = h * H;                    // g_proj Linear(hidden_size, num_heads) = 40,960 — 가중치만, KV 무관
  const mlp = 3 * h * I;                  // 78,643,200
  const norms = 2 * h;                    // input_layernorm + post_attention_layernorm
  const perLayer = qkv + out + gProj + mlp + norms; // 104,903,680
  assert.equal(L * perLayer + V * h + h, TOTAL_PARAMS);
  assert.equal(2 * TOTAL_PARAMS, 8224158720); // index metadata.total_size (BF16)
  assert.equal(CHECKPOINT_BYTES - 2 * TOTAL_PARAMS, 33688); // safetensors 헤더 slack
});

test('test_pinned_config_parses_exact_fields', () => {
  const m = parseHfConfig('XHToken/Spark-X2.5-4B', CONFIG, CHECKPOINT_BYTES, EVIDENCE);
  assert.equal(m.layerCount, 36);
  assert.equal(m.attnHeads, 16);
  assert.equal(m.kvHeads, 4);
  assert.equal(m.kvHeadDim, 256);
  assert.equal(m.hiddenSize, 2560);
  assert.equal(m.slidingWindow, 512);
  assert.equal(m.globalAttnLayers, 9);
  assert.equal(m.slidingPattern, '3:1');
  assert.equal(m.maxContext, 1048576);
  assert.equal(m.totalParams, 4.1); // 4,112,079,360 → parseHfConfig 관례(소수 1자리)
  assert.equal(m.parameterSource, 'hf-safetensors-parameters');
  assert.deepEqual(m.tags, ['dense']);
  for (const absent of ['fullAttnLayers', 'linearAttn', 'mlaKvLoraRank', 'mlaRopeDim', 'pleParams', 'mtpLayerCount', 'numExperts', 'globalHeadDim']) {
    assert.equal(m[absent], undefined, `${absent} must be absent`);
  }
  assert.equal(m.pleOffloadVerified, false);
  // 증거 없이 index total_size(8,224,158,720)만 있어도 같은 자릿수(uniform bf16 경로)
  const byIndex = parseHfConfig('XHToken/Spark-X2.5-4B', CONFIG, 2 * TOTAL_PARAMS);
  assert.equal(byIndex.totalParams, 4.1);
  assert.equal(byIndex.parameterSource, 'uniform-checkpoint');
  // 파싱 결과와 카탈로그 행이 같은 KV를 낸다(같은 구조를 두 경로가 같게 읽는다)
  assert.equal(calcKVCache(m, 1048576, 16).totalBytes, calcKVCache(spark(), 1048576, 16).totalBytes);
});

test('test_gate_keys_rejected_outside_spark2_5', () => {
  // 같은 이름의 게이트 필드라도 검증 안 된 계열에서는 의미가 같다는 보장이 없다(#87) — 정확히 spark2_5만 허용.
  for (const modelType of ['llama', 'qwen3_5', 'gemma4_text', 'spark2_5_text', 'spark2_5_moe', 'spark3', 'Spark2_5', '']) {
    assert.throws(
      () => parseHfConfig(`x/${modelType || 'none'}`, { ...CONFIG, model_type: modelType }, CHECKPOINT_BYTES, EVIDENCE),
      /구조를 알 수 없는 config 필드\(gate_attn_act_mode, headwise_attn_output_gate\)/,
      `model_type=${JSON.stringify(modelType)} must stay fail-closed`,
    );
  }
  // 각 키 단독으로도 계열 밖에서는 거부
  for (const [k, v] of Object.entries(GATE_KEYS)) {
    assert.throws(
      () => parseHfConfig('x/llama-gated', { ...without(CONFIG, 'gate_attn_act_mode', 'headwise_attn_output_gate'), model_type: 'llama', [k]: v }, CHECKPOINT_BYTES, EVIDENCE),
      new RegExp(`구조를 알 수 없는 config 필드\\(${k}\\)`),
    );
  }
  // spark2_5 허용은 이 두 키에 한정 — 다른 미지 구조 키는 여전히 거부
  assert.throws(
    () => parseHfConfig('x/spark-mhc', { ...CONFIG, mhc_enabled: true }, CHECKPOINT_BYTES, EVIDENCE),
    /구조를 알 수 없는 config 필드\(mhc_enabled\)/,
  );
  // 래퍼(text_config) 최상위에 게이트 키가 있고 본체가 다른 계열이면 거부(스코프 2개 모두 스캔)
  assert.throws(
    () => parseHfConfig('x/wrapped', { model_type: 'spark2_5', ...GATE_KEYS, text_config: { ...without(CONFIG, 'gate_attn_act_mode', 'headwise_attn_output_gate'), model_type: 'llama' } }, CHECKPOINT_BYTES, EVIDENCE),
    /구조를 알 수 없는 config 필드/,
  );
});

test('test_sliding_pattern_derived_from_layer_types', () => {
  // 종래엔 슬라이딩이면 무조건 '5:1'(Gemma 4 관례)을 붙였다. 라벨은 layer_types 실카운트(gcd 약분 sliding:full)로 유도하고,
  // KV 계산은 전부터 globalAttnLayers(실카운트)를 썼으므로 바이트는 불변이어야 한다.
  const base = {
    model_type: 'llama', num_attention_heads: 48, num_key_value_heads: 8, hidden_size: 2048, head_dim: 128,
    intermediate_size: 8192, vocab_size: 131072, max_position_embeddings: 262144, torch_dtype: 'bfloat16', sliding_window: 512,
  };
  const layout = (n, period) => Array.from({ length: n }, (_, i) => ((i + 1) % period === 0 ? 'full_attention' : 'sliding_attention'));
  const per = 2 * 8 * 128 * 2; // 4,096 B/layer/token
  for (const [layers, period, pattern, globalLayers] of [
    [40, 4, '3:1', 10], // Laguna XS 2.1 배치
    [48, 4, '3:1', 12], // Laguna S 2.1 배치
    [42, 6, '5:1', 7],  // Gemma 4 E4B 배치 — 종래 라벨과 동일
    [35, 5, '4:1', 7],  // Gemma 4 E2B 배치
    [24, 2, '1:1', 12], // gpt-oss-20b 배치
    [26, 6, '11:2', 4], // Gemma-3-1B 배치(22:4) — 약분 불가 비율은 정수쌍 그대로
  ]) {
    const m = parseHfConfig('x/swa', { ...base, num_hidden_layers: layers, layer_types: layout(layers, period) }, 8e9);
    assert.equal(m.globalAttnLayers, globalLayers, `${layers}L/${period}`);
    assert.equal(m.slidingPattern, pattern, `${layers}L/${period}`);
    assert.equal(calcKVCache(m, 262144, 16).totalBytes, per * globalLayers * 262144 + per * (layers - globalLayers) * 512, `${layers}L/${period} KV`);
  }
  // pinned Spark 배치 27:9 → '3:1'
  assert.equal(parseHfConfig('XHToken/Spark-X2.5-4B', CONFIG, CHECKPOINT_BYTES, EVIDENCE).slidingPattern, '3:1');
  // MLA 경로는 종래대로 슬라이딩 라벨 없음
  const mla = parseHfConfig('x/mla', { ...base, num_hidden_layers: 40, kv_lora_rank: 512, qk_rope_head_dim: 64, sliding_window: 0 }, 8e9);
  assert.equal(mla.slidingPattern, undefined);
});

test('test_catalog_kv_hand_derived_1m', () => {
  const m = spark();
  assert.ok(m, 'Spark-X2.5-4B must be in LOCAL_MODELS');
  assert.equal(PER_LAYER_TOKEN, 4096);
  const globalBytes = PER_LAYER_TOKEN * 9 * 1048576; // 38,654,705,664 — 글로벌 9레이어 × 전체 ctx
  const localBytes = PER_LAYER_TOKEN * 27 * 512;     //     56,623,104 — 슬라이딩 27레이어 × window 512
  assert.equal(globalBytes + localBytes, 38711328768);
  const kv = calcKVCache(m, 1048576, 16);
  assert.equal(kv.totalBytes, 38711328768);
  assert.equal(kv.kvPerToken, 36864); // 윈도우 초과 후 토큰당 증가분 = 4,096 × 9 글로벌 레이어
  assert.equal(calcKVCache(m, 131072, 16).totalBytes, PER_LAYER_TOKEN * 9 * 131072 + localBytes); // 4,888,461,312
  assert.equal(calcKVCache(m, 512, 16).totalBytes, PER_LAYER_TOKEN * 36 * 512); // 윈도우 이내면 전 레이어 동일
  // 순진 계산(36레이어 전부 전체 ctx)은 4× 과대: 36 × 2^32 B = 144 GiB
  assert.equal(naiveKVCache(m, 1048576, 16), 144);
  assert.equal(naiveKVCache(m, 1048576, 16) * 1024 ** 3 / kv.totalBytes > 3.99, true);
});

test('test_catalog_row_exact_pinned_fields', () => {
  const m = spark();
  assert.deepEqual(
    (({ group, tags, totalParams, layerCount, globalAttnLayers, slidingWindow, slidingPattern, kvHeads, kvHeadDim, attnHeads, hiddenSize, maxContext, benchmarks }) =>
      ({ group, tags, totalParams, layerCount, globalAttnLayers, slidingWindow, slidingPattern, kvHeads, kvHeadDim, attnHeads, hiddenSize, maxContext, benchmarks }))(m),
    {
      group: 'Spark', tags: ['dense'], totalParams: 4.112, layerCount: 36, globalAttnLayers: 9, slidingWindow: 512, slidingPattern: '3:1',
      kvHeads: 4, kvHeadDim: 256, attnHeads: 16, hiddenSize: 2560, maxContext: 1048576, benchmarks: null,
    },
  );
  assert.equal(Math.round(m.totalParams * 1e9), Math.round(TOTAL_PARAMS / 1e6) * 1e6); // 4.112B = 4,112,079,360 소수 3자리
  // 직접 상주 의미 — PLE/MLA/MTP 전제 없음(어느 플랫폼에서도 structuralAssumptions 키가 붙지 않는다)
  for (const absent of ['pleParams', 'pleOffloadVerified', 'mlaKvLoraRank', 'mtpLayerCount', 'linearAttn', 'fullAttnLayers', 'globalKvHeads', 'globalHeadDim', 'numExperts', 'isCloud']) {
    assert.equal(m[absent], undefined, `${absent} must be absent`);
  }
  assert.deepEqual(structuralAssumptions(m, gpu4090()), []);
  assert.deepEqual(structuralAssumptions(m, 64), []);
});

test('test_catalog_append_index_and_group_order_stable', () => {
  const BASE_NAMES = [
    'GLM-4.7-Flash', 'GLM-5.2', 'gpt-oss-20b', 'gpt-oss-120b', 'Qwen 3.6 27B', 'Qwen 3.6 35B-A3B', 'Qwen-AgentWorld-35B-A3B',
    'Gemma 4 e2b', 'Gemma 4 e4b', 'Gemma 4 12b', 'Gemma 4 26b A4B', 'Gemma 4 31b', 'Llama-3.2-3B-Instruct', 'Llama-3.1-8B-Instruct',
    'MiniCPM5-1B', 'Qwen3-0.6B', 'Qwen3-1.7B', 'Llama-3.2-1B-Instruct', 'Gemma-3-1B-it', 'Claude Opus 4.7', 'Hy3',
    'Qwen 3.8 27B', 'Qwen 3.8 2.4T-A95B', 'Laguna XS 2.1', 'Laguna S 2.1',
  ];
  // ?m= 공유링크 인덱스 0–24 불변, Spark는 배열 끝 append(인덱스 25)
  assert.deepEqual(MODELS.slice(0, BASE_NAMES.length).map((m) => m.name), BASE_NAMES);
  assert.equal(MODELS.length, BASE_NAMES.length + 3); // 2026-09-05 Granite-4.2-30B append(인덱스 26, #93) + 2.15.0 GLM-5.3 append(인덱스 27)
  assert.equal(MODELS[BASE_NAMES.length].name, 'Spark-X2.5-4B');
  assert.equal(LOCAL_MODELS[24].name, 'Spark-X2.5-4B'); // ?m=24 불변
  assert.equal(MODELS.filter((m) => /spark/i.test(m.name)).length, 1); // 1.7B sibling은 이번 변경에 넣지 않는다
  assert.equal(LOCAL_MODELS.length, 27); // 26(2.14.x) + GLM-5.3(2.15.0, 인덱스 26)
  // 기존 그룹의 상대 순서 불변 + Spark 그룹 등재
  const BASE_ORDER = ['Qwen 3.8', 'Laguna', 'GLM', 'gpt-oss', 'Qwen 3.6', 'Qwen3.5', 'Hunyuan', 'Gemma 4', 'Llama', 'MiniCPM', 'Draft'];
  assert.deepEqual(MODEL_GROUP_ORDER.filter((g) => g !== 'Spark' && g !== 'Granite'), BASE_ORDER);
  assert.ok(MODEL_GROUP_ORDER.includes('Spark'));
  // 이름 해석: 정확 일치 + 토큰 일치, 다른 카탈로그 항목과 충돌 없음
  assert.equal(resolveLocalModel('Spark-X2.5-4B').matchedBy, 'exact');
  assert.equal(resolveLocalModel('spark x2.5 4b').match.name, 'Spark-X2.5-4B');
  assert.equal(resolveLocalModel('spark').match.name, 'Spark-X2.5-4B');
});

test('test_cli_contract_fits_8k_kv_bound_1m', () => {
  // 기본(8K ctx, Q4_K_M): 4.1B 가중치는 24GB 카드에 여유
  const ok = run(['Spark-X2.5-4B', '--gpu', 'RTX 4090', '--json']);
  assert.equal(ok.code, 0, ok.out);
  const j = JSON.parse(ok.out);
  assert.equal(j.verdict, 'yes');
  assert.equal('structuralAssumptions' in j, false); // 직접 상주 GQA — 전제 키 없음(기존 JSON shape)
  // 네이티브 1M ctx: 글로벌 9레이어의 KV만으로 카드 용량을 넘긴다 — 가중치가 아니라 KV 상주가 병목
  const far = run(['Spark-X2.5-4B', '--gpu', 'RTX 4090', '--ctx', '1048576', '--json']);
  assert.equal(far.code, 1, far.out);
  const k = JSON.parse(far.out);
  assert.equal(k.verdict, 'no');
  assert.ok(k.breakdown.kvGB > k.breakdown.paramGB, `kv ${k.breakdown.kvGB} must exceed weights ${k.breakdown.paramGB}`);
  assert.ok(k.breakdown.kvGB > 24, `kv ${k.breakdown.kvGB} GiB alone exceeds the 24 GiB card`);
  assert.equal(k.breakdown.kvGB, +(38711328768 / 1024 ** 3).toFixed(2)); // 36.05 GiB — 손계산 벡터와 같은 바이트
  // --why: 감사 근거가 3:1 슬라이딩 구조를 그대로 말한다
  const why = JSON.parse(run(['Spark-X2.5-4B', '--gpu', 'RTX 4090', '--json', '--why']).out);
  assert.deepEqual(why.basis.attention, {
    kind: 'sliding-window', totalLayers: 36, kvLayers: 36, kvHeads: 4, kvHeadDim: 256, slidingWindow: 512, slidingPattern: '3:1',
  });
});

test('test_census_contains_spark_rows', () => {
  const census = readJson('../census/census-v1.json');
  const manifest = readJson('../census/manifest.json');
  const sparkRows = census.data.filter((r) => r.model === 'Spark-X2.5-4B');
  assert.equal(sparkRows.length, 57 * 3 + 36 * 5); // Mac 57구성 × 3 tiers + GPU 36종 × 5 tiers = 351 rows/model
  const total = LOCAL_MODELS.length * (57 * 3 + 36 * 5); // 27 × 351 = 9,477 (2.15.0: GLM-5.3 1행 추가)
  assert.equal(census.verdicts, total);
  assert.equal(census.data.length, total);
  assert.equal(manifest.rows, total);
  assert.ok(sparkRows.every((r) => r.params_b === 4.112 && r.linear_state_gb === 0 && r.ctx === 8192));
});

test('test_release_version_and_readme_counts', () => {
  const pkg = readJson('../package.json');
  const lock = readJson('../package-lock.json');
  assert.equal(pkg.version, '2.15.0');
  assert.equal(lock.version, '2.15.0');
  assert.equal(lock.packages[''].version, '2.15.0');
  assert.equal(ENGINE_VERSION, '2.15.0');
  const readme = readFileSync(new URL('../README.md', import.meta.url), 'utf8');
  const agents = readFileSync(new URL('../AGENTS.md', import.meta.url), 'utf8');
  const verdicts = `${readJson('../census/manifest.json').rows.toLocaleString('en-US')} verdicts`; // 산출물에서 유도
  for (const s of ['uses: click6067-ship-it/fitllm-engine@v2.15.0', 'conformance_vectors-30%2F30', '30 language-neutral', verdicts, `${LOCAL_MODELS.length} models incl. draft tier`]) {
    assert.ok(readme.includes(s), `README missing: ${s}`);
  }
  for (const s of [verdicts, '30 byte-exact anchors']) assert.ok(agents.includes(s), `AGENTS missing: ${s}`);
  const vectors = readJson('../vectors/fit-vectors-v1.json');
  assert.equal(vectors.vectors.length, 30); // 2026-09-05 Granite 벡터 1개 추가(#93) — Spark 벡터는 id로 고정
  const sparkVector = vectors.vectors.find((v) => v.id === 'spark-x25-4b-kv-1m-f16');
  assert.deepEqual(sparkVector, {
    id: 'spark-x25-4b-kv-1m-f16', kind: 'kv_total_bytes', model: 'Spark-X2.5-4B', ctx: 1048576, kvBits: 16, expect: 38711328768,
    note: sparkVector.note,
  });
  assert.doesNotMatch(readme, /Spark[^\n]*tok(?:ens)?\/s/i); // 속도 주장 금지
});
