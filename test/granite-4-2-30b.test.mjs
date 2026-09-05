// Granite-4.2-30B (IBM, model_type granite / GraniteForCausalLM) — 2026-09-05 Day-0 (#93). 기대값은 전부 손계산 리터럴.
// 1차 출처(전부 pinned revision 9e668ce1c538387ef24d3644e9b0606647762636 — 2026-09-04 모델카드 갱신 커밋.
// config.json·generation_config.json·model.safetensors.index.json blob은 이슈 코멘트의 8b445a5c315f32da0f89e1f648bfec0cd601b154 와 byte-identical):
//   config.json  https://huggingface.co/ibm-granite/granite-4.2-30b/blob/9e668ce1c538387ef24d3644e9b0606647762636/config.json
//   index        https://huggingface.co/ibm-granite/granite-4.2-30b/blob/9e668ce1c538387ef24d3644e9b0606647762636/model.safetensors.index.json
//   HF API       https://huggingface.co/api/models/ibm-granite/granite-4.2-30b/revision/9e668ce1c538387ef24d3644e9b0606647762636
//   모델카드     https://huggingface.co/ibm-granite/granite-4.2-30b/blob/9e668ce1c538387ef24d3644e9b0606647762636/README.md
// 네트워크 의존 없음 — config는 fixtures/day0에 pinned 사본(sha256 82c834ac6bf6a2ecbf9801b49c3fb927008df24ea415c35e931b6091cab21277),
// 파라미터 증거·shard 크기는 API 응답 리터럴.
import { test } from 'node:test';
import { strict as assert } from 'node:assert';
import { readFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import {
  MODELS, LOCAL_MODELS, MODEL_GROUP_ORDER, GPUS, GPU_QUANTS, ENGINE_VERSION, gpuDevice, simulate,
  parseHfConfig, calcKVCache, naiveKVCache, structuralAssumptions, resolveLocalModel,
} from '../engine.js';

const BIN = new URL('../bin/fitllm.mjs', import.meta.url).pathname;
const run = (args) => {
  try { return { out: execFileSync(process.execPath, [BIN, ...args], { encoding: 'utf8' }), code: 0 }; }
  catch (e) { return { out: (e.stdout || '') + (e.stderr || ''), code: e.status }; }
};
const readJson = (rel) => JSON.parse(readFileSync(new URL(rel, import.meta.url), 'utf8'));

const REVISION = '9e668ce1c538387ef24d3644e9b0606647762636';
const CONFIG = readJson('./fixtures/day0/granite-4.2-30b.config.json');
// HF API tree(pinned revision)의 11개 shard 크기. 합계 58,553,607,904 B = 텐서 58,553,540,608 B(= 29,276,770,304 × 2) + 헤더 67,296 B.
const SHARD_BYTES = [
  5201015192, 5335259280, 5335259272, 5335259280, 5335259272, 5335259280,
  5335259272, 5335259272, 5335259280, 5335259272, 5335259232,
];
const CHECKPOINT_BYTES = SHARD_BYTES.reduce((a, b) => a + b, 0);
// HF API safetensors: { parameters: { BF16: 29276770304 }, total: 29276770304 } == index metadata.total_size ÷ 2
const EVIDENCE = { revision: REVISION, safetensorsParameters: { BF16: 29276770304 }, safetensorsTotal: 29276770304 };
const TOTAL_PARAMS = 29276770304;
const INDEX_TOTAL_SIZE = 58553540608;
const PER_LAYER_TOKEN = 2 * 8 * 128 * 2; // K+V × 8 kv-heads × head_dim 128 × 2 B(F16) = 4,096 B/layer/token
const KV_128K_F16 = 34359738368;         // 4,096 × 64 layers × 131,072 ctx = 정확히 32 GiB
const granite = () => LOCAL_MODELS.find((m) => m.name === 'Granite-4.2-30B');
const gpu = (name, env) => gpuDevice(GPUS.find((g) => g.name === name), env);
const Q4_K_M = GPU_QUANTS.find((q) => q.tier === 'Q4_K_M').bpw;

test('test_pinned_config_reproduces_contract_claims', () => {
  // 킬 조건 — pinned 아티팩트가 계약 수치를 정확히 재현하지 않으면 모델을 넣지 않는다.
  assert.equal(CONFIG.model_type, 'granite');
  assert.deepEqual(CONFIG.architectures, ['GraniteForCausalLM']);
  assert.equal(CONFIG.num_hidden_layers, 64);
  assert.equal(CONFIG.hidden_size, 4096);
  assert.equal(CONFIG.num_attention_heads, 32);
  assert.equal(CONFIG.num_key_value_heads, 8);
  assert.equal(CONFIG.head_dim, undefined); // head_dim 키 없음 → hidden_size / num_attention_heads = 128 (shard 헤더 k_proj [1024, 4096] = 8 × 128 로 확인)
  assert.equal(CONFIG.intermediate_size, 32768);
  assert.equal(CONFIG.vocab_size, 100352);
  assert.equal(CONFIG.tie_word_embeddings, false);
  assert.equal(CONFIG.max_position_embeddings, 131072);
  assert.equal(CONFIG.torch_dtype, 'bfloat16');
  assert.equal(CONFIG.dtype, 'bfloat16');
  // 구조를 바꾸는 키가 하나도 없다 — dense GQA 균일 구조
  for (const absent of ['sliding_window', 'layer_types', 'layers_block_type', 'kv_lora_rank', 'num_local_experts', 'num_experts', 'n_routed_experts',
    'vocab_size_per_layer_input', 'num_nextn_predict_layers', 'text_config', 'quantization_config', 'auto_map']) {
    assert.equal(absent in CONFIG, false, `${absent} must be absent`);
  }
  // Granite 고유 스칼라 — 값 스케일링만, 텐서 치수·KV 레이아웃 불변
  assert.equal(CONFIG.attention_multiplier, 0.0078125); // = 1/128
  assert.equal(CONFIG.embedding_multiplier, 1.0);
  assert.equal(CONFIG.residual_multiplier, 1.0);
  assert.equal(CONFIG.logits_scaling, 1.0);
  // 파라미터 손계산(safetensors 텐서명 기준: q/k/v/o_proj · gate/up/down_proj · norm 2종 / 임베딩 + untied lm_head / 최종 norm)
  const h = 4096, H = 32, K = 8, d = 128, I = 32768, V = 100352, L = 64;
  const attn = h * (H * d) + 2 * (K * d) * h + (H * d) * h; // q 16,777,216 + k,v 2 × 4,194,304 + o 16,777,216 = 41,943,040
  const mlp = 3 * h * I;                                     // 402,653,184
  const norms = 2 * h;                                       // 8,192
  const perLayer = attn + mlp + norms;                       // 444,604,416
  assert.equal(perLayer, 444604416);
  assert.equal(L * perLayer + V * h + V * h + h, TOTAL_PARAMS); // 28,454,682,624 + 411,041,792 × 2 + 4,096
  assert.equal(2 * TOTAL_PARAMS, INDEX_TOTAL_SIZE); // index metadata.total_size (BF16)
  assert.equal(CHECKPOINT_BYTES - INDEX_TOTAL_SIZE, 67296); // 11개 shard safetensors 헤더 slack 합
});

test('test_pinned_config_parses_exact_fields', () => {
  const m = parseHfConfig('ibm-granite/granite-4.2-30b', CONFIG, CHECKPOINT_BYTES, EVIDENCE);
  assert.equal(m.layerCount, 64);
  assert.equal(m.attnHeads, 32);
  assert.equal(m.kvHeads, 8);
  assert.equal(m.kvHeadDim, 128);
  assert.equal(m.hiddenSize, 4096);
  assert.equal(m.maxContext, 131072);
  assert.equal(m.totalParams, 29.3); // 29,276,770,304 → parseHfConfig 관례(소수 1자리)
  assert.equal(m.parameterSource, 'hf-safetensors-parameters');
  assert.deepEqual(m.tags, ['dense']);
  for (const absent of ['slidingWindow', 'slidingPattern', 'globalAttnLayers', 'fullAttnLayers', 'linearAttn', 'mlaKvLoraRank', 'mlaRopeDim',
    'pleParams', 'mtpLayerCount', 'numExperts', 'expertsPerToken', 'globalHeadDim']) {
    assert.equal(m[absent], undefined, `${absent} must be absent`);
  }
  assert.equal(m.pleOffloadVerified, false);
  // 증거 없이 index total_size(58,553,540,608)만 있어도 같은 자릿수(uniform bf16 경로)
  const byIndex = parseHfConfig('ibm-granite/granite-4.2-30b', CONFIG, INDEX_TOTAL_SIZE);
  assert.equal(byIndex.totalParams, 29.3);
  assert.equal(byIndex.parameterSource, 'uniform-checkpoint');
  // 파싱 결과와 카탈로그 행이 같은 KV를 낸다(같은 구조를 두 경로가 같게 읽는다)
  assert.equal(calcKVCache(m, 131072, 16).totalBytes, KV_128K_F16);
  assert.equal(calcKVCache(m, 131072, 16).totalBytes, calcKVCache(granite(), 131072, 16).totalBytes);
  assert.deepEqual(structuralAssumptions(m, gpu('RTX 4090')), []);
});

test('test_granite_family_stays_fail_closed_for_unknown_structure', () => {
  // 이번 변경은 parseHfConfig 게이트를 넓히지 않는다 — granite 계열에 미지 구조 키가 오면 종전대로 거부.
  assert.throws(
    () => parseHfConfig('x/granite-mhc', { ...CONFIG, mhc_enabled: true }, CHECKPOINT_BYTES, EVIDENCE),
    /구조를 알 수 없는 config 필드\(mhc_enabled\)/,
  );
  assert.throws(
    () => parseHfConfig('x/granite-swa', { ...CONFIG, sliding_window: 4096 }, CHECKPOINT_BYTES, EVIDENCE),
    /레이어별 sliding\/full 구성이 없는 모델은 정확히 계산할 수 없어요/,
  );
  // 같은 치수에 num_key_value_heads만 32(MHA)로 바꾸면 KV가 정확히 4배 — 숫자를 결정하는 것이 GQA 8 kv-heads임을 고정
  const mha = parseHfConfig('x/granite-mha', { ...CONFIG, num_key_value_heads: 32 }, CHECKPOINT_BYTES, EVIDENCE);
  assert.equal(calcKVCache(mha, 131072, 16).totalBytes, 4 * KV_128K_F16); // 137,438,953,472 B = 128 GiB
});

test('test_catalog_kv_hand_derived_128k', () => {
  const m = granite();
  assert.ok(m, 'Granite-4.2-30B must be in LOCAL_MODELS');
  assert.equal(PER_LAYER_TOKEN, 4096);
  assert.equal(PER_LAYER_TOKEN * 64, 262144); // 256 KiB/token — 전 레이어 균일
  assert.equal(PER_LAYER_TOKEN * 64 * 131072, KV_128K_F16);
  const kv = calcKVCache(m, 131072, 16);
  assert.equal(kv.totalBytes, KV_128K_F16);
  assert.equal(kv.kvPerToken, 262144);
  assert.equal(kv.totalGB, 32); // 정확히 32 GiB — 네이티브 128K 컨텍스트의 F16 KV만으로 24 GiB 카드를 넘긴다
  assert.equal(calcKVCache(m, 8192, 16).totalBytes, 2147483648);   // 8K = 정확히 2 GiB
  assert.equal(calcKVCache(m, 131072, 8).totalBytes, 17179869184); // Q8 KV = 절반
  assert.equal(calcKVCache(m, 1, 16).totalBytes, 262144);
  // dense 전층 full-attention이라 순진식과 엔진이 같다 — 슬라이딩·하이브리드 할인이 *없어야* 맞다(거짓 할인 = 거짓 fits)
  assert.equal(naiveKVCache(m, 131072, 16), 32);
  assert.equal(naiveKVCache(m, 131072, 16) * 1024 ** 3, kv.totalBytes);
});

test('test_catalog_row_exact_pinned_fields', () => {
  const m = granite();
  assert.deepEqual(
    (({ group, tags, totalParams, activeParams, layerCount, kvHeads, kvHeadDim, attnHeads, hiddenSize, maxContext, benchmarks }) =>
      ({ group, tags, totalParams, activeParams, layerCount, kvHeads, kvHeadDim, attnHeads, hiddenSize, maxContext, benchmarks }))(m),
    {
      group: 'Granite', tags: ['dense'], totalParams: 29.277, activeParams: 29.277, layerCount: 64,
      kvHeads: 8, kvHeadDim: 128, attnHeads: 32, hiddenSize: 4096, maxContext: 131072, benchmarks: null,
    },
  );
  assert.equal(Math.round(m.totalParams * 1e9), Math.round(TOTAL_PARAMS / 1e6) * 1e6); // 29.277B = 29,276,770,304 소수 3자리
  // 직접 상주 의미 — 슬라이딩/PLE/MLA/MTP/선형 전제 없음(어느 플랫폼에서도 structuralAssumptions 키가 붙지 않는다)
  for (const absent of ['slidingWindow', 'slidingPattern', 'globalAttnLayers', 'pleParams', 'pleOffloadVerified', 'mlaKvLoraRank', 'mlaRopeDim',
    'mtpLayerCount', 'linearAttn', 'fullAttnLayers', 'globalKvHeads', 'globalHeadDim', 'numExperts', 'expertsPerToken', 'isCloud']) {
    assert.equal(m[absent], undefined, `${absent} must be absent`);
  }
  assert.deepEqual(structuralAssumptions(m, gpu('RTX 4090')), []);
  assert.deepEqual(structuralAssumptions(m, 64), []);
  assert.doesNotMatch(m.desc, /tok(?:ens)?\/s/i); // 속도 주장 금지
});

test('test_catalog_append_index_and_group_order_stable', () => {
  const BASE_NAMES = [
    'GLM-4.7-Flash', 'GLM-5.2', 'gpt-oss-20b', 'gpt-oss-120b', 'Qwen 3.6 27B', 'Qwen 3.6 35B-A3B', 'Qwen-AgentWorld-35B-A3B',
    'Gemma 4 e2b', 'Gemma 4 e4b', 'Gemma 4 12b', 'Gemma 4 26b A4B', 'Gemma 4 31b', 'Llama-3.2-3B-Instruct', 'Llama-3.1-8B-Instruct',
    'MiniCPM5-1B', 'Qwen3-0.6B', 'Qwen3-1.7B', 'Llama-3.2-1B-Instruct', 'Gemma-3-1B-it', 'Claude Opus 4.7', 'Hy3',
    'Qwen 3.8 27B', 'Qwen 3.8 2.4T-A95B', 'Laguna XS 2.1', 'Laguna S 2.1', 'Spark-X2.5-4B',
  ];
  // ?m= 공유링크 인덱스 0–25 불변, Granite는 2.14.0 시점 배열 끝 append(MODELS 인덱스 26) — 이후 모델이 뒤에 붙어도 이 인덱스는 불변
  assert.deepEqual(MODELS.slice(0, BASE_NAMES.length).map((m) => m.name), BASE_NAMES);
  assert.equal(MODELS.length, BASE_NAMES.length + 2); // 2.15.0 GLM-5.3 append(MODELS 인덱스 27)
  assert.equal(MODELS[BASE_NAMES.length].name, 'Granite-4.2-30B');
  assert.equal(LOCAL_MODELS[25].name, 'Granite-4.2-30B'); // ?m=25
  assert.equal(MODELS.filter((m) => /granite/i.test(m.name)).length, 1); // 3B/8B 형제·base 모델은 이번 변경에 넣지 않는다
  assert.equal(LOCAL_MODELS.length, 27); // 26(2.14.x) + GLM-5.3(2.15.0, 인덱스 26) — 앞 26개는 그대로
  // 기존 그룹의 상대 순서 불변 + Granite 그룹 등재(신규 그룹은 앞 — Spark 전례)
  const BASE_ORDER = ['Spark', 'Qwen 3.8', 'Laguna', 'GLM', 'gpt-oss', 'Qwen 3.6', 'Qwen3.5', 'Hunyuan', 'Gemma 4', 'Llama', 'MiniCPM', 'Draft'];
  assert.deepEqual(MODEL_GROUP_ORDER.filter((g) => g !== 'Granite'), BASE_ORDER);
  assert.equal(MODEL_GROUP_ORDER[0], 'Granite');
  // 이름 해석: 정확 일치 + 토큰 일치(표기 차이 무관). 별칭 테이블은 추가하지 않는다 — 정본 이름 하나뿐.
  assert.equal(resolveLocalModel('Granite-4.2-30B').matchedBy, 'exact');
  for (const q of ['granite-4.2-30b', 'granite 4.2 30b', 'Granite 4.2 30B', 'GRANITE_4_2_30B']) {
    assert.equal(resolveLocalModel(q).canonicalName, 'Granite-4.2-30B', q);
  }
  // HF id 형태(org 접두)는 카탈로그가 아니라 종전대로 라이브 HF 경로로 간다 — 토큰 'ibm'이 이름에 없어 카탈로그 일치가 아니다.
  assert.equal(resolveLocalModel('ibm-granite/granite-4.2-30b').status, 'unknown');
  // 다른 카탈로그 항목과 충돌 없음 — 기존 모호 질의는 그대로 모호
  for (const q of ['llama', 'gemma', 'qwen', 'glm']) assert.equal(resolveLocalModel(q).status, 'ambiguous', q);
});

test('test_fit_4090_q4_8k_hand_derived_env_boundary', () => {
  // 손계산(카탈로그 29.277B): 가중치 29.277e9 × 4.8944/8 B ÷ 2^30 = 16.6815 GiB · KV 8K F16 = 2 GiB
  // rtDyn = 0.12 × 16.6815 + 0.15 × 2 + 8192 × 0.00003 = 2.0018 + 0.3 + 0.2458 = 2.5475 GiB
  // Windows+display reserve 2.0 → used 23.229 / 24 → free 0.771 < headroom 1.2 → 'tight'
  // Linux headless reserve 0.6 → used 21.829 → free 2.171 ≥ 1.2 → 'yes' — 판정이 환경 reserve 하나로 갈리는 경계 사례
  const m = granite();
  const paramGiB = (29.277e9 * (Q4_K_M / 8)) / 1024 ** 3;
  assert.ok(Math.abs(paramGiB - 16.6815) < 1e-3, `param ${paramGiB}`);
  const rtDyn = paramGiB * 0.12 + 2 * 0.15 + 8192 * 0.00003;
  const win = simulate(m, gpu('RTX 4090', 'windows-display'), 8192, { weightBpw: Q4_K_M, kvBits: 16 });
  assert.ok(Math.abs(win.param - paramGiB) < 1e-9);
  assert.equal(win.kv, 2);
  assert.ok(Math.abs(win.rtDyn - rtDyn) < 1e-9);
  assert.equal(win.reserve, 2);
  assert.ok(Math.abs(win.used - (paramGiB + 2 + rtDyn + 2)) < 1e-9);
  assert.equal(win.verdict, 'tight');
  assert.equal('structuralAssumptions' in win, false);
  const headless = simulate(m, gpu('RTX 4090', 'linux-headless'), 8192, { weightBpw: Q4_K_M, kvBits: 16 });
  assert.equal(headless.verdict, 'yes');
  assert.ok(Math.abs(headless.used - (paramGiB + 2 + rtDyn + 0.6)) < 1e-9);
  // 경계 확장: 32K는 어느 환경에서도 초과, 네이티브 128K는 KV(32 GiB)만으로 카드를 넘긴다
  assert.equal(simulate(m, gpu('RTX 4090', 'linux-headless'), 32768, { weightBpw: Q4_K_M, kvBits: 16 }).verdict, 'no');
  const far = simulate(m, gpu('RTX 4090', 'linux-headless'), 131072, { weightBpw: Q4_K_M, kvBits: 16 });
  assert.equal(far.verdict, 'no');
  assert.equal(far.kv, 32);
  assert.ok(far.kv > far.param);
});

test('test_cli_contract_tight_8k_kv_bound_128k', () => {
  // 기본(8K ctx, Q4_K_M, Windows+display reserve): 29.3B 가중치 16.68 GiB로 24GB 카드에 빠듯 — exit 0(TIGHT는 실패가 아니다)
  const ok = run(['Granite-4.2-30B', '--gpu', 'RTX 4090', '--json']);
  assert.equal(ok.code, 0, ok.out);
  const j = JSON.parse(ok.out);
  assert.equal(j.verdict, 'tight');
  assert.equal('structuralAssumptions' in j, false); // 직접 상주 GQA — 전제 키 없음(기존 JSON shape)
  assert.equal(j.breakdown.paramGB, 16.68);
  assert.equal(j.breakdown.kvGB, 2);
  assert.equal(j.breakdown.linearStateGB, 0);
  assert.equal(j.breakdown.reserveGB, 2);
  const text = run(['Granite-4.2-30B', '--gpu', 'RTX 4090']);
  assert.equal(text.code, 0, text.out);
  assert.match(text.out, /△ TIGHT — Granite-4\.2-30B on RTX 4090/);
  assert.match(text.out, /receipt: https:\/\/fitllm\.run\/r\/granite-4-2-30b-q4_k_m-on-rtx-4090/);
  // 네이티브 128K ctx: 가중치가 아니라 KV 상주(정확히 32 GiB)가 카드를 넘긴다
  const far = run(['Granite-4.2-30B', '--gpu', 'RTX 4090', '--ctx', '131072', '--json']);
  assert.equal(far.code, 1, far.out);
  const k = JSON.parse(far.out);
  assert.equal(k.verdict, 'no');
  assert.equal(k.breakdown.kvGB, 32);
  assert.ok(k.breakdown.kvGB > k.breakdown.paramGB, `kv ${k.breakdown.kvGB} must exceed weights ${k.breakdown.paramGB}`);
  assert.ok(k.breakdown.kvGB > 24, `kv ${k.breakdown.kvGB} GiB alone exceeds the 24 GiB card`);
  // --why: 감사 근거가 dense GQA 구조를 그대로 말한다(64/64 레이어가 KV 보유)
  const why = JSON.parse(run(['Granite-4.2-30B', '--gpu', 'RTX 4090', '--json', '--why']).out);
  assert.deepEqual(why.basis.attention, { kind: 'gqa', totalLayers: 64, kvLayers: 64, kvHeads: 8, kvHeadDim: 128 });
  assert.equal(why.basis.model.totalParamsB, 29.277);
});

test('test_census_contains_granite_rows', () => {
  const census = readJson('../census/census-v1.json');
  const manifest = readJson('../census/manifest.json');
  const rows = census.data.filter((r) => r.model === 'Granite-4.2-30B');
  assert.equal(rows.length, 57 * 3 + 36 * 5); // Mac 57구성 × 3 tiers + GPU 36종 × 5 tiers = 351 rows/model
  const total = LOCAL_MODELS.length * (57 * 3 + 36 * 5); // 27 × 351 = 9,477 (2.15.0: GLM-5.3 1행 추가)
  assert.equal(census.verdicts, total);
  assert.equal(census.data.length, total);
  assert.equal(manifest.rows, total);
  assert.ok(rows.every((r) => r.params_b === 29.277 && r.linear_state_gb === 0 && r.ctx === 8192));
});

test('test_release_version_bumped_and_readme_counts', () => {
  // 카탈로그 1행 추가 = minor bump 2.13.0 → 2.14.0 (#93) — 버전 표면 3곳과 수치 표면을 함께 현행화.
  // 2.14.1 = gpt-oss totalParams를 HF 증거값(20.9/116.8)으로 맞춘 patch bump (#98 잔여).
  // 2.15.0 = PLE parser fail-closed + residency-policy 정정 + text-only GLM-5.3 (minor bump, 카탈로그 27행).
  const pkg = readJson('../package.json');
  const lock = readJson('../package-lock.json');
  assert.equal(pkg.version, '2.15.0');
  assert.equal(lock.version, '2.15.0');
  assert.equal(ENGINE_VERSION, '2.15.0');
  const readme = readFileSync(new URL('../README.md', import.meta.url), 'utf8');
  const agents = readFileSync(new URL('../AGENTS.md', import.meta.url), 'utf8');
  const verdicts = `${readJson('../census/manifest.json').rows.toLocaleString('en-US')} verdicts`; // 산출물에서 유도
  for (const s of ['uses: click6067-ship-it/fitllm-engine@v2.15.0', 'conformance_vectors-30%2F30', '30 language-neutral', verdicts, `${LOCAL_MODELS.length} models incl. draft tier`]) {
    assert.ok(readme.includes(s), `README missing: ${s}`);
  }
  for (const s of [verdicts, '30 byte-exact anchors']) assert.ok(agents.includes(s), `AGENTS missing: ${s}`);
  assert.doesNotMatch(readme, /Granite[^\n]*tok(?:ens)?\/s/i); // 속도 주장 금지
  const vectors = readJson('../vectors/fit-vectors-v1.json');
  assert.equal(vectors.vectors.length, 30);
  const v = vectors.vectors.find((x) => x.id === 'granite42-30b-kv-128k-f16');
  assert.deepEqual(v, {
    id: 'granite42-30b-kv-128k-f16', kind: 'kv_total_bytes', model: 'Granite-4.2-30B', ctx: 131072, kvBits: 16, expect: KV_128K_F16,
    note: v.note,
  });
  assert.match(v.note, new RegExp(REVISION));
  assert.equal(vectors.vectors.filter((x) => x.model === 'Granite-4.2-30B').length, 1); // 정확히 벡터 1개
});
