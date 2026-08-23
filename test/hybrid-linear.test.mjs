// 하이브리드 선형 어텐션(Gated DeltaNet) + Qwen 3.8 / Laguna 2.1 배치 회귀 테스트 — 2026-08-24.
// 기대값은 전부 손계산 리터럴. config 필드는 공식 HF config.json 실측치를 합성으로 재현한다.
// 거부 케이스는 "그럴듯한 오답"이 나오던 실제 아키텍처에서 뽑았다(라이브 config로 재현 확인).
import { test } from 'node:test';
import { strict as assert } from 'node:assert';
import { MODELS, LOCAL_MODELS, calcKVCache, calcLinearState, parseHfConfig, simulate, gpuDevice, GPUS } from '../engine.js';

const byName = (n) => MODELS.find((m) => m.name === n);

// ── 1) 신규 모델 KV — 전부 손계산 ──────────────────────────────────────────────
test('Qwen 3.8 27B: KV는 풀어텐션 16레이어만 = 2×4kvh×256d×2B×16L×262144 = 17,179,869,184 B', () => {
  assert.equal(calcKVCache(byName('Qwen 3.8 27B'), 262144, 16).totalBytes, 2 * 4 * 256 * 2 * 16 * 262144);
  // 64레이어 전부를 full attention으로 오산하면 4배(68.7GB)가 된다 — 순진한 계산기의 전형적 오류
  assert.equal(2 * 4 * 256 * 2 * 64 * 262144, 68719476736);
});

test('Qwen 3.8 2.4T-A95B: KV = 2×4kvh×256d×2B×23L×262144 = 24,696,061,952 B', () => {
  assert.equal(calcKVCache(byName('Qwen 3.8 2.4T-A95B'), 262144, 16).totalBytes, 2 * 4 * 256 * 2 * 23 * 262144);
});

test('Laguna XS 2.1: 슬라이딩512 — global 10L×전체ctx + local 30L×512 = 10,800,332,800 B', () => {
  const per = 2 * 8 * 128 * 2; // 4096 B/layer/token
  assert.equal(calcKVCache(byName('Laguna XS 2.1'), 262144, 16).totalBytes, per * 10 * 262144 + per * 30 * 512);
});

test('Laguna S 2.1: global 12L + local 36L(window 512) = 12,960,399,360 B', () => {
  const per = 2 * 8 * 128 * 2;
  assert.equal(calcKVCache(byName('Laguna S 2.1'), 262144, 16).totalBytes, per * 12 * 262144 + per * 36 * 512);
});

// ── 2) 선형 어텐션 고정 상태 ─────────────────────────────────────────────────
test('선형 상태 = conv(conv_dim×kernel×2B) + recurrent(numV×kDim×vDim×4B), 레이어당 × 레이어수', () => {
  // Qwen 3.8 27B: key_dim 16×128=2048, value_dim 48×128=6144 → conv_dim 2048×2+6144 = 10240
  const conv = 10240 * 4 * 2;                 //     81,920 B
  const rec = 48 * 128 * 128 * 4;             //  3,145,728 B
  assert.equal(calcLinearState(byName('Qwen 3.8 27B')).totalBytes, (conv + rec) * 48); // 154,927,104
  assert.equal(calcLinearState(byName('Qwen 3.8 27B')).totalBytes, 154927104);
  // 2.4T-A95B: value_dim 128×128=16384 → conv_dim 20480, recurrent 128×128×128×4
  assert.equal(calcLinearState(byName('Qwen 3.8 2.4T-A95B')).totalBytes, (20480 * 4 * 2 + 128 * 128 * 128 * 4) * 69);
});

test('선형 상태는 ctx에 비례하지 않는다(고정) — KV와 근본적으로 다른 항목', () => {
  const m = byName('Qwen 3.8 27B');
  const a = calcLinearState(m).totalBytes;
  assert.equal(a, calcLinearState(m).totalBytes); // 순수 함수, ctx 인자 자체가 없다
  assert.ok(calcKVCache(m, 8192, 16).totalBytes < calcKVCache(m, 262144, 16).totalBytes); // KV는 늘어난다
});

test('표준 어텐션 모델의 선형 상태는 정확히 0 (신규 항목이 기존 모델을 오염시키지 않음)', () => {
  for (const n of ['Hy3', 'Llama-3.1-8B-Instruct', 'Gemma 4 31b', 'GLM-5.2', 'Laguna XS 2.1', 'gpt-oss-120b']) {
    assert.equal(calcLinearState(byName(n)).totalBytes, 0, `${n}: 선형 상태가 0이 아님`);
  }
});

test('linearState는 simulate의 used에 반영된다 (분해 합 == used)', () => {
  const s = simulate(byName('Qwen 3.8 27B'), gpuDevice(GPUS.find((g) => g.name === 'RTX 5090')), 32768, { weightBpw: 4, kvBits: 16 });
  assert.ok(s.linearState > 0);
  assert.ok(Math.abs(s.param + s.kv + s.linearState + s.rtDyn + s.reserve - s.used) < 1e-9);
});

// ── 3) parseHfConfig: qwen3_5 하이브리드 허용 ────────────────────────────────
const QWEN38_27B = {
  model_type: 'qwen3_5',
  image_token_id: 248056,
  text_config: {
    model_type: 'qwen3_5_text', num_hidden_layers: 64, num_attention_heads: 24, num_key_value_heads: 4,
    head_dim: 256, hidden_size: 5120, intermediate_size: 17408, vocab_size: 248320, max_position_embeddings: 262144,
    linear_num_key_heads: 16, linear_num_value_heads: 48, linear_key_head_dim: 128, linear_value_head_dim: 128,
    linear_conv_kernel_dim: 4, mamba_ssm_dtype: 'float32',
    layer_types: Array.from({ length: 64 }, (_, i) => ((i + 1) % 4 === 0 ? 'full_attention' : 'linear_attention')),
  },
  vision_config: { depth: 27, hidden_size: 1152 },
};

test('붙여넣기: qwen3_5 하이브리드는 내장 Qwen 3.8 27B와 KV·선형상태가 동일해야 한다', () => {
  const m = parseHfConfig('Qwen/Qwen3.8-27B', QWEN38_27B, 55562855904);
  const b = byName('Qwen 3.8 27B');
  assert.equal(m.fullAttnLayers, 16);
  assert.equal(m.layerCount, 64);
  assert.equal(calcKVCache(m, 262144, 16).totalBytes, calcKVCache(b, 262144, 16).totalBytes);
  assert.equal(calcLinearState(m).totalBytes, calcLinearState(b).totalBytes);
  assert.ok(Math.abs(m.totalParams - 27.8) < 0.1, `params ${m.totalParams}`); // 55,562,855,904 / 2B
});

test('붙여넣기: linear 레이어인데 linear_* 치수가 없으면 거부(상태 크기를 모른다)', () => {
  const noDims = { ...QWEN38_27B, text_config: { ...QWEN38_27B.text_config, linear_num_value_heads: undefined } };
  assert.throws(() => parseHfConfig('x/y', noDims, 1e10), /linear\/recurrent/);
});

// ── 4) parseHfConfig: 실제로 오답을 내던 아키텍처 거부 ────────────────────────
test('거부: Mamba/SSM 블록 스케줄(layers_block_type) — 전 레이어 full attention 오산 방지', () => {
  // nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B: 52레이어 중 attention은 6개뿐인데
  // layer_types가 없어 52개 전부로 계산되던 케이스(KV 약 8.7배 과대)
  const cfg = { model_type: 'nemotron_h', num_hidden_layers: 52, num_attention_heads: 32, num_key_value_heads: 2,
    head_dim: 128, hidden_size: 2688, ssm_state_size: 128, mamba_num_heads: 64,
    layers_block_type: ['mamba', 'moe', 'attention'] };
  assert.throws(() => parseHfConfig('nvidia/nemotron', cfg, 65842365568), /Mamba\/SSM/);
});

test('거부: MLA + 선형 혼합(bailing_hybrid) — MLA 균일 경로로 계산하면 틀린다', () => {
  const cfg = { model_type: 'bailing_hybrid', num_hidden_layers: 24, num_attention_heads: 16, num_key_value_heads: 16,
    head_dim: 128, hidden_size: 1536, kv_lora_rank: 512, qk_rope_head_dim: 64,
    num_kv_heads_for_linear_attn: 0, layer_group_size: 4, short_conv_kernel_size: 4 };
  assert.throws(() => parseHfConfig('inclusionAI/Ling-3.0-tiny', cfg, 1e11), /MLA \+ 선형/);
});

test("거부: 모르는 레이어 타입('conv' 등) — lfm2류", () => {
  const cfg = { model_type: 'lfm2', num_hidden_layers: 30, num_attention_heads: 32, num_key_value_heads: 8,
    hidden_size: 2048, conv_L_cache: 3,
    layer_types: ['conv', 'conv', 'full_attention'] };
  assert.throws(() => parseHfConfig('LiquidAI/LFM2.5-2.6B', cfg, 5394397184), /알 수 없는 어텐션 레이어 타입/);
});

test('거부: 레포 이름이 타깃 모델을 가리키는 드래프트 헤드 (z-lab/Qwen3.8-27B-DFlash2 = 5레이어)', () => {
  const draft = { model_type: 'qwen3', num_hidden_layers: 5, num_attention_heads: 32, num_key_value_heads: 8,
    head_dim: 128, hidden_size: 5120, intermediate_size: 17408, vocab_size: 248320,
    sliding_window: 2048, use_sliding_window: true, max_position_embeddings: 262144,
    layer_types: Array(5).fill('sliding_attention') };
  // totalSize 없음 → 이름에서 "27B"를 줍는 경로. config 치수는 ~4B라 2배 가드에 걸려야 한다.
  assert.throws(() => parseHfConfig('z-lab/Qwen3.8-27B-DFlash2', draft, null), /드래프트\/어댑터/);
  // 실제 크기(1.92B×2B)를 주면 정상 계산된다
  const ok = parseHfConfig('z-lab/Qwen3.8-27B-DFlash2', draft, 3848817896);
  assert.ok(ok.totalParams > 1.5 && ok.totalParams < 2.5, `params ${ok.totalParams}B`);
});

// ── 5) 카탈로그 정합 ─────────────────────────────────────────────────────────
test('신규 4종이 카탈로그에 있고 필수 필드가 채워져 있다', () => {
  for (const n of ['Qwen 3.8 27B', 'Qwen 3.8 2.4T-A95B', 'Laguna XS 2.1', 'Laguna S 2.1']) {
    const m = byName(n);
    assert.ok(m, `${n} 없음`);
    assert.ok(LOCAL_MODELS.includes(m), `${n}이 LOCAL_MODELS에 없음`);
    for (const k of ['totalParams', 'layerCount', 'kvHeads', 'kvHeadDim', 'attnHeads', 'hiddenSize', 'maxContext']) {
      assert.ok(m[k] > 0, `${n}.${k} 누락`);
    }
  }
});

test('linearAttn을 가진 모델은 fullAttnLayers도 반드시 갖는다(선형 레이어만 세고 KV를 빼먹는 실수 방지)', () => {
  for (const m of LOCAL_MODELS) {
    if (!m.linearAttn) continue;
    assert.ok(m.fullAttnLayers > 0, `${m.name}: linearAttn은 있는데 fullAttnLayers가 없음`);
    assert.equal(m.linearAttn.layers + m.fullAttnLayers, m.layerCount, `${m.name}: linear+full != layerCount`);
  }
});

test('ENGINE_VERSION == package.json version (소비처가 표시하는 버전의 단일 출처)', async () => {
  const { ENGINE_VERSION } = await import('../engine.js');
  const { readFileSync } = await import('node:fs');
  const pkg = JSON.parse(readFileSync(new URL('../package.json', import.meta.url), 'utf8'));
  assert.equal(ENGINE_VERSION, pkg.version);
});

test('표시 순서: 배열은 append-only여도 신규 그룹이 카탈로그 앞에 온다', async () => {
  const { groupedForDisplay, MODEL_GROUP_ORDER } = await import('../engine.js');
  const order = groupedForDisplay(LOCAL_MODELS).map((g) => g.group);
  assert.equal(order[0], 'Qwen 3.8', `첫 그룹이 ${order[0]}`);
  assert.equal(order[1], 'Laguna');
  assert.ok(order.indexOf('Draft') > order.indexOf('Qwen 3.6'), 'Draft는 실모델 뒤');
  // 모델 유실 없음
  assert.equal(groupedForDisplay(LOCAL_MODELS).reduce((n, g) => n + g.items.length, 0), LOCAL_MODELS.length);
  // 목록에 없는 그룹도 사라지지 않는다
  const withNew = [...LOCAL_MODELS, { name: 'X', group: 'ZZZ-New' }];
  assert.ok(groupedForDisplay(withNew).map((g) => g.group).includes('ZZZ-New'));
  // 카탈로그의 모든 그룹이 순서 목록에 등재돼 있는지(신규 추가 시 알림)
  const missing = [...new Set(LOCAL_MODELS.map((m) => m.group))].filter((g) => !MODEL_GROUP_ORDER.includes(g));
  assert.deepEqual(missing, [], `MODEL_GROUP_ORDER에 없는 그룹: ${missing.join(', ')}`);
});
