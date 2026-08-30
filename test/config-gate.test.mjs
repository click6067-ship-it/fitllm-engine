// 미지 구조 필드 게이트(issue #87) + total_size sanity check 회귀 테스트.
// 전부 합성 config다 — 외부 레포는 언제든 바뀌므로 테스트가 네트워크에 의존하면 안 된다.
// 각 케이스에 실제로 이 동작을 요구한 실측 사례를 주석으로 남긴다.
import { test } from 'node:test';
import { strict as assert } from 'node:assert';
import { parseHfConfig } from '../engine.js';

// 표준 dense 모델 — 게이트를 그냥 통과해야 하는 기준선.
const BASE = {
  model_type: 'llama',
  num_hidden_layers: 32, num_attention_heads: 32, num_key_value_heads: 8,
  hidden_size: 4096, head_dim: 128, intermediate_size: 14336, vocab_size: 128256,
  max_position_embeddings: 131072, torch_dtype: 'bfloat16',
};
// BASE 치수의 실제 파라미터 규모 ≈ 8B → total_size 16GB가 정합한다.
const SIZE_8B = 16e9;

test('기준선: 표준 dense config는 통과한다', () => {
  const m = parseHfConfig('meta/Llama-3-8B', BASE, SIZE_8B);
  assert.equal(m.layerCount, 32);
  assert.ok(m.totalParams > 7 && m.totalParams < 9, `totalParams ${m.totalParams}B`);
});

// ── 미지 구조 필드 게이트 ────────────────────────────────────────────────────
test('거부: 구조에 영향 줄 수 있는 모르는 필드 (Breeze-TTS-2가 LLM으로 통과하던 경로)', () => {
  // 실측 2026-08-31: TTS 모델이 num_hidden_layers/head_dim만 표준처럼 생겨서 3.5B LLM으로 통과했다.
  assert.throws(
    () => parseHfConfig('x/tts', { ...BASE, text_encoder_lora_config: { r: 8 } }, SIZE_8B),
    /구조를 알 수 없는 config 필드/
  );
});

test('거부: MHC(Multi-Head Compression) 같은 미지 어텐션 변형 — Motif-3', () => {
  // MLA 필드가 있다는 이유로 MLA 균일 경로를 타면 틀린다. mhc_* 를 모르면 계산하지 않는다.
  assert.throws(
    () => parseHfConfig('Motif/Motif-3', { ...BASE, kv_lora_rank: 512, qk_rope_head_dim: 64, mhc_enabled: true }, SIZE_8B),
    /구조를 알 수 없는 config 필드/
  );
});

test('통과: 메모리 레이아웃을 안 바꾸는 무해 필드 (Granite attention_multiplier)', () => {
  // 어텐션 스코어에 곱하는 스칼라 — 텐서 치수 불변이므로 거부하면 거짓 거부가 된다.
  const m = parseHfConfig('ibm/granite', { ...BASE, attention_multiplier: 0.0078, rope_parameters: { type: 'linear' } }, SIZE_8B);
  assert.ok(m.totalParams > 7, `totalParams ${m.totalParams}B`);
});

test('계열 전용 키는 검증된 계열에서만 통과한다 (attn_output_gate)', () => {
  // 같은 이름의 필드라도 미검증 아키텍처에서 의미가 같다는 보장이 없다 — #87의 요지.
  const hybrid = {
    ...BASE, model_type: 'qwen3_5', attn_output_gate: true,
    layer_types: [...Array(8).fill('full_attention'), ...Array(24).fill('linear_attention')],
    linear_num_key_heads: 16, linear_num_value_heads: 32,
    linear_key_head_dim: 128, linear_value_head_dim: 128, linear_conv_kernel_dim: 4,
  };
  const ok = parseHfConfig('Qwen/Qwen3.8-9B', hybrid, SIZE_8B);
  assert.equal(ok.linearAttn.layers, 24);
  assert.equal(ok.fullAttnLayers, 8);

  // 미지 계열(g9v3류)에서 같은 게이팅 필드가 나오면 거부한다
  assert.throws(
    () => parseHfConfig('x/unknown', { ...BASE, model_type: 'g9v3', attn_output_gate: true }, SIZE_8B),
    /구조를 알 수 없는 config 필드/
  );
});

test('MLA 조건부 키: kv_lora_rank가 있을 때만 통과 (v_head_dim)', () => {
  // MLA에선 V도 압축 latent에서 복원돼 별도 캐시가 없다 → 무해.
  const mla = { ...BASE, kv_lora_rank: 512, qk_rope_head_dim: 64, v_head_dim: 128, q_lora_rank: 1536, qk_nope_head_dim: 128 };
  const ok = parseHfConfig('zai-org/GLM-4.7-Flash', mla, SIZE_8B);
  assert.equal(ok.mlaKvLoraRank, 512);

  // 비-MLA에서 v_head_dim은 V의 head_dim이 K와 다르다는 뜻이라 2×kvHeads×headDim 공식이 깨진다
  assert.throws(
    () => parseHfConfig('x/nonmla', { ...BASE, v_head_dim: 256 }, SIZE_8B),
    /구조를 알 수 없는 config 필드/
  );
});

// ── 레이어별 상이 슬라이딩 윈도우 ────────────────────────────────────────────
test('거부: sliding_windows(복수형) — KV 과소계산으로 거짓 fits가 나던 경로', () => {
  // 실측: K-EXAONE-2.0은 sliding_window(단수)가 없어 sliding=0이 되고, 슬라이딩 레이어 58개가
  // 'KV 없는 linear'로 취급돼 78레이어 중 20레이어만 KV를 갖는 것으로 산출됐다.
  const cfg = {
    ...BASE, model_type: 'exaone_moe', num_hidden_layers: 78,
    sliding_windows: [4096, 4096, null], mtp_sliding_windows: [4096],
    layer_types: [...Array(20).fill('full_attention'), ...Array(58).fill('sliding_attention')],
  };
  assert.throws(() => parseHfConfig('LGAI/K-EXAONE-2.0', cfg, 1.5e12), /레이어별로 다른 슬라이딩 윈도우/);
});

// ── total_size 자릿수 붕괴 ───────────────────────────────────────────────────
test('거부: index의 total_size가 config 치수와 자릿수째 다르면 계산하지 않는다', () => {
  // 실측: InternScience/Agents-A1-4B는 shard 2개(텍스트 32L·hidden 2560)인데 total_size가
  // 550.7GB로 적혀 있어 275.3B가 산출됐다. 2샤드에 550GB는 물리적으로 불가능하다.
  assert.throws(
    () => parseHfConfig('x/small-but-huge-metadata', { ...BASE, hidden_size: 2560, intermediate_size: 9728 }, 550.7e9),
    /total_size 메타데이터를 믿을 수 없어/
  );
});

test('통과: 이름이 틀리고 크기가 맞는 경우는 크기를 믿는다 (드래프트 헤드)', () => {
  // 심판을 레포 이름으로 두면 이 케이스가 깨진다 — 그래서 config 치수를 심판으로 쓴다.
  const draft = { ...BASE, num_hidden_layers: 5, hidden_size: 5120, intermediate_size: 17408, vocab_size: 248320 };
  const m = parseHfConfig('z-lab/Qwen3.8-27B-DFlash2', draft, 3848817896);
  assert.ok(m.totalParams > 1.5 && m.totalParams < 2.5, `params ${m.totalParams}B`);
});

test('MoE 전문가 폭이 intermediate_size에 있으면 치수 심판을 건너뛴다 (gpt_oss 거짓 거부 방지)', () => {
  // 실측: gpt-oss-120b는 moe_intermediate_size가 없어 paramsFromDims가 FFN을 0으로 세고
  // ~2.1B로 추정한다. 그 상태로 32.6B와 비교하면 정상 모델을 거짓 거부한다.
  const gptOss = {
    ...BASE, model_type: 'gpt_oss', num_hidden_layers: 36, hidden_size: 2880,
    intermediate_size: 2880, num_local_experts: 128, experts_per_token: 4,
    vocab_size: 201088, sliding_window: 128,
    layer_types: [...Array(18).fill('sliding_attention'), ...Array(18).fill('full_attention')],
  };
  const m = parseHfConfig('openai/gpt-oss-120b', gptOss, 65.2e9);
  assert.ok(m.totalParams > 30, `totalParams ${m.totalParams}B — 거짓 거부되면 안 된다`);
});
