// 미지 구조 필드 게이트(issue #87) + total_size sanity check 회귀 테스트.
// 전부 합성 config다 — 외부 레포는 언제든 바뀌므로 테스트가 네트워크에 의존하면 안 된다.
// 각 케이스에 실제로 이 동작을 요구한 실측 사례를 주석으로 남긴다.
import { test } from 'node:test';
import { strict as assert } from 'node:assert';
import { parseHfConfig, simulate, gpuDevice, GPUS } from '../engine.js';

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

test('MoE 전문가 폭이 intermediate_size에 있으면 치수 심판을 건너뛴다 (거짓 거부 방지)', () => {
  // paramsFromDims는 MoE일 때 moe_intermediate_size만 보므로, 전문가 폭이 intermediate_size에
  // 있는 config는 FFN이 0으로 잡혀 추정이 자릿수째 무너진다. 그 추정으로 심판하면 거짓 거부가 난다.
  const moe = {
    ...BASE, model_type: 'gpt_oss', num_hidden_layers: 36, hidden_size: 2880,
    intermediate_size: 2880, num_local_experts: 128, experts_per_token: 4,
    vocab_size: 201088, sliding_window: 128, num_attention_heads: 64, head_dim: 64,
    layer_types: [...Array(18).fill('sliding_attention'), ...Array(18).fill('full_attention')],
  };
  const m = parseHfConfig('x/moe', moe, 65.2e9);
  assert.ok(m.totalParams > 30, `totalParams ${m.totalParams}B — 거짓 거부되면 안 된다`);
});

// ── 저장 비트폭 (Codex 리뷰 P0) ───────────────────────────────────────────────
test('거부: dtype도 quantization도 없으면 2바이트로 추정하지 않는다 (gpt-oss mxfp4, #98)', () => {
  // 실측: gpt-oss-120b는 torch_dtype·dtype·bits가 전부 없고 quant_method만 "mxfp4"라,
  // 종래엔 기본 2바이트로 가정해 117B 모델을 32.6B로 계산했다(3.6배 과소 = 거짓 fits).
  const { torch_dtype, ...noDtype } = BASE;
  const gptOss = {
    ...noDtype, model_type: 'gpt_oss', num_hidden_layers: 36, hidden_size: 2880,
    num_attention_heads: 64, head_dim: 64, intermediate_size: 2880,
    num_local_experts: 128, vocab_size: 201088,
    quantization_config: { quant_method: 'mxfp4', modules_to_not_convert: ['model.layers.*.self_attn'] },
  };
  assert.throws(() => parseHfConfig('openai/gpt-oss-120b', gptOss, 65248815744), /저장 비트폭/);
});

test('최상위 quantization_config + 중첩 text_config를 함께 읽는다 (Qwen GPTQ-Int4 4배 과소계산)', () => {
  // 실측: Qwen/Qwen3.5-397B-A17B-GPTQ-Int4 는 quantization_config.bits=4가 raw 최상위에 있고
  // text_config는 중첩이라, inner만 읽던 종래 코드가 /2 를 적용해 397B를 117.8B로 계산했다.
  // 치수 심판도 117.8 vs ~392.9 = 3.3배라 4배 경계 안이라 못 막았다.
  const raw = {
    architectures: ['Qwen3VLMoeForConditionalGeneration'],
    quantization_config: { bits: 4, quant_method: 'gptq' },
    text_config: {
      model_type: 'qwen3_moe', num_hidden_layers: 60, num_attention_heads: 64,
      num_key_value_heads: 4, head_dim: 128, hidden_size: 4096,
      moe_intermediate_size: 1536, num_experts: 128, vocab_size: 151936,
      max_position_embeddings: 262144,
    },
  };
  const m = parseHfConfig('Qwen/Qwen3.5-397B-A17B-GPTQ-Int4', raw, 235657499488);
  // 4bit = 0.5B/param → 235.7GB / 0.5 ≈ 471B. /2 로 읽으면 117.8B가 나온다.
  assert.ok(m.totalParams > 400, `totalParams ${m.totalParams}B — 4비트를 못 읽고 2바이트로 나눴다`);
});

test('거부: 양자화 정보가 최상위와 text_config에서 서로 다르면 어느 쪽도 믿지 않는다', () => {
  const raw = {
    quantization_config: { bits: 4, quant_method: 'gptq' },
    text_config: { ...BASE, quantization_config: { bits: 8, quant_method: 'gptq' } },
  };
  assert.throws(() => parseHfConfig('x/conflict', raw, 16e9), /서로 달라요/);
});

// ── 필수 KV 치수·값 검증 (Codex 리뷰 P0) ─────────────────────────────────────
test('거부: num_attention_heads가 없으면 kvHeads를 1로 떨어뜨리지 않는다', () => {
  // 종래: kvHeads = num_key_value_heads ?? num_attention_heads ?? 1 → 32-head MHA가 1로 계산돼
  // 128K KV가 64GiB인데 2GiB로 나왔다(32배 과소 = 거짓 fits).
  const { num_attention_heads, num_key_value_heads, ...noHeads } = BASE;
  assert.throws(() => parseHfConfig('x/noheads', noHeads, 16e9), /num_attention_heads/);
});

test('거부: num_key_value_heads가 0이면 KV가 0으로 계산되는 것을 막는다', () => {
  assert.throws(() => parseHfConfig('x/zerokv', { ...BASE, num_key_value_heads: 0 }, 16e9), /양의 정수가 아니에요/);
});

test('거부: layer_types 길이가 num_hidden_layers와 다르면 계산하지 않는다', () => {
  const cfg = { ...BASE, num_hidden_layers: 32, layer_types: Array(8).fill('full_attention') };
  assert.throws(() => parseHfConfig('x/mismatch', cfg, 16e9), /layer_types 길이/);
});

test('거부: 슬라이딩 레이어가 있는데 윈도우 크기를 못 읽으면 KV를 삭제하지 않는다', () => {
  // sliding_windows(복수형) 외에도 swa_size처럼 이름이 다른 경우가 있다. 개별 필드명이 아니라
  // "윈도우를 못 읽었다"는 상태로 막아야 한다 — 안 그러면 sliding 레이어가 KV 0으로 취급된다.
  const cfg = {
    ...BASE, num_hidden_layers: 32, swa_size: 4096,
    layer_types: [...Array(8).fill('full_attention'), ...Array(24).fill('sliding_attention')],
  };
  assert.throws(() => parseHfConfig('x/swa', cfg, 16e9), /윈도우 크기를 읽을 수 없어|구조를 알 수 없는/);
});

// ── 2차 Codex 리뷰 반영 (기형·악성 config에서 거짓 fits) ─────────────────────
test('거부: 양자화 비트폭이 별칭 간에 어긋난다 (최상위 bits=4 + 중첩 quantization.bits=8)', () => {
  // 같은 키끼리만 비교하면 이 우회를 놓친다 — 종래엔 8비트로 계산됐다.
  const raw = {
    quantization_config: { bits: 4, quant_method: 'gptq' },
    text_config: { ...BASE, quantization: { bits: 8 } },
  };
  assert.throws(() => parseHfConfig('x/alias-conflict', raw, 16e9), /양자화 비트폭 선언이 서로 달라요/);
});

test('거부: dtype이 별칭 간에 어긋난다 (최상위 torch_dtype=float32 + 중첩 dtype=bfloat16)', () => {
  const { torch_dtype, ...base } = BASE;
  const raw = { torch_dtype: 'float32', text_config: { ...base, dtype: 'bfloat16' } };
  assert.throws(() => parseHfConfig('x/dtype-conflict', raw, 16e9), /dtype 선언이 서로 달라요/);
});

test('거부: 음수 비트폭 (bits=-4 → totalParams가 음수가 되어 verdict yes)', () => {
  assert.throws(
    () => parseHfConfig('x/negbits', { ...BASE, quantization_config: { bits: -4 } }, 16e9),
    /양수가 아니에요/
  );
});

test('통과: 같은 양자화 선언이 키 순서만 다르면 거부하지 않는다', () => {
  // JSON.stringify 비교는 프로퍼티 순서에 의존해 거짓 거부를 냈다 — 값 기반으로 바꿨다.
  const raw = {
    quantization_config: { bits: 4, quant_method: 'gptq' },
    text_config: { ...BASE, quantization_config: { quant_method: 'gptq', bits: 4 } },
  };
  const m = parseHfConfig('x/order', raw, 8e9);
  assert.ok(m.totalParams > 15, `totalParams ${m.totalParams}B — 4비트로 읽혀야 한다`);
});

test('거부: linear_num_key_heads 누락 (선형 상태가 NaN → verdict yes)', () => {
  const cfg = {
    ...BASE, model_type: 'qwen3_5',
    layer_types: [...Array(8).fill('full_attention'), ...Array(24).fill('linear_attention')],
    linear_num_value_heads: 32, linear_key_head_dim: 128, linear_value_head_dim: 128,
  };
  assert.throws(() => parseHfConfig('x/nolinearkey', cfg, 16e9), /linear\/recurrent/);
});

test('거부: global_head_dim 음수 (KV가 음수가 되어 verdict yes)', () => {
  assert.throws(
    () => parseHfConfig('x/negglobal', { ...BASE, global_head_dim: -1024 }, 16e9),
    /양의 정수가 아니에요/
  );
});

test('거부: layer_types가 배열이 아니면 타입·길이 검사를 우회한다', () => {
  assert.throws(
    () => parseHfConfig('x/strlayers', { ...BASE, layer_types: 'sliding_attention' }, 16e9),
    /layer_types가 배열이 아니에요/
  );
});

test('거부: 최상위 cross_attention_config (종래엔 text_config만 검사해 통과했다)', () => {
  const raw = { cross_attention_config: { layers: [3, 8] }, text_config: { ...BASE } };
  assert.throws(() => parseHfConfig('x/xattn', raw, 16e9), /cross-attention/);
});

test('거부: 구조 필드가 최상위에만 있어도 잡는다', () => {
  const raw = { mhc_enabled: true, text_config: { ...BASE } };
  assert.throws(() => parseHfConfig('x/outer-structural', raw, 16e9), /구조를 알 수 없는 config 필드/);
});

test('거부: 최상위와 text_config의 model_type 계열이 갈린다', () => {
  // 래퍼만 qwen3_5고 본체가 미지 계열이면 linear 상태 공식을 허용해선 안 된다.
  const raw = { model_type: 'qwen3_5', text_config: { ...BASE, model_type: 'llama' } };
  assert.throws(() => parseHfConfig('x/family-conflict', raw, 16e9), /model_type 계열이 서로 달라요/);
});

test('거부: total_size가 유효한 양수가 아니면 계산하지 않는다', () => {
  assert.throws(() => parseHfConfig('x/negsize', BASE, -16e9), /total_size.*양수가 아니에요/);
  assert.throws(() => parseHfConfig('x/nansize', BASE, NaN), /total_size.*양수가 아니에요/);
});

test('통과: MoE 배치 필드가 있으면 치수 심판을 건너뛴다 (거짓 거부 방지)', () => {
  // 추정식이 모든 레이어에 모든 전문가가 있다고 세어 과대 추정 → 정상 모델을 거부하던 경로.
  const cfg = {
    ...BASE, num_hidden_layers: 48, hidden_size: 2048, vocab_size: 151936,
    num_attention_heads: 16, head_dim: 128,
    num_experts: 128, moe_intermediate_size: 768, moe_layer_freq: 8,
  };
  const m = parseHfConfig('x/moe-placement', cfg, 105e9);
  assert.ok(m.totalParams > 50, `totalParams ${m.totalParams}B — 거짓 거부되면 안 된다`);
});

test('최종 방어선: used/free가 비유한수면 verdict가 yes가 되지 않는다', () => {
  // 상류 게이트가 모두 뚫려도 "모르는데 된다"고 답하지 않게 하는 마지막 가드.
  // Codex가 재현한 실제 경로: linear_num_key_heads가 빠지면 numKHeads가 undefined가 되고
  // calcLinearState의 곱셈이 NaN이 된다. parseHfConfig는 이제 막지만, 모델 객체가 다른 경로로
  // 들어와도 'yes'가 나오면 안 된다.
  const broken = {
    name: 'broken', totalParams: 8, layerCount: 32, kvHeads: 8, kvHeadDim: 128,
    attnHeads: 32, hiddenSize: 4096, maxContext: 8192,
    linearAttn: { layers: 24, numVHeads: 32, headKDim: 128, headVDim: 128, convKernel: 4 }, // numKHeads 누락
  };
  const s = simulate(broken, gpuDevice(GPUS[0]), 8192, { weightBpw: 4, kvBits: 16 });
  assert.notEqual(s.verdict, 'yes', `verdict ${s.verdict} — NaN인데 yes가 나왔다`);
});
