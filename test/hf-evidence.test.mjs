// HF safetensors 파라미터 증거 회귀 — v2 미러(src/lib/hf-evidence.test.js)와 같은 계약.
// 계기(P0): zai-org/GLM-4.7-Flash의 index metadata.total_size는 *바이트가 아니라*
// 파라미터 수(31,221,488,576)인데 실제 48개 shard 합계는 62,444,175,504바이트다.
// total_size를 바이트로 믿고 저장 dtype으로 나누면 파라미터가 정확히 절반이 되어
// 붙여넣기 경로가 내장 카탈로그와 반대되는 FITS를 냈다.
import test from 'node:test';
import assert from 'node:assert/strict';
import { MODELS, parseHfConfig, resolveParameterCount, simulate, gpuDevice, GPUS } from '../engine.js';

const REVISION = '7dd20894a642a0aa287e9827cb1a1f7f91386b67';
const GLM_PARAMETERS = { BF16: 31_221_485_568, F32: 3_008 };
const GLM_PARAMETER_TOTAL = 31_221_488_576;
const GLM_CHECKPOINT_BYTES = 62_444_175_504;

const GLM_CONFIG = {
  model_type: 'glm4_moe_lite',
  hidden_size: 2048,
  intermediate_size: 10240,
  moe_intermediate_size: 1536,
  max_position_embeddings: 202752,
  num_attention_heads: 20,
  n_routed_experts: 64,
  n_shared_experts: 1,
  num_experts_per_tok: 4,
  first_k_dense_replace: 1,
  num_hidden_layers: 47,
  num_key_value_heads: 20,
  num_nextn_predict_layers: 1,
  dtype: 'bfloat16',
  q_lora_rank: 768,
  kv_lora_rank: 512,
  qk_nope_head_dim: 192,
  qk_rope_head_dim: 64,
  v_head_dim: 256,
  vocab_size: 154880,
};

const DENSE_CONFIG = {
  model_type: 'llama',
  hidden_size: 1024,
  intermediate_size: 4096,
  max_position_embeddings: 32768,
  num_attention_heads: 8,
  num_key_value_heads: 8,
  num_hidden_layers: 8,
  vocab_size: 32000,
  dtype: 'bfloat16',
};
const DENSE_DIMENSION_COUNT = 199_753_728;

test('GLM 내장 경로와 HF 증거 경로가 같은 파라미터·판정을 낸다', () => {
  const builtIn = MODELS.find((model) => model.name === 'GLM-4.7-Flash');
  assert.ok(builtIn, 'GLM-4.7-Flash 카탈로그 항목이 있어야 한다');
  const fromHf = parseHfConfig('zai-org/GLM-4.7-Flash', GLM_CONFIG, GLM_CHECKPOINT_BYTES, {
    revision: REVISION,
    safetensorsParameters: GLM_PARAMETERS,
    safetensorsTotal: GLM_PARAMETER_TOTAL,
  });

  assert.equal(fromHf.parameterSource, 'hf-safetensors-parameters');
  assert.equal(fromHf.totalParams, 31.2);
  assert.ok(Math.abs(fromHf.totalParams - builtIn.totalParams) / builtIn.totalParams <= 0.03);

  const device = gpuDevice(GPUS.find((g) => g.name === 'RTX 5080'), 'linux-headless');
  const builtInResult = simulate(builtIn, device, 32768, { weightBpw: 4, kvBits: 16 });
  const hfResult = simulate(fromHf, device, 32768, { weightBpw: 4, kvBits: 16 });
  assert.equal(hfResult.verdict, builtInResult.verdict);
  assert.ok(Math.abs(hfResult.param - builtInResult.param) < 1e-9);
  assert.ok(Math.abs(hfResult.kv - builtInResult.kv) < 1e-9);
});

test('index total_size를 바이트로 믿던 옛 경로는 절반값을 만든다 (회귀 기준)', () => {
  // 증거 없이 total_size(=파라미터 수)만 바이트로 넘기면 31.2B가 아니라 15.6B가 된다.
  const halved = parseHfConfig('zai-org/GLM-4.7-Flash', GLM_CONFIG, GLM_PARAMETER_TOTAL);
  assert.equal(halved.parameterSource, 'uniform-checkpoint');
  assert.ok(halved.totalParams > 15 && halved.totalParams < 16, `got ${halved.totalParams}B`);
  // 이것이 실제 사용자 피해였다: 같은 하드웨어에서 카탈로그는 no, 붙여넣기는 yes.
  const device = gpuDevice(GPUS.find((g) => g.name === 'RTX 5080'), 'linux-headless');
  assert.equal(simulate(MODELS.find((m) => m.name === 'GLM-4.7-Flash'), device, 32768, { weightBpw: 4, kvBits: 16 }).verdict, 'no');
  assert.equal(simulate(halved, device, 32768, { weightBpw: 4, kvBits: 16 }).verdict, 'yes');
  // 같은 입력에 파라미터 증거를 붙이면 정본이 증거 쪽으로 넘어간다.
  const corrected = parseHfConfig('zai-org/GLM-4.7-Flash', GLM_CONFIG, GLM_CHECKPOINT_BYTES, {
    revision: REVISION,
    safetensorsParameters: GLM_PARAMETERS,
    safetensorsTotal: GLM_PARAMETER_TOTAL,
  });
  assert.ok(corrected.totalParams / halved.totalParams > 1.9);
});

test('packed(GPTQ/AWQ) 혼합 dtype은 논리 파라미터 합계를 쓰고, 의미 드리프트는 막는다', () => {
  const parsed = parseHfConfig(
    'org/packed-int4',
    { ...GLM_CONFIG, quantization_config: { bits: 4 } },
    5_575_381_144,
    {
      revision: REVISION,
      safetensorsParameters: { I32: 6_525_288_448, F16: 1_090_328_064 },
      safetensorsTotal: 7_615_616_512,
    },
  );
  assert.equal(parsed.parameterSource, 'hf-safetensors-parameters');
  assert.equal(parsed.totalParams, 7.6);

  assert.throws(() => parseHfConfig('org/packed-semantics-drift', GLM_CONFIG, 5_575_381_144, {
    revision: REVISION,
    safetensorsParameters: { I32: 815_661_056, F16: 1_090_328_064 },
    safetensorsTotal: 1_905_989_120,
  }), /bytes\/parameter/);
});

test('checkpoint byte가 없으면 config 치수 교차검증이 필수다', () => {
  const ok = parseHfConfig('org/dense-evidence', DENSE_CONFIG, null, {
    revision: REVISION,
    safetensorsParameters: { BF16: DENSE_DIMENSION_COUNT },
    safetensorsTotal: DENSE_DIMENSION_COUNT,
  });
  assert.equal(ok.totalParams, 0.2);
  assert.equal(ok.parameterSource, 'hf-safetensors-parameters');

  assert.throws(() => parseHfConfig('org/unverifiable-moe', GLM_CONFIG, null, {
    revision: REVISION,
    safetensorsParameters: GLM_PARAMETERS,
    safetensorsTotal: GLM_PARAMETER_TOTAL,
  }), /byte.*교차검증/);
});

test('packed 경로는 config 치수 ×2 백스톱을 실제로 통과한다', () => {
  const packedConfig = { ...DENSE_CONFIG, quantization_config: { bits: 4 } };
  const checkpointBytes = Math.round(DENSE_DIMENSION_COUNT * 0.7);
  const ok = parseHfConfig('org/dense-packed', packedConfig, checkpointBytes, {
    revision: REVISION,
    safetensorsParameters: { U8: DENSE_DIMENSION_COUNT },
    safetensorsTotal: DENSE_DIMENSION_COUNT,
  });
  assert.equal(ok.totalParams, 0.2);

  const drifted = DENSE_DIMENSION_COUNT * 3;
  assert.throws(() => parseHfConfig('org/dense-packed-drift', packedConfig, Math.round(drifted * 0.7), {
    revision: REVISION,
    safetensorsParameters: { U8: drifted },
    safetensorsTotal: drifted,
  }), /config 구조 추정/);
});

test('float-only 레포는 shard byte 등식과 헤더 slack 경계를 지킨다', () => {
  const oneMbFloor = { revision: REVISION, safetensorsParameters: { BF16: 500_000 }, safetensorsTotal: 500_000 };
  assert.doesNotThrow(() => resolveParameterCount({ ...oneMbFloor, checkpointBytes: 2_000_000 }));
  assert.throws(() => resolveParameterCount({ ...oneMbFloor, checkpointBytes: 2_000_001 }), /shard byte/);

  const onePercent = { revision: REVISION, safetensorsParameters: { BF16: 100_000_000 }, safetensorsTotal: 100_000_000 };
  assert.doesNotThrow(() => resolveParameterCount({ ...onePercent, checkpointBytes: 202_000_000 }));
  assert.throws(() => resolveParameterCount({ ...onePercent, checkpointBytes: 202_000_001 }), /shard byte/);

  const absoluteCap = { revision: REVISION, safetensorsParameters: { BF16: 10_000_000_000 }, safetensorsTotal: 10_000_000_000 };
  assert.doesNotThrow(() => resolveParameterCount({ ...absoluteCap, checkpointBytes: 20_064_000_000 }));
  assert.throws(() => resolveParameterCount({ ...absoluteCap, checkpointBytes: 20_064_000_001 }), /shard byte/);
});

test('F32 레포에 작은 비-float 보조 텐서가 섞여도 거짓 거부하지 않는다', () => {
  const ok = resolveParameterCount({
    checkpointBytes: 400_000_100,
    revision: REVISION,
    safetensorsParameters: { F32: 100_000_000, I64: 1 },
    safetensorsTotal: 100_000_001,
  });
  // I64는 폭이 확정된 1:1 dtype이라 강한 byte-equality 검증을 그대로 받는다
  // (작은 보조 텐서 하나가 31B BF16 레포를 약한 밴드로 떨어뜨리면 안 된다).
  assert.equal(ok.validation, 'byte-equality');
});

test('FP8 레포는 약한 밴드가 아니라 byte-equality 검증을 받는다', () => {
  // FP8은 packing 컨테이너가 아니라 1 element = 1 byte다(DeepSeek-V3-FP8 류).
  const params = { F8_E4M3: 31_000_000_000, BF16: 100_000_000 };
  const tensorBytes = 31_000_000_000 + 100_000_000 * 2;
  const ok = resolveParameterCount({
    checkpointBytes: tensorBytes + 2_000_000,
    revision: REVISION,
    safetensorsParameters: params,
    safetensorsTotal: 31_100_000_000,
  });
  assert.equal(ok.validation, 'byte-equality');
  assert.ok(Math.abs(ok.totalParamsB - 31.1) < 1e-9);
  // 파라미터 수가 절반으로 잘못 보고되면 등식이 깨져 거부한다 — 밴드였다면 통과했을 값이다.
  assert.throws(() => resolveParameterCount({
    checkpointBytes: tensorBytes + 2_000_000,
    revision: REVISION,
    safetensorsParameters: { F8_E4M3: 15_500_000_000, BF16: 50_000_000 },
    safetensorsTotal: 15_550_000_000,
  }), /shard byte/);
});

test('기형·모순 증거는 계산하지 않는다', () => {
  assert.throws(() => parseHfConfig('org/model', GLM_CONFIG, GLM_CHECKPOINT_BYTES, {
    revision: REVISION,
    safetensorsParameters: { BF16: GLM_PARAMETER_TOTAL },
    safetensorsTotal: GLM_PARAMETER_TOTAL + 1,
  }), /파라미터.*합계/);

  for (const revision of ['main', `${REVISION}a`, '']) {
    assert.throws(() => parseHfConfig('org/model', GLM_CONFIG, GLM_CHECKPOINT_BYTES, {
      revision,
      safetensorsParameters: GLM_PARAMETERS,
      safetensorsTotal: GLM_PARAMETER_TOTAL,
    }), /revision/);
  }

  assert.throws(() => resolveParameterCount({
    checkpointBytes: 1_000,
    revision: REVISION,
    safetensorsParameters: { BF16: -1 },
  }), /비음수 정수/);
});

test('저장 dtype을 확정할 수 없으면 BF16으로 가정하지 않는다', () => {
  assert.throws(
    () => parseHfConfig('org/unknown-storage', { ...GLM_CONFIG, dtype: 'vendor_packed_v1' }, GLM_CHECKPOINT_BYTES),
    /저장 dtype/,
  );
});
