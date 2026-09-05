// gpt-oss 20B/120B — HF safetensors.parameters(논리 파라미터 수) ↔ 내장 카탈로그 parity (#98 잔여).
// #98의 원 사고(MXFP4 total_size÷2 → 6.9B/32.6B)는 v2.8.1에서 fail-closed됐고, v2.10.0부터 immutable revision의
// HF safetensors.parameters 증거가 파라미터 수의 정본이다. 남은 잔여는 *카탈로그 행*이 모델카드 반올림(21/117)을
// 들고 있어, 같은 모델을 HF 즉석 파싱으로 계산하면 다른 판정(no↔tight)이 나오는 것 — 이 파일이 그 등식을 고정한다.
// 파서의 0.1B 계약(totalParams.toFixed(1))은 넓히지 않는다. 카탈로그가 파서와 같은 값을 갖는지만 본다.
//
// 1차 출처(immutable revision, 2026-09-05 실측 — 응답을 그대로 상수로 옮겼고 테스트는 네트워크를 쓰지 않는다):
//   openai/gpt-oss-20b  @ 6cee5e81ee83917806bbde320786a8fb61efebee
//     https://huggingface.co/api/models/openai/gpt-oss-20b/revision/6cee5e81ee83917806bbde320786a8fb61efebee
//     https://huggingface.co/openai/gpt-oss-20b/resolve/6cee5e81ee83917806bbde320786a8fb61efebee/config.json
//       (config.json content sha256 3a2a26ded679375b7928ddeca59764df7cea83220c1961035f6d6e232659e9ce)
//   openai/gpt-oss-120b @ b5c939de8f754692c1647ca79fbf85e8c1e70f8a
//     https://huggingface.co/api/models/openai/gpt-oss-120b/revision/b5c939de8f754692c1647ca79fbf85e8c1e70f8a
//     https://huggingface.co/openai/gpt-oss-120b/resolve/b5c939de8f754692c1647ca79fbf85e8c1e70f8a/config.json
//       (config.json content sha256 933aeb666a3fd851133ddd7686414f369bc564c4185fb5704416550879f10566)
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  ENV_PRESETS, GPUS, LOCAL_MODELS, appleDevice, gpuDevice, parseHfConfig, resolveParameterCount, simulate,
} from '../engine.js';

// 공식 config.json을 키 순서까지 그대로 옮긴 것. 두 모델은 레이어 수·전문가 수만 다르다.
// layer_types는 sliding→full 교대(짝수 index = sliding_attention) — 공식 파일의 배열과 동일.
const gptOssConfig = (layers, experts) => Object.freeze({
  architectures: ['GptOssForCausalLM'],
  attention_bias: true,
  attention_dropout: 0.0,
  eos_token_id: 200002,
  experts_per_token: 4,
  head_dim: 64,
  hidden_act: 'silu',
  hidden_size: 2880,
  initial_context_length: 4096,
  initializer_range: 0.02,
  layer_types: Array.from({ length: layers }, (_, i) => (i % 2 === 0 ? 'sliding_attention' : 'full_attention')),
  max_position_embeddings: 131072,
  model_type: 'gpt_oss',
  num_attention_heads: 64,
  num_experts_per_tok: 4,
  num_hidden_layers: layers,
  num_key_value_heads: 8,
  num_local_experts: experts,
  output_router_logits: false,
  pad_token_id: 199999,
  quantization_config: {
    modules_to_not_convert: ['model.layers.*.self_attn', 'model.layers.*.mlp.router', 'model.embed_tokens', 'lm_head'],
    quant_method: 'mxfp4',
  },
  rms_norm_eps: 1e-05,
  rope_scaling: {
    beta_fast: 32.0, beta_slow: 1.0, factor: 32.0, original_max_position_embeddings: 4096, rope_type: 'yarn', truncate: false,
  },
  rope_theta: 150000,
  router_aux_loss_coef: 0.9,
  sliding_window: 128,
  swiglu_limit: 7.0,
  tie_word_embeddings: false,
  transformers_version: '4.55.0.dev0',
  use_cache: true,
  vocab_size: 201088,
});

const CASES = Object.freeze([
  {
    id: 'openai/gpt-oss-20b',
    catalogName: 'gpt-oss-20b',
    revision: '6cee5e81ee83917806bbde320786a8fb61efebee',
    config: gptOssConfig(24, 32),
    // HF API safetensors.parameters — BF16(attention·router·embedding·lm_head) + U8(MXFP4 packed expert weights, *논리* 수)
    parameters: Object.freeze({ BF16: 1804459584, U8: 19110297600 }),
    total: 20914757184,
    // API siblings size 합 — model-0000{0,1,2}-of-00002.safetensors 3개 샤드(original/ 제외)
    shardBytes: 13761316904,
    shape: {
      layerCount: 24, kvHeads: 8, kvHeadDim: 64, attnHeads: 64, hiddenSize: 2880,
      numExperts: 32, expertsPerToken: 4, maxContext: 131072, slidingWindow: 128, globalAttnLayers: 12,
    },
  },
  {
    id: 'openai/gpt-oss-120b',
    catalogName: 'gpt-oss-120b',
    revision: 'b5c939de8f754692c1647ca79fbf85e8c1e70f8a',
    config: gptOssConfig(36, 128),
    parameters: Object.freeze({ BF16: 2167371072, U8: 114661785600 }),
    total: 116829156672,
    // model-000{00..14}-of-00014.safetensors 15개 샤드 합(original/ 제외)
    shardBytes: 65248893184,
    shape: {
      layerCount: 36, kvHeads: 8, kvHeadDim: 64, attnHeads: 64, hiddenSize: 2880,
      numExperts: 128, expertsPerToken: 4, maxContext: 131072, slidingWindow: 128, globalAttnLayers: 18,
    },
  },
]);

const evidenceOf = (c) => ({ revision: c.revision, safetensorsParameters: c.parameters, safetensorsTotal: c.total });
const catalogOf = (c) => LOCAL_MODELS.find((m) => m.name === c.catalogName);
const parsedOf = (c) => parseHfConfig(c.id, c.config, c.shardBytes, evidenceOf(c));
// 엔진의 기존 0.1B 계약 — parseHfConfig가 totalParams에 적용하는 것과 같은 반올림.
const oneDecimalB = (params) => +(params / 1e9).toFixed(1);

for (const c of CASES) {
  test(`${c.id} — dtype별 합계·total·샤드 byte가 서로 설명되고 exact 증거로 채택된다 (U8 packed → byte-band)`, () => {
    assert.equal(c.parameters.BF16 + c.parameters.U8, c.total);
    assert.deepEqual(resolveParameterCount({
      checkpointBytes: c.shardBytes, safetensorsParameters: c.parameters, safetensorsTotal: c.total, revision: c.revision,
    }), {
      totalParamsB: c.total / 1e9,
      tensorBytes: null,
      source: 'hf-safetensors-parameters',
      confidence: 'exact',
      validation: 'byte-band',
    });
  });

  test(`${c.id} — parseHfConfig의 totalParams(0.1B 계약) == 내장 카탈로그 totalParams`, () => {
    const parsed = parsedOf(c);
    const catalog = catalogOf(c);
    assert.ok(catalog, `카탈로그에 ${c.catalogName} 행이 있어야 한다`);
    assert.equal(parsed.parameterSource, 'hf-safetensors-parameters');
    assert.equal(parsed.totalParams, oneDecimalB(c.total)); // 20.9 / 116.8 — 증거에서 직접 유도
    assert.equal(catalog.totalParams, parsed.totalParams);
  });

  test(`${c.id} — 메모리에 들어가는 구조 필드가 카탈로그와 같다`, () => {
    const parsed = parsedOf(c);
    const catalog = catalogOf(c);
    const keys = Object.keys(c.shape);
    const pick = (m) => Object.fromEntries(keys.map((k) => [k, m[k]]));
    assert.deepEqual(pick(parsed), c.shape);
    assert.deepEqual(pick(catalog), c.shape);
    assert.deepEqual(parsed.tags, ['moe']);
    assert.deepEqual(catalog.tags, ['moe']);
    // 카탈로그는 활성 파라미터를 모델카드 값(3.6/5.1)으로 유지한다 — 판정엔 들어가지 않는다.
    assert.equal(typeof catalog.activeParams, 'number');
  });

  test(`${c.id} — 설명 문구가 총 파라미터를 모델카드 반올림(21B/117B)으로 단정하지 않는다`, () => {
    const catalog = catalogOf(c);
    assert.ok(catalog.desc.includes(`${oneDecimalB(c.total)}B`), `desc에 ${oneDecimalB(c.total)}B가 있어야 한다: ${catalog.desc}`);
    assert.ok(!/(^|[^\d.])(21|117)B\b/.test(catalog.desc), `desc가 모델카드 반올림을 단정하면 안 된다: ${catalog.desc}`);
  });
}

// 판정 매트릭스 — 카탈로그 GPU 전부(환경 프리셋 3종) + Apple 통합메모리 10단계 × weight bpw 5종 × ctx 4종,
// KV 비트는 (a) weight와 동일(Apple 숫자 인자 관례) (b) F16 고정(GPU GGUF 관례) 두 가지 모두.
// 카탈로그 행과 HF 즉석 파싱이 한 케이스라도 다른 verdict를 내면 실패하고, 그 케이스를 그대로 나열한다.
const APPLE_RAM_GB = [16, 24, 32, 48, 64, 96, 128, 192, 256, 512];
const WEIGHT_BPW = [4, 5, 6, 8, 16];
const CONTEXTS = [8192, 32768, 65536, 131072];
const DEVICES = [
  ...Object.keys(ENV_PRESETS).flatMap((env) => GPUS.map((g) => ({ label: `gpu:${g.name}@${env}`, device: gpuDevice(g, env) }))),
  ...APPLE_RAM_GB.map((ram) => ({ label: `apple:${ram}GB`, device: appleDevice(ram) })),
];
const QUANTS = WEIGHT_BPW.flatMap((bpw) => [
  { label: `quant=${bpw}(kv=weight)`, quant: bpw },
  { label: `quant={w${bpw},kv16}`, quant: { weightBpw: bpw, kvBits: 16 } },
]);

for (const c of CASES) {
  test(`${c.id} — 모든 디바이스·정밀도·컨텍스트에서 카탈로그 행과 HF 즉석 파싱의 verdict 불일치 0`, () => {
    const catalog = catalogOf(c);
    const parsed = parsedOf(c);
    const mismatches = [];
    const seen = new Set();
    let cases = 0;
    for (const { label: dev, device } of DEVICES) {
      for (const { label: q, quant } of QUANTS) {
        for (const ctx of CONTEXTS) {
          cases += 1;
          const a = simulate(catalog, device, ctx, quant).verdict;
          const b = simulate(parsed, device, ctx, quant).verdict;
          seen.add(a);
          if (a !== b) mismatches.push(`${dev} ${q} ctx=${ctx}: catalog=${a} hf=${b}`);
        }
      }
    }
    assert.equal(cases, DEVICES.length * QUANTS.length * CONTEXTS.length);
    assert.ok(cases >= 920, `매트릭스가 920케이스 이상이어야 한다 (got ${cases})`);
    // 매트릭스가 판별력을 갖는지 — 전부 'no'거나 전부 'yes'인 퇴화 매트릭스가 아니어야 한다.
    assert.ok(seen.has('yes') && seen.has('no'), `매트릭스가 퇴화했다: ${[...seen].join(',')}`);
    assert.deepEqual(mismatches, []);
  });
}

for (const c of CASES) {
  // 증거 없는 MXFP4 경로는 여전히 fail-closed (#98 원 사고 비회귀)
  test(`${c.id} — 저장 비트폭을 확정할 수 없으면 total_size÷2로 추정하지 않고 거부한다`, () => {
    assert.throws(() => parseHfConfig(c.id, c.config, c.shardBytes), /저장 비트폭을 확정할 수 없어/);
  });
}
