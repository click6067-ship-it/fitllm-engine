// PLE(Per-Layer Embeddings) 파서 fail-closed — 2.15.0 public port (private docs/plans/2026-09-05-ple-parser-fail-closed.md).
//
// 배경: 정확한 gemma4_text config가 hidden_size_per_layer_input을 부풀리면 종래 파서(2.14.1)는 교차검증 없이
// pleOffloadVerified=true를 주어 GPU 가중치를 최대 99% 차감했다(거짓 fits). 이 파일은 그 구멍(RED)과
// 공식 pinned Gemma 4 e2b/e4b 수치의 불변(보존)을 함께 고정한다. 기대값은 전부 손계산·pinned 리터럴이다.
//
// pinned 공식 출처(2026-09-05, 무자격 공개 fetch — 테스트는 네트워크를 쓰지 않는다):
//   google/gemma-4-E2B-it @ 3e22461f65e89153144f8adb70e3b8c2cc9845a7 config.json
//     sha256 1b28f3d2c3100f6c594754b81107428bd7b822a7f48272ca681dae9d2ec38330 · safetensors BF16 5,123,178,051 params
//   google/gemma-4-E4B-it @ ee0ef6023621cff504d758262d4e04895a5af4a2 config.json
//     sha256 33b10c02df3c2e8536cf323d29d53262aaa2f4d11dbe19bc729373fbe90295d4 · safetensors BF16 7,996,156,490 params
//   PLE 텐서 형상 [vocab_size_per_layer_input, num_hidden_layers × hidden_size_per_layer_input]:
//     transformers@4177486a9f199bd7be520eff14431071d5d41ec5 src/transformers/models/gemma4/configuration_gemma4.py
import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import {
  GPUS, LOCAL_MODELS, calcMaxContext, calcParamMemory, gpuDevice,
  parseHfConfig, simulate, simulateStack, structuralAssumptions,
} from '../engine.js';

const PINNED = {
  E2B: { sha256: '1b28f3d2c3100f6c594754b81107428bd7b822a7f48272ca681dae9d2ec38330', params: 5123178051 },
  E4B: { sha256: '33b10c02df3c2e8536cf323d29d53262aaa2f4d11dbe19bc729373fbe90295d4', params: 7996156490 },
};
// 현재 미지 구조 키 정책(STRUCTURAL_KEY_RE)이 거부하는 공식 text_config 키. 이 정책은 이 작업의 범위 밖이라
// 전체 공식 파일은 기존 거부를 그대로 유지하고, 파서가 받는 부분집합으로 PLE 수치를 검사한다.
const REJECTED_BY_KEY_POLICY = [
  'attention_k_eq_v', 'enable_moe_block', 'expert_intermediate_size', 'num_global_key_value_heads',
  'num_kv_shared_layers', 'top_k_experts', 'use_bidirectional_attention',
];
const FULL_OFFICIAL_REJECTION =
  '구조를 알 수 없는 config 필드(attention_k_eq_v, enable_moe_block, expert_intermediate_size, num_global_key_value_heads 외 4개)가 있어 계산하지 않아요';
const PLE_PREMISE_ID = 'ple-llamacpp-non-gpu-residency';
const PLE_STATEMENT = 'GPU weight memory excludes the verified Gemma 4 PLE tensors only because the pinned llama.cpp/GGUF path assigns the per_layer_token_embd input-layer tensor to CPU/host buffers instead of accelerator memory; that host memory is not budgeted here, and a runtime that loads PLE onto the accelerator invalidates this estimate.';

const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');
const officialBytes = (n) => readFileSync(new URL(`./fixtures/google__gemma-4-${n}-it.config.json`, import.meta.url));
const officialConfig = (n) => JSON.parse(officialBytes(n).toString('utf8'));
const subset = (n) => Object.fromEntries(
  Object.entries(officialConfig(n).text_config).filter(([k]) => !REJECTED_BY_KEY_POLICY.includes(k))
);
const checkpointBytes = (n) => PINNED[n].params * 2; // 단일 BF16 model.safetensors
const gpu = (name, env) => gpuDevice(GPUS.find((g) => g.name === name), env);
const gpu3060 = gpu('RTX 3060 8GB', 'linux-headless');
const gpu4090 = gpu('RTX 4090');
const FP16 = { weightBpw: 16, kvBits: 16 };
const fullWeightsGB = (model) => (model.totalParams * 1e9 * 2) / 1024 ** 3;
// vitest toBeCloseTo(expected, digits) == |actual − expected| < 10^−digits / 2
const close = (actual, expected, digits, label = '') =>
  assert.ok(Math.abs(actual - expected) < 10 ** -digits / 2, `${label} ${actual} ≉ ${expected} (${digits} digits)`);
const matchObject = (actual, expected, label = '') => {
  for (const [key, value] of Object.entries(expected)) assert.deepEqual(actual[key], value, `${label} ${key}`);
};

// 다섯 계산 표면 전부에서 "전체 가중치 상주 · pleOffloadGB 0 · PLE 전제 없음"을 단언한다.
// 기준은 같은 모델에서 PLE 메타데이터를 아예 지운 사본 — 값이 그 사본과 같아야 한다.
function expectFullResidency(model, label = '') {
  const full = fullWeightsGB(model);
  const reference = { ...model, pleParams: undefined, pleOffloadVerified: false };
  assert.equal(model.pleOffloadVerified, false, `${label} pleOffloadVerified`);
  const sim = simulate(model, gpu3060, 8192, FP16);
  close(sim.param, full, 6, `${label} simulate.param`);
  assert.equal(sim.pleOffloadGB, 0, `${label} pleOffloadGB`);
  assert.equal(sim.structuralAssumptions, undefined, `${label} structuralAssumptions`);
  assert.equal(sim.used, simulate(reference, gpu3060, 8192, FP16).used, `${label} used`);
  close(calcParamMemory(model, 16, gpu3060).totalGB, full, 6, `${label} calcParamMemory`);
  assert.deepEqual(structuralAssumptions(model, gpu3060), [], `${label} structuralAssumptions()`);
  assert.equal(calcMaxContext(model, gpu3060, FP16), calcMaxContext(reference, gpu3060, FP16), `${label} calcMaxContext`);
  const stack = simulateStack([{ model, ctx: 8192, weightBpw: 16, kvBits: 16 }], gpu3060);
  const referenceStack = simulateStack([{ model: reference, ctx: 8192, weightBpw: 16, kvBits: 16 }], gpu3060);
  close(stack.param, full, 6, `${label} simulateStack.param`);
  assert.equal(stack.used, referenceStack.used, `${label} stack.used`);
  assert.equal(stack.structuralAssumptions, undefined, `${label} stack.structuralAssumptions`);
  assert.equal(stack.parts[0].structuralAssumptions, undefined, `${label} stack.parts[0].structuralAssumptions`);
  return sim;
}

test('RED: inflated hidden_size_per_layer_input on an exact gemma4_text config fails closed on every calculation surface', () => {
  // 기준 엔진(2.14.1): h=552 → pleParams 5.065, verified true, RTX 3060 8GB FP16 param 0.0652, verdict yes.
  for (const [h, declaredPle] of [[552, 5.065], [512, 4.698]]) {
    const hostile = parseHfConfig('google/gemma-4-E2B-it', { ...subset('E2B'), hidden_size_per_layer_input: h }, checkpointBytes('E2B'));
    assert.equal(hostile.totalParams, 5.1);
    assert.equal(hostile.pleParams, declaredPle); // 선언값은 메타데이터로 남는다
    const sim = expectFullResidency(hostile, `h=${h}`);
    close(sim.param, 9.4995, 4, `h=${h} param`); // 5.1e9 × 2 B / 1024³ — 전체 가중치
    assert.equal(sim.verdict, 'no'); // 8GB 카드에 FP16 5.1B는 들어가지 않는다
  }
});

test('pinned official E2B/E4B configs keep the baseline outputs, the full official file keeps its rejection, and wrapper resolution follows the text body', () => {
  const expected = {
    E2B: {
      totalParams: 5.1, pleParams: 2.349, layerCount: 35, kvHeads: 1, globalAttnLayers: 7, slidingPattern: '4:1',
      gpu3060: { param: 5.1241, kv: 0.123, pleOffloadGB: 4.3754, used: 6.7263, verdict: 'yes', maxContext: 27456 },
      gpu4090: { param: 5.1241, pleOffloadGB: 4.3754, used: 8.1263, verdict: 'yes', maxContext: 131072 },
      apple64: { param: 4.7497, used: 13.6362, verdict: 'yes' },
    },
    E4B: {
      totalParams: 8, pleParams: 2.819, layerCount: 42, kvHeads: 2, globalAttnLayers: 7, slidingPattern: '5:1',
      gpu3060: { param: 9.6504, kv: 0.2529, pleOffloadGB: 5.2508, used: 11.945, verdict: 'no', maxContext: 0 },
      gpu4090: { param: 9.6504, pleOffloadGB: 5.2508, used: 13.345, verdict: 'yes', maxContext: 131072 },
      apple64: { param: 7.4506, used: 16.7358, verdict: 'yes' },
    },
  };
  for (const n of ['E2B', 'E4B']) {
    assert.equal(sha256(officialBytes(n)), PINNED[n].sha256);
    const official = officialConfig(n);
    assert.equal(official.model_type, 'gemma4');
    assert.equal(official.text_config.model_type, 'gemma4_text');
    assert.equal(official.text_config.vocab_size_per_layer_input, 262144);
    assert.equal(official.text_config.hidden_size_per_layer_input, 256);
    assert.equal(official.text_config.vocab_size, 262144);
    // 전체 공식 파일: 기존 미지 구조 키 거부가 글자 하나 다르지 않게 유지된다(래퍼·본체 모두).
    assert.throws(() => parseHfConfig(`google/gemma-4-${n}-it`, official, checkpointBytes(n)), (error) => {
      assert.equal(error.message, FULL_OFFICIAL_REJECTION);
      return true;
    });

    const e = expected[n];
    const text = parseHfConfig(`google/gemma-4-${n}-it`, subset(n), checkpointBytes(n));
    const wrapped = parseHfConfig(`google/gemma-4-${n}-it`, { model_type: 'gemma4', text_config: subset(n) }, checkpointBytes(n));
    for (const [shape, m] of [['text', text], ['wrapped', wrapped]]) {
      const label = `${n}/${shape}`;
      matchObject(m, {
        totalParams: e.totalParams, pleParams: e.pleParams, pleOffloadVerified: true, layerCount: e.layerCount,
        kvHeads: e.kvHeads, globalAttnLayers: e.globalAttnLayers, slidingPattern: e.slidingPattern,
        parameterSource: 'uniform-checkpoint', tags: ['dense'],
      }, label);
      const s3060 = simulate(m, gpu3060, 8192, FP16);
      close(s3060.param, e.gpu3060.param, 4, `${label} 3060 param`);
      close(s3060.kv, e.gpu3060.kv, 4, `${label} 3060 kv`);
      close(s3060.pleOffloadGB, e.gpu3060.pleOffloadGB, 4, `${label} 3060 pleOffloadGB`);
      close(s3060.used, e.gpu3060.used, 4, `${label} 3060 used`);
      assert.equal(s3060.verdict, e.gpu3060.verdict, `${label} 3060 verdict`);
      assert.equal(s3060.maxContext, e.gpu3060.maxContext, `${label} 3060 maxContext`);
      assert.deepEqual(s3060.structuralAssumptions, [{ id: PLE_PREMISE_ID, statement: PLE_STATEMENT }], `${label} premise`);
      assert.doesNotMatch(JSON.stringify(s3060), /lazy-or-host|SSD|NVMe/i);
      const s4090 = simulate(m, gpu4090, 8192, FP16);
      close(s4090.param, e.gpu4090.param, 4, `${label} 4090 param`);
      close(s4090.pleOffloadGB, e.gpu4090.pleOffloadGB, 4, `${label} 4090 pleOffloadGB`);
      close(s4090.used, e.gpu4090.used, 4, `${label} 4090 used`);
      assert.equal(s4090.verdict, e.gpu4090.verdict, `${label} 4090 verdict`);
      assert.equal(s4090.maxContext, e.gpu4090.maxContext, `${label} 4090 maxContext`);
      // Apple: 통합 메모리라 PLE 차감·전제 없음(기존 fail-closed 유지)
      const apple = simulate(m, 64, 8192, 8);
      close(apple.param, e.apple64.param, 4, `${label} apple param`);
      assert.equal(apple.pleOffloadGB, 0, `${label} apple pleOffloadGB`);
      close(apple.used, e.apple64.used, 4, `${label} apple used`);
      assert.equal(apple.verdict, e.apple64.verdict, `${label} apple verdict`);
      assert.equal(apple.structuralAssumptions, undefined, `${label} apple premise`);
    }
    // 래퍼에만 PLE 키가 있고 본체에 없으면 본체 기준 — PLE 없음(기존 동작)
    const { vocab_size_per_layer_input, hidden_size_per_layer_input, ...withoutPle } = subset(n);
    const wrapperOnlyPle = parseHfConfig(`google/gemma-4-${n}-it`, {
      model_type: 'gemma4', vocab_size_per_layer_input, hidden_size_per_layer_input, text_config: withoutPle,
    }, checkpointBytes(n));
    assert.equal(wrapperOnlyPle.pleParams, undefined);
    assert.equal(wrapperOnlyPle.pleOffloadVerified, false);
    // 본체가 부풀려졌으면 래퍼가 정직해도 fail-closed
    const hostileWrapped = parseHfConfig(`google/gemma-4-${n}-it`, {
      model_type: 'gemma4', text_config: { ...subset(n), hidden_size_per_layer_input: 512 },
    }, checkpointBytes(n));
    expectFullResidency(hostileWrapped, `${n} hostile wrapped`);
  }
});

test('malformed, mismatched, or non-allowlisted PLE declarations fail closed', () => {
  const honest = subset('E2B');
  const bytes = checkpointBytes('E2B');
  // (a) 타입·범위·오버플로 — pleParams 자체를 만들지 않는다
  const malformed = [
    ['hidden_size_per_layer_input', '256'], ['hidden_size_per_layer_input', 0], ['hidden_size_per_layer_input', -256],
    ['hidden_size_per_layer_input', 1.5], ['hidden_size_per_layer_input', Number.NaN], ['hidden_size_per_layer_input', Number.POSITIVE_INFINITY],
    ['hidden_size_per_layer_input', null], ['hidden_size_per_layer_input', 2 ** 53], ['hidden_size_per_layer_input', 2 ** 40], // 2^40: 곱이 2^53 초과
    ['vocab_size_per_layer_input', '262144'], ['vocab_size_per_layer_input', 0], ['vocab_size_per_layer_input', -1],
    ['vocab_size_per_layer_input', 262144.5], ['vocab_size_per_layer_input', Number.NaN], ['vocab_size_per_layer_input', null],
    ['vocab_size_per_layer_input', 2 ** 53], ['vocab_size_per_layer_input', 2 ** 45],
  ];
  for (const [key, value] of malformed) {
    const m = parseHfConfig('google/gemma-4-E2B-it', { ...honest, [key]: value }, bytes);
    assert.equal(m.pleParams, undefined, `${key}=${String(value)}`);
    expectFullResidency(m, `${key}=${String(value)}`);
  }
  // (b) 양의 안전 정수지만 pinned 검증 치수(262144 × 256)가 아니다 — 선언은 남고 검증만 거부
  const mismatched = [
    ['hidden_size_per_layer_input', 128], ['hidden_size_per_layer_input', 512],
    ['vocab_size_per_layer_input', 262145], ['vocab_size_per_layer_input', 131072],
  ];
  for (const [key, value] of mismatched) {
    const m = parseHfConfig('google/gemma-4-E2B-it', { ...honest, [key]: value }, bytes);
    assert.ok(m.pleParams > 0, `${key}=${value}`);
    expectFullResidency(m, `${key}=${value}`);
  }
  // (c) 허용목록 밖 계열은 정직한 치수라도 기존처럼 검증되지 않는다(PLE_OFFLOAD_FAMILIES 불변)
  for (const modelType of ['gemma4', 'gemma3n_text', 'gemma4_text_v2', 'gemma4-text', 'llama', undefined]) {
    const m = parseHfConfig('example/lookalike', { ...honest, model_type: modelType }, bytes);
    assert.equal(m.pleParams, 2.349, String(modelType));
    expectFullResidency(m, String(modelType));
  }
});

test('PLE verification requires the checkpoint to reconcile with the config body', () => {
  const honest = subset('E2B');
  // (a) 수축된 체크포인트: 4.8e9 B → 2.4B 총량, PLE 2.349B를 빼면 0.05B만 남아 config가 선언한 dense body(~1.64B)를 담을 수 없다.
  //     기준 엔진(2.14.1)은 verified true·GPU param 0.095 GiB였다.
  const deflated = parseHfConfig('google/gemma-4-E2B-it', honest, 4.8e9);
  assert.equal(deflated.totalParams, 2.4);
  assert.equal(deflated.pleParams, 2.349);
  const sim = expectFullResidency(deflated, 'deflated');
  close(sim.param, 4.4703, 4, 'deflated param'); // 2.4e9 × 2 B / 1024³
  // (b) body 추정을 만들 수 없으면(intermediate_size·vocab_size 부재) 정합을 확인할 수 없어 검증하지 않는다
  for (const key of ['intermediate_size', 'vocab_size']) {
    const { [key]: _omitted, ...withoutKey } = honest;
    const m = parseHfConfig('google/gemma-4-E2B-it', withoutKey, checkpointBytes('E2B'));
    assert.equal(m.pleParams, 2.349, key);
    expectFullResidency(m, `without ${key}`);
  }
  // (c) 층 수 부풀리기: 70L이면 PLE 4.698B, 잔여 0.42B < body(70L) — fail-closed
  const inflatedLayers = parseHfConfig('google/gemma-4-E2B-it', {
    ...honest, num_hidden_layers: 70,
    layer_types: [...Array(56).fill('sliding_attention'), ...Array(14).fill('full_attention')],
  }, checkpointBytes('E2B'));
  assert.equal(inflatedLayers.pleParams, 4.698);
  expectFullResidency(inflatedLayers, '70L');
  // (d) 정직한 공식 E4B는 잔여 5.177B ≥ body 4.525B로 통과한다
  assert.equal(parseHfConfig('google/gemma-4-E4B-it', subset('E4B'), checkpointBytes('E4B')).pleOffloadVerified, true);
});

test('RED: layer-count inflation that passes the residual-body floor and cross-profile body mixes fail closed on every calculation surface', () => {
  // 독립 리뷰 P1(2026-09-05): PLE 곱의 세 번째 인자 num_hidden_layers는 pinned 대조에서 빠져 있었다.
  // 공식 E2B 체크포인트에 45층을 선언하면 잔여 하한을 통과하고 verified true → RTX 3060 8GB FP16 48K param 3.8743, verdict yes.
  // 파서의 checkpointSanityParams(paramsFromDims)와 같은 식 — 아래 입력이 "잔여 하한은 통과한다"는 것을 테스트가 스스로 증명한다.
  const denseBodyB = (c, layers) => {
    const q = c.num_attention_heads * c.head_dim;
    const kv = c.num_key_value_heads * c.head_dim;
    const perLayer = 2 * c.hidden_size * q + 2 * c.hidden_size * kv + 3 * c.hidden_size * c.intermediate_size;
    return (layers * perLayer + c.vocab_size * c.hidden_size * (c.tie_word_embeddings ? 1 : 2)) / 1e9;
  };
  const layerTypes = (n, period) => Array.from({ length: n }, (_, i) => (i % period === period - 1 ? 'full_attention' : 'sliding_attention'));
  const pleB = (layers) => (262144 * 256 * layers) / 1e9;
  const passesFloor = (n, c, layers) => PINNED[n].params / 1e9 - pleB(layers) >= denseBodyB(c, layers);
  const expectClosedAt48k = (model, label) => {
    const sim = expectFullResidency(model, label);
    close(sim.param, fullWeightsGB(model), 6, `${label} param`);
    const long = simulate(model, gpu3060, 48000, FP16);
    close(long.param, fullWeightsGB(model), 6, `${label} 48K param`);
    assert.equal(long.pleOffloadGB, 0, `${label} 48K pleOffloadGB`);
    assert.equal(long.verdict, 'no', `${label} 48K verdict`);
    assert.equal(long.maxContext, 0, `${label} 48K maxContext`);
    assert.equal(long.structuralAssumptions, undefined, `${label} 48K premise`);
  };
  // (a) E2B 체크포인트 + 40/45/46층: 하한 통과, 그러나 두 pinned profile 어느 쪽 텐서 형상도 아니다
  for (const [layers, declaredPle] of [[40, 2.684], [45, 3.02], [46, 3.087]]) {
    const cfg = { ...subset('E2B'), num_hidden_layers: layers, layer_types: layerTypes(layers, 5) };
    assert.equal(passesFloor('E2B', cfg, layers), true, `${layers}L passes residual floor`);
    const m = parseHfConfig('google/gemma-4-E2B-it', cfg, checkpointBytes('E2B'));
    assert.equal(m.layerCount, layers);
    assert.equal(m.totalParams, 5.1);
    assert.equal(m.pleParams, declaredPle); // 선언값은 메타데이터로 남는다
    expectClosedAt48k(m, `E2B ${layers}L`);
  }
  // (b) profile 교차·혼합: 각 필드는 어느 공식 profile엔가 있지만 한 profile로 완전히 맞지 않는다 — 전부 하한은 통과하는 입력
  const mixes = [
    ['E2B', { num_hidden_layers: 42, layer_types: layerTypes(42, 6) }, 42], // E2B 본체 + E4B 층 수
    ['E4B', { num_hidden_layers: 35, layer_types: layerTypes(35, 5) }, 35], // E4B 본체 + E2B 층 수
    ['E2B', { intermediate_size: 10240 }, 35], // E2B + E4B intermediate
    ['E4B', { intermediate_size: 6144 }, 42], // E4B + E2B intermediate
    ['E4B', { hidden_size: 1536 }, 42], // E4B + E2B hidden (E2B 본체 필드 둘을 다 넣으면 기존 체크포인트 자릿수 게이트가 먼저 throw한다)
  ];
  for (const [n, patch, layers] of mixes) {
    const cfg = { ...subset(n), ...patch };
    const label = `${n}+${JSON.stringify(patch)}`;
    assert.equal(passesFloor(n, cfg, layers), true, `${label} passes residual floor`);
    const m = parseHfConfig(`google/gemma-4-${n}-it`, cfg, checkpointBytes(n));
    assert.equal(m.pleParams, +pleB(layers).toFixed(3), label);
    expectClosedAt48k(m, label);
  }
  // (c) 완전 profile 일치만 통과한다 — 공식 E2B 35층·1536·6144, E4B 42층·2560·10240
  for (const [n, layers, ple] of [['E2B', 35, 2.349], ['E4B', 42, 2.819]]) {
    const m = parseHfConfig(`google/gemma-4-${n}-it`, subset(n), checkpointBytes(n));
    matchObject(m, { layerCount: layers, pleParams: ple, pleOffloadVerified: true }, n);
  }
});

test('catalog rows and plain GQA are unchanged by the parser guard', () => {
  assert.deepEqual(LOCAL_MODELS.filter((m) => m.pleOffloadVerified === true).map((m) => m.name), ['Gemma 4 e2b', 'Gemma 4 e4b']);
  const rows = {
    'Gemma 4 e2b': { totalParams: 5.1, activeParams: 2.3, pleParams: 2.349, layerCount: 35, kvHeads: 1, hiddenSize: 1536,
      sim: { param: 5.1241, pleOffloadGB: 4.3754, used: 6.6634, verdict: 'yes', maxContext: 33050 } },
    'Gemma 4 e4b': { totalParams: 8, activeParams: 4.5, pleParams: 2.819, layerCount: 42, kvHeads: 2, hiddenSize: 2560,
      sim: { param: 9.6504, pleOffloadGB: 5.2508, used: 11.8193, verdict: 'no', maxContext: 0 } },
  };
  for (const [name, { sim: expectedSim, ...fields }] of Object.entries(rows)) {
    const model = LOCAL_MODELS.find((m) => m.name === name);
    matchObject(model, { ...fields, pleOffloadVerified: true, tags: ['dense', 'ple'] }, name);
    const s = simulate(model, gpu3060, 8192, FP16);
    close(s.param, expectedSim.param, 4, `${name} param`);
    close(s.pleOffloadGB, expectedSim.pleOffloadGB, 4, `${name} pleOffloadGB`);
    close(s.used, expectedSim.used, 4, `${name} used`);
    assert.equal(s.verdict, expectedSim.verdict, `${name} verdict`);
    assert.equal(s.maxContext, expectedSim.maxContext, `${name} maxContext`);
    assert.deepEqual(s.structuralAssumptions, [{ id: PLE_PREMISE_ID, statement: PLE_STATEMENT }], `${name} premise`);
    assert.doesNotMatch(JSON.stringify(s), /lazy-or-host|SSD|NVMe/i);
    const apple = simulate(model, 64, 8192, 8);
    assert.equal(apple.structuralAssumptions, undefined, `${name} apple premise`);
    assert.equal(apple.pleOffloadGB, 0, `${name} apple pleOffloadGB`);
  }
  // plain GQA: PLE 키가 없으면 pleParams도 전제도 생기지 않는다
  const plain = parseHfConfig('meta/plain-gqa', {
    model_type: 'llama', torch_dtype: 'bfloat16', num_hidden_layers: 32, num_attention_heads: 32,
    num_key_value_heads: 8, hidden_size: 4096, intermediate_size: 14336, vocab_size: 128256, max_position_embeddings: 131072,
  }, 16e9);
  assert.equal(plain.pleParams, undefined);
  assert.equal(plain.pleOffloadVerified, false);
  assert.equal(simulate(plain, gpu4090, 8192, FP16).structuralAssumptions, undefined);
});
