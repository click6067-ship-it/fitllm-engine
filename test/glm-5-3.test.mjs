// GLM-5.3 (zai-org) — 2.15.0 public port (private docs/plans/2026-09-05-glm-5-3-model-refresh.md AC-1..AC-11).
// 원칙: 공식 모델카드가 "GLM-5.2와 같은 base model, 모든 향상은 post-training"이라고 밝히고 두 리비전의
// safetensors total이 753,329,940,480으로 정확히 같으며 config diff가 quantization_config뿐이므로,
// GLM-5.3 행은 GLM-5.2 행과 **수학·형상 동일**해야 한다. 새 측정을 지어내지 않고, 기존 인덱스·GLM-5.2 결과·
// fail-closed 경계는 바이트 단위로 보존한다. 기대값은 전부 pinned 리터럴 — 테스트는 네트워크를 쓰지 않는다.
//
// 1차 출처(immutable revision, 2026-09-05 실측):
//   zai-org/GLM-5.3 @ aca966e4e02791568aa6a4ced368624b3d897f42
//     https://huggingface.co/api/models/zai-org/GLM-5.3/revision/aca966e4e02791568aa6a4ced368624b3d897f42
//     https://huggingface.co/zai-org/GLM-5.3/resolve/aca966e4e02791568aa6a4ced368624b3d897f42/config.json
//     https://huggingface.co/zai-org/GLM-5.3/resolve/aca966e4e02791568aa6a4ced368624b3d897f42/README.md
//   zai-org/GLM-5.2 @ cf457fa734ab149ffef225f80893eb38c6ff5cdc
//     https://huggingface.co/api/models/zai-org/GLM-5.2/revision/cf457fa734ab149ffef225f80893eb38c6ff5cdc
//     https://huggingface.co/zai-org/GLM-5.2/resolve/cf457fa734ab149ffef225f80893eb38c6ff5cdc/config.json
import test from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync, spawnSync } from 'node:child_process';
import { mkdtempSync, readFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import {
  ENV_PRESETS, GPUS, LOCAL_MODELS, MODELS, MODEL_GROUP_ORDER,
  appleDevice, calcKVCache, calcMaxContext, gpuDevice, groupedForDisplay, naiveKVCache,
  parseHfConfig, resolveLocalModel, simulate,
} from '../engine.js';

const ROOT = new URL('..', import.meta.url).pathname;
const BIN = new URL('../bin/fitllm.mjs', import.meta.url).pathname;
const ENTRY = new URL('../scripts/action-entry.sh', import.meta.url).pathname;
const run = (args) => {
  try { return { out: execFileSync(process.execPath, [BIN, ...args], { encoding: 'utf8' }), code: 0 }; }
  catch (e) { return { out: (e.stdout || '') + (e.stderr || ''), code: e.status }; }
};
function runAction(inputs) {
  const outputFile = join(mkdtempSync(join(tmpdir(), 'fitllm-action-')), 'output');
  const result = spawnSync('bash', [ENTRY], {
    cwd: ROOT,
    encoding: 'utf8',
    env: {
      ...process.env,
      GITHUB_ACTION_PATH: ROOT,
      GITHUB_OUTPUT: outputFile,
      INPUT_MODEL: '', INPUT_GPU: '', INPUT_MAC: '', INPUT_QUANT: '', INPUT_CTX: '8192', INPUT_KV: '16', INPUT_COUNT: '1',
      ...inputs,
    },
  });
  return { ...result, output: readFileSync(outputFile, 'utf8') };
}
const byName = (name) => LOCAL_MODELS.find((m) => m.name === name);
const glm53 = () => byName('GLM-5.3');
const glm52 = () => byName('GLM-5.2');
const matchObject = (actual, expected, label = '') => {
  for (const [key, value] of Object.entries(expected)) assert.deepEqual(actual?.[key], value, `${label} ${key}`);
};

// ── 핀 사실 (HF API safetensors.parameters — dtype별 *논리* 파라미터 수) ─────────────────────────────
const GLM53_REVISION = 'aca966e4e02791568aa6a4ced368624b3d897f42';
const GLM52_REVISION = 'cf457fa734ab149ffef225f80893eb38c6ff5cdc';
const GLM53_PARAMETERS = Object.freeze({ BF16: 2103729152, F8_E4M3: 751226191872, F32: 19456 });
const GLM52_PARAMETERS = Object.freeze({ BF16: 753329921024, F32: 19456 });
const SAFETENSORS_TOTAL = 753329940480; // 두 리비전 동일

// 두 config에 공통인 형상 키(공식 config.json 값 그대로). 두 파일의 diff는 quantization_config·transformers_version뿐.
const SHARED_GEOMETRY = Object.freeze({
  architectures: ['GlmMoeDsaForCausalLM'],
  model_type: 'glm_moe_dsa',
  num_hidden_layers: 78,
  hidden_size: 6144,
  num_attention_heads: 64,
  num_key_value_heads: 64,
  head_dim: 192,          // == qk_nope_head_dim (MLA의 non-RoPE q/k 차원)
  qk_nope_head_dim: 192,
  qk_rope_head_dim: 64,
  qk_head_dim: 256,       // nope 192 + rope 64
  v_head_dim: 256,        // 카탈로그 kvHeadDim 256의 실체
  kv_lora_rank: 512,
  q_lora_rank: 2048,
  n_routed_experts: 256,
  n_shared_experts: 1,
  num_experts_per_tok: 8,
  moe_intermediate_size: 2048,
  intermediate_size: 12288,
  first_k_dense_replace: 3,
  num_nextn_predict_layers: 1,
  max_position_embeddings: 1048576,
  vocab_size: 154880,
  // DSA(DeepSeek Sparse Attention) indexer — 엔진이 fail-closed로 거부하는 키
  index_topk: 2048,
  index_n_heads: 32,
  index_head_dim: 128,
});
const GLM53_QUANTIZATION = Object.freeze({ quant_method: 'fp8', fmt: 'e4m3', activation_scheme: 'dynamic', weight_block_size: [128, 128] });

// 공개 2.14.1(619585f)의 GLM-5.2 행 — 이 변경이 GLM-5.2를 한 글자도 건드리지 않았음을 고정한다.
const GLM52_BASELINE_ROW = Object.freeze({
  name: 'GLM-5.2', group: 'GLM', tags: ['moe', 'mla'],
  totalParams: 753, activeParams: 40, layerCount: 78, kvHeads: 64, kvHeadDim: 256, attnHeads: 64, hiddenSize: 6144,
  numExperts: 256, expertsPerToken: 8, mlaKvLoraRank: 512, mlaRopeDim: 64, maxContext: 1048576, benchmarks: null,
  desc: 'MoE · MLA · 753B / ~40B active · 256 experts(top-8) · 압축 KV · 최대 1M (4bit도 512GB급만 fit)',
});

// 공개 2.14.1의 ?m= 인덱스 표(LOCAL_MODELS 0..25).
const PINNED_INDEX = Object.freeze([
  'GLM-4.7-Flash', 'GLM-5.2', 'gpt-oss-20b', 'gpt-oss-120b',
  'Qwen 3.6 27B', 'Qwen 3.6 35B-A3B', 'Qwen-AgentWorld-35B-A3B',
  'Gemma 4 e2b', 'Gemma 4 e4b', 'Gemma 4 12b', 'Gemma 4 26b A4B', 'Gemma 4 31b',
  'Llama-3.2-3B-Instruct', 'Llama-3.1-8B-Instruct', 'MiniCPM5-1B',
  'Qwen3-0.6B', 'Qwen3-1.7B', 'Llama-3.2-1B-Instruct', 'Gemma-3-1B-it',
  'Hy3', 'Qwen 3.8 27B', 'Qwen 3.8 2.4T-A95B', 'Laguna XS 2.1', 'Laguna S 2.1',
  'Spark-X2.5-4B', 'Granite-4.2-30B',
]);
const BASELINE_GROUP_ORDER = Object.freeze([
  'Granite', 'Spark', 'Qwen 3.8', 'Laguna', 'GLM', 'gpt-oss', 'Qwen 3.6', 'Qwen3.5',
  'Hunyuan', 'Gemma 4', 'Llama', 'MiniCPM', 'Draft',
]);

test('AC1_identity: GLM-5.3 resolves across spellings; GLM-5.2 and ambiguous rules are unchanged', () => {
  for (const q of ['GLM-5.3', 'glm-5.3', 'GLM 5.3', '  glm_5_3  ']) {
    matchObject(resolveLocalModel(q), { status: 'resolved', canonicalName: 'GLM-5.3' }, q);
  }
  matchObject(resolveLocalModel('glm 5.2'), { status: 'resolved', canonicalName: 'GLM-5.2' });
  matchObject(resolveLocalModel('GLM-5.2'), { status: 'resolved', canonicalName: 'GLM-5.2' });
  // 'glm'·'glm 5'는 후보가 여럿이라 첫 항목을 승격하지 않는다 — 5.3 추가로 'glm 5'는 정상적으로 ambiguous가 된다.
  for (const q of ['glm', 'glm 5']) {
    const r = resolveLocalModel(q);
    assert.equal(r.status, 'ambiguous', q);
    assert.equal(r.canonicalName, undefined);
    assert.ok(r.candidates.map((c) => c.name).includes('GLM-5.3'), `${q} candidates include GLM-5.3`);
    assert.ok(r.candidates.map((c) => c.name).includes('GLM-5.2'), `${q} candidates include GLM-5.2`);
  }
});

test('AC2_append_only_index: 0..25 stay exact, GLM-5.3 is last (26/27), and the MODELS cloud slot is unchanged', () => {
  PINNED_INDEX.forEach((name, i) => assert.equal(LOCAL_MODELS[i]?.name, name, `?m=${i}`));
  assert.equal(LOCAL_MODELS.length, 27);
  assert.equal(LOCAL_MODELS[26]?.name, 'GLM-5.3');
  assert.equal(LOCAL_MODELS.at(-1)?.name, 'GLM-5.3');
  assert.equal(MODELS.length, 28);
  assert.equal(MODELS.at(-1)?.name, 'GLM-5.3');
  matchObject(MODELS[19], { name: 'Claude Opus 4.7', isCloud: true });
  assert.equal(MODELS.filter((m) => m.name === 'GLM-5.3').length, 1);
});

test('AC3_geometry_mla_premise: row fields equal the pinned config, kvHeadDim=v_head_dim(256) and is inert on the MLA path', () => {
  const m = glm53();
  assert.ok(m, 'GLM-5.3 in LOCAL_MODELS');
  matchObject(m, {
    group: 'GLM', tags: ['moe', 'mla'],
    layerCount: SHARED_GEOMETRY.num_hidden_layers,
    hiddenSize: SHARED_GEOMETRY.hidden_size,
    attnHeads: SHARED_GEOMETRY.num_attention_heads,
    kvHeads: SHARED_GEOMETRY.num_key_value_heads,
    kvHeadDim: SHARED_GEOMETRY.v_head_dim,
    mlaKvLoraRank: SHARED_GEOMETRY.kv_lora_rank,
    mlaRopeDim: SHARED_GEOMETRY.qk_rope_head_dim,
    numExperts: SHARED_GEOMETRY.n_routed_experts,
    expertsPerToken: SHARED_GEOMETRY.num_experts_per_tok,
    maxContext: SHARED_GEOMETRY.max_position_embeddings,
  });
  for (const key of ['slidingWindow', 'globalAttnLayers', 'fullAttnLayers', 'linearAttn', 'pleParams']) {
    assert.equal(m[key], undefined, key);
  }
  assert.equal(SHARED_GEOMETRY.head_dim, SHARED_GEOMETRY.qk_nope_head_dim);
  assert.equal(SHARED_GEOMETRY.qk_nope_head_dim + SHARED_GEOMETRY.qk_rope_head_dim, SHARED_GEOMETRY.qk_head_dim);
  assert.equal(SHARED_GEOMETRY.qk_head_dim, SHARED_GEOMETRY.v_head_dim);
  // MLA KV = (512 + 64) elem × 2 B × 78 layers = 89,856 B/token — kvHeads·kvHeadDim은 식에 없다.
  assert.equal(calcKVCache(m, 1, 16).kvPerToken, (512 + 64) * 2 * 78);
  assert.equal(calcKVCache(m, 1048576, 16).totalGB, 87.75);
  const with192 = { ...m, kvHeadDim: 192 };
  for (const ctx of [8192, 131072, 1048576]) {
    assert.equal(calcKVCache(with192, ctx, 16).totalBytes, calcKVCache(m, ctx, 16).totalBytes);
    assert.equal(calcKVCache(with192, ctx, 8).totalBytes, calcKVCache(m, ctx, 8).totalBytes);
  }
  assert.equal(calcMaxContext(with192, appleDevice(512), 4), calcMaxContext(m, appleDevice(512), 4));
  const rtx4090 = gpuDevice(GPUS.find((g) => g.name === 'RTX 4090'));
  assert.equal(calcMaxContext(with192, rtx4090, { weightBpw: 4.8944, kvBits: 16 }), calcMaxContext(m, rtx4090, { weightBpw: 4.8944, kvBits: 16 }));
  assert.equal(naiveKVCache(m, 32768, 16), naiveKVCache(glm52(), 32768, 16));
});

// 판정 매트릭스 — 카탈로그 GPU 전부(환경 프리셋 3종) + Apple 통합메모리 10단계 × weight bpw 6종 × KV 2모드 × ctx 4종.
const APPLE_RAM_GB = [16, 24, 32, 48, 64, 96, 128, 192, 256, 512];
const WEIGHT_BPW = [4, 4.8944, 5, 6, 8, 16];
const CONTEXTS = [8192, 32768, 131072, 1048576];
const DEVICES = [
  ...Object.keys(ENV_PRESETS).flatMap((env) => GPUS.map((g) => ({ label: `gpu:${g.name}@${env}`, device: gpuDevice(g, env) }))),
  ...APPLE_RAM_GB.map((ram) => ({ label: `apple:${ram}GB`, device: appleDevice(ram) })),
];
const QUANTS = WEIGHT_BPW.flatMap((bpw) => [
  { label: `quant=${bpw}(kv=weight)`, quant: bpw },
  { label: `quant={w${bpw},kv16}`, quant: { weightBpw: bpw, kvBits: 16 } },
]);
const withoutModel = (sim) => JSON.stringify({ ...sim, model: undefined });

test('AC4_parity_with_glm52: simulate/calcKVCache/calcMaxContext are JSON-identical except the model field over ≥5,000 cases', () => {
  assert.deepEqual(glm52(), GLM52_BASELINE_ROW);
  const a = glm52();
  const b = glm53();
  assert.ok(b, 'GLM-5.3 in LOCAL_MODELS');
  const mismatches = [];
  const seen = new Set();
  let cases = 0;
  for (const { label: dev, device } of DEVICES) {
    for (const { label: q, quant } of QUANTS) {
      for (const ctx of CONTEXTS) {
        cases += 1;
        const sa = simulate(a, device, ctx, quant);
        const sb = simulate(b, device, ctx, quant);
        seen.add(sa.verdict);
        if (withoutModel(sa) !== withoutModel(sb)) mismatches.push(`${dev} ${q} ctx=${ctx}: 5.2=${sa.verdict} 5.3=${sb.verdict}`);
        if (sb.model?.name !== 'GLM-5.3') mismatches.push(`${dev} ${q} ctx=${ctx}: model name ${sb.model?.name}`);
      }
      if (calcMaxContext(a, device, quant) !== calcMaxContext(b, device, quant)) mismatches.push(`${dev} ${q}: maxContext`);
    }
  }
  for (const ctx of CONTEXTS) for (const bits of [4, 8, 16]) {
    assert.deepEqual(calcKVCache(b, ctx, bits), calcKVCache(a, ctx, bits));
  }
  assert.equal(cases, DEVICES.length * QUANTS.length * CONTEXTS.length);
  assert.ok(cases >= 5000, `only ${cases} cases`);
  assert.equal(seen.has('yes') && seen.has('no'), true); // 퇴화 매트릭스가 아니다(512GB Mac 4bit는 yes, 24GB GPU는 no)
  assert.deepEqual(mismatches, []);
  const { name: _n1, desc: _d1, ...shape52 } = a;
  const { name: _n2, desc: _d2, ...shape53 } = b;
  assert.deepEqual(shape53, shape52);
});

test('AC5_no_invented_claims: benchmarks null, 753 integer convention, inherited activeParams, no SSD/NVMe/speed copy', () => {
  const m = glm53();
  assert.ok(m, 'GLM-5.3 in LOCAL_MODELS');
  assert.equal(m.benchmarks, null);
  assert.equal(m.totalParams, 753);
  assert.equal(m.totalParams, glm52().totalParams);
  assert.equal(Number((SAFETENSORS_TOTAL / 1e9).toFixed(1)), 753.3); // 정수 753은 카탈로그 관례(GLM-5.2와 동일), 새 측정치가 아니다
  assert.equal(Object.values(GLM53_PARAMETERS).reduce((s, v) => s + v, 0), SAFETENSORS_TOTAL);
  assert.equal(Object.values(GLM52_PARAMETERS).reduce((s, v) => s + v, 0), SAFETENSORS_TOTAL);
  assert.equal(m.activeParams, glm52().activeParams); // 상속 — 별도 핀 측정 없음
  assert.match(m.desc, /~40B/);
  assert.doesNotMatch(JSON.stringify(m), /tok\/s|ssd|nvme|offload|stream|nvmai/i);
  const engineSrc = readFileSync(join(ROOT, 'engine.js'), 'utf8');
  assert.doesNotMatch(engineSrc, /NVMAI/);
  const at = engineSrc.indexOf("name: 'GLM-5.3'");
  assert.ok(at > 0, 'GLM-5.3 row present in engine.js');
  const glmBlock = engineSrc.slice(at - 1200, at + 800);
  assert.doesNotMatch(glmBlock, /SSD|NVMe|offload|tok\/s/i);
  assert.match(glmBlock, /same base|같은 base|동일 base/i); // 상속 근거를 코드에 남긴다
  assert.ok(glmBlock.includes(GLM53_REVISION));
});

test('AC7_fail_closed_unchanged: Flash-Next stays absent and both pinned GLM configs are rejected as DSA', () => {
  assert.equal(LOCAL_MODELS.some((m) => /flash[\s-]?next/i.test(JSON.stringify(m))), false);
  assert.notEqual(resolveLocalModel('Qwen3.8-Flash-Next').status, 'resolved');
  assert.notEqual(resolveLocalModel('Qwen/Qwen3.8-Flash-Next-NVFP4').status, 'resolved');
  assert.ok(byName('Qwen 3.8 27B'));
  const glm53Config = { ...SHARED_GEOMETRY, quantization_config: GLM53_QUANTIZATION, torch_dtype: 'bfloat16' };
  const glm52Config = { ...SHARED_GEOMETRY, torch_dtype: 'bfloat16' };
  assert.throws(() => parseHfConfig('zai-org/GLM-5.3', glm53Config, null, {
    revision: GLM53_REVISION, safetensorsParameters: GLM53_PARAMETERS, safetensorsTotal: SAFETENSORS_TOTAL,
  }), /비표준 압축·희소·혼합 어텐션/);
  assert.throws(() => parseHfConfig('zai-org/GLM-5.2', glm52Config, null, {
    revision: GLM52_REVISION, safetensorsParameters: GLM52_PARAMETERS, safetensorsTotal: SAFETENSORS_TOTAL,
  }), /비표준 압축·희소·혼합 어텐션/);
});

test('AC9_dropdown_grouping: GLM group = [4.7-Flash, 5.2, 5.3], group order unchanged, all 27 addressable', () => {
  assert.deepEqual(MODEL_GROUP_ORDER, BASELINE_GROUP_ORDER);
  const groups = groupedForDisplay(LOCAL_MODELS);
  assert.deepEqual(groups.find((g) => g.group === 'GLM').items.map((m) => m.name), ['GLM-4.7-Flash', 'GLM-5.2', 'GLM-5.3']);
  assert.deepEqual(groups.slice(0, 5).map((g) => g.group), ['Granite', 'Spark', 'Qwen 3.8', 'Laguna', 'GLM']);
  const flat = groups.flatMap((g) => g.items);
  assert.equal(flat.length, 27);
  for (const m of flat) assert.ok(LOCAL_MODELS.indexOf(m) >= 0);
});

test('AC10_cli_parity: CLI text/JSON/--why for GLM-5.3 equal GLM-5.2 with only the name and receipt slug substituted', () => {
  const substitute = (s) => s.replaceAll('GLM-5.2', 'GLM-5.3').replaceAll('glm-5-2-', 'glm-5-3-');
  const cases = [
    { args: ['--gpu', '4090'], code: 1 },
    { args: ['--mac', '512', '--quant', '4'], code: 0 },
    { args: ['--gpu', 'RTX 5090 + RTX 3090', '--ctx', '32768'], code: 1 },
  ];
  for (const { args, code } of cases) {
    const label = args.join(' ');
    const a = run(['GLM-5.2', ...args]);
    const b = run(['GLM-5.3', ...args]);
    assert.equal(a.code, code, `GLM-5.2 ${label} exit`);
    assert.equal(b.code, code, `GLM-5.3 ${label} exit`);
    assert.equal(b.out, substitute(a.out), `text ${label}`);
    const aj = run(['GLM-5.2', ...args, '--json']);
    const bj = run(['GLM-5.3', ...args, '--json']);
    assert.equal(bj.code, aj.code, `json ${label} exit`);
    assert.deepEqual(JSON.parse(bj.out), { ...JSON.parse(aj.out), model: 'GLM-5.3' }, `json ${label}`);
    const aw = run(['GLM-5.2', ...args, '--json', '--why']);
    const bw = run(['GLM-5.3', ...args, '--json', '--why']);
    assert.equal(bw.out, substitute(aw.out), `why ${label}`);
    assert.ok(bw.out.includes('mla-compressed-latent-cache'), `why ${label} premise`);
  }
  const text = run(['GLM-5.3', '--gpu', '4090']);
  assert.match(text.out, /premise \[mla-compressed-latent-cache\]/);
  assert.match(text.out, /receipt: https:\/\/fitllm\.run\/r\/glm-5-3-/);
  const list = run(['--list']);
  assert.equal(list.code, 0);
  assert.match(list.out, /GLM-5\.3/);
  assert.match(list.out, /GLM-5\.2/);
});

test('AC11_action_parity: composite Action carries GLM-5.3 through the 0/1 exit contract like GLM-5.2', () => {
  const noFit = runAction({ INPUT_MODEL: 'GLM-5.3', INPUT_GPU: 'RTX 4090' });
  assert.equal(noFit.status, 1);
  assert.match(noFit.output, /exit-code=1/);
  assert.match(noFit.output, /"verdict": "no"/);
  assert.match(noFit.output, /"model": "GLM-5\.3"/);
  assert.match(noFit.output, /mla-compressed-latent-cache/);
  const fits = runAction({ INPUT_MODEL: 'GLM-5.3', INPUT_MAC: '512', INPUT_QUANT: '4' });
  assert.equal(fits.status, 0);
  assert.match(fits.output, /exit-code=0/);
  assert.match(fits.output, /"verdict": "yes"/);
  const reference = runAction({ INPUT_MODEL: 'GLM-5.2', INPUT_GPU: 'RTX 4090' });
  assert.equal(reference.status, 1);
  const strip = (out) => out.replace(/^result<<.*$/m, 'result<<X').replace(/^FITLLM_RESULT_.*$/m, 'X');
  assert.equal(strip(noFit.output), strip(reference.output).replaceAll('GLM-5.2', 'GLM-5.3'));
});
