// 출시 FP4 전문가 가중치 — gpt-oss(MXFP4)·DeepSeek-V4 계열(FP4)은 전문가 텐서를 E2M1 4bit + 32원소당 E8M0 스케일 = 4.25 bit로
// 내놓고, 공개 GGUF는 어느 단계에서도 그 텐서를 다시 펴지 않는다. 2.21.4까지 엔진은 모델 전체에 단계 평균 bit를 곱해
// gpt-oss-120b Q8_0을 115.6 GiB(실파일 59.0)로 세어 GB10(128GB)에서 거짓 "안 들어감"을 냈다.
// 기대 바이트는 HF 파일 목록(2026-10-23 조회, 리비전 고정)에서 옮겼다 — 테스트는 네트워크를 쓰지 않는다.
//   https://huggingface.co/api/models/openai/gpt-oss-120b/revision/b5c939de8f754692c1647ca79fbf85e8c1e70f8a?blobs=true
//   https://huggingface.co/api/models/openai/gpt-oss-20b/revision/6cee5e81ee83917806bbde320786a8fb61efebee?blobs=true
//   https://huggingface.co/api/models/unsloth/gpt-oss-120b-GGUF/revision/ff1a82da6ad466e32284fa3d2b86694db3204789?blobs=true
//   https://huggingface.co/api/models/unsloth/gpt-oss-20b-GGUF/revision/d449b42d93e1c2c7bda5312f5c25c8fb91dfa9b4?blobs=true
//   https://huggingface.co/api/models/ggml-org/gpt-oss-120b-GGUF/revision/7200b74bea78b0a2964ef9f3897eefa401e890ff?blobs=true
//   https://huggingface.co/api/models/unsloth/DeepSeek-V4-Flash-GGUF/revision/e3aa0d6a5fa4f820d9e132ac1fd1d01e1b2b49e0?blobs=true
//   https://huggingface.co/api/models/unsloth/GLM-5.3-GGUF/revision/346b3591c7f28d1a23716f97a065ecf12ec14771?blobs=true
//   https://huggingface.co/api/models/bartowski/Nex-N2.5-Pro-GGUF/revision/f7c3c07945cf420f149226c010e10b1343f5d149?blobs=true
import test from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import {
  GPUS, GPU_QUANTS, LOCAL_MODELS, MACBOOK_RAM_GROUPS, appleDevice, calcMaxContext, calcParamMemory, gpuDevice, parseHfConfig,
  simulate, structuralAssumptions,
} from '../engine.js';

const BIN = new URL('../bin/fitllm.mjs', import.meta.url).pathname;
const GIB = 1024 ** 3;
const byName = (name) => LOCAL_MODELS.find((m) => m.name === name);
const gpu = (name) => gpuDevice(GPUS.find((g) => g.name === name));
const tier = (t) => GPU_QUANTS.find((q) => q.tier === t).bpw;
const weightBytes = (model, bits, device) => calcParamMemory(model, bits, device).totalGB * GIB;

const RELEASE_BYTES = { 'gpt-oss-120b': 65248893184, 'gpt-oss-20b': 13761316904 }; // safetensors 샤드 합(original/ 제외)
const GGUF_BYTES = { // unsloth 단계별 파일(분할 파일 합)
  'gpt-oss-120b': { Q4_K_M: 62768723552, Q5_K_M: 62889522272, Q6_K: 63284496960, Q8_0: 63387347520, FP16: 65369017728 },
  'gpt-oss-20b': { Q4_K_M: 11624759488, Q5_K_M: 11717357248, Q6_K: 12041000128, Q8_0: 12109567168, FP16: 13792639168 },
};
const FP4_ROWS = ['gpt-oss-20b', 'gpt-oss-120b', 'DeepSeek-V4.1-Flash-1017', 'DeepSeek-V4-Flash', 'DeepSeek-V4-Pro'];

test('catalog: exactly the five FP4-release rows carry nativeFp4Params, equal to the HF logical count of their FP4 tensors', () => {
  assert.deepEqual(LOCAL_MODELS.filter((m) => m.nativeFp4Params).map((m) => m.name), FP4_ROWS);
  const handCounts = {
    'gpt-oss-20b': 24 * 32 * (2880 * 5760 + 2880 * 2880), // HF U8 19,110,297,600
    'gpt-oss-120b': 36 * 128 * (2880 * 5760 + 2880 * 2880), // HF U8 114,661,785,600
    'DeepSeek-V4.1-Flash-1017': (40 * 384 + 3 * 128) * 3 * 5120 * 2304, // HF I8 557,171,343,360
    'DeepSeek-V4-Flash': (43 + 1) * 256 * 3 * 4096 * 2048, // HF I8 283,467,841,536
    'DeepSeek-V4-Pro': 1572763336704, // HF I8
  };
  for (const [name, count] of Object.entries(handCounts)) assert.equal(Math.round(byName(name).nativeFp4Params * 1e9), count, name);
  // FP8 출시본(GLM-5.3 · Nex-N2.5-Pro)은 이 부류가 아니다 — GGUF에 FP8 타입이 없어 단계별로 다시 쓴다
  assert.equal(byName('GLM-5.3').nativeFp4Params, undefined);
  assert.equal(byName('Nex-N2.5-Pro').nativeFp4Params, undefined);
});

test('gpt-oss: no GGUF tier counts more than the release file, and each tier lands within 4% under the published GGUF (2.21.4: 6–240% over)', () => {
  const discrete = gpu('RTX 4090');
  for (const name of ['gpt-oss-20b', 'gpt-oss-120b']) {
    const m = byName(name);
    for (const [t, file] of Object.entries(GGUF_BYTES[name])) {
      const w = weightBytes(m, tier(t), discrete);
      assert.ok(w <= RELEASE_BYTES[name], `${name} ${t} ${w} > release ${RELEASE_BYTES[name]}`);
      assert.ok(w <= file && (file - w) / file < 0.04, `${name} ${t}: engine ${(w / GIB).toFixed(2)} vs file ${(file / GIB).toFixed(2)} GiB`);
      const old = m.totalParams * 1e9 * tier(t) / 8; // 2.21.4: 전부 단계 평균 bit
      assert.ok(old / file > 1.06, `${name} ${t} old ${(old / GIB).toFixed(2)}`);
    }
    // Mac 8·16bit도 출시 크기를 넘지 않는다(4bit는 4.25 미만이라 종래 그대로)
    for (const bits of [8, 16]) assert.ok(weightBytes(m, bits, appleDevice(128)) <= RELEASE_BYTES[name], `${name} mac ${bits}`);
    assert.equal(calcParamMemory(m, 4, appleDevice(128)).totalGB, (m.totalParams * 1e9 * 0.5) / GIB);
  }
  // 바이트 확인: MXFP4 전문가 × 4.25/8 + BF16 × 2 = 원본 샤드 합(차는 safetensors 헤더 77 KB)
  assert.equal(RELEASE_BYTES['gpt-oss-120b'] - (114661785600 * 4.25 / 8 + 2167371072 * 2), 77440);
});

test('GB10 one unit: gpt-oss-120b fits at 4-bit and 8-bit (2.21.4 said 8-bit does not fit), weights at the real file size', () => {
  const spark = gpu('DGX Spark 128GB');
  const m = byName('gpt-oss-120b');
  const q4 = simulate(m, spark, 8192, { weightBpw: tier('Q4_K_M'), kvBits: 16 });
  const q8 = simulate(m, spark, 8192, { weightBpw: tier('Q8_0'), kvBits: 16 });
  // 가중치: 114.6617856e9 × 4.25/8 + 2.1382144e9 × bpw/8 — 실파일 Q4_K_M 58.46 · Q8_0(=ggml-org MXFP4) 59.03 GiB
  assert.equal(+q4.param.toFixed(2), 57.95); // 2.21.4: 66.55
  assert.equal(+q8.param.toFixed(2), 58.85); // 2.21.4: 115.59
  assert.equal(q4.verdict, 'yes');
  assert.equal(+q4.used.toFixed(2), 71.83); // 2.21.4: 80.43
  assert.equal(q8.verdict, 'yes'); // 2.21.4: no (129.47 of 128)
  assert.equal(+q8.used.toFixed(2), 72.73);
  assert.equal(calcMaxContext(m, spark, { weightBpw: tier('Q8_0'), kvBits: 16 }), m.maxContext); // 128K 전부
  const fp16 = simulate(m, spark, 131072, { weightBpw: tier('FP16'), kvBits: 16 });
  assert.equal(fp16.verdict, 'yes');
  assert.ok(structuralAssumptions(m, spark).some((a) => a.id === 'native-fp4-expert-weights'));
});

test('16 GB GPU: gpt-oss-20b fits at Q4_K_M and Q8_0 (8K), F16 still does not; 2.21.4 said tight / no', () => {
  for (const name of ['RTX 4060 Ti 16GB', 'RTX 5060 Ti 16GB']) {
    const d = gpu(name);
    const m = byName('gpt-oss-20b');
    const q4 = simulate(m, d, 8192, { weightBpw: tier('Q4_K_M'), kvBits: 16 });
    const q8 = simulate(m, d, 8192, { weightBpw: tier('Q8_0'), kvBits: 16 });
    const f16 = simulate(m, d, 8192, { weightBpw: tier('FP16'), kvBits: 16 });
    assert.equal(+q4.param.toFixed(2), 10.47); // 2.21.4: 11.91 — 실파일 10.83
    assert.equal(q4.verdict, 'yes', name); // 2.21.4: tight (15.80 of 16)
    assert.equal(+q4.used.toFixed(2), 14.2);
    assert.equal(+q8.param.toFixed(2), 11.23); // 2.21.4: 20.68 — 실파일 11.28
    assert.equal(q8.verdict, 'yes', name); // 2.21.4: no (25.63)
    assert.equal(f16.verdict, 'no', name); // F16 실파일 12.85 GiB + 12% + Windows reserve 2.0 → 16.79
    assert.ok(calcMaxContext(m, d, { weightBpw: tier('Q4_K_M'), kvBits: 16 }) > 16384);
  }
});

test('rows without nativeFp4Params keep the 2.21.4 weight formula bit-for-bit on every device kind and tier', () => {
  const multiplier = { 'Gemma 4 e2b': { 16: 1.0, 8: 1.15, 4: 1.39 }, 'Gemma 4 e4b': { 16: 1.0, 8: 1.11, 4: 1.29 }, 'Gemma 4 31b': { 16: 1.02, 8: 1.1, 4: 1.2 }, 'Gemma 4 26b A4B': { 16: 1.01, 8: 1.1, 4: 1.2 } };
  const devices = [gpu('RTX 4090'), gpu('DGX Spark 128GB'), gpu('4× DGX Spark 128GB'), appleDevice(64), undefined];
  const bitsList = [...GPU_QUANTS.map((q) => q.bpw), 4, 8, 16];
  let checked = 0;
  for (const m of LOCAL_MODELS.filter((x) => !x.nativeFp4Params)) {
    for (const d of devices) {
      for (const bits of bitsList) {
        // 2.21.4 calcParamMemory 원문 식(PLE 차감 · Engram 하한 · quantAdjust 포함)
        const ple = d && d.type === 'gpu' && !d.unifiedMemory && m.pleOffloadVerified === true && m.pleParams && m.pleParams < m.totalParams ? m.pleParams : 0;
        const floor = m.engramParams ? m.engramParams * 1e9 * Math.max(0, 8.25 - bits) / 8 : 0;
        const expected = (((m.totalParams - ple) * 1e9 * (bits / 8) + floor) / GIB) * ((multiplier[m.name] && multiplier[m.name][bits]) || 1.0);
        assert.equal(calcParamMemory(m, bits, d).totalGB, expected, `${m.name} ${bits} ${d?.type}`);
        checked++;
      }
    }
    assert.ok(!structuralAssumptions(m, gpu('RTX 4090')).some((a) => a.id === 'native-fp4-expert-weights'), m.name);
  }
  assert.ok(checked >= 29 * 5 * 8, `checked ${checked}`);
  // FP4 행도 Mac 4bit(4 < 4.25)는 종래와 같다
  for (const name of FP4_ROWS) {
    const m = byName(name);
    const floor = m.engramParams ? m.engramParams * 1e9 * (8.25 - 4) / 8 : 0;
    assert.equal(calcParamMemory(m, 4, appleDevice(512)).totalGB, (m.totalParams * 1e9 * 0.5 + floor) / GIB, name);
  }
});

test('FP8 releases stay on the tier formula: their published Q8_0 / BF16 GGUFs re-encode FP8 and match it within 0.2%; a release-size cap would understate them', () => {
  const glm = byName('GLM-5.3');
  const nex = byName('Nex-N2.5-Pro');
  const d = gpu('RTX 4090');
  const files = [
    [glm, 'Q8_0', 801357677216], [glm, 'FP16', 1507988027936], // unsloth/GLM-5.3-GGUF @346b359 (Q8_0 · BF16 폴더 합)
    [nex, 'Q8_0', 421576669344], // bartowski/Nex-N2.5-Pro-GGUF @f7c3c07
  ];
  for (const [m, t, file] of files) {
    const w = weightBytes(m, tier(t), d);
    assert.ok(Math.abs(w - file) / file < 0.002, `${m.name} ${t}: ${(w / GIB).toFixed(1)} vs ${(file / GIB).toFixed(1)} GiB`);
  }
  // FP8 출시본 바이트(GLM-5.3 755,632,050,320 · Nex 407,130,522,144)는 Q8_0 GGUF보다 5–6% 작다 — 상한으로 쓰면 거짓 fits 방향
  assert.ok(755632050320 < 801357677216 * 0.95);
  assert.ok(407130522144 < 421576669344 * 0.97);
});

test('DeepSeek-V4: FP4 experts at their shipped size — V4-Flash Q4_K_M matches the published UD-Q4_K_XL within 0.1%; Q8_0 halves vs 2.21.4', () => {
  const f = byName('DeepSeek-V4-Flash');
  const d = gpu('4× DGX Spark 128GB');
  const q4 = weightBytes(f, tier('Q4_K_M'), d);
  assert.ok(Math.abs(q4 - 155095240320) / 155095240320 < 0.001, `${q4 / GIB}`); // unsloth UD-Q4_K_XL 144.44 GiB
  const q8 = weightBytes(f, tier('Q8_0'), d);
  assert.ok(q8 <= 159617149040, 'V4-Flash Q8_0 within the 148.66 GiB release'); // 출시 샤드 합
  assert.equal(+(q8 / GIB).toFixed(2), 147.61); // 2.21.4: 287.88
  // FP8·BF16 몫은 단계를 따른다(GGUF에 FP8 타입 없음) — FP16 단계는 그 몫만큼 출시본보다 크다
  assert.ok(weightBytes(f, tier('FP16'), d) > 159617149040);
  // 4.25 bit 미만(Mac 4bit)은 다시 양자화한 빌드라 선택 정밀도 그대로
  assert.equal(calcParamMemory(f, 4, appleDevice(256)).totalGB, (f.totalParams * 1e9 * 0.5) / GIB);
});

// gpt-oss-hf-parity.test.mjs와 같은 공식 config(필요한 키만) — 즉석 파싱 경로도 같은 FP4 수를 읽는지, 못 읽으면 상한을 두지 않는지
const gptOss120bConfig = Object.freeze({
  architectures: ['GptOssForCausalLM'], experts_per_token: 4, head_dim: 64, hidden_size: 2880, intermediate_size: 2880,
  layer_types: Array.from({ length: 36 }, (_, i) => (i % 2 === 0 ? 'sliding_attention' : 'full_attention')),
  max_position_embeddings: 131072, model_type: 'gpt_oss', num_attention_heads: 64, num_experts_per_tok: 4, num_hidden_layers: 36,
  num_key_value_heads: 8, num_local_experts: 128,
  quantization_config: { modules_to_not_convert: ['model.layers.*.self_attn', 'model.layers.*.mlp.router', 'model.embed_tokens', 'lm_head'], quant_method: 'mxfp4' },
  sliding_window: 128, tie_word_embeddings: false, vocab_size: 201088,
});
const evidence120b = (parameters = { BF16: 2167371072, U8: 114661785600 }) => ({
  revision: 'b5c939de8f754692c1647ca79fbf85e8c1e70f8a', safetensorsParameters: parameters,
  safetensorsTotal: Object.values(parameters).reduce((a, b) => a + b, 0),
});

test('HF live parse reads the FP4 count only when the shard bytes are explained by it; otherwise no cap (counts everything at the tier, the larger answer)', () => {
  const parsed = parseHfConfig('openai/gpt-oss-120b', gptOss120bConfig, RELEASE_BYTES['gpt-oss-120b'], evidence120b());
  assert.equal(parsed.nativeFp4Params, byName('gpt-oss-120b').nativeFp4Params);
  assert.ok(structuralAssumptions(parsed, gpu('DGX Spark 128GB')).some((a) => a.id === 'native-fp4-expert-weights'));
  // 바이트가 FP4 4.25 bit로 설명되지 않으면(100 MB 초과 차이) 상한을 두지 않는다
  const off = parseHfConfig('x/gpt-oss-120b', gptOss120bConfig, RELEASE_BYTES['gpt-oss-120b'] + 100_000_000, evidence120b());
  assert.equal(off.nativeFp4Params, undefined);
  assert.ok(calcParamMemory(off, tier('Q8_0')).totalGB > calcParamMemory(parsed, tier('Q8_0')).totalGB * 1.9);
  // 선언 없는 U8(예: 다른 packed 형식)은 FP4로 읽지 않는다
  const noDecl = { ...gptOss120bConfig, quantization_config: { quant_method: 'bitsandbytes', bits: 4 } };
  assert.equal(parseHfConfig('x/y', noDecl, RELEASE_BYTES['gpt-oss-120b'], evidence120b()).nativeFp4Params, undefined);
  // 증거가 없으면 종래처럼 fail-closed(#98)
  assert.throws(() => parseHfConfig('openai/gpt-oss-120b', gptOss120bConfig, RELEASE_BYTES['gpt-oss-120b']), /저장 비트폭을 확정할 수 없어/);
});

test('CLI: the seller-card case answers FITS — gpt-oss-120b on one DGX Spark at Q8_0, with the FP4 premise in --json', () => {
  const r = spawnSync(process.execPath, [BIN, 'gpt-oss-120b', '--gpu', 'DGX Spark 128GB', '--quant', 'Q8_0', '--json'], { encoding: 'utf8' });
  assert.equal(r.status, 0, r.stdout + r.stderr);
  const j = JSON.parse(r.stdout);
  assert.equal(j.verdict, 'yes');
  assert.equal(j.usedGB, 72.73);
  assert.deepEqual(j.structuralAssumptions.map((a) => a.id), ['native-fp4-expert-weights', 'gb10-unified-memory-reserve']);
  // 맥 64GB는 여전히 안 들어간다(README 예시 그대로)
  const mac = spawnSync(process.execPath, [BIN, 'gpt-oss-120b', '--mac', '64'], { encoding: 'utf8' });
  assert.equal(mac.status, 1, mac.stdout);
  assert.ok(Object.values(MACBOOK_RAM_GROUPS).flat().includes(64));
});
