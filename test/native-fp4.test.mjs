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
//   https://huggingface.co/api/models/bartowski/openai_gpt-oss-120b-GGUF/revision/8d64014efc314685c68ad448590f905fcb20e534?blobs=true
//   https://huggingface.co/api/models/bartowski/openai_gpt-oss-20b-GGUF/revision/e39ba3aa000c47c83dacdc9e1ca2c9dd0808c205?blobs=true
import test from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import {
  GPUS, GPU_QUANTS, LOCAL_MODELS, MACBOOK_RAM_GROUPS, appleDevice, calcMaxContext, calcParamMemory, gpuDevice, parseHfConfig,
  simulate, simulateStack, structuralAssumptions,
} from '../engine.js';

const BIN = new URL('../bin/fitllm.mjs', import.meta.url).pathname;
const GIB = 1024 ** 3;
const byName = (name) => LOCAL_MODELS.find((m) => m.name === name);
const gpu = (name) => gpuDevice(GPUS.find((g) => g.name === name));
const tier = (t) => GPU_QUANTS.find((q) => q.tier === t).bpw;
const weightBytes = (model, bits, device) => calcParamMemory(model, bits, device).totalGB * GIB;

const RELEASE_BYTES = { 'gpt-oss-120b': 65248893184, 'gpt-oss-20b': 13761316904 }; // safetensors 샤드 합(original/ 제외)
// 단계별 공개 GGUF(분할 파일 합) — unsloth @ff1a82d/@d449b42 · bartowski @8d64014/@e39ba3a. 엔진은 단계마다 가장 큰 파일을 쓴다.
const GGUF_BYTES = {
  'gpt-oss-120b': {
    Q4_K_M: [62768723552, 62841713344], Q5_K_M: [62889522272], Q6_K: [63284496960, 63284496064], Q8_0: [63387347520, 63387346368],
    FP16: [65369017728, 65369016768],
  },
  'gpt-oss-20b': {
    Q4_K_M: [11624759488, 11673418816], Q5_K_M: [11717357248, 11728415296], Q6_K: [12041000128, 12040998976], Q8_0: [12109567168, 12109566016],
    FP16: [13792639168, 13792637760],
  },
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

test('gpt-oss: each GGUF tier counts the largest public file of that tier; only F16 is above the release, by its F32 small tensors (2.21.4: 9.5–257% over)', () => {
  const discrete = gpu('RTX 4090');
  const over = [];
  for (const name of ['gpt-oss-20b', 'gpt-oss-120b']) {
    const m = byName(name);
    for (const [t, files] of Object.entries(GGUF_BYTES[name])) {
      const file = Math.max(...files);
      assert.equal(m.ggufFileBytes[t], file, `${name} ${t} row = largest public file`);
      const w = weightBytes(m, tier(t), discrete);
      assert.ok(Math.abs(w - file) < 1, `${name} ${t}: engine ${w} vs file ${file}`);
      // 단계 공식(MXFP4 4.25 bit + 나머지 단계 bpw)은 파일 아래로 0.1–3.7% — 그래서 공개 파일을 하한으로 둔다
      const formula = (m.totalParams - m.nativeFp4Params) * 1e9 * tier(t) / 8 + m.nativeFp4Params * 1e9 * 4.25 / 8;
      assert.ok(formula <= file && (file - formula) / file < 0.04, `${name} ${t} formula ${(formula / GIB).toFixed(2)}`);
      if (t === 'FP16') assert.ok(file > RELEASE_BYTES[name] && (file - RELEASE_BYTES[name]) / RELEASE_BYTES[name] < 0.0025, `${name} F16`);
      else assert.ok(file <= RELEASE_BYTES[name], `${name} ${t} file above the release`);
      over.push(m.totalParams * 1e9 * tier(t) / 8 / file - 1); // 2.21.4: 전부 단계 평균 bit
    }
    // Mac 8·16bit는 GGUF 단계가 아니라 출시본이 하한이다(4bit는 4.25 미만이라 종래 그대로)
    for (const bits of [8, 16]) assert.equal(weightBytes(m, bits, appleDevice(128)), RELEASE_BYTES[name], `${name} mac ${bits}`);
    assert.equal(calcParamMemory(m, 4, appleDevice(128)).totalGB, (m.totalParams * 1e9 * 0.5) / GIB);
  }
  assert.equal(+(Math.min(...over) * 100).toFixed(2), 9.54); // 20b Q4_K_M
  assert.equal(+(Math.max(...over) * 100).toFixed(2), 257.37); // 120b FP16
  // 바이트 확인: MXFP4 전문가 × 4.25/8 + BF16 × 2 = 원본 샤드 합(차는 safetensors 헤더 77 KB)
  assert.equal(RELEASE_BYTES['gpt-oss-120b'] - (114661785600 * 4.25 / 8 + 2167371072 * 2), 77440);
});

test('GB10 one unit: gpt-oss-120b fits at 4-bit and 8-bit (2.21.4 said 8-bit does not fit), weights at the real file size', () => {
  const spark = gpu('DGX Spark 128GB');
  const m = byName('gpt-oss-120b');
  const q4 = simulate(m, spark, 8192, { weightBpw: tier('Q4_K_M'), kvBits: 16 });
  const q8 = simulate(m, spark, 8192, { weightBpw: tier('Q8_0'), kvBits: 16 });
  // 가중치 = 단계별 가장 큰 공개 파일(공식 57.95·58.85보다 크다): Q4_K_M 62,841,713,344 B · Q8_0 63,387,347,520 B(= ggml-org MXFP4)
  assert.equal(+q4.param.toFixed(2), 58.53); // 2.21.4: 66.55
  assert.equal(+q8.param.toFixed(2), 59.03); // 2.21.4: 115.59
  assert.equal(q4.verdict, 'yes');
  assert.equal(+q4.used.toFixed(2), 72.41); // 2.21.4: 80.43
  assert.equal(q8.verdict, 'yes'); // 2.21.4: no (129.47 of 128)
  assert.equal(+q8.used.toFixed(2), 72.92);
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
    assert.equal(+q4.param.toFixed(2), 10.87); // 2.21.4: 11.91 — bartowski Q4_K_M 11,673,418,816 B
    assert.equal(q4.verdict, 'yes', name); // 2.21.4: tight (15.80 of 16)
    assert.equal(+q4.used.toFixed(2), 14.64);
    assert.equal(+q8.param.toFixed(2), 11.28); // 2.21.4: 20.68 — unsloth Q8_0 12,109,567,168 B
    assert.equal(q8.verdict, 'yes', name); // 2.21.4: no (25.63)
    assert.equal(+q8.used.toFixed(2), 15.1);
    assert.equal(f16.verdict, 'no', name); // F16 파일 12.85 GiB + 12% + Windows reserve 2.0 → 16.85
    assert.ok(calcMaxContext(m, d, { weightBpw: tier('Q4_K_M'), kvBits: 16 }) > 16384);
  }
});

test('rows without nativeFp4Params keep the 2.21.4 weight formula bit-for-bit on every device kind and tier, except the published GGUF floor of that exact tier', () => {
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
        // 공개 GGUF 하한(gguf-file-floor.test.mjs) — 그 단계 bpw 요청에만, PLE를 host에 두는 경로는 제외
        const q = GPU_QUANTS.find((x) => x.bpw === bits);
        const published = q && m.ggufFileBytes && !ple ? m.ggufFileBytes[q.tier] || 0 : 0;
        const expected = (Math.max((m.totalParams - ple) * 1e9 * (bits / 8) + floor, published) / GIB) * ((multiplier[m.name] && multiplier[m.name][bits]) || 1.0);
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
  // FP8 출시본 바이트(GLM-5.3 755,632,050,320 · Nex 407,130,522,144)는 Q8_0 GGUF보다 5.71% · 3.43% 작다 — 상한으로 쓰면 거짓 fits 방향
  assert.equal(+((1 - 755632050320 / 801357677216) * 100).toFixed(2), 5.71);
  assert.equal(+((1 - 407130522144 / 421576669344) * 100).toFixed(2), 3.43);
});

test('DeepSeek-V4: no same-tier public GGUF is known, so Q4_K_M–Q8_0 count the release itself (148.66 GiB; 2.21.4: 165.75–287.88); FP16 counts the FP8 share at 16 bits', () => {
  const f = byName('DeepSeek-V4-Flash');
  const d = gpu('4× DGX Spark 128GB');
  for (const t of ['Q4_K_M', 'Q5_K_M', 'Q6_K', 'Q8_0']) assert.equal(weightBytes(f, tier(t), d), 159617149040, t); // 출시 샤드 합
  // 공식(FP4 4.25 bit + 나머지 단계 bpw)은 Q4_K_M 144.48 GiB로 unsloth UD-Q4_K_XL(155,095,240,320 B = 144.44)과 맞지만, UD-Q8_K_XL
  // (161,869,614,720 B = 150.75)은 출시본보다도 크다 — 같은 이름 단계의 파일이 없으니 확인된 바이트(출시본)를 하한으로 둔다
  const formulaQ4 = (f.totalParams - f.nativeFp4Params) * 1e9 * tier('Q4_K_M') / 8 + f.nativeFp4Params * 1e9 * 4.25 / 8;
  assert.ok(Math.abs(formulaQ4 - 155095240320) / 155095240320 < 0.001);
  // FP8·BF16 몫은 단계를 따른다(GGUF에 FP8 타입 없음) — FP16 단계는 그 몫만큼 출시본보다 크다
  assert.equal(+(weightBytes(f, tier('FP16'), d) / GIB).toFixed(2), 154.09);
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
  // FP4 수·형상이 같으면 GGUF 단계 파일 크기도 같다 — 카탈로그 행의 공개 크기를 하한으로 물려받는다(복사본)
  assert.deepEqual(parsed.ggufFileBytes, byName('gpt-oss-120b').ggufFileBytes);
  assert.notEqual(parsed.ggufFileBytes, byName('gpt-oss-120b').ggufFileBytes);
  assert.equal(calcParamMemory(parsed, tier('Q4_K_M')).totalGB, calcParamMemory(byName('gpt-oss-120b'), tier('Q4_K_M')).totalGB);
  assert.ok(structuralAssumptions(parsed, gpu('DGX Spark 128GB')).some((a) => a.id === 'native-fp4-expert-weights'));
  // 바이트가 FP4 4.25 bit로 설명되지 않으면(100 MB 초과 차이) 상한을 두지 않는다
  const off = parseHfConfig('x/gpt-oss-120b', gptOss120bConfig, RELEASE_BYTES['gpt-oss-120b'] + 100_000_000, evidence120b());
  assert.equal(off.nativeFp4Params, undefined);
  assert.equal(off.ggufFileBytes, undefined);
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
  assert.equal(j.usedGB, 72.92);
  assert.deepEqual(j.structuralAssumptions.map((a) => a.id), ['native-fp4-expert-weights', 'gb10-unified-memory-reserve']);
  // 맥 64GB는 여전히 안 들어간다(README 예시 그대로)
  const mac = spawnSync(process.execPath, [BIN, 'gpt-oss-120b', '--mac', '64'], { encoding: 'utf8' });
  assert.equal(mac.status, 1, mac.stdout);
  assert.ok(Object.values(MACBOOK_RAM_GROUPS).flat().includes(64));
});

test('Codex R1: a verified PLE model never also takes the FP4 discount — no negative or double-deducted weights', () => {
  // E4B 형상 + mxfp4 선언 + U8 7.8B·BF16 0.196B — 샤드 바이트가 FP4 4.25 bit로 정확히 설명되는 합성 입력(4,536,062,980 B)
  // text_config에서 현재 구조 키 정책이 거부하는 키만 뺀 부분집합(ple-parser-fail-closed.test.mjs의 subset과 같다)
  const rejected = ['attention_k_eq_v', 'enable_moe_block', 'expert_intermediate_size', 'num_global_key_value_heads',
    'num_kv_shared_layers', 'top_k_experts', 'use_bidirectional_attention'];
  const text = JSON.parse(readFileSync(new URL('./fixtures/google__gemma-4-E4B-it.config.json', import.meta.url), 'utf8')).text_config;
  const cfg = { ...Object.fromEntries(Object.entries(text).filter(([k]) => !rejected.includes(k))), quantization_config: { quant_method: 'mxfp4' } };
  const parameters = { U8: 7800000000, BF16: 196156490 };
  const bytes = 7800000000 * 4.25 / 8 + 196156490 * 2;
  assert.equal(bytes, 4536062980);
  const m = parseHfConfig('x/e4b-mxfp4', cfg, bytes, {
    revision: 'b'.repeat(40), safetensorsParameters: parameters, safetensorsTotal: 7996156490,
  });
  const d = gpu('RTX 3060 8GB');
  for (const q of GPU_QUANTS) {
    const w = calcParamMemory(m, q.bpw, d).totalGB;
    assert.ok(w > 0, `${q.tier} ${w}`);
    if (m.pleOffloadVerified) {
      // PLE를 뺀 상주분 전체를 선택 정밀도로 센다 — FP4 몫을 따로 빼지 않는다
      assert.equal(w, ((m.totalParams - m.pleParams) * 1e9 * q.bpw / 8) / GIB, q.tier);
    }
  }
  assert.notEqual(simulate(m, d, 131072, { weightBpw: tier('FP16'), kvBits: 16 }).verdict, 'yes');
});

test('Codex R1: ten gpt-oss-20b Q6_K instances on one GB10 add up to the public file sizes — no', () => {
  const m = byName('gpt-oss-20b');
  const stack = simulateStack(Array.from({ length: 10 }, () => ({ model: m, ctx: 8192, weightBpw: tier('Q6_K'), kvBits: 16 })), gpu('DGX Spark 128GB'));
  assert.equal(+stack.param.toFixed(2), +((10 * 12041000128) / GIB).toFixed(2));
  assert.equal(stack.verdict, 'no');
  assert.equal(+stack.used.toFixed(2), 130.1);
});

// gpt-oss-20b 공식 config(필요한 키) — Codex 2차 검수 재현용
const gptOss20bConfig = Object.freeze({
  ...gptOss120bConfig, num_hidden_layers: 24, num_local_experts: 32,
  layer_types: Array.from({ length: 24 }, (_, i) => (i % 2 === 0 ? 'sliding_attention' : 'full_attention')),
});

test('Codex R2: a vocab-extended gpt-oss-20b derivative does not inherit the original file sizes; it falls back to its own release bytes', () => {
  // 어휘 +5,000 → BF16 +28,800,000(embed·lm_head 2 × 5,000 × 2,880), 샤드 +57,600,000 B
  const cfg = { ...gptOss20bConfig, vocab_size: 201088 + 5000 };
  const parameters = { BF16: 1804459584 + 28800000, U8: 19110297600 };
  const bytes = RELEASE_BYTES['gpt-oss-20b'] + 57600000;
  const m = parseHfConfig('x/gpt-oss-20b-ext', cfg, bytes, {
    revision: 'c'.repeat(40), safetensorsParameters: parameters, safetensorsTotal: parameters.BF16 + parameters.U8,
  });
  assert.equal(m.nativeFp4Params, byName('gpt-oss-20b').nativeFp4Params);
  assert.equal(m.totalParams, 20.9); // 반올림하면 원본과 같다 — 그래서 반올림 값으로는 쌍을 판정하지 않는다
  assert.equal(m.ggufFileBytes, undefined);
  assert.equal(m.fp4ReleaseBytes, bytes);
  for (const q of GPU_QUANTS) {
    const w = weightBytes(m, q.bpw, gpu('RTX 4090'));
    // 원본 공개 파일 + 늘어난 임베딩 바이트(그 단계에서 원소당 많아야 2 B)보다 작게 세지 않는다
    const lower = byName('gpt-oss-20b').ggufFileBytes[q.tier] + 2 * 5000 * 2880 * 2;
    assert.ok(w >= lower, `${q.tier} ${w} < ${lower}`);
  }
  // 재현: GB10 한 대에 F16 8개 · 25,459 ctx — 공개 파일 기준으로 안 들어가면 엔진도 안 들어간다고 한다
  const stack = simulateStack(Array.from({ length: 8 }, () => ({ model: m, ctx: 25459, weightBpw: tier('FP16'), kvBits: 16 })), gpu('DGX Spark 128GB'));
  assert.equal(stack.verdict, 'no');
});

test('Codex R2: evidence that a one-byte reading also explains (small U8 share inside the header slack) gets no FP4 discount', () => {
  const parameters = { U8: 100000000, BF16: 6900000000 };
  const bytes = 13900000000; // U8을 원소당 1바이트로 저장한 크기 그대로 — FP4로 읽어도 차이 46,875,000 B가 64 MB 여유 안에 든다
  const m = parseHfConfig('x/ambiguous-mxfp4', gptOss20bConfig, bytes, {
    revision: 'd'.repeat(40), safetensorsParameters: parameters, safetensorsTotal: 7000000000,
  });
  assert.equal(m.nativeFp4Params, undefined);
  assert.equal(m.fp4ReleaseBytes, undefined);
  assert.ok(weightBytes(m, tier('FP16'), gpu('RTX 5090')) >= bytes);
  // 공식 gpt-oss는 1바이트 해석과 53.7 GB 차이라 그대로 인정된다
  assert.equal(parseHfConfig('openai/gpt-oss-20b', gptOss20bConfig, RELEASE_BYTES['gpt-oss-20b'], {
    revision: '6cee5e81ee83917806bbde320786a8fb61efebee', safetensorsParameters: { BF16: 1804459584, U8: 19110297600 }, safetensorsTotal: 20914757184,
  }).nativeFp4Params, byName('gpt-oss-20b').nativeFp4Params);
});
