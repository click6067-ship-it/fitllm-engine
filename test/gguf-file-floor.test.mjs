// 공개 GGUF 하한(ggufFileBytes) — 단계 평균 bpw 공식이 실제 공개 파일보다 작아 "들어간다"고 잘못 답하던 부류(2026-10-23 전수 조사).
// 픽스처 = 카탈로그 34개 × 5단계의 기준 파일 중 가장 큰 것(HF 파일 목록 ?blobs=true, 리비전 고정) + Gemma 4 e2b·e4b GGUF 헤더에서 읽은
// per_layer_token_embd 바이트. 기준 배포자·제외 규칙은 픽스처 method와 engine.js ggufFileFloorBytes 주석에 있다.
// 출시 FP4 전문가 행(gpt-oss · DeepSeek-V4)은 native-fp4.test.mjs가 맡는다.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import {
  GPUS, GPU_QUANTS, LOCAL_MODELS, appleDevice, calcParamMemory, gpuDevice, parseHfConfig, simulate,
} from '../engine.js';

const GIB = 1024 ** 3;
const FIX = JSON.parse(readFileSync(new URL('./fixtures/gguf-file-sizes-2026-10-23.json', import.meta.url), 'utf8'));
const byName = (name) => LOCAL_MODELS.find((m) => m.name === name);
const gpu = (name) => gpuDevice(GPUS.find((g) => g.name === name));
const tier = (t) => GPU_QUANTS.find((q) => q.tier === t).bpw;
const weightBytes = (model, bits, device) => calcParamMemory(model, bits, device).totalGB * GIB;
const formulaBytes = (m, bits) => m.totalParams * 1e9 * (bits / 8) + (m.engramParams ? m.engramParams * 1e9 * Math.max(0, 8.25 - bits) / 8 : 0);
const SPARK = gpu('DGX Spark 128GB'); // 통합 메모리 — PLE를 빼지 않고 12% 몫도 없다: calcParamMemory = 가중치 바이트 그대로
const TOLERANCE = 1.01;

test('fixture covers every catalog row with public GGUFs except the shipped-FP4 rows', () => {
  const names = Object.keys(FIX.largestReferenceFile).sort();
  assert.deepEqual(names, LOCAL_MODELS.filter((m) => !m.nativeFp4Params).map((m) => m.name).sort());
  // 기준 배포자에 그 단계 파일이 없으면 칸이 없다(Qwen-AgentWorld Q4_K_M·Q5_K_M은 vultr 파일뿐 — 문서의 '그 밖의 업로더' 표)
  assert.equal(FIX.largestReferenceFile['Qwen-AgentWorld-35B-A3B'].Q4_K_M, undefined);
});

test('every reference GGUF is at most 1% above the engine weights of its tier (GB10 and discrete GPU paths)', () => {
  let checked = 0;
  for (const [name, tiers] of Object.entries(FIX.largestReferenceFile)) {
    const m = byName(name);
    for (const [t, { bytes, source }] of Object.entries(tiers)) {
      const onSpark = weightBytes(m, tier(t), SPARK);
      assert.ok(onSpark * TOLERANCE >= bytes, `${name} ${t}: engine ${(onSpark / GIB).toFixed(2)} GiB < ${source} ${(bytes / GIB).toFixed(2)} GiB / 1.01`);
      if (!m.pleOffloadVerified) {
        assert.equal(weightBytes(m, tier(t), gpu('RTX 4090')), onSpark, `${name} ${t}: weights do not depend on the card`);
      }
      checked++;
    }
  }
  assert.equal(checked, 128);
});

test('a floor is stored only where the tier average is more than 1% short, and then the weights equal it exactly', () => {
  let floors = 0;
  for (const m of LOCAL_MODELS.filter((x) => x.ggufFileBytes && !x.nativeFp4Params)) {
    for (const [t, bytes] of Object.entries(m.ggufFileBytes)) {
      assert.ok(Number.isSafeInteger(bytes) && bytes > 0, `${m.name} ${t}`);
      assert.ok(bytes > formulaBytes(m, tier(t)) * TOLERANCE, `${m.name} ${t}: floor not needed`);
      assert.equal(weightBytes(m, tier(t), SPARK), bytes, `${m.name} ${t}`);
      // 픽스처의 그 모델 최대 파일 이상이다(같은 텐서 목록인 행끼리는 큰 쪽을 함께 쓴다)
      const own = FIX.largestReferenceFile[m.name]?.[t]?.bytes;
      assert.ok(own && bytes >= own, `${m.name} ${t}: floor below its own largest file`);
      floors++;
    }
  }
  assert.equal(floors, 61);
});

test('rows that share a tensor list share the same floors (Qwen 3.6 27B · Qwen 3.8 27B)', () => {
  const a = byName('Qwen 3.6 27B');
  const b = byName('Qwen 3.8 27B');
  assert.deepEqual(a.ggufFileSource, b.ggufFileSource);
  assert.deepEqual(a.ggufFileBytes, b.ggufFileBytes);
  const sources = LOCAL_MODELS.filter((m) => m.ggufFileSource).map((m) => JSON.stringify(m.ggufFileSource));
  const dupes = sources.filter((s, i) => sources.indexOf(s) !== i);
  assert.equal(dupes.length, 1, 'only the Qwen 27B pair shares a tensor list');
});

test('PLE rows: on a discrete GPU the resident weights stay above every reference file minus its per_layer_token_embd', () => {
  for (const [name, tiers] of Object.entries(FIX.pleResidentMax)) {
    const m = byName(name);
    assert.equal(m.pleOffloadVerified, true);
    for (const [t, { bytes, fileBytes, perLayerTokenEmbdBytes }] of Object.entries(tiers)) {
      assert.equal(bytes, fileBytes - perLayerTokenEmbdBytes);
      const resident = weightBytes(m, tier(t), gpu('RTX 4060 Ti 8GB'));
      assert.equal(resident, (m.totalParams - m.pleParams) * 1e9 * (tier(t) / 8), `${name} ${t}: discrete path uses no floor`);
      assert.ok(resident >= bytes, `${name} ${t}: resident ${(resident / GIB).toFixed(3)} < ${(bytes / GIB).toFixed(3)}`);
    }
  }
  // GB10은 같은 풀이라 PLE까지 센다 — 파일 전체가 하한(e2b Q4_K_M bartowski 3.22 GiB > 공식 2.91)
  const e2b = byName('Gemma 4 e2b');
  assert.equal(weightBytes(e2b, tier('Q4_K_M'), SPARK), e2b.ggufFileBytes.Q4_K_M);
});

test('the floor is a GGUF-tier fact: Mac 4/8/16-bit and off-tier bpw keep the formula', () => {
  const m = byName('Qwen3-0.6B');
  for (const bits of [4, 8, 16]) {
    assert.equal(calcParamMemory(m, bits, appleDevice(16)).totalGB, (m.totalParams * 1e9 * (bits / 8)) / GIB);
  }
  assert.equal(weightBytes(m, 8.5, SPARK), m.totalParams * 1e9 * (8.5 / 8)); // Q8_0 단계 bpw(8.5008)와 다르면 하한 없음
  assert.equal(weightBytes(m, tier('Q8_0'), SPARK), m.ggufFileBytes.Q8_0);
});

test('verdicts only move toward "won\'t fit": the 2026-10-23 trigger rows', () => {
  // 들어간다고 하던 조합 중 공개 파일로는 안 들어가는 것 — 고친 뒤에는 그 파일 기준으로 답한다
  const nexPro = simulate(byName('Nex-N2.5-Pro'), gpu('4× DGX Spark 128GB'), 131072, { weightBpw: tier('Q5_K_M'), kvBits: 16 });
  assert.equal(+(nexPro.param).toFixed(2), 284.62); // bartowski Q5_K_M 305,612,552,288 B (공식 263.47)
  const gemma31 = simulate(byName('Gemma 4 31b'), gpu('RTX 4090'), 8192, { weightBpw: tier('Q4_K_M'), kvBits: 16 });
  assert.equal(+(gemma31.param).toFixed(2), 18.25); // bartowski Q4_K_M 19,598,489,952 B (공식 17.49, lmstudio 17.40 · unsloth 17.07)
  assert.equal(gemma31.verdict, 'no');
});

test('HF live parse inherits the floors only for the exact tensor list, and merges rows that share it', () => {
  const qwen06 = JSON.parse(readFileSync(new URL('./fixtures/Qwen__Qwen3-0.6B.config.json', import.meta.url), 'utf8'));
  const evidence = (n) => ({ revision: 'c1899de289a04d12100db370d81485cdf75e47ca', safetensorsParameters: { BF16: n }, safetensorsTotal: n });
  const parsed = parseHfConfig('Qwen/Qwen3-0.6B', qwen06, 751632384 * 2, evidence(751632384));
  assert.deepEqual(parsed.ggufFileBytes, byName('Qwen3-0.6B').ggufFileBytes);
  assert.ok(weightBytes(parsed, tier('Q8_0'), SPARK) >= byName('Qwen3-0.6B').ggufFileBytes.Q8_0);
  // 어휘를 1,000 늘린 파생(임베딩 1,024 × 1,000 × 2) — 같은 이름 단계 크기를 물려받지 않는다
  const ext = parseHfConfig('x/qwen3-0.6b-ext', { ...qwen06, vocab_size: qwen06.vocab_size + 1000 }, (751632384 + 2048000) * 2, evidence(751632384 + 2048000));
  assert.equal(ext.ggufFileBytes, undefined);
  // 증거 없이(바이트만) 들어온 체크포인트도 물려받지 않는다
  assert.equal(parseHfConfig('Qwen/Qwen3-0.6B', qwen06, 751632384 * 2).ggufFileBytes, undefined);
  // Qwen3.6-27B와 Qwen3.8-27B는 텐서 목록이 같다 — 두 행 하한의 단계별 최대(= 두 행에 저장된 같은 값)
  const qwen27 = JSON.parse(readFileSync(new URL('./fixtures/Qwen__Qwen3.6-27B.config.json', import.meta.url), 'utf8'));
  const p27 = parseHfConfig('Qwen/Qwen3.6-27B', qwen27, 27781427952 * 2, {
    revision: '6a9e13bd6fc8f0983b9b99948120bc37f49c13e9', safetensorsParameters: { BF16: 27781427952 }, safetensorsTotal: 27781427952,
  });
  assert.deepEqual(p27.ggufFileBytes, byName('Qwen 3.8 27B').ggufFileBytes);
  assert.equal(calcParamMemory(p27, tier('Q4_K_M'), SPARK).totalGB, calcParamMemory(byName('Qwen 3.6 27B'), tier('Q4_K_M'), SPARK).totalGB);
});

test('every floored row carries the tensor list its files were measured on; gated originals use a same-count public mirror config', () => {
  for (const m of LOCAL_MODELS.filter((x) => x.ggufFileBytes && !x.nativeFp4Params)) {
    assert.ok(m.ggufFileSource && Object.keys(m.ggufFileSource.parameters).length && Object.keys(m.ggufFileSource.config).length, m.name);
  }
  // Codex R1: Llama-3.2-1B/3B·Gemma-3-1B는 원본 config가 gated라 빠져 있었다 — 같은 형상의 공개 미러가 하한을 못 물려받았다
  //   (Llama-3.2-1B Q4_K_M 공식 734,160,000 B vs 공개 파일 807,694,464 B, RTX 4060 Ti 8GB · 79,232 ctx에서 tight ↔ no)
  for (const [file, name] of [['unsloth__Llama-3.2-1B-Instruct', 'Llama-3.2-1B-Instruct'], ['unsloth__Llama-3.2-3B-Instruct', 'Llama-3.2-3B-Instruct']]) {
    const cfg = JSON.parse(readFileSync(new URL(`./fixtures/${file}.config.json`, import.meta.url), 'utf8'));
    const n = byName(name).ggufFileSource.parameters.BF16;
    const m = parseHfConfig(`unsloth/${name}`, cfg, n * 2, { revision: 'e'.repeat(40), safetensorsParameters: { BF16: n }, safetensorsTotal: n });
    assert.deepEqual(m.ggufFileBytes, byName(name).ggufFileBytes, name);
    assert.ok(weightBytes(m, tier('Q4_K_M'), gpu('RTX 4060 Ti 8GB')) >= byName(name).ggufFileBytes.Q4_K_M, name);
  }
});

test('Codex R2: a twin parsed from HF stays within 1% of every reference file on all five tiers, also when the config omits derivable keys', () => {
  const load = (f) => JSON.parse(readFileSync(new URL(`./fixtures/${f}.config.json`, import.meta.url), 'utf8'));
  const parse = (id, cfg, n) => parseHfConfig(id, cfg, n * 2, { revision: 'f'.repeat(40), safetensorsParameters: { BF16: n }, safetensorsTotal: n });
  const cases = [
    ['unsloth/Llama-3.2-1B-Instruct', load('unsloth__Llama-3.2-1B-Instruct'), 'Llama-3.2-1B-Instruct'],
    ['unsloth/Llama-3.2-3B-Instruct', load('unsloth__Llama-3.2-3B-Instruct'), 'Llama-3.2-3B-Instruct'],
    ['Qwen/Qwen3-0.6B', load('Qwen__Qwen3-0.6B'), 'Qwen3-0.6B'],
    ['Qwen/Qwen3.6-27B', load('Qwen__Qwen3.6-27B'), 'Qwen 3.6 27B'],
  ];
  for (const [id, cfg, name] of cases) {
    const n = byName(name).ggufFileSource.parameters.BF16;
    const variants = [cfg];
    // 파서가 계산하는 키를 지운 같은 체크포인트(head_dim = hidden / heads, 묶임 여부는 정확한 수가 가른다)
    const inner = cfg.text_config ? 'text_config' : null;
    const strip = (keys) => {
      const copy = structuredClone(cfg);
      for (const k of keys) { delete copy[k]; if (inner) delete copy[inner][k]; }
      return copy;
    };
    const body = inner ? cfg[inner] : cfg;
    if (body.head_dim === body.hidden_size / body.num_attention_heads) variants.push(strip(['head_dim']));
    variants.push(strip(['tie_word_embeddings']));
    for (const v of variants) {
      const m = parse(id, v, n);
      assert.ok(m.ggufFileBytes, `${id}: inherits`);
      for (const [t, { bytes, source }] of Object.entries(FIX.largestReferenceFile[name])) {
        const w = weightBytes(m, tier(t), SPARK);
        assert.ok(w * TOLERANCE >= bytes, `${id} ${t}: parsed ${(w / GIB).toFixed(3)} GiB vs ${source} ${(bytes / GIB).toFixed(3)}`);
      }
    }
  }
  // FP4 행은 엄격한 대조를 유지한다 — 물려받은 GGUF 하한이 출시본 하한을 대신하므로, 키 하나라도 빠지면 물려받지 않는다
  const gptOss120b = {
    architectures: ['GptOssForCausalLM'], experts_per_token: 4, head_dim: 64, hidden_size: 2880, intermediate_size: 2880,
    layer_types: Array.from({ length: 36 }, (_, i) => (i % 2 === 0 ? 'sliding_attention' : 'full_attention')),
    max_position_embeddings: 131072, model_type: 'gpt_oss', num_attention_heads: 64, num_experts_per_tok: 4, num_hidden_layers: 36,
    num_key_value_heads: 8, num_local_experts: 128, quantization_config: { quant_method: 'mxfp4' },
    sliding_window: 128, tie_word_embeddings: false, vocab_size: 201088,
  };
  const ev = { revision: 'b'.repeat(40), safetensorsParameters: { BF16: 2167371072, U8: 114661785600 }, safetensorsTotal: 116829156672 };
  assert.deepEqual(parseHfConfig('openai/gpt-oss-120b', gptOss120b, 65248893184, ev).ggufFileBytes, byName('gpt-oss-120b').ggufFileBytes);
  const { tie_word_embeddings: _omit, ...noTie } = gptOss120b;
  const strict = parseHfConfig('x/gpt-oss-120b', noTie, 65248893184, ev);
  assert.equal(strict.ggufFileBytes, undefined);
  assert.equal(strict.totalParams, 116.8); // FP4 쌍은 종래 0.1B 반올림 그대로
});
