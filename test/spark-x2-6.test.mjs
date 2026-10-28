// Spark-X2.6-4B (XHToken) — 2026-10-27 Day-0 #173(지시형)·#172(Base). 신규 수학 없음: X2.5와 구조가 바이트 단위로 같다.
// 1차 출처(지시형 pinned 66e3ae7bee54e0a523d692cb8ef841f35409807b · Base pinned e1bd659ae2b1c2263b2fc2544485976c1b516115):
//   config.json  https://huggingface.co/XHToken/Spark-X2.6-4B/blob/66e3ae7bee54e0a523d692cb8ef841f35409807b/config.json
//   modeling     https://huggingface.co/XHToken/Spark-X2.6-4B/blob/66e3ae7bee54e0a523d692cb8ef841f35409807b/modeling_spark.py
//   HF API       https://huggingface.co/api/models/XHToken/Spark-X2.6-4B/revision/66e3ae7bee54e0a523d692cb8ef841f35409807b
// 세 리비전(X2.5@5e10fcc · X2.6@66e3ae7 · X2.6-Base@e1bd659)의 config.json sha256이 아래 리터럴로 같다 — 그래서 X2.5의 pinned 사본을 쓴다.
// 네트워크 의존 없음.
import { test } from 'node:test';
import { strict as assert } from 'node:assert';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import {
  LOCAL_MODELS, GPUS, GPU_QUANTS, gpuDevice, parseHfConfig, calcKVCache, calcParamMemory, resolveLocalModel,
} from '../engine.js';
import { catalogModelNamesFrom, isAlreadyInCatalog } from '../.github/scripts/day0-core.mjs';

const BIN = new URL('../bin/fitllm.mjs', import.meta.url).pathname;
const run = (args) => {
  try { return { out: execFileSync(process.execPath, [BIN, ...args], { encoding: 'utf8' }), code: 0 }; }
  catch (e) { return { out: (e.stdout || '') + (e.stderr || ''), code: e.status }; }
};
const CONFIG_BYTES = readFileSync(new URL('./fixtures/day0/spark-x2.5-4b.config.json', import.meta.url));
const CONFIG = JSON.parse(CONFIG_BYTES);
const CONFIG_SHA256 = '767161ece8ed2891e44345afdf98d29c1cacb964116cb6db7d20dc1c757490ad'; // X2.5·X2.6·X2.6-Base 공통
const EVIDENCE = { revision: '66e3ae7bee54e0a523d692cb8ef841f35409807b', safetensorsParameters: { BF16: 4112079360 }, safetensorsTotal: 4112079360 };
const GIB = 1024 ** 3;
const byName = (name) => LOCAL_MODELS.find((m) => m.name === name);
const tier = (t) => GPU_QUANTS.find((q) => q.tier === t).bpw;
const SPARK = gpuDevice(GPUS.find((g) => g.name === 'DGX Spark 128GB'));

test('the pinned config is the same file for X2.5 and X2.6', () => {
  assert.equal(createHash('sha256').update(CONFIG_BYTES).digest('hex'), CONFIG_SHA256);
  assert.equal(CONFIG.model_type, 'spark2_5');
});

test('the X2.6 row is the X2.5 row under a new name: same structure, same tensor list, same floors', () => {
  const x25 = byName('Spark-X2.5-4B');
  const x26 = byName('Spark-X2.6-4B');
  const own = new Set(['name', 'added', 'aliases']);
  const keys = new Set([...Object.keys(x25), ...Object.keys(x26)].filter((k) => !own.has(k)));
  for (const k of keys) assert.deepEqual(x26[k], x25[k], k);
  assert.equal(x26.added, '2026-10-27');
  assert.deepEqual(x26.ggufFileBytes, { Q4_K_M: 3543441312, Q5_K_M: 3164766848, Q6_K: 3611148928 });
});

test('weights equal the largest public file where the tier average is short (GB10 path counts bytes as they are)', () => {
  const m = byName('Spark-X2.6-4B');
  // Q4_K_M = 공식 XHToken 파일(BF16 임베딩·Q8_0 output) — 같은 단계 mradermacher 2,600,225,568 B보다 36% 크다
  assert.equal(calcParamMemory(m, tier('Q4_K_M'), SPARK).totalGB * GIB, 3543441312);
  assert.equal(calcParamMemory(m, tier('Q5_K_M'), SPARK).totalGB * GIB, 3164766848);
  assert.equal(calcParamMemory(m, tier('Q6_K'), SPARK).totalGB * GIB, 3611148928);
});

test('HF live parse of the X2.6 id reads the same structure and inherits the shared floors', () => {
  const m = parseHfConfig('XHToken/Spark-X2.6-4B', CONFIG, 2 * 4112079360, EVIDENCE);
  assert.equal(m.totalParams, 4.112);
  assert.deepEqual(m.ggufFileBytes, byName('Spark-X2.6-4B').ggufFileBytes);
  assert.equal(calcKVCache(m, 1048576, 16).totalBytes, calcKVCache(byName('Spark-X2.6-4B'), 1048576, 16).totalBytes);
  assert.equal(calcKVCache(m, 1048576, 16).totalBytes, 38711328768); // X2.5 벡터 spark-x25-4b-kv-1m-f16과 같은 바이트
});

test('the instruct and Base ids resolve to the row, and the bare family name stays ambiguous', () => {
  for (const q of ['Spark-X2.6-4B', 'spark x2.6 4b', 'XHToken/Spark-X2.6-4B', 'Spark-X2.6-4B-Base', 'XHToken/Spark-X2.6-4B-Base']) {
    assert.equal(resolveLocalModel(q).canonicalName, 'Spark-X2.6-4B', q);
  }
  assert.equal(resolveLocalModel('Spark-X2.5-4B').canonicalName, 'Spark-X2.5-4B');
  assert.equal(resolveLocalModel('spark').status, 'ambiguous');
});

test('day0-watch treats both issue ids and their quantized repos as already in the catalog', () => {
  const names = catalogModelNamesFrom(LOCAL_MODELS);
  for (const id of ['XHToken/Spark-X2.6-4B', 'XHToken/Spark-X2.6-4B-Base', 'XHToken/Spark-X2.6-4B-GGUF', 'XHToken/Spark-X2.6-4B-FP8']) {
    assert.equal(isAlreadyInCatalog(id, names), true, id);
  }
  assert.equal(isAlreadyInCatalog('XHToken/Spark-X2.7-4B', names), false);
});

test('CLI: an 8GB card fits Q4_K_M at 8K with the 3.3 GiB official file', () => {
  const r = run(['Spark-X2.6-4B', '--gpu', 'RTX 4060 Ti 8GB']);
  assert.equal(r.code, 0, r.out);
  assert.match(r.out, /✓ FITS — Spark-X2\.6-4B on RTX 4060 Ti 8GB/);
  assert.match(r.out, /weights 3\.3 \+ KV 0\.3/);
  assert.match(r.out, /receipt: https:\/\/fitllm\.run\/r\/spark-x2-6-4b-q4_k_m-on-rtx-4060-ti-8gb/);
});
