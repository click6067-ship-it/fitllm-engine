// DeepSeek-V4 계열 (2.21.0) — V4-Flash · V4-Pro(CSA + HCA)와 V4.1-Flash-1017(CSA2, 레이어 간 KV 공유).
// 원칙: 캐시 크기는 공식 추론 코드가 실제로 잡는 버퍼만 센다(engine.js compressedKvBytes 주석). 기대값은 pinned config에서
// 테스트가 *따로* 손계산한다 — 엔진 파서를 기대값 생성에 쓰지 않는다. 네트워크를 쓰지 않는다.
//   deepseek-ai/DeepSeek-V4-Flash         @ 60d8d70770c6776ff598c94bb586a859a38244f1
//   deepseek-ai/DeepSeek-V4-Pro           @ b5968e9190ef611bbf34a7229255be88a0e937c1
//   deepseek-ai/DeepSeek-V4.1-Flash-1017  @ 1bae0e1088993b9b82de08f59de5be5a34e42170
import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import {
  GPUS, LOCAL_MODELS, MACBOOK_RAM_GROUPS, GPU_QUANTS, calcKVCache, calcMaxContext, gpuDevice, naiveKVCache,
  parseHfConfig, resolveLocalModel, simulate, structuralAssumptions, clusterTensorParallelOk, calcParamMemory,
} from '../engine.js';

const BIN = new URL('../bin/fitllm.mjs', import.meta.url).pathname;
const STUB = new URL('./fixtures/hf-pinned-stub.mjs', import.meta.url).href;
const fixture = (name) => readFileSync(new URL(`./fixtures/${name}`, import.meta.url));
const config = (repo) => JSON.parse(fixture(`${repo.replace('/', '__')}.config.json`));
const EVIDENCE = JSON.parse(fixture('hf-api-evidence-2026-10-18.json'));
const gitBlob = (buf) => createHash('sha1').update(`blob ${buf.length}\0`).update(buf).digest('hex');
const byName = (name) => LOCAL_MODELS.find((m) => m.name === name);
const device = (name) => gpuDevice(GPUS.find((g) => g.name === name));
const parse = (repo, raw = config(repo)) => {
  const e = EVIDENCE[repo];
  return parseHfConfig(repo, raw, e.checkpointBytes, { revision: e.sha, safetensorsParameters: e.safetensors.parameters, safetensorsTotal: e.safetensors.total });
};
const ROWS = Object.freeze({
  'deepseek-ai/DeepSeek-V4.1-Flash-1017': 'DeepSeek-V4.1-Flash-1017',
  'deepseek-ai/DeepSeek-V4-Flash': 'DeepSeek-V4-Flash',
  'deepseek-ai/DeepSeek-V4-Pro': 'DeepSeek-V4-Pro',
});

test('fixtures are the pinned revisions: git blob of each config == the blob the HF API reports', () => {
  for (const repo of Object.keys(ROWS)) {
    assert.equal(gitBlob(fixture(`${repo.replace('/', '__')}.config.json`)), EVIDENCE[repo].configBlob, repo);
  }
});

// 테스트 쪽 독립 손계산 — 공식 config의 compress_ratios / kv_source_layer_ids에서 원소 수를 직접 센다.
function handCount(raw) {
  const c = raw.text_config || raw;
  const L = c.num_hidden_layers;
  const ratios = c.compress_ratios.slice(0, L);
  let elems = 0;
  if (c.kv_source_layer_ids) {
    for (const s of c.kv_source_layer_ids) elems += (c.head_dim + c.index_head_dim) / ratios[s];
  } else {
    for (const r of ratios) if (r) elems += c.head_dim / r + (r === 4 ? c.index_head_dim / r : 0);
  }
  return { elems, windowLayers: c.compress_ratios.length, window: c.sliding_window, headDim: c.head_dim };
}

test('cached elements per token match a hand count of the official layout (V4.1 1,600 · V4-Flash 3,440 · V4-Pro 4,924)', () => {
  const expected = { 'deepseek-ai/DeepSeek-V4.1-Flash-1017': 1600, 'deepseek-ai/DeepSeek-V4-Flash': 3440, 'deepseek-ai/DeepSeek-V4-Pro': 4924 };
  for (const [repo, name] of Object.entries(ROWS)) {
    const hand = handCount(config(repo));
    assert.equal(hand.elems, expected[repo], repo);
    const m = byName(name);
    const tokenState = repo.includes('V4.1') ? 8 : 0; // Engram NgramHashState.cache int64 / 토큰
    assert.equal(calcKVCache(m, 1, 16).kvPerToken, hand.elems * 2 + tokenState, `${name} bf16`);
    assert.equal(calcKVCache(m, 1, 8).kvPerToken, hand.elems + tokenState, `${name} 8-bit`);
    // 고정분: 창 128 × 512 × (본체 + MTP/DSpark) + 압축기 fp32 상태 + 인덱서 프리필 작업 공간 512 MiB(vLLM 기본 상한) — ctx와 무관
    const fixed16 = calcKVCache(m, 0, 16).totalBytes;
    assert.equal(m.compressedKv.workspaceBytes, 512 * 1024 * 1024);
    assert.equal(fixed16, hand.window * hand.headDim * hand.windowLayers * 2 + m.compressedKv.stateBytes + 512 * 1024 * 1024, `${name} fixed`);
  }
});

test('V4.1 cross-check: the same 1,600 elements in DeepSeek FP4 formats are the 890 bytes/token the tech report states', () => {
  const m = byName('DeepSeek-V4.1-Flash-1017');
  const main = m.compressedKv.caches.filter((c) => c.dim === 512).reduce((s, c) => s + (c.layers * c.dim) / c.ratio, 0);
  const index = m.compressedKv.caches.filter((c) => c.dim === 128).reduce((s, c) => s + (c.layers * c.dim) / c.ratio, 0);
  assert.equal(main, 1280);
  assert.equal(index, 320);
  // main: E2M1 + E4M3 scale per 16 channels = 4.5 bit · indexer: MXFP4 (E2M1 + E8M0 per 32) = 4.25 bit
  assert.equal(main * (4 + 8 / 16) / 8 + index * (4 + 8 / 32) / 8, 890);
  // naive 2 × layers × heads × head_dim 공식은 40층 전부에 512 캐시를 둔다 → 1M bf16에서 80 GiB vs 3.14 GiB
  assert.equal(naiveKVCache(m, 1048576, 16), 80);
  assert.equal(+calcKVCache(m, 1048576, 16).totalGB.toFixed(3), 3.638); // 3.138 캐시 + 0.5 작업 공간
  // vLLM 블로그(2026-04-24): V4-Pro bf16 KV는 1M에서 시퀀스당 9.62 GiB — 토큰당 원소 수가 같은 값을 낸다
  assert.equal(+((calcKVCache(byName('DeepSeek-V4-Pro'), 1, 16).kvPerToken * 1048576) / 1024 ** 3).toFixed(2), 9.62);
});

test('catalog rows == live parse of the pinned configs + pinned safetensors evidence (no drift between CLI paths)', () => {
  const gpus = GPUS.map((g) => gpuDevice(g));
  const macs = [64, 128, 192, 256, 512];
  for (const [repo, name] of Object.entries(ROWS)) {
    const row = byName(name);
    const parsed = parse(repo);
    for (const key of ['totalParams', 'layerCount', 'kvHeads', 'kvHeadDim', 'attnHeads', 'hiddenSize', 'numExperts',
      'expertsPerToken', 'mtpLayerCount', 'maxContext', 'slidingWindow', 'mlaKvLoraRank', 'linearAttn', 'engramParams']) {
      assert.deepEqual(parsed[key], row[key], `${name} ${key}`);
    }
    assert.deepEqual(parsed.compressedKv, row.compressedKv, `${name} compressedKv`);
    assert.equal(parsed.parameterSource, 'hf-safetensors-parameters');
    for (const d of [...gpus, ...macs]) {
      for (const [ctx, q] of [[8192, GPU_QUANTS[0].bpw], [262144, 8.5008], [1048576, 4]]) {
        const a = simulate(row, d, ctx, { weightBpw: q, kvBits: 16 });
        const b = simulate(parsed, d, ctx, { weightBpw: q, kvBits: 16 });
        assert.equal(a.used, b.used, `${name} ${d.gpu?.name ?? d} ${ctx}`);
        assert.equal(a.verdict, b.verdict);
        assert.equal(a.maxContext, b.maxContext);
      }
    }
  }
});

test('totalParams is the HF logical safetensors sum; FP4 experts reported as I8 equal a hand count of the expert tensors', () => {
  const v41 = EVIDENCE['deepseek-ai/DeepSeek-V4.1-Flash-1017'].safetensors.parameters;
  // (40층 × 384 + DSpark 3층 × 128) 전문가 × gate/up/down 3 × hidden 5120 × moe_intermediate 2304
  assert.equal(v41.I8, (40 * 384 + 3 * 128) * 3 * 5120 * 2304);
  const v4f = EVIDENCE['deepseek-ai/DeepSeek-V4-Flash'].safetensors.parameters;
  assert.equal(v4f.I8, (43 + 1) * 256 * 3 * 4096 * 2048); // 본체 43층 + MTP 1층
  assert.equal(byName('DeepSeek-V4.1-Flash-1017').totalParams, 763.2);
  assert.equal(byName('DeepSeek-V4-Flash').totalParams, 290.9);
  assert.equal(byName('DeepSeek-V4-Pro').totalParams, 1598.8);
});

test('Engram tables are never counted below their FP8 + scale storage (8.25 bit); other weights follow the chosen quant', () => {
  const v41 = byName('DeepSeek-V4.1-Flash-1017');
  const engram = (384006168 + 384016682) * 256; // config engram_num_embeddings × engram_head_dim
  assert.equal(Math.round(v41.engramParams * 1e9), engram);
  const gib = (b) => b / 1024 ** 3;
  assert.equal(calcParamMemory(v41, 4).totalGB, gib((763.2e9 - engram) * 0.5 + engram * 8.25 / 8));
  assert.equal(calcParamMemory(v41, 16).totalGB, gib(763.2e9 * 2)); // 16-bit은 하한보다 높다 — 그대로
  // Mac 512GB 4bit: 하한 적용 시 안 맞음(515.85), Engram도 4bit로 깎으면 맞는다고 잘못 말한다(406.89)
  const mac = simulate(v41, 512, 8192, 4);
  assert.equal(mac.verdict, 'no');
  assert.equal(simulate({ ...v41, engramParams: undefined }, 512, 8192, 4).verdict, 'yes');
  // V4·Muse 등 다른 모델은 하한이 없다
  assert.equal(byName('DeepSeek-V4-Flash').engramParams, undefined);
});

test('4× DGX Spark: V4.1-Flash-1017 does not fit at Q4_K_M; KV is replicated per unit; maxContext agrees with simulate', () => {
  const spark4 = device('4× DGX Spark 128GB');
  const v41 = byName('DeepSeek-V4.1-Flash-1017');
  const s = simulate(v41, spark4, 8192, { weightBpw: 4.8944, kvBits: 16 });
  assert.equal(s.verdict, 'no');
  assert.equal(+s.used.toFixed(2), 629.72);
  // TP 4: 단일 KV 헤드의 창·압축 캐시·인덱서 K를 대마다 둔다
  assert.equal(calcKVCache({ ...v41, compressedKv: { ...v41.compressedKv, replicas: 4 } }, 8192, 16).totalBytes,
    4 * calcKVCache(v41, 8192, 16).totalBytes);
  assert.equal(s.kv, 4 * calcKVCache(v41, 8192, 16).totalGB);
  for (const id of ['deepseek-compressed-sparse-kv', 'mtp-ordinary-generation', 'gb10-unified-memory-reserve', 'gb10-cluster-tensor-parallel']) {
    assert.ok(s.structuralAssumptions.some((a) => a.id === id), id);
  }
  // calcMaxContext는 'yes'(free ≥ headroom)를 지키는 최대 ctx — 그 경계에서 simulate와 맞물린다
  const v4 = byName('DeepSeek-V4-Flash');
  const q = { weightBpw: 8.5008, kvBits: 16 };
  const max = calcMaxContext(v4, spark4, q);
  assert.ok(max > 100000 && max < v4.maxContext, `max ${max}`);
  assert.equal(simulate(v4, spark4, max, q).verdict, 'yes');
  assert.notEqual(simulate(v4, spark4, max + 1, q).verdict, 'yes');
});

test('tensor parallel split must also divide o_groups, indexer heads, experts and vocab (reference code asserts) — otherwise no cluster fit', () => {
  const spark4 = device('4× DGX Spark 128GB');
  const v4 = byName('DeepSeek-V4-Flash');
  assert.equal(clusterTensorParallelOk(v4, spark4), true);
  for (const name of Object.values(ROWS)) assert.equal(clusterTensorParallelOk(byName(name), spark4), true, name);
  for (const patch of [{ oGroups: 6 }, { indexHeads: 30 }, { routedExperts: [255] }, { vocabSize: 129281 }, { routedExperts: [384, 126] }, { routedExperts: [] }]) {
    assert.equal(clusterTensorParallelOk({ ...v4, compressedKv: { ...v4.compressedKv, ...patch } }, spark4), false, JSON.stringify(patch));
  }
  // 파서도 config 값에서 같은 필드를 만든다(Codex 리뷰 재현: n_routed_experts 255 · vocab 129281이 4대에서 yes였다)
  const raw = config('deepseek-ai/DeepSeek-V4-Flash');
  for (const patch of [{ n_routed_experts: 255 }, { vocab_size: 129281 }]) {
    const parsed = parseHfConfig('x/y', { ...raw, ...patch }, EVIDENCE['deepseek-ai/DeepSeek-V4-Flash'].checkpointBytes, {
      revision: EVIDENCE['deepseek-ai/DeepSeek-V4-Flash'].sha, safetensorsParameters: EVIDENCE['deepseek-ai/DeepSeek-V4-Flash'].safetensors.parameters,
    });
    assert.equal(simulate(parsed, spark4, 8192, { weightBpw: 4.8944, kvBits: 16 }).verdict, 'no', JSON.stringify(patch));
  }
  const odd = { ...v4, compressedKv: { ...v4.compressedKv, oGroups: 6 } };
  assert.equal(clusterTensorParallelOk(odd, spark4), false);
  const s = simulate(odd, spark4, 8192, { weightBpw: 4.8944, kvBits: 16 });
  assert.equal(s.verdict, 'no');
  assert.equal(calcMaxContext(odd, spark4, { weightBpw: 4.8944, kvBits: 16 }), 0);
  assert.ok(structuralAssumptions(odd, spark4).some((a) => a.id === 'gb10-cluster-tensor-parallel-incompatible'));
});

test('fail closed: configs that leave the official layout are not computed', () => {
  const v4 = config('deepseek-ai/DeepSeek-V4-Flash');
  const v41 = config('deepseek-ai/DeepSeek-V4.1-Flash-1017');
  const t41 = (patch) => ({ ...v41, text_config: { ...v41.text_config, ...patch } });
  const v41Ratios = v41.text_config.compress_ratios;
  const cases = [
    ['V4 unverified ratio', 'deepseek-ai/DeepSeek-V4-Flash', { ...v4, compress_ratios: v4.compress_ratios.map((r) => (r === 128 ? 64 : r)) }, /검증된 값/],
    ['V4 ratios length', 'deepseek-ai/DeepSeek-V4-Flash', { ...v4, compress_ratios: v4.compress_ratios.slice(1) }, /길이/],
    ['V4 two KV heads', 'deepseek-ai/DeepSeek-V4-Flash', { ...v4, num_key_value_heads: 2 }, /num_key_value_heads/],
    ['V4 MLA rank', 'deepseek-ai/DeepSeek-V4-Flash', { ...v4, kv_lora_rank: 512 }, /kv_lora_rank/],
    ['V4 layer_types', 'deepseek-ai/DeepSeek-V4-Flash', { ...v4, layer_types: Array(43).fill('full_attention') }, /layer_types/],
    ['V4 MTP compresses', 'deepseek-ai/DeepSeek-V4-Flash', { ...v4, compress_ratios: [...v4.compress_ratios.slice(0, 43), 4] }, /MTP/],
    ['V4.1 source not indexing', 'deepseek-ai/DeepSeek-V4.1-Flash-1017', t41({ index_source_layer_ids: [8, 14, 20, 24, 28, 32, 36] }), /인덱스 소스/],
    ['V4.1 reuse across ratios', 'deepseek-ai/DeepSeek-V4.1-Flash-1017', t41({ kv_source_layer_ids: [2, 8, 14] }), /같은 비율/],
    ['V4.1 unsorted sources', 'deepseek-ai/DeepSeek-V4.1-Flash-1017', t41({ kv_source_layer_ids: [8, 2, 14, 20] }), /오름차순/],
    ['V4.1 ratio 4 (V4 value)', 'deepseek-ai/DeepSeek-V4.1-Flash-1017', t41({ compress_ratios: v41Ratios.map((r) => (r === 2 ? 4 : r)) }), /검증된 값/],
    ['V4.1 DSpark compresses', 'deepseek-ai/DeepSeek-V4.1-Flash-1017', t41({ compress_ratios: [...v41Ratios.slice(0, 42), 1] }), /MTP/],
    ['V4.1 candidate not an index layer', 'deepseek-ai/DeepSeek-V4.1-Flash-1017', t41({ candidate_source_layer_id: 21 }), /candidate/],
    // 후보 마스크(ratio 2 폭)를 ratio 1 인덱서가 쓰게 되는 배치 — Codex 리뷰 재현
    ['V4.1 candidate across ratios', 'deepseek-ai/DeepSeek-V4.1-Flash-1017', t41({ candidate_source_layer_id: 2 }), /후보 마스크/],
    ['V4.1 engram rows mismatch', 'deepseek-ai/DeepSeek-V4.1-Flash-1017', t41({ engram_num_embeddings: [384006168] }), /engram_num_embeddings/],
    ['V4.1 DSpark experts', 'deepseek-ai/DeepSeek-V4.1-Flash-1017', t41({ dspark_n_routed_experts: -1 }), /dspark_n_routed_experts/],
  ];
  for (const [label, repo, raw, re] of cases) assert.throws(() => parse(repo, raw), re, label);
  // 바이트만 있고 safetensors 증거가 없으면 FP4 전문가를 절반으로 세게 된다 — 계산하지 않는다
  assert.throws(() => parseHfConfig('deepseek-ai/DeepSeek-V4.1-Flash-1017', v41, EVIDENCE['deepseek-ai/DeepSeek-V4.1-Flash-1017'].checkpointBytes),
    /safetensors 파라미터 증거/);
});

test('other deepseek_v4-typed repos keep failing closed (Vision-Exp extra keys, transformers-format Nex-N2.5-Max, GLM DSA)', () => {
  const v4 = config('deepseek-ai/DeepSeek-V4-Flash');
  const e = EVIDENCE['deepseek-ai/DeepSeek-V4-Flash'];
  const ev = { revision: e.sha, safetensorsParameters: e.safetensors.parameters, safetensorsTotal: e.safetensors.total };
  // deepseek-ai/DeepSeek-V4-Flash-Vision-Exp(#101): 같은 model_type에 DSpark·vision 키 — 참조 V4 코드가 읽지 않는 구조
  assert.throws(() => parseHfConfig('x/vision-exp', { ...v4, dspark_block_size: 5, dspark_target_layer_ids: [40, 41, 42], vision_n_layers: 32, vision_n_heads: 16 }, e.checkpointBytes, ev),
    /구조를 알 수 없는 config 필드/);
  // nex-agi/Nex-N2.5-Max: transformers DeepseekV4Config 형식(compress_rates + layer_types)
  assert.throws(() => parseHfConfig('x/nex-max', { ...v4, compress_rates: { compressed_sparse_attention: 4 }, layer_types: Array(43).fill('compressed_sparse_attention') }, e.checkpointBytes, ev));
  // 같은 키 이름(index_topk·compress_ratios)이라도 다른 계열이면 그대로 거부
  assert.throws(() => parseHfConfig('x/glm', { ...v4, model_type: 'glm_moe_dsa' }, e.checkpointBytes, ev), /비표준 압축·희소·혼합 어텐션/);
});

test('names: spaced and glued spellings resolve; family-only queries stay ambiguous; HF IDs keep the live HF path', () => {
  for (const [q, name] of [
    ['DeepSeek-V4.1-Flash-1017', 'DeepSeek-V4.1-Flash-1017'], ['deepseek v4.1 flash 1017', 'DeepSeek-V4.1-Flash-1017'],
    ['DeepSeek V4.1 Flash', 'DeepSeek-V4.1-Flash-1017'], // 같은 config·같은 safetensors(V4.1-Flash == 1017 형상)
    ['DeepSeek V4 Flash', 'DeepSeek-V4-Flash'], ['deepseek-v4-pro', 'DeepSeek-V4-Pro'],
  ]) assert.equal(resolveLocalModel(q).canonicalName, name, q);
  for (const q of ['deepseek', 'DeepSeek V4']) assert.equal(resolveLocalModel(q).status, 'ambiguous', q);
  for (const repo of [...Object.keys(ROWS), 'deepseek-ai/DeepSeek-V4.1-Flash']) assert.equal(resolveLocalModel(repo).status, 'unknown', repo);
});

test('CLI: the reported command now answers — HF ID × 4× DGX Spark via the pinned HF path, exit 1 with the same numbers as the catalog', () => {
  const runStub = (...args) => spawnSync(process.execPath, ['--import', STUB, BIN, ...args], { encoding: 'utf8' });
  const hf = runStub('deepseek-ai/DeepSeek-V4.1-Flash-1017', '--gpu', 'DGX Spark', '--count', '4', '--json', '--why');
  assert.equal(hf.status, 1, hf.stderr);
  const j = JSON.parse(hf.stdout);
  assert.equal(j.verdict, 'no');
  assert.equal(j.modelSource.type, 'huggingface');
  assert.equal(j.modelSource.revision, '1bae0e1088993b9b82de08f59de5be5a34e42170');
  assert.equal(j.usedGB, 629.72);
  assert.equal(j.basis.attention.kind, 'compressed-sparse');
  assert.equal(j.basis.attention.kvLayers, 4);
  assert.equal(j.basis.attention.compressedKv.cachedElementsPerToken, 1600);
  const cat = spawnSync(process.execPath, [BIN, 'DeepSeek-V4.1-Flash-1017', '--gpu', 'DGX Spark', '--count', '4', '--json'], { encoding: 'utf8' });
  assert.equal(cat.status, 1);
  const c = JSON.parse(cat.stdout);
  for (const k of ['verdict', 'usedGB', 'freeGB', 'maxContext', 'breakdown']) assert.deepEqual(c[k], j[k], k);
  // 텍스트 출력은 전제와 압축 KV 구성을 말한다
  const text = runStub('deepseek-ai/DeepSeek-V4.1-Flash-1017', '--gpu', 'DGX Spark', '--count', '4', '--why');
  assert.equal(text.status, 1);
  assert.match(text.stdout, /WON'T FIT — DeepSeek-V4\.1-Flash-1017 on 4× DGX Spark 128GB/);
  assert.match(text.stdout, /premise \[deepseek-compressed-sparse-kv\]/);
  assert.match(text.stdout, /compressed KV: 1600 cached elements\/token \(1×512\/1 \+ 1×128\/1 \+ 3×512\/2 \+ 3×128\/2\) \+ 128-token window × 43 layers \+ 8 B\/token Engram history/);
  // V4-Flash는 2×에서 빠듯, 4×에서 Q8_0까지 들어간다
  const flash = spawnSync(process.execPath, [BIN, 'DeepSeek-V4-Flash', '--gpu', 'DGX Spark', '--count', '4', '--quant', 'Q8_0', '--json'], { encoding: 'utf8' });
  assert.equal(flash.status, 0, flash.stdout);
  assert.equal(JSON.parse(flash.stdout).verdict, 'yes');
});

test('census rows exist for the three DeepSeek models with the whole-checkpoint parameter count', () => {
  const census = JSON.parse(readFileSync(new URL('../census/census-v1.json', import.meta.url), 'utf8'));
  const macConfigs = Object.values(MACBOOK_RAM_GROUPS).reduce((n, r) => n + r.length, 0);
  for (const name of Object.values(ROWS)) {
    const rows = census.data.filter((r) => r.model === name);
    assert.equal(rows.length, GPUS.length * GPU_QUANTS.length + macConfigs * 3, name);
    assert.ok(rows.every((r) => r.params_b === byName(name).totalParams && r.linear_state_gb === 0), name);
  }
  assert.ok(readFileSync(new URL('../census/README.md', import.meta.url), 'utf8').includes('deepseek-compressed-sparse-kv'));
  assert.ok(readFileSync(new URL('../README.md', import.meta.url), 'utf8').includes('`deepseek-compressed-sparse-kv`'));
});
