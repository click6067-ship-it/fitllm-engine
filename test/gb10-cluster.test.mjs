// GB10 클러스터(DGX Spark 2대·4대) — 한 모델이 여러 대의 메모리를 같이 쓰는 경우.
// ① 같은 GB10 행 2·4대만(NVIDIA: 2대 직결, 4대 스위치) — 3대(링)·5대 이상·다른 카드와 섞기는 fail closed
// ② 대당 reserve·활성화 버퍼는 대수만큼, KV는 tensor parallel 복제(KV 헤드 < 대수, MLA)까지 센다
// ③ 합산이 성립하는 조건은 전제 'gb10-cluster-tensor-parallel'로 판정과 함께 나간다
import test from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import {
  GPUS, LOCAL_MODELS, gpuDevice, combineGpus, simulate, simulateStack, calcMaxContext, calcKVCache,
  clusterKvModel, clusterTensorParallelOk, resolveGpuByName, structuralAssumptions,
} from '../engine.js';

const CLI = new URL('../bin/fitllm.mjs', import.meta.url).pathname;
const run = (...args) => spawnSync(process.execPath, [CLI, ...args], { encoding: 'utf8' });
const spark = GPUS.find((g) => g.name === 'DGX Spark 128GB');
const two = GPUS.find((g) => g.name === '2× DGX Spark 128GB');
const four = GPUS.find((g) => g.name === '4× DGX Spark 128GB');
const rtx = GPUS.find((g) => g.name === 'RTX 4090');
const model = (name) => LOCAL_MODELS.find((m) => m.name === name);
const host = (u) => new URL(u).hostname.replace(/^www\./, '');
const PER_UNIT_RESERVE = (128 - 121.69) + 5.0 + 2.0;

test('클러스터 행: 2× 256GB · 4× 512GB, 대당 reserve × 대수, 연결 출처 ≥2 독립 도메인', () => {
  for (const [row, n] of [[two, 2], [four, 4]]) {
    assert.ok(row, `${n}x row missing`);
    assert.equal(row.vramGB, 128 * n);
    assert.equal(row.count, n);
    assert.equal(row.perCardVramGB, 128);
    assert.equal(row.clusterOf, 'DGX Spark 128GB');
    assert.equal(row.unifiedMemory, spark.unifiedMemory); // 같은 profile을 공유한다
    assert.ok(new Set(row.sources.interconnect.map(host)).size >= 2);
    const d = gpuDevice(row);
    assert.equal(d.memoryGB, 128 * n);
    assert.equal(d.clusterUnits, n);
    assert.equal(d.gpuCount, n);
    assert.ok(Math.abs(d.reserveGB - PER_UNIT_RESERVE * n) < 1e-9);
    assert.equal(d.headroomRatio, gpuDevice(spark).headroomRatio);
  }
  assert.equal(gpuDevice(spark).clusterUnits, undefined); // 한 대는 종전 그대로
});

test('같은 GB10 2·4대만 묶인다 — 3대·5대 이상·섞기·클러스터 행끼리는 fail closed', () => {
  assert.deepEqual(combineGpus([spark, spark]), gpuDevice(two));
  assert.deepEqual(combineGpus([spark, spark, spark, spark]), gpuDevice(four));
  for (const list of [[spark, spark, spark], Array(5).fill(spark), [spark, rtx], [rtx, spark], [two, spark], [two, two], [four, rtx]]) {
    assert.throws(() => combineGpus(list), /unified-memory system/, list.map((g) => g.name).join(' + '));
  }
});

test('KV 복제: 층 종류별로 KV 헤드가 대수보다 적으면 복제, MLA는 순위마다 통째로', () => {
  const d2 = gpuDevice(two);
  const d4 = gpuDevice(four);
  const kv = (name, d, ctx = 8192) => calcKVCache(clusterKvModel(model(name), d), ctx, 16).totalGB;
  const base = (name, ctx = 8192) => calcKVCache(model(name), ctx, 16).totalGB;
  assert.equal(kv('gpt-oss-120b', d4), base('gpt-oss-120b')); // 8 KV heads ≥ 4
  assert.equal(kv('Qwen 3.6 35B-A3B', d2), base('Qwen 3.6 35B-A3B')); // 2 heads, 2 units
  assert.equal(kv('Qwen 3.6 35B-A3B', d4), base('Qwen 3.6 35B-A3B') * 2); // 2 heads, 4 units → ×2
  assert.equal(kv('GLM-4.7-Flash', d4), base('GLM-4.7-Flash') * 4); // MLA
  // Gemma 4 12b: 로컬 8헤드는 나뉘고 전역 1헤드만 ×4 — 한 배율(×4)로 묶으면 1.75 GiB, 층별이면 0.8125 GiB (Codex 리뷰 재현값)
  assert.equal(kv('Gemma 4 12b', d4), 0.8125);
  assert.equal(clusterKvModel(model('Qwen 3.6 35B-A3B'), gpuDevice(spark)), model('Qwen 3.6 35B-A3B')); // 한 대·카드는 원래 모델
  assert.equal(clusterKvModel(model('Qwen 3.6 35B-A3B'), gpuDevice(rtx)), model('Qwen 3.6 35B-A3B'));
});

test('tensor parallel로 나눌 수 없는 모델은 클러스터에서 fits가 나오지 않는다 (Codex 리뷰 P1)', () => {
  // Qwen2.5-0.5B 형상: 어텐션 14헤드 — 4대로 나눠지지 않아 vLLM/TRT-LLM이 거부한다
  const odd = { ...model('Llama-3.2-1B-Instruct'), name: 'odd-heads', attnHeads: 14, kvHeads: 2 };
  const d4 = gpuDevice(four);
  assert.equal(clusterTensorParallelOk(odd, d4), false);
  assert.equal(clusterTensorParallelOk(odd, gpuDevice(two)), true); // 14 % 2 = 0
  const s = simulate(odd, d4, 8192, { weightBpw: 16.0005, kvBits: 16 });
  assert.equal(s.verdict, 'no');
  assert.equal(s.maxContext, 0);
  assert.ok(s.structuralAssumptions.some((a) => a.id === 'gb10-cluster-tensor-parallel-incompatible'));
  assert.equal(simulateStack([{ model: odd, ctx: 8192, weightBpw: 16.0005, kvBits: 16 }], d4).verdict, 'no');
  // 헤드 수를 모르면 성립한다고 하지 않는다
  assert.equal(clusterTensorParallelOk({ ...odd, attnHeads: undefined }, d4), false);
  // 카탈로그 전 모델은 2·4대 TP가 성립한다(census에 'incompatible' 행이 없다)
  for (const m of LOCAL_MODELS) for (const d of [gpuDevice(two), d4]) assert.ok(clusterTensorParallelOk(m, d), `${m.name} ×${d.clusterUnits}`);
  // 한 대·디스크리트 카드는 검사하지 않는다
  assert.equal(clusterTensorParallelOk(odd, gpuDevice(spark)), true);
});

test('simulate: used = weights + KV×복제 + 선형상태 + 동적(활성화×대수) + reserve×대수', () => {
  const m = model('Qwen 3.6 35B-A3B');
  const d = gpuDevice(four);
  const ctx = 131072;
  const s = simulate(m, d, ctx, { weightBpw: 8.5008, kvBits: 16 });
  const kv = calcKVCache(m, ctx, 16).totalGB * 2; // 2 KV heads, 4 units
  assert.ok(Math.abs(s.kv - kv) < 1e-12);
  const rtDyn = s.param * 0.12 + kv * 0.15 + ctx * 0.00003 * 4;
  assert.ok(Math.abs(s.rtDyn - rtDyn) < 1e-9);
  assert.ok(Math.abs(s.used - (s.param + kv + s.linearState + rtDyn + PER_UNIT_RESERVE * 4)) < 1e-9);
  // 클러스터는 가중치 12% 몫을 유지하고(위 rtDyn), 한 대는 2.21.4부터 0이다(GGUF 파일 크기 그대로 — 공개 GB10 실측):
  //   가중치 = 출시 MXFP4 전문가 114.6617856e9 × 4.25/8 + 나머지 2.1382144e9 × 4.8944/8 = 57.95 GiB(unsloth Q4_K_M 실파일 58.46)
  //   + KV 0.29 + 0.15×KV 0.04 + 8192×0.00003 0.25 + reserve 13.31 = 71.83 (2.21.4: 단계 평균 bit 66.55 GiB로 80.43 · 2.19.0–2.21.3: 88.42)
  const one = simulate(model('gpt-oss-120b'), gpuDevice(spark), 8192, { weightBpw: 4.8944, kvBits: 16 });
  assert.equal(+one.used.toFixed(2), 71.83);
});

test('calcMaxContext는 같은 복제·활성화 규칙을 쓴다 (경계에서 simulate와 일치)', () => {
  const m = model('Qwen 3.6 35B-A3B');
  const d = gpuDevice(four);
  const q = { weightBpw: 8.5008, kvBits: 16 };
  const max = calcMaxContext(m, d, q);
  assert.equal(max, m.maxContext); // 512GB면 최대 컨텍스트까지
  const g = model('Gemma 4 31b');
  const d2 = gpuDevice(two);
  const maxG = calcMaxContext(g, d2, { weightBpw: 16.0005, kvBits: 16 });
  const at = simulate(g, d2, maxG, { weightBpw: 16.0005, kvBits: 16 });
  assert.ok(at.free >= at.headroom - 1e-6, 'max context stays inside the comfortable budget');
});

test('전제: 클러스터는 통합 메모리 전제 + 클러스터 전제를 함께 단다', () => {
  const ids = structuralAssumptions(model('Llama-3.1-8B-Instruct'), gpuDevice(two)).map((a) => a.id);
  assert.deepEqual(ids, ['gb10-unified-memory-reserve', 'gb10-cluster-tensor-parallel']);
  const [, cluster] = structuralAssumptions(model('Llama-3.1-8B-Instruct'), gpuDevice(four));
  for (const phrase of ['vLLM with Ray or TensorRT-LLM', 'tensor parallel equal to the unit count', 'two units by a direct QSFP cable and four through a RoCE switch', 'Ollama and LM Studio use one unit', 'replicated']) {
    assert.ok(cluster.statement.includes(phrase), phrase);
  }
  assert.deepEqual(structuralAssumptions(model('Llama-3.1-8B-Instruct'), gpuDevice(spark)).map((a) => a.id), ['gb10-unified-memory-reserve']);
});

test('simulateStack도 클러스터 규칙을 쓴다', () => {
  const st = simulateStack([{ model: model('Qwen 3.6 35B-A3B'), ctx: 131072, weightBpw: 8.5008, kvBits: 16 }], gpuDevice(four));
  const s = simulate(model('Qwen 3.6 35B-A3B'), gpuDevice(four), 131072, { weightBpw: 8.5008, kvBits: 16 });
  assert.ok(Math.abs(st.parts[0].kv - s.kv) < 1e-12); // 복제된 KV
  assert.ok(Math.abs(st.parts[0].rtDyn - s.rtDyn) < 1e-12); // 대수만큼의 활성화 버퍼
  assert.equal(st.reserve, s.reserve);
});

test('이름: 2대·4대 행과 별칭, 한 대 질의는 여전히 한 대', () => {
  assert.equal(resolveGpuByName('dgx spark').canonicalName, 'DGX Spark 128GB');
  assert.equal(resolveGpuByName('2x DGX Spark 128GB').canonicalName, '2× DGX Spark 128GB');
  assert.equal(resolveGpuByName('DGX Spark 128GB x4').canonicalName, '4× DGX Spark 128GB');
  assert.equal(resolveGpuByName('4 dgx spark').canonicalName, '4× DGX Spark 128GB');
  assert.notEqual(resolveGpuByName('2 dgx').status, 'resolved');
});

test('CLI: --count 2·4와 2× 행은 같은 판정, 3대·섞기·클러스터 행 --count는 exit 2', () => {
  const a = run('gpt-oss-120b', '--gpu', 'DGX Spark 128GB', '--count', '2', '--quant', 'Q8_0', '--json');
  const b = run('gpt-oss-120b', '--gpu', '2× DGX Spark 128GB', '--quant', 'Q8_0', '--json');
  assert.equal(a.status, 0, a.stderr);
  const ja = JSON.parse(a.stdout);
  assert.deepEqual(JSON.parse(b.stdout), ja);
  assert.equal(ja.hardware, '2× DGX Spark 128GB (256GB unified memory)');
  assert.equal(ja.verdict, 'yes');
  // gpt-oss는 출시 FP4 전문가 전제(native-fp4-expert-weights)가 함께 붙는다
  assert.deepEqual(ja.structuralAssumptions.map((x) => x.id), ['native-fp4-expert-weights', 'gb10-unified-memory-reserve', 'gb10-cluster-tensor-parallel']);
  const c = run('gpt-oss-120b', '--gpu', 'DGX Spark 128GB', '--count', '4');
  assert.equal(c.status, 0, c.stderr);
  assert.match(c.stdout, /4× DGX Spark 128GB \(512GB unified memory\)/);
  for (const args of [['--gpu', 'DGX Spark 128GB', '--count', '3'], ['--gpu', 'DGX Spark 128GB + RTX 4090'], ['--gpu', '2× DGX Spark 128GB', '--count', '2']]) {
    const r = run('gpt-oss-20b', ...args);
    assert.equal(r.status, 2, args.join(' '));
    assert.match(r.stderr, /unified-memory system/);
  }
  const why = run('gpt-oss-120b', '--gpu', '4× DGX Spark 128GB', '--why');
  assert.match(why.stdout, /unified-memory reserve \(4 units\): 25\.2 not visible to the OS \+ 20\.0 idle OS \+ 8\.0 runtime = 53\.2 GiB/);
});

test('measure --from-ollama: GB10 클러스터 지정은 거부된다 (Ollama는 한 대)', async () => {
  const { deviceFromFlags } = await import('../bin/measure-from-runtime.mjs');
  assert.throws(() => deviceFromFlags({ gpu: 'DGX Spark 128GB', count: '4' }), /Ollama runs on one unit/);
  assert.throws(() => deviceFromFlags({ gpu: '2× DGX Spark 128GB' }), /Ollama runs on one unit/);
  assert.equal(deviceFromFlags({ gpu: 'DGX Spark 128GB' }).clusterUnits, undefined);
  assert.equal(deviceFromFlags({ gpu: 'RTX 4090', count: '2' }).gpuCount, 2); // 디스크리트는 종전대로
});
