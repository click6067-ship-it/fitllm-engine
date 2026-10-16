// GB10 통합 메모리 시스템(DGX Spark 계열) — 디스크리트 GPU와 다른 점만 고정한다.
// ① 128GB 한 풀을 DGX OS·CPU·GPU가 같이 쓴다 → env 프리셋 대신 근거 붙은 고정 reserve
// ② host 버퍼도 같은 풀 → Gemma 4 PLE 차감 없음
// ③ 완제품 컴퓨터 → 다른 카드와 풀링·여러 대 연결은 모델링하지 않음(fail closed)
// ④ 같은 칩·같은 메모리 파트너 제품은 별칭으로 같은 행에 해석
import test from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import {
  GPUS, LOCAL_MODELS, ENV_PRESETS, HEADROOM_RATIO, gpuDevice, combineGpus, simulate, calcMaxContext,
  suggestFixGpu, resolveGpuByName, structuralAssumptions, unifiedReserveGB,
} from '../engine.js';

const CLI = new URL('../bin/fitllm.mjs', import.meta.url).pathname;
const run = (...args) => spawnSync(process.execPath, [CLI, ...args], { encoding: 'utf8' });
const spark = GPUS.find((g) => g.name === 'DGX Spark 128GB');
const model = (name) => LOCAL_MODELS.find((m) => m.name === name);
const host = (u) => new URL(u).hostname.replace(/^www\./, '');

test('DGX Spark 행: 128GB 통합 메모리 · 출처 ≥2 독립 도메인 · reserve 측정 항목마다 출처', () => {
  assert.ok(spark, 'DGX Spark 128GB row missing');
  assert.equal(spark.vramGB, 128);
  assert.equal(spark.series, 'unified');
  for (const field of ['vramGB', 'bandwidthGBs']) {
    assert.ok(new Set(spark.sources[field].map(host)).size >= 2, `${field} needs 2 independent domains`);
  }
  const u = spark.unifiedMemory;
  assert.equal(u.osVisibleGB, 121.69); // DGX OS 7.6 MemTotal = CUDA total
  assert.equal(u.osIdleGB, 5.0);
  assert.equal(u.runtimeFixedGB, 2.0);
  for (const part of ['osVisibleGB', 'osIdleGB']) {
    assert.ok(new Set((u.sources[part] || []).map(host)).size >= 2, `${part} needs 2 independent domains`);
  }
  // 측정이 아닌 항목은 추정이라고 명시돼 있어야 한다
  assert.match(u.estimates.runtimeFixedGB, /not a GB10 measurement/);
});

test('reserve = (128 − OS 가시 총량) + OS 유휴 + 런타임 · env 프리셋 무시 · headroom은 맥과 같은 공유 풀 비율', () => {
  const expected = (128 - 121.69) + 5.0 + 2.0;
  for (const env of [undefined, ...Object.keys(ENV_PRESETS)]) {
    const d = gpuDevice(spark, env);
    assert.equal(d.type, 'gpu');
    assert.equal(d.unifiedMemory, true);
    assert.equal(d.memoryGB, 128);
    assert.equal(d.gpuCount, 1);
    assert.ok(Math.abs(d.reserveGB - expected) < 1e-9, `env=${env}`);
    assert.ok(Math.abs(unifiedReserveGB(spark) - 13.31) < 1e-9);
    assert.equal(d.headroomRatio, HEADROOM_RATIO);
    assert.equal(d.env, 'dgx-os-unified');
  }
  // combineGpus([spark])는 단일 장치와 같다
  assert.deepEqual(combineGpus([spark], 'linux-headless'), gpuDevice(spark));
});

test('simulate: used = weights + KV + 동적 오버헤드 + reserve (고정 2GB 이중계상 없음)', () => {
  const m = model('gpt-oss-120b');
  const d = gpuDevice(spark);
  const s = simulate(m, d, 8192, { weightBpw: 4.8944, kvBits: 16 });
  assert.ok(Math.abs(s.used - (s.param + s.kv + s.linearState + s.rtDyn + d.reserveGB)) < 1e-9);
  assert.equal(s.reserve, d.reserveGB);
  assert.equal(s.rt, s.rtDyn); // GPU 경로: 고정 런타임 2GB는 reserve에 이미 포함
  assert.equal(s.memoryGB, 128);
});

test('Gemma 4 PLE는 통합 메모리에서 차감하지 않는다 (host 버퍼도 같은 풀)', () => {
  for (const name of ['Gemma 4 e2b', 'Gemma 4 e4b']) {
    const m = model(name);
    const onSpark = simulate(m, gpuDevice(spark), 8192, { weightBpw: 16, kvBits: 16 });
    const asPlain = simulate({ ...m, pleOffloadVerified: false }, gpuDevice(spark), 8192, { weightBpw: 16, kvBits: 16 });
    assert.equal(onSpark.pleOffloadGB, 0, name);
    assert.equal(onSpark.param, asPlain.param, name);
    const ids = (onSpark.structuralAssumptions || []).map((a) => a.id);
    assert.equal(ids.includes('ple-llamacpp-non-gpu-residency'), false, name);
    // 디스크리트 카드에서는 기존대로 차감된다(회귀 방지)
    const onCard = simulate(m, gpuDevice(GPUS.find((g) => g.name === 'RTX 4090')), 8192, { weightBpw: 16, kvBits: 16 });
    assert.ok(onCard.pleOffloadGB > 0, name);
  }
});

test('통합 메모리 전제가 판정과 함께 나온다', () => {
  const ids = structuralAssumptions(model('Llama-3.1-8B-Instruct'), gpuDevice(spark)).map((a) => a.id);
  assert.deepEqual(ids, ['gb10-unified-memory-reserve']);
  const onCard = structuralAssumptions(model('Llama-3.1-8B-Instruct'), gpuDevice(GPUS.find((g) => g.name === 'RTX 4090')));
  assert.deepEqual(onCard, []);
});

test('다른 카드와 풀링·여러 대 연결은 fail closed', () => {
  const rtx = GPUS.find((g) => g.name === 'RTX 4090');
  for (const list of [[spark, spark], [spark, rtx], [rtx, spark]]) {
    assert.throws(() => combineGpus(list), /unified-memory system/);
  }
});

test('더 큰 카드 제안에 통합 메모리 완제품은 끼지 않는다', () => {
  const m = model('gpt-oss-120b');
  const fix = suggestFixGpu(m, gpuDevice(GPUS.find((g) => g.name === 'RTX 5090')), 131072, { weightBpw: 16.0005, kvBits: 16 }, (ko, en) => en);
  assert.notEqual(fix.gpu, 'DGX Spark 128GB');
});

test('같은 GB10·같은 128GB 파트너 제품은 별칭(이름 전체 일치)으로 DGX Spark 128GB 행에 해석된다', () => {
  assert.equal(spark.aliases.length, 15);
  for (const q of [...spark.aliases, 'dgx spark', 'DGX Spark', 'spark', 'nvidia dgx spark 128gb', 'asus-ascent-gx10-128gb', 'Dell Pro Max GB10 128GB']) {
    const r = resolveGpuByName(q);
    assert.equal(r.status, 'resolved', q);
    assert.equal(r.canonicalName, 'DGX Spark 128GB', q);
  }
  // 별칭은 전부 용량을 품는다 — 용량 없이 적힌 파트너 이름(64GB 구성이 2026-10-23 판매)은 고르지 않는다
  for (const alias of spark.aliases) assert.match(alias, /\b128GB\b/, alias);
  for (const q of ['DGX Spark 64GB', 'ASUS Ascent GX10 64GB', 'ASUS Ascent GX10', 'gx10', 'gb10', 'ThinkStation PGX']) {
    const r = resolveGpuByName(q);
    assert.notEqual(r.status, 'resolved', q);
  }
});

test('제조사·용도 토큰 하나로는 어떤 장치로도 확정되지 않는다 (별칭 부분 일치 금지 — Codex P1 회귀)', () => {
  for (const q of ['NVIDIA', 'ASUS', 'Dell', 'HP', 'Lenovo', 'MSI', 'Acer', 'GIGABYTE', 'nano', 'workstation', 'edition', 'server', 'max-q', 'sff']) {
    assert.notEqual(resolveGpuByName(q).status, 'resolved', q);
  }
  const r = run('gpt-oss-120b', '--gpu', 'NVIDIA', '--json');
  assert.equal(r.status, 2, 'unknown hardware must not pass the download guard');
  assert.match(r.stderr, /unknown GPU: "NVIDIA" — did you mean: DGX Spark 128GB\?/);
});

test('별칭은 카탈로그 전체에서 유일하다 (두 행이 같은 이름을 갖지 않음)', () => {
  const seen = new Map();
  const key = (s) => s.normalize('NFKC').toLowerCase().split(/[^a-z0-9]+/).filter(Boolean).join(' ');
  for (const g of GPUS) {
    for (const n of [g.name, ...(g.aliases || [])]) {
      const k = key(n);
      assert.equal(seen.has(k), false, `${n} (${g.name}) collides with ${seen.get(k)}`);
      seen.set(k, g.name);
    }
  }
});

test('CLI: --list에 DGX Spark와 별칭이 보이고, --gpu 별칭으로 같은 판정이 나온다', () => {
  const list = run('--list');
  assert.equal(list.status, 0);
  assert.match(list.stdout, /DGX Spark 128GB {2}\(128GB unified memory\)/);
  for (const alias of spark.aliases) assert.ok(list.stdout.includes(alias), alias);
  const a = run('gpt-oss-120b', '--gpu', 'DGX Spark 128GB', '--json');
  const b = run('gpt-oss-120b', '--gpu', spark.aliases[0], '--json');
  assert.equal(a.status, 0, a.stderr);
  assert.equal(b.status, a.status);
  const ja = JSON.parse(a.stdout);
  const jb = JSON.parse(b.stdout);
  assert.deepEqual(jb, ja);
  assert.equal(ja.hardware, 'DGX Spark 128GB (128GB unified memory)');
  assert.deepEqual(ja.structuralAssumptions.map((x) => x.id), ['gb10-unified-memory-reserve']);
});

test('CLI: 풀링·--count 2는 exit 2로 거절된다', () => {
  for (const args of [['--gpu', 'DGX Spark 128GB', '--count', '2'], ['--gpu', 'DGX Spark + RTX 4090']]) {
    const r = run('gpt-oss-20b', ...args);
    assert.equal(r.status, 2, args.join(' '));
    assert.match(r.stderr, /unified-memory system/);
  }
});

test('maxContext: 통합 메모리 예산식은 디스크리트와 같은 식에 reserve·headroom만 다르다', () => {
  const m = model('Llama-3.1-8B-Instruct');
  const d = gpuDevice(spark);
  const max = calcMaxContext(m, d, { weightBpw: 4.8944, kvBits: 16 });
  assert.equal(max, m.maxContext); // 8B Q4는 128GB에서 최대 컨텍스트까지 들어간다
});

test('CLI --why: reserve 구성과 근거가 판정과 같은 값으로 나온다', () => {
  const r = run('gpt-oss-120b', '--gpu', 'DGX Spark 128GB', '--json', '--why');
  assert.equal(r.status, 0, r.stderr);
  const { basis, breakdown } = JSON.parse(r.stdout);
  const um = basis.hardware.unifiedMemory;
  assert.equal(basis.hardware.environment, 'dgx-os-unified');
  assert.ok(Math.abs(um.notVisibleGiB - 6.31) < 1e-9);
  assert.equal(um.osIdleGiB, 5.0);
  assert.equal(um.runtimeFixedGiB, 2.0);
  assert.ok(Math.abs(um.reserveGiB - 13.31) < 1e-9);
  assert.equal(breakdown.reserveGB, 13.31);
  assert.ok(um.sources.osVisible.length >= 2 && um.sources.osIdle.length >= 2);
  const text = run('gpt-oss-120b', '--gpu', 'DGX Spark 128GB', '--why');
  assert.match(text.stdout, /unified-memory reserve: 6\.3 not visible to the OS \+ 5\.0 idle OS \+ 2\.0 runtime = 13\.3 GiB/);
});

test('census: GB10 행은 PLE를 차감하지 않고, README 공개 문구도 그 범위를 밝힌다', async () => {
  const { readFileSync } = await import('node:fs');
  const census = JSON.parse(readFileSync(new URL('../census/census-v1.json', import.meta.url), 'utf8'));
  const readme = readFileSync(new URL('../census/README.md', import.meta.url), 'utf8');
  const e2b = model('Gemma 4 e2b');
  const row = census.data.find((r) => r.model === 'Gemma 4 e2b' && r.device === 'DGX Spark 128GB' && r.quant === 'Q4_K_M');
  assert.ok(Math.abs(row.predicted_param_gb - (e2b.totalParams * 1e9 * (4.8944 / 8)) / 1024 ** 3) < 0.01);
  const line = readme.split('\n').find((l) => l.includes('`ple-llamacpp-non-gpu-residency`'));
  assert.ok(line.includes('every discrete-GPU row for'));
  assert.ok(line.includes('the GB10 unified-memory rows (DGX Spark 128GB) and the Mac rows (Apple unified memory) count the full weights'));
});
