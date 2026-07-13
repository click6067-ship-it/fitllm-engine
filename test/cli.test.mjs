import { test } from 'node:test';
import { strict as assert } from 'node:assert';
import { execFileSync } from 'node:child_process';
const BIN = new URL('../bin/fitllm.mjs', import.meta.url).pathname;
const run = (args) => {
  try { return { out: execFileSync(process.execPath, [BIN, ...args], { encoding: 'utf8' }), code: 0 }; }
  catch (e) { return { out: (e.stdout || '') + (e.stderr || ''), code: e.status }; }
};

test('fits case exits 0 with verdict json', () => {
  const { out, code } = run(['gpt-oss-20b', '--gpu', 'RTX 4090', '--json']);
  assert.equal(code, 0);
  const j = JSON.parse(out);
  assert.equal(j.verdict, 'yes');
  assert.ok(j.breakdown.paramGB > 0);
});

test("won't-fit case exits 1 and suggests a fix", () => {
  const { out, code } = run(['gpt-oss-120b', '--gpu', 'RTX 4090']);
  assert.equal(code, 1);
  assert.match(out, /WON'T FIT/);
  assert.match(out, /→ /);
});

test('mac path works', () => {
  const { code } = run(['Qwen 3.6 35B-A3B', '--mac', '64', '--json']);
  assert.equal(code, 0);
});

test('--list contains new catalog', () => {
  const { out, code } = run(['--list']);
  assert.equal(code, 0);
  assert.match(out, /GLM-5\.2/);
  assert.match(out, /RX 7900 XTX/);
});

test('unknown model exits 2', () => {
  const { code } = run(['definitely-not-a-model', '--mac', '64']);
  assert.equal(code, 2);
});

test('multi-GPU combo "5090 + 3090" pools VRAM (56GB)', () => {
  const { out, code } = run(['Qwen 3.6 35B-A3B', '--gpu', 'RTX 5090 + RTX 3090', '--json']);
  assert.equal(code, 0);
  const j = JSON.parse(out);
  assert.equal(j.memoryGB, 56);                       // 32 + 24 풀링
  assert.match(j.hardware, /RTX 5090 \+ RTX 3090.*pooled/);
});

test('--count 2 equals the 2× preset verdict', () => {
  const a = JSON.parse(run(['Qwen 3.6 35B-A3B', '--gpu', 'RTX 3090', '--count', '2', '--json']).out);
  const b = JSON.parse(run(['Qwen 3.6 35B-A3B', '--gpu', '2× RTX 3090', '--json']).out);
  assert.equal(a.memoryGB, b.memoryGB);               // 48 = 48
  assert.equal(a.verdict, b.verdict);
});

test('combo with unknown part exits 2 naming the part', () => {
  const { out, code } = run(['gpt-oss-20b', '--gpu', 'RTX 4090 + notacard']);
  assert.equal(code, 2);
  assert.match(out, /notacard/);
});

// ── --top: "이 하드웨어에서 뭘 돌릴 수 있나" ──
test('--top: scans catalog, sorted by params desc, all fit, honest ranking label', () => {
  const { out, code } = run(['--top', '--gpu', 'RTX 4090', '--json']);
  assert.equal(code, 0);
  const j = JSON.parse(out);
  assert.ok(j.fits.length >= 5);
  for (let i = 1; i < j.fits.length; i++) assert.ok(j.fits[i - 1].params_b >= j.fits[i].params_b, 'sorted by params desc');
  assert.ok(j.fits.every((f) => f.verdict !== 'no'));
  assert.match(j.rankedBy, /not a quality ranking/);
});
test('--top N limits results', () => {
  const { out, code } = run(['--top', '3', '--mac', '64', '--json']);
  assert.equal(code, 0);
  assert.equal(JSON.parse(out).fits.length, 3);
});
test('--top respects --quant filter', () => {
  const { out, code } = run(['--top', '--gpu', 'RTX 4090', '--quant', 'FP16', '--json']);
  assert.equal(code, 0);
  const j = JSON.parse(out);
  assert.ok(j.fits.length >= 1);
  assert.ok(j.fits.every((f) => f.quant === 'FP16'));
});
test('--top exits 1 when nothing in catalog fits', () => {
  const { code } = run(['--top', '--mac', '8']);
  assert.equal(code, 1);
});
