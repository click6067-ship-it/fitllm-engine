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
  const { code } = run(['Qwen3-30B-A3B', '--mac', '64', '--json']);
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
