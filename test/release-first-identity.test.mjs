// release-first GLM identity boundary — 2.15.0 public port of private src/lib/release-first-identity.test.js.
// 정확한 `GLM-5.3`(text-only)만 해석되고, 기존 `GLM-4.7-Flash` identity는 그대로이며, 모델링하지 않은
// `GLM-5.3-Flash`·`zai-org/GLM-5.3-Flash`는 정규화 이름 일치로도 GLM-5.3에 흡수되지 않는다(숫자 판정 0건).
import test from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync, spawnSync } from 'node:child_process';
import { mkdtempSync, readFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { LOCAL_MODELS, resolveLocalModel } from '../engine.js';

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

const BASELINE_NAMES = Object.freeze([
  'GLM-4.7-Flash', 'GLM-5.2', 'gpt-oss-20b', 'gpt-oss-120b',
  'Qwen 3.6 27B', 'Qwen 3.6 35B-A3B', 'Qwen-AgentWorld-35B-A3B',
  'Gemma 4 e2b', 'Gemma 4 e4b', 'Gemma 4 12b', 'Gemma 4 26b A4B', 'Gemma 4 31b',
  'Llama-3.2-3B-Instruct', 'Llama-3.1-8B-Instruct', 'MiniCPM5-1B',
  'Qwen3-0.6B', 'Qwen3-1.7B', 'Llama-3.2-1B-Instruct', 'Gemma-3-1B-it',
  'Hy3', 'Qwen 3.8 27B', 'Qwen 3.8 2.4T-A95B', 'Laguna XS 2.1', 'Laguna S 2.1',
  'Spark-X2.5-4B', 'Granite-4.2-30B',
]);

test('resolves exact GLM-5.3 text while preserving the existing Flash identity', () => {
  const text = resolveLocalModel('GLM-5.3');
  assert.equal(text.status, 'resolved');
  assert.equal(text.canonicalName, 'GLM-5.3');
  const flash = resolveLocalModel('GLM-4.7-Flash');
  assert.equal(flash.status, 'resolved');
  assert.equal(flash.canonicalName, 'GLM-4.7-Flash');
});

test('GLM-5.3-Flash and zai-org/GLM-5.3-Flash never resolve to the catalog', () => {
  for (const query of ['GLM-5.3-Flash', 'zai-org/GLM-5.3-Flash', 'glm 5.3 flash']) {
    const r = resolveLocalModel(query);
    assert.notEqual(r.status, 'resolved', query);
    assert.equal(r.canonicalName, undefined, query);
  }
});

test('CLI and Action refuse a fake Flash derivative with exit 2 and no numeric verdict', () => {
  // HF id 형태(zai-org/…)는 라이브 HF 경로라 네트워크가 필요하므로 여기서는 카탈로그 이름 형태만 CLI로 검사한다.
  const cli = run(['GLM-5.3-Flash', '--gpu', 'RTX 4090']);
  assert.equal(cli.code, 2, cli.out);
  assert.doesNotMatch(cli.out, /FITS|WON'T FIT|TIGHT|"verdict"|receipt:/);
  const json = run(['GLM-5.3-Flash', '--gpu', 'RTX 4090', '--json']);
  assert.equal(json.code, 2, json.out);
  assert.doesNotMatch(json.out, /"verdict"|"usedGB"|"memoryGB"/);
  const action = runAction({ INPUT_MODEL: 'GLM-5.3-Flash', INPUT_GPU: 'RTX 4090' });
  assert.equal(action.status, 2);
  assert.match(action.output, /exit-code=2/);
  assert.doesNotMatch(action.output, /"verdict"|"memoryGB"|receipt/i);
});

test('keeps indices 0 through 25 and appends GLM-5.3 at 26', () => {
  assert.deepEqual(LOCAL_MODELS.slice(0, 26).map((model) => model.name), BASELINE_NAMES);
  assert.equal(LOCAL_MODELS[26]?.name, 'GLM-5.3');
  assert.equal(LOCAL_MODELS.length, 27);
});
