// README/CONTRIBUTING 전환-감사 주장 드리프트 가드 — 수치·과장·링크가 코드 실재와 어긋나면 실패.
// (demo.gif 내 문구는 이미지 재생성 비용으로 보류 — 텍스트 표면만 여기서 고정)
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';

const README = readFileSync(new URL('../README.md', import.meta.url), 'utf8');
const CONTRIB = readFileSync(new URL('../CONTRIBUTING.md', import.meta.url), 'utf8');
const SERVER = readFileSync(new URL('../server.json', import.meta.url), 'utf8');
const CENSUS_GENERATOR = readFileSync(new URL('../census/generate.mjs', import.meta.url), 'utf8');

test('README: 낡은/과장 문구 금지', () => {
  for (const banned of [
    'M1–M5', '16 language-neutral', '× 88 GPUs', 'matches real local runs',
    'Every hardware number is cross-verified', '16%2F16',
  ]) assert.equal(README.includes(banned), false, `금지 문구 잔존: ${banned}`);
  assert.doesNotMatch(README, /every number from official config\.json/i);
  assert.doesNotMatch(SERVER, /exact VRAM|exact.*math/i);
  assert.doesNotMatch(CENSUS_GENERATOR, /every number[^\n]*official/i);
  assert.equal(README.includes('8,000+'), false, 'census 수치는 8,424 exact');
});

test('Quick start 첫 예시는 실제로 성공(FITS) 데모 — Gemma 4 12b × RTX 4090 E2E', () => {
  assert.ok(README.includes('npx fitllm "Gemma 4 12b" --gpu "RTX 4090"'), 'Quick start 예시 드리프트');
  const r = spawnSync(process.execPath, [new URL('../bin/fitllm.mjs', import.meta.url).pathname, 'Gemma 4 12b', '--gpu', 'RTX 4090'], { encoding: 'utf8' });
  assert.equal(r.status, 0); // 첫 데모가 TIGHT/실패면 혼란 — exit 0 고정
  assert.match(r.stdout, /✓ FITS/);
  assert.match(r.stdout, /receipt: https:\/\/fitllm\.run\/r\/gemma-4-12b-q4_k_m-on-rtx-4090/);
});

test('README: 현행 사실 필수 표기', () => {
  for (const required of [
    '28 language-neutral', '× 93 GPUs', 'M1–M6', '28%2F28', '8,424 verdicts',
    'npm install fitllm-engine', '?template=measurement.yml',
    "from 'fitllm-engine'",
    'The CLI, API, and MCP use a curated catalog pinned to official configs.', // 문법 파손 정정문(감사 제안) 고정
  ]) assert.equal(README.includes(required), true, `필수 문구 누락: ${required}`);
  // Quick start가 MCP 섹션보다 위
  assert.ok(README.indexOf('## Quick start') < README.indexOf('## Remote MCP server'));
});

test('measurement 링크는 template 지정 — README·CONTRIBUTING 모두', () => {
  assert.equal(README.includes('labels=measurement)'), false);
  assert.equal(CONTRIB.includes('labels=measurement)'), false);
  assert.ok(CONTRIB.includes('?template=measurement.yml'));
});
test('test_approved_cli_workflows', () => {
  const readme = readFileSync(new URL('../README.md', import.meta.url), 'utf8');
  assert.match(readme, /npx fitllm --top --detect --json/);
  assert.match(readme, /npx fitllm measure/);
  assert.match(readme, /npm run benchmark:accuracy/);
  assert.match(readme, /Hugging Face ID|org\/model/);
  assert.doesNotMatch(readme, /most accurate|#1 accuracy/i);
});

test('distribution preflight examples stay present and reproducible', () => {
  for (const required of [
    'uses: click6067-ship-it/fitllm-engine@v2.9.0',
    'npx fitllm "Gemma 4 12b" --detect --json --why',
    '&& ollama pull gemma4:12b',
    '&& llama-cli -m',
    'npm run benchmark:differential',
    'npm run benchmark:capture:llmfit -- --binary',
    'architecture_differential_not_runtime_accuracy',
  ]) assert.ok(README.includes(required), `distribution example missing: ${required}`);
});
