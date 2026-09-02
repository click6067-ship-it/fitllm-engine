// README/CONTRIBUTING 전환-감사 주장 드리프트 가드 — 수치·과장·링크가 코드 실재와 어긋나면 실패.
// (demo.gif 내 문구는 이미지 재생성 비용으로 보류 — 텍스트 표면만 여기서 고정)
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const README = readFileSync(new URL('../README.md', import.meta.url), 'utf8');
const CONTRIB = readFileSync(new URL('../CONTRIBUTING.md', import.meta.url), 'utf8');

test('README: 낡은/과장 문구 금지', () => {
  for (const banned of [
    'M1–M5', '16 language-neutral', '× 88 GPUs', 'matches real local runs',
    'Every hardware number is cross-verified', '16%2F16',
  ]) assert.equal(README.includes(banned), false, `금지 문구 잔존: ${banned}`);
  assert.doesNotMatch(README, /every number from official config\.json/i);
});

test('README: 현행 사실 필수 표기', () => {
  for (const required of [
    '28 language-neutral', '× 93 GPUs', 'M1–M6', '28%2F28',
    'npm install fitllm-engine', '?template=measurement.yml',
    "from 'fitllm-engine'",
  ]) assert.equal(README.includes(required), true, `필수 문구 누락: ${required}`);
  // Quick start가 MCP 섹션보다 위
  assert.ok(README.indexOf('## Quick start') < README.indexOf('## Remote MCP server'));
});

test('measurement 링크는 template 지정 — README·CONTRIBUTING 모두', () => {
  assert.equal(README.includes('labels=measurement)'), false);
  assert.equal(CONTRIB.includes('labels=measurement)'), false);
  assert.ok(CONTRIB.includes('?template=measurement.yml'));
});
