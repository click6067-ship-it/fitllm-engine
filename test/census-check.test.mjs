// census 결정성 — 커밋 산출물이 CENSUS_DATE 고정 재생성과 byte-identical (npm test 경로에서 상시 집행).
import test from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';

test('census:check — 커밋된 census 4개 파일이 결정적 재생성과 byte-identical', () => {
  const out = execFileSync(process.execPath, [new URL('../census/check.mjs', import.meta.url).pathname], { encoding: 'utf8' });
  assert.match(out, /census-v1\.json byte-identical/);
  assert.match(out, /census-v1\.csv byte-identical/);
  assert.match(out, /manifest\.json byte-identical/);
  assert.match(out, /README\.md byte-identical/);
});
