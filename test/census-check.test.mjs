// census 결정성 — 커밋 산출물이 CENSUS_DATE 고정 재생성과 byte-identical (npm test 경로에서 상시 집행).
import test from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { mkdtempSync, readdirSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const CHECK = new URL('../census/check.mjs', import.meta.url).pathname;

test('census:check — byte-identical 4/4 그리고 임시 산출물 완전 제거(/tmp 누수 차단)', () => {
  // TMPDIR 주입으로 check.mjs의 mkdtemp 부모를 격리 — 종료 후 부모가 비어 있어야 한다
  const parent = mkdtempSync(join(tmpdir(), 'census-check-parent-'));
  try {
    const out = execFileSync(process.execPath, [CHECK], { encoding: 'utf8', env: { ...process.env, TMPDIR: parent } });
    assert.match(out, /census-v1\.json byte-identical/);
    assert.match(out, /census-v1\.csv byte-identical/);
    assert.match(out, /manifest\.json byte-identical/);
    assert.match(out, /README\.md byte-identical/);
    assert.deepEqual(readdirSync(parent), [], 'check.mjs가 임시 디렉터리를 남김');
  } finally {
    rmSync(parent, { recursive: true, force: true });
  }
});

test('generate.mjs — CENSUS_DATE 형식 위반은 즉시 exit 2', () => {
  const gen = new URL('../census/generate.mjs', import.meta.url).pathname;
  const parent = mkdtempSync(join(tmpdir(), 'census-badenv-'));
  try {
    assert.throws(() => execFileSync(process.execPath, [gen], {
      env: { ...process.env, CENSUS_DATE: '2026-9-3', CENSUS_OUT: parent + '/' }, stdio: 'pipe',
    }), /status.*2|Command failed/s);
  } finally {
    rmSync(parent, { recursive: true, force: true });
  }
});
