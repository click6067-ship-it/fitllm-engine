// --ctx 가드 회귀(감독 재현: `--ctx -1`이 exit 0 + FITS + KV 0 + 파싱 불가 영수증) — 서브프로세스 실검.
import test from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';

const BIN = new URL('../bin/fitllm.mjs', import.meta.url).pathname;
const run = (...args) => spawnSync(process.execPath, [BIN, ...args], { encoding: 'utf8' });

test('--ctx -1: exit 2, 판정/영수증 미출력 (거짓 FITS 차단)', () => {
  const r = run('Gemma 4 31b', '--gpu', 'RTX 4090', '--ctx', '-1');
  assert.equal(r.status, 2);
  assert.doesNotMatch(r.stdout, /FITS|receipt:/);
  assert.match(r.stderr, /--ctx must be an integer >= 16/);
});

test('--ctx 0 / 비수치: exit 2', () => {
  assert.equal(run('Gemma 4 31b', '--gpu', 'RTX 4090', '--ctx', '0').status, 2);
  assert.equal(run('Gemma 4 31b', '--gpu', 'RTX 4090', '--ctx', 'abc').status, 2);
});

test('--top 경로도 같은 가드 (독립 파싱이던 ctxReq 공용화 회귀)', () => {
  const r = run('--top', '3', '--gpu', 'RTX 4090', '--ctx', '-1');
  assert.equal(r.status, 2);
  assert.match(r.stderr, /--ctx must be an integer >= 16/);
});

test('대조군: 유효 --ctx 32768은 정상 판정 + canonical 영수증', () => {
  const r = run('Gemma 4 31b', '--gpu', 'RTX 4090', '--ctx', '32768');
  assert.ok(r.status === 0 || r.status === 1); // 판정 결과에 따른 정상 exit
  assert.match(r.stdout, /receipt: https:\/\/fitllm\.run\/r\/gemma-4-31b-q4_k_m-ctx32768-on-rtx-4090/);
});
