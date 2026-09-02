// CLI 영수증 ↔ v2 /r 파서 계약: CLI가 발급하는 모든 영수증은 v2가 200을 줄 수 있어야 하고,
// v2가 표현 못 하는 입력(카탈로그 밖 GPU·9카드+·Mac>2048GB)엔 URL을 아예 내지 않는다(깨진 링크 금지).
// 실 파서 왕복은 fitllm-v2 scripts/receipt-roundtrip.test.js가 집행 — 여기서는 발급 게이트와
// 슬러그 문법(정본: fitllm-v2 api/r.js 파서 규칙 미러)을 서브프로세스로 고정한다.
import test from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { receiptRepresentable } from '../bin/receipt-slug.mjs';

const BIN = new URL('../bin/fitllm.mjs', import.meta.url).pathname;
const run = (...args) => spawnSync(process.execPath, [BIN, ...args], { encoding: 'utf8' });
// v2 api/r.js 파서 수용 문법 미러(quant 토큰·-ctx/-kv 순서·mac 전체 앵커) — v2 쪽 변경 시 함께 갱신
const V2_SLUG = /^[a-z0-9][a-z0-9._-]*-(q4_k_m|q5_k_m|q6_k|q8_0|fp16|4bit|8bit|16bit)(-ctx\d{2,7})?(-kv(8|4))?-on-((mac-\d+gb)|[a-z0-9][a-z0-9-]*)$/;
const receiptOf = (out) => out.match(/receipt: https:\/\/fitllm\.run\/r\/(\S+)/)?.[1];

test('Mac 4096GB(감독 반례): 계산은 보존, 영수증 URL 미발급 + 명시 안내', () => {
  const r = run('Gemma 4 31b', '--mac', '4096');
  assert.ok([0, 1].includes(r.status));
  assert.match(r.stdout, /FITS|TIGHT|WON'T FIT/); // 계산 기능 보존
  assert.equal(receiptOf(r.stdout), undefined);
  assert.match(r.stdout, /receipt: n\/a — .*Macs 8–2048GB/);
});

test('16카드(5090+3090 ×8, 감독 반례): 영수증 미발급 + 안내', () => {
  const r = run('Qwen 3.8 27B', '--gpu', 'RTX 5090 + RTX 3090', '--count', '8');
  assert.ok([0, 1].includes(r.status));
  assert.match(r.stdout, /FITS|TIGHT|WON'T FIT/);
  assert.equal(receiptOf(r.stdout), undefined);
  assert.match(r.stdout, /up to 8 cards/);
});

test('경계: Mac 2048GB는 발급, 카드 8장(4×2)도 발급 — 슬러그는 v2 문법 통과', () => {
  for (const args of [
    ['Gemma 4 31b', '--mac', '2048'],
    ['Qwen 3.8 27B', '--gpu', 'RTX 5090 + RTX 3090', '--count', '4'],
    ['Gemma 4 31b', '--gpu', 'RTX 4090', '--ctx', '131072', '--kv', '4'],
  ]) {
    const r = run(...args);
    const slug = receiptOf(r.stdout);
    assert.ok(slug, `${args.join(' ')} — 영수증이 발급되어야 함`);
    assert.match(slug, V2_SLUG, `${slug} — v2 파서 문법 위반`);
  }
});

test('receiptRepresentable 단위: 미등록(detected) GPU·한계 초과는 false', () => {
  assert.equal(receiptRepresentable({ isGpu: true, catalogGpu: false }), false);      // --detect 미등록
  assert.equal(receiptRepresentable({ isGpu: true, totalCards: 9 }), false);
  assert.equal(receiptRepresentable({ isGpu: true, totalCards: 8 }), true);
  assert.equal(receiptRepresentable({ isGpu: false, ramGB: 4096 }), false);
  assert.equal(receiptRepresentable({ isGpu: false, ramGB: 2048 }), true);
  assert.equal(receiptRepresentable({ isGpu: false, ramGB: 4 }), false);
});
