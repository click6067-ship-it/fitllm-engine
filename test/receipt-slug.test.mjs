// CLI 영수증 슬러그 = fitllm-v2 canonical 규칙과의 패리티 고정 (기대 문자열은 v2 receipt-roundtrip 테스트와 동일).
import test from 'node:test';
import assert from 'node:assert/strict';
import { canonicalReceiptSlug, quantToken, defaultCtxFor } from '../bin/receipt-slug.mjs';

test('GPU 기본: quant 토큰만 (ctx/kv 기본은 생략)', () => {
  assert.equal(
    canonicalReceiptSlug({ modelName: 'Gemma 4 31b', quantLabel: 'Q4_K_M', isGpu: true, hwLabel: 'RTX 4090', ctx: 8192, kvBits: 16, maxContext: 262144 }),
    'gemma-4-31b-q4_k_m-on-rtx-4090',
  );
});

test('Mac 기본 8-bit → 8bit 토큰 (v2 /r 파서 수용형 — "8-bit" 원형이 404 나던 결함의 패리티)', () => {
  assert.equal(
    canonicalReceiptSlug({ modelName: 'Gemma 4 31b', quantLabel: '8-bit', isGpu: false, ramGB: 64, ctx: 8192, kvBits: 16, maxContext: 262144 }),
    'gemma-4-31b-8bit-on-mac-64gb',
  );
});

test('비기본 ctx/kv는 조건부 토큰 — 순서 quant→ctx→kv', () => {
  assert.equal(
    canonicalReceiptSlug({ modelName: 'Gemma 4 31b', quantLabel: 'Q4_K_M', isGpu: true, hwLabel: 'RTX 4090', ctx: 131072, kvBits: 4, maxContext: 262144 }),
    'gemma-4-31b-q4_k_m-ctx131072-kv4-on-rtx-4090',
  );
});

test('멀티GPU " + " → -plus- · 점 모델명 → 대시', () => {
  assert.equal(
    canonicalReceiptSlug({ modelName: 'Qwen 3.8 27B', quantLabel: 'Q4_K_M', isGpu: true, hwLabel: 'RTX 5090 + RTX 3090', ctx: 8192, kvBits: 16, maxContext: 262144 }),
    'qwen-3-8-27b-q4_k_m-on-rtx-5090-plus-rtx-3090',
  );
});

test('maxContext<8192 모델은 그 값이 기본 ctx — 생략 대칭', () => {
  assert.equal(defaultCtxFor(4096), 4096);
  assert.equal(
    canonicalReceiptSlug({ modelName: 'X', quantLabel: 'Q4_K_M', isGpu: true, hwLabel: 'RTX 4090', ctx: 4096, kvBits: 16, maxContext: 4096 }),
    'x-q4_k_m-on-rtx-4090',
  );
});

test('quantToken: GPU 티어는 소문자 유지·bit형만 변환', () => {
  assert.equal(quantToken('Q4_K_M'), 'q4_k_m');
  assert.equal(quantToken('8-bit'), '8bit');
  assert.equal(quantToken('FP16'), 'fp16');
});
