// parseHfConfig 회귀 테스트 — 선-양자화 레포 파라미터 역산 (issue #2의 ÷2 과소계산)
// 전부 합성 config — 외부 제보 수치에 의존하지 않는다(제보는 주장, 테스트는 합성으로 증명).
import { test } from 'node:test';
import { strict as assert } from 'node:assert';
import { parseHfConfig } from '../engine.js';

const BASE = {
  num_hidden_layers: 48, num_attention_heads: 16, num_key_value_heads: 8,
  hidden_size: 4096, head_dim: 128, max_position_embeddings: 32768,
};
const GiB = 1024 ** 3;

test('MLX 8-bit (quantization.bits): total_size 32GiB → ~34B params, NOT halved to ~17B', () => {
  const m = parseHfConfig('test/mlx-8bit', { ...BASE, torch_dtype: 'bfloat16', quantization: { bits: 8 } }, 32 * GiB);
  // 8bit = 1 byte/param → 32GiB ≈ 34.4B. ÷2 버그면 17.2B로 나온다.
  assert.ok(m.totalParams > 30, `totalParams ${m.totalParams}B — bfloat16(2B) 경로로 절반 계산됨`);
  assert.ok(m.totalParams < 40, `totalParams ${m.totalParams}B — 과대`);
});

test('AWQ 4-bit (quantization_config.bits): total_size 8GB → ~16B params, NOT quartered', () => {
  const m = parseHfConfig('test/awq-4bit', { ...BASE, torch_dtype: 'float16', quantization_config: { bits: 4 } }, 8e9);
  // 4bit = 0.5 byte/param → 8GB ≈ 16B. torch_dtype(2B) 경로면 4B로 나온다.
  assert.ok(m.totalParams > 14 && m.totalParams < 18, `totalParams ${m.totalParams}B`);
});

test('bf16 path preserved: no quantization key → total_size 16GB / 2 bytes = 8B params', () => {
  const m = parseHfConfig('test/bf16', { ...BASE, torch_dtype: 'bfloat16' }, 16e9);
  assert.ok(Math.abs(m.totalParams - 8) < 0.1, `totalParams ${m.totalParams}B ≠ 8B`);
});

test('bitsandbytes load_in_4bit (no bits field): total_size 8GB → ~16B params, NOT ~4B', () => {
  const m = parseHfConfig('test/bnb-4bit', { ...BASE, torch_dtype: 'float16', quantization_config: { load_in_4bit: true, bnb_4bit_quant_type: 'nf4' } }, 8e9);
  assert.ok(m.totalParams > 14 && m.totalParams < 18, `totalParams ${m.totalParams}B — load_in_4bit 미인식으로 torch_dtype 경로 탔음`);
});

test('bitsandbytes load_in_8bit: total_size 16GB → ~16B params', () => {
  const m = parseHfConfig('test/bnb-8bit', { ...BASE, torch_dtype: 'float16', quantization_config: { load_in_8bit: true } }, 16e9);
  assert.ok(m.totalParams > 14 && m.totalParams < 18, `totalParams ${m.totalParams}B`);
});

test('fp32 and int8 dtype paths preserved', () => {
  const fp32 = parseHfConfig('test/fp32', { ...BASE, torch_dtype: 'float32' }, 16e9);
  assert.ok(Math.abs(fp32.totalParams - 4) < 0.1, `fp32 ${fp32.totalParams}B ≠ 4B`);
  const int8 = parseHfConfig('test/int8', { ...BASE, torch_dtype: 'int8' }, 16e9);
  assert.ok(Math.abs(int8.totalParams - 16) < 0.1, `int8 ${int8.totalParams}B ≠ 16B`);
});

test('PLE and MTP metadata are normalized without broadening the family allowlist', () => {
  const base = {
    dtype: 'bfloat16', num_hidden_layers: 35, num_attention_heads: 8,
    num_key_value_heads: 1, head_dim: 256, hidden_size: 1536,
    max_position_embeddings: 131072, vocab_size_per_layer_input: 262144,
    hidden_size_per_layer_input: 256,
  };
  const verified = parseHfConfig('google/gemma-4-E2B-it', {
    model_type: 'gemma4', text_config: { ...base, model_type: 'gemma4_text' },
  }, 10246356102);
  const unverified = parseHfConfig('example/field-lookalike', {
    ...base, model_type: 'llama', num_nextn_predict_layers: 1,
  }, 10246356102);
  assert.equal(verified.pleOffloadVerified, true);
  assert.equal(unverified.pleOffloadVerified, false);
  assert.equal(unverified.mtpLayerCount, 1);
});
