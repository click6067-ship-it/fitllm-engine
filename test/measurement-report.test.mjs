import test from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { buildMeasurementReport } from '../bin/measurement-report.mjs';

test('test_validated_prefill_without_submit', () => {
  const report = buildMeasurementReport({
    model: 'Qwen 3.6 27B', hardware: 'RTX 4090', quant: 'Q4_K_M', ctx: 8192, kvBits: 16,
    measured: 15.3, kind: 'generation_peak', unit: 'GiB', runtime: 'llama.cpp b6400',
    predicted: 17.1,
  });
  const url = new URL(report.issueUrl);
  assert.equal(url.origin, 'https://github.com');
  assert.equal(url.pathname, '/click6067-ship-it/fitllm-engine/issues/new');
  assert.match(url.searchParams.get('command_and_artifact'), /generation_peak/);
  assert.equal(url.searchParams.get('model'), 'Qwen 3.6 27B');
  assert.equal(url.searchParams.get('measurement_kind'), 'generation_peak');
  assert.equal(report.candidate.measurementKind, 'generation_peak');
  assert.equal(report.submitted, false);

  assert.throws(() => buildMeasurementReport({ model: 'x', hardware: 'y', measured: 1, kind: 'generation_peak', unit: 'GiB', runtime: 'Ollama' }), /version/i);
  assert.throws(() => buildMeasurementReport({ model: 'x', hardware: 'y', measured: -1, kind: 'unknown', unit: 'MB', runtime: 'Ollama 1.0' }), /measured|kind|unit/i);
});

test('measure CLI prints an exact local candidate and never submits it', () => {
  const bin = new URL('../bin/fitllm.mjs', import.meta.url).pathname;
  const result = spawnSync(process.execPath, [
    bin, 'measure', 'Qwen 3.6 27B', '--gpu', 'RTX 4090',
    '--measured', '15.3', '--kind', 'system_total_peak', '--unit', 'GiB',
    '--runtime', 'llama.cpp b6400', '--json',
  ], { encoding: 'utf8' });
  assert.equal(result.status, 0, result.stderr);
  const report = JSON.parse(result.stdout);
  assert.equal(report.candidate.device, 'RTX 4090');
  assert.equal(report.candidate.predictedMetric, 'predicted_total_to_run_gb');
  assert.equal(report.submitted, false);

  for (const args of [
    ['measure', '--gpu', 'RTX 4090'],
    ['measure', 'Qwen', '--gpu', 'RTX 4090'],
  ]) {
    const rejected = spawnSync(process.execPath, [
      bin, ...args, '--measured', '15.3', '--kind', 'system_total_peak', '--unit', 'GiB', '--runtime', 'llama.cpp b6400', '--json',
    ], { encoding: 'utf8' });
    assert.equal(rejected.status, 2);
    assert.doesNotMatch(rejected.stdout, /candidate|GLM-4\.7-Flash/);
  }
});
