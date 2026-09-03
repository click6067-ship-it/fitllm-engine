import test from 'node:test';
import assert from 'node:assert/strict';
import { buildAccuracyReport, renderMarkdown } from '../scripts/accuracy-report.mjs';

const census = { data: [
  { model: 'A', device: 'GPU', quant: 'Q4', ctx: 8192, predicted_resident_weights_gb: 10, predicted_total_to_run_gb: 14, verdict: 'yes' },
  { model: 'B', device: 'Mac', quant: '8bit', ctx: 8192, predicted_resident_weights_gb: 20, predicted_total_to_run_gb: 24, verdict: 'no' },
] };

test('test_typed_metric_pairing', () => {
  const report = buildAccuracyReport(census, [
    { model: 'A', device: 'GPU', quant: 'Q4', ctx: 32768, measuredPeakGB: 10.73741824, unit: 'GB', measurementKind: 'idle_resident', evidenceLevel: 'verified', runtime: 'r1 1.0' },
    { model: 'B', device: 'Mac', quant: '8bit', ctx: 8192, measuredPeakGB: 24, unit: 'GiB', measurementKind: 'system_total_peak', evidenceLevel: 'verified', runtime: 'r2 2.0' },
    { model: 'A', device: 'GPU', quant: 'Q4', ctx: 8192, measuredPeakGB: 12, unit: 'GiB', measurementKind: 'unknown', evidenceLevel: 'verified', runtime: 'r1 1.0' },
  ]);
  assert.equal(report.included.length, 2);
  assert.equal(report.excluded.length, 1);
  assert.equal(report.included[0].predictedMetric, 'predicted_resident_weights_gb');
  assert.equal(report.included[1].predictedMetric, 'predicted_total_to_run_gb');
  assert.equal(report.metrics.mapePct, 0);
  const markdown = renderMarkdown(report);
  assert.match(markdown, /verified paired claim cases/i);
  assert.match(markdown, /idle_resident=1/);
  assert.match(markdown, /system_total_peak=1/);
  assert.match(markdown, /verified=2/);
});

test('test_public_claim_gate', () => {
  const report = buildAccuracyReport(census, []);
  assert.equal(report.claimGate.allowed, false);
  assert.ok(report.claimGate.minimums.measurements >= 30);
  assert.match(report.claimGate.reason, /insufficient/i);
});

test('test_claim_gate_requires_same_immutable_cases_and_independent_verification', () => {
  const measurements = Array.from({ length: 30 }, (_, index) => ({
    caseId: `case-${index}`,
    model: 'A',
    device: 'GPU',
    quant: 'Q4',
    ctx: 8192,
    measuredPeakGB: 14,
    unit: 'GiB',
    measurementKind: 'system_total_peak',
    evidenceLevel: 'verified',
    runtime: `runtime-${index % 3} 1.0`,
    architectureFamily: `family-${index % 6}`,
    modelRevision: '0123456789abcdef0123456789abcdef01234567',
    rawOutputSha256: 'a'.repeat(64),
    source: `https://example.test/raw/case-${index}`,
    reporter: 'runner-a',
    verifiedBy: 'reviewer-b',
  }));
  const competitors = measurements.map((measurement, index) => ({
    caseId: `other-${index}`,
    competitor: 'competitor@0123456',
    absolutePercentageErrorPct: 20,
    falseFit: true,
    rawOutputSha256: 'b'.repeat(64),
    source: `https://example.test/competitor/other-${index}`,
  }));
  const mismatched = buildAccuracyReport(census, measurements, competitors);
  assert.equal(mismatched.claimGate.allowed, false);
  assert.equal(mismatched.claimGate.actual.measurements, 0);

  for (let index = 0; index < competitors.length; index += 1) competitors[index].caseId = `case-${index}`;
  const paired = buildAccuracyReport(census, measurements, competitors);
  assert.equal(paired.claimGate.allowed, true);
  assert.equal(paired.claimGate.actual.measurements, 30);

  measurements[0].verifiedBy = 'runner-a';
  const selfVerified = buildAccuracyReport(census, measurements, competitors);
  assert.equal(selfVerified.claimGate.allowed, false);
  assert.equal(selfVerified.claimGate.actual.measurements, 29);
});
