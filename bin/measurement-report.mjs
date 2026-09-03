const KINDS = new Set(['idle_resident', 'load_peak', 'generation_peak', 'system_total_peak']);
const UNITS = new Set(['GiB', 'GB']);

function requiredText(value, label) {
  const text = String(value || '').trim();
  if (!text) throw new Error(`${label} is required`);
  return text;
}

export function buildMeasurementReport(input) {
  const measured = Number(input.measured);
  if (!Number.isFinite(measured) || measured <= 0) throw new Error('measured memory must be a positive number');
  const measurementKind = requiredText(input.kind, 'measurement kind');
  if (!KINDS.has(measurementKind)) throw new Error(`kind must be one of: ${[...KINDS].join(', ')}`);
  const unit = requiredText(input.unit, 'unit');
  if (!UNITS.has(unit)) throw new Error('unit must be GiB or GB');
  const runtime = requiredText(input.runtime, 'runtime');
  if (!/\d/.test(runtime)) throw new Error('runtime version or commit is required');
  const predicted = Number(input.predicted);
  const candidate = {
    model: requiredText(input.model, 'model'),
    device: requiredText(input.hardware, 'hardware'),
    quant: requiredText(input.quant || 'unknown', 'quant'),
    ctx: Number.isInteger(Number(input.ctx)) ? Number(input.ctx) : 8192,
    kvBits: [4, 8, 16].includes(Number(input.kvBits)) ? Number(input.kvBits) : 16,
    measuredPeakGB: measured,
    measurementKind,
    unit,
    runtime,
    evidenceLevel: 'community_unverified',
    predictedMetric: input.predictedMetric || null,
    predictedGB: Number.isFinite(predicted) ? predicted : null,
  };
  const body = [
    '## Reproducible measurement candidate',
    '',
    '```json',
    JSON.stringify(candidate, null, 2),
    '```',
    '',
    '- [ ] I took this measurement on the named hardware.',
    '- [ ] I will attach the exact command/flags and raw log or screenshot.',
    '- [ ] The run completed prefill and decode when reporting a generation peak.',
    '',
    '> This command only prepared a candidate. It did not submit or trust the measurement.',
  ].join('\n');
  const issue = new URL('https://github.com/click6067-ship-it/fitllm-engine/issues/new');
  issue.searchParams.set('template', 'measurement.yml');
  issue.searchParams.set('title', `[measured] ${candidate.model} on ${candidate.device}`);
  issue.searchParams.set('command_and_artifact', body);
  issue.searchParams.set('model', candidate.model);
  issue.searchParams.set('hardware', candidate.device);
  issue.searchParams.set('quant', candidate.quant);
  issue.searchParams.set('context', String(candidate.ctx));
  issue.searchParams.set('runtime', candidate.runtime);
  issue.searchParams.set('measured', String(candidate.measuredPeakGB));
  issue.searchParams.set('measurement_kind', candidate.measurementKind);
  issue.searchParams.set('unit', candidate.unit);
  issue.searchParams.set('kv_bits', String(candidate.kvBits));
  if (candidate.predictedGB != null) issue.searchParams.set('estimate', String(candidate.predictedGB));
  return { candidate, body, issueUrl: issue.toString(), submitted: false };
}
