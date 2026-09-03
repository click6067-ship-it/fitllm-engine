import { readFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';

const MINIMUMS = Object.freeze({ measurements: 30, runtimes: 3, architectureFamilies: 6, competitors: 1, relativeMapeImprovementPct: 10 });
const PAIRS = Object.freeze({
  idle_resident: 'predicted_resident_weights_gb',
  system_total_peak: 'predicted_total_to_run_gb',
});

const field = (row, snake, camel) => row[snake] ?? row[camel];
const normalizedMeasuredGiB = (measurement) => measurement.unit === 'GB'
  ? Number(measurement.measuredPeakGB) / 1.073741824
  : Number(measurement.measuredPeakGB);
const isSha256 = (value) => /^[a-f0-9]{64}$/i.test(String(value || ''));
const isHttpsUrl = (value) => {
  try {
    return new URL(String(value || '')).protocol === 'https:';
  } catch {
    return false;
  }
};

function isIndependentlyVerified(row) {
  return ['maintainer_verified', 'verified'].includes(row.evidenceLevel)
    && Boolean(row.caseId)
    && Boolean(row.modelRevision)
    && isSha256(row.rawOutputSha256)
    && isHttpsUrl(row.source)
    && Boolean(row.reporter)
    && Boolean(row.verifiedBy)
    && row.reporter !== row.verifiedBy;
}

function validCompetitorRow(row, eligibleCaseIds) {
  return eligibleCaseIds.has(row.caseId)
    && /^[^@\s]+@[^@\s]+$/.test(String(row.competitor || ''))
    && Number.isFinite(Number(row.absolutePercentageErrorPct))
    && typeof row.falseFit === 'boolean'
    && isSha256(row.rawOutputSha256)
    && isHttpsUrl(row.source);
}

function sameIdentity(row, measurement) {
  if (row.model !== measurement.model || row.device !== measurement.device || row.quant !== measurement.quant) return false;
  const rowKv = field(row, 'kv', 'kvBits');
  if (measurement.kvBits != null && rowKv != null && !String(rowKv).includes(String(measurement.kvBits))) return false;
  if (measurement.measurementKind !== 'idle_resident' && measurement.ctx != null && Number(row.ctx) !== Number(measurement.ctx)) return false;
  return true;
}

export function buildAccuracyReport(census, measurements, competitorRows = []) {
  const rows = Array.isArray(census) ? census : census?.data || [];
  const included = [];
  const excluded = [];
  for (const measurement of measurements || []) {
    const predictedMetric = PAIRS[measurement.measurementKind];
    if (!predictedMetric) {
      excluded.push({ measurement, reason: `unsupported measurement kind: ${measurement.measurementKind || 'missing'}` });
      continue;
    }
    const row = rows.find((candidate) => sameIdentity(candidate, measurement));
    if (!row) {
      excluded.push({ measurement, reason: 'no exact census identity match' });
      continue;
    }
    const predictedGiB = Number(field(row, predictedMetric, predictedMetric.replace(/_([a-z])/g, (_, letter) => letter.toUpperCase())));
    const measuredGiB = normalizedMeasuredGiB(measurement);
    if (!Number.isFinite(predictedGiB) || !Number.isFinite(measuredGiB) || measuredGiB <= 0) {
      excluded.push({ measurement, reason: 'invalid predicted or measured value' });
      continue;
    }
    included.push({
      caseId: measurement.caseId || null,
      model: measurement.model,
      device: measurement.device,
      measurementKind: measurement.measurementKind,
      predictedMetric,
      predictedGiB,
      measuredGiB,
      absolutePercentageErrorPct: Math.abs(predictedGiB - measuredGiB) / measuredGiB * 100,
      evidenceLevel: measurement.evidenceLevel || 'community_unverified',
      runtime: measurement.runtime,
      source: measurement.source || null,
      modelRevision: measurement.modelRevision || null,
      rawOutputSha256: measurement.rawOutputSha256 || null,
      reporter: measurement.reporter || null,
      verifiedBy: measurement.verifiedBy || null,
      architectureFamily: measurement.architectureFamily || row.architecture_family || row.architectureFamily || null,
      falseFit: measurement.measurementKind === 'system_total_peak'
        && row.verdict === 'yes'
        && Number.isFinite(Number(field(row, 'memory_gb', 'memoryGB')))
        && measuredGiB > Number(field(row, 'memory_gb', 'memoryGB')),
    });
  }
  const mapePct = included.length
    ? included.reduce((sum, row) => sum + row.absolutePercentageErrorPct, 0) / included.length
    : null;
  const verifiedGroups = new Map();
  for (const row of included.filter(isIndependentlyVerified)) {
    const group = verifiedGroups.get(row.caseId) || [];
    group.push(row);
    verifiedGroups.set(row.caseId, group);
  }
  const verifiedByCase = new Map(
    [...verifiedGroups].filter(([, rowsForCase]) => rowsForCase.length === 1).map(([caseId, rowsForCase]) => [caseId, rowsForCase[0]]),
  );
  const eligibleCaseIds = new Set(verifiedByCase.keys());
  const competitors = new Map();
  for (const row of competitorRows.filter((candidate) => validCompetitorRow(candidate, eligibleCaseIds))) {
    const cases = competitors.get(row.competitor) || new Map();
    if (cases.has(row.caseId)) cases.set(row.caseId, null);
    else cases.set(row.caseId, row);
    competitors.set(row.competitor, cases);
  }
  const selectedCompetitor = [...competitors]
    .map(([competitor, cases]) => [competitor, new Map([...cases].filter(([, row]) => row))])
    .sort((a, b) => b[1].size - a[1].size || a[0].localeCompare(b[0]))[0] || null;
  const paired = selectedCompetitor
    ? [...selectedCompetitor[1]].map(([caseId, competitor]) => ({ fit: verifiedByCase.get(caseId), competitor }))
    : [];
  const runtimes = new Set(paired.map(({ fit }) => fit.runtime).filter(Boolean));
  const families = new Set(paired.map(({ fit }) => fit.architectureFamily).filter(Boolean));
  const pairedFitMapePct = paired.length
    ? paired.reduce((sum, { fit }) => sum + fit.absolutePercentageErrorPct, 0) / paired.length
    : null;
  const competitorMapePct = paired.length
    ? paired.reduce((sum, { competitor }) => sum + Number(competitor.absolutePercentageErrorPct), 0) / paired.length
    : null;
  const comparablePairs = paired.filter(({ fit }) => fit.predictedMetric === 'predicted_total_to_run_gb');
  const fitFalseFitRatePct = comparablePairs.length
    ? comparablePairs.filter(({ fit }) => fit.falseFit).length / comparablePairs.length * 100
    : null;
  const competitorFalseFitRatePct = comparablePairs.length
    ? comparablePairs.filter(({ competitor }) => competitor.falseFit).length / comparablePairs.length * 100
    : null;
  const relativeMapeImprovementPct = competitorMapePct > 0 && pairedFitMapePct != null
    ? (competitorMapePct - pairedFitMapePct) / competitorMapePct * 100
    : null;
  const counts = {
    measurements: paired.length,
    runtimes: runtimes.size,
    architectureFamilies: families.size,
    competitors: selectedCompetitor && paired.length ? 1 : 0,
  };
  const allowed = counts.measurements >= MINIMUMS.measurements
    && counts.runtimes >= MINIMUMS.runtimes
    && counts.architectureFamilies >= MINIMUMS.architectureFamilies
    && counts.competitors >= MINIMUMS.competitors
    && relativeMapeImprovementPct >= MINIMUMS.relativeMapeImprovementPct
    && fitFalseFitRatePct != null
    && competitorFalseFitRatePct != null
    && fitFalseFitRatePct < competitorFalseFitRatePct;
  return {
    included,
    excluded,
    metrics: { mapePct, pairedFitMapePct, competitorMapePct, relativeMapeImprovementPct, fitFalseFitRatePct, competitorFalseFitRatePct },
    claimGate: {
      allowed,
      minimums: MINIMUMS,
      actual: counts,
      competitor: selectedCompetitor?.[0] || null,
      reason: allowed ? 'evidence thresholds passed' : 'insufficient independently verified, typed, comparable evidence',
    },
  };
}

export function renderMarkdown(report) {
  const fmt = (value) => value == null ? 'n/a' : `${value.toFixed(2)}%`;
  const countBy = (rows, key) => [...rows.reduce((counts, row) => {
    const value = row[key] || 'missing';
    counts.set(value, (counts.get(value) || 0) + 1);
    return counts;
  }, new Map())].map(([name, count]) => `${name}=${count}`).join(', ') || 'none';
  return [
    '# FitLLM accuracy report',
    '',
    `- Typed comparable measurements: ${report.included.length}`,
    `- Measurement kinds: ${countBy(report.included, 'measurementKind')}`,
    `- Evidence levels: ${countBy(report.included, 'evidenceLevel')}`,
    `- Excluded claims: ${report.excluded.length}`,
    `- Descriptive FitLLM MAPE (all typed rows; not a comparative claim): ${fmt(report.metrics.mapePct)}`,
    `- Verified paired claim cases: ${report.claimGate.actual.measurements} / ${report.claimGate.minimums.measurements}`,
    `- Public comparative claim gate: ${report.claimGate.allowed ? 'PASS' : 'BLOCKED'}`,
    `- Reason: ${report.claimGate.reason}`,
  ].join('\n');
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const json = process.argv.includes('--json');
  const census = JSON.parse(readFileSync(new URL('../census/census-v1.json', import.meta.url), 'utf8'));
  const measurements = JSON.parse(readFileSync(new URL('../fixtures/measured.json', import.meta.url), 'utf8'));
  const competitors = JSON.parse(readFileSync(new URL('../benchmarks/competitors.json', import.meta.url), 'utf8'));
  const report = buildAccuracyReport(census, measurements, competitors);
  console.log(json ? JSON.stringify(report, null, 2) : renderMarkdown(report));
}
