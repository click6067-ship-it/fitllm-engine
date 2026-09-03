import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';

import { calcKVCache, LOCAL_MODELS } from '../engine.js';

const DEFAULT_DIR = new URL('../benchmarks/llmfit-v1.1.12/', import.meta.url);
const sha256 = (value) => createHash('sha256').update(value).digest('hex');

export function buildDifferentialReport(benchmarkDir = DEFAULT_DIR) {
  const manifest = JSON.parse(readFileSync(new URL('manifest.json', benchmarkDir), 'utf8'));
  const rows = manifest.cases.map((entry) => {
    const rawText = readFileSync(new URL(entry.stdoutFile, benchmarkDir), 'utf8');
    if (sha256(rawText) !== entry.stdoutSha256) {
      throw new Error(`SHA-256 mismatch for ${entry.id}: ${entry.stdoutFile}`);
    }
    const raw = JSON.parse(rawText);
    const model = LOCAL_MODELS.find((candidate) => candidate.name === entry.fitllmModel);
    if (!model) throw new Error(`FitLLM model missing: ${entry.fitllmModel}`);
    const competitorKvCacheGiB = raw.kv_alternatives.find((row) => row.kv_quant === 'fp16')?.kv_cache_gb;
    if (!Number.isFinite(competitorKvCacheGiB)) throw new Error(`llmfit fp16 KV value missing: ${entry.id}`);
    const fitllmKvCacheGiB = calcKVCache(model, 8192, 16).totalGB;
    return {
      id: entry.id,
      role: entry.role,
      architecture: entry.architecture,
      contextTokens: 8192,
      kvBits: 16,
      competitor: `${manifest.competitor.repository}@${manifest.competitor.releaseTag}`,
      competitorKvCacheGiB,
      fitllmKvCacheGiB,
      deltaGiB: competitorKvCacheGiB - fitllmKvCacheGiB,
      sourceOutput: entry.stdoutFile,
      sourceOutputSha256: entry.stdoutSha256,
    };
  });
  return {
    schemaVersion: 1,
    evidenceClass: 'architecture_differential_not_runtime_accuracy',
    claimEligible: false,
    limitation: 'These are estimator-to-estimator architecture differentials, not measurements of runtime accuracy.',
    rows,
  };
}

export function renderDifferentialMarkdown(report) {
  return [
    '# Architecture differential: FitLLM vs llmfit v1.1.12',
    '',
    `Evidence class: \`${report.evidenceClass}\` (public accuracy claim eligible: no).`,
    '',
    '| Case | Role | Architecture | llmfit FP16 KV GiB | FitLLM FP16 KV GiB | Delta GiB |',
    '|---|---|---:|---:|---:|---:|',
    ...report.rows.map((row) => `| ${row.id} | ${row.role} | ${row.architecture} | ${row.competitorKvCacheGiB} | ${row.fitllmKvCacheGiB} | ${row.deltaGiB} |`),
    '',
    report.limitation,
  ].join('\n');
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const report = buildDifferentialReport();
  console.log(process.argv.includes('--json') ? JSON.stringify(report, null, 2) : renderDifferentialMarkdown(report));
}
