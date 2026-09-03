import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { appendFileSync, cpSync, mkdtempSync, readFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';
import test from 'node:test';

import { buildDifferentialReport } from '../scripts/differential-report.mjs';

const ROOT = new URL('../benchmarks/llmfit-v1.1.12/', import.meta.url);
const sha256 = (value) => createHash('sha256').update(value).digest('hex');

test('manifest pins the official release checksum and three uncut raw outputs', () => {
  const manifest = JSON.parse(readFileSync(new URL('manifest.json', ROOT), 'utf8'));
  assert.equal(manifest.competitor.binaryVersion, 'llmfit 1.1.12');
  assert.equal(manifest.competitor.releaseTag, 'v1.1.12');
  assert.equal(manifest.competitor.artifactSha256, '6a97338862c87e497c844ccd29a16512a147335631c179744b4f6cc87a36ead1');
  assert.match(manifest.competitor.artifactUrl, /releases\/download\/v1\.1\.12\/llmfit-v1\.1\.12-x86_64-unknown-linux-gnu\.tar\.gz$/);
  assert.equal(manifest.cases.length, 3);

  const expectedKv = new Map([
    ['llama31-8b-gqa', 1],
    ['gemma4-31b-sliding-window', 7.5],
    ['glm47-flash-mla', 2.926025390625],
  ]);
  for (const entry of manifest.cases) {
    const raw = readFileSync(new URL(entry.stdoutFile, ROOT), 'utf8');
    assert.equal(sha256(raw), entry.stdoutSha256);
    const parsed = JSON.parse(raw);
    assert.equal(parsed.context, 8192);
    assert.equal(parsed.quantization, 'Q4_K_M');
    assert.equal(parsed.kv_alternatives.find((row) => row.kv_quant === 'fp16').kv_cache_gb, expectedKv.get(entry.id));
    assert.match(raw, /estimate_notice/);
  }
});

test('differential report separates the GQA control from architecture counterexamples', () => {
  const report = buildDifferentialReport();
  assert.equal(report.evidenceClass, 'architecture_differential_not_runtime_accuracy');
  assert.equal(report.claimEligible, false);
  assert.equal(report.rows.length, 3);
  assert.equal(report.rows.find((row) => row.id === 'llama31-8b-gqa').role, 'control');
  assert.equal(report.rows.find((row) => row.id === 'llama31-8b-gqa').deltaGiB, 0);
  for (const id of ['gemma4-31b-sliding-window', 'glm47-flash-mla']) {
    const row = report.rows.find((candidate) => candidate.id === id);
    assert.equal(row.role, 'counterexample');
    assert.ok(row.competitorKvCacheGiB > row.fitllmKvCacheGiB);
  }
});

test('differential report rejects raw output that no longer matches its manifest hash', () => {
  const temp = mkdtempSync(join(tmpdir(), 'fitllm-differential-'));
  cpSync(ROOT, temp, { recursive: true });
  appendFileSync(join(temp, 'llama31-8b-gqa.json'), '\n');
  assert.throws(
    () => buildDifferentialReport(pathToFileURL(`${temp}/`)),
    /SHA-256 mismatch.*llama31-8b-gqa/i,
  );
});
