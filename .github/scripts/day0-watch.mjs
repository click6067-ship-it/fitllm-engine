#!/usr/bin/env node
// Thin CLI/env adapter. Discovery, evidence, classification and mutation planning live in day0-core.mjs.
import { appendFile, mkdtemp, readFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  applyIssuePlan,
  attachArtifactRef,
  fetchJsonWithRetry,
  loadSourcePolicy,
  runDay0Watch,
} from './day0-core.mjs';

const SOURCE_ROOT = fileURLToPath(new URL('../..', import.meta.url));
const REPO = process.env.GITHUB_REPOSITORY || 'click6067-ship-it/fitllm-engine';
const TOKEN = process.env.GH_TOKEN || process.env.GITHUB_TOKEN || null;
const args = new Set(process.argv.slice(2));

function valueFor(name) {
  const prefix = `${name}=`;
  const inline = [...args].find((arg) => arg.startsWith(prefix));
  if (inline) return inline.slice(prefix.length);
  const index = process.argv.indexOf(name);
  return index >= 0 ? process.argv[index + 1] : null;
}

function githubClient() {
  const base = `https://api.github.com/repos/${REPO}`;
  const call = async (relativePath, init = {}) => {
    const method = init.method || 'GET';
    if (!['GET', 'POST', 'PATCH'].includes(method)) throw new Error(`GitHub method not allowed: ${method}`);
    if (method !== 'GET'
        && !((method === 'POST' && ['/issues', '/labels'].includes(relativePath))
          || (method === 'PATCH' && /^\/issues\/\d+$/.test(relativePath)))) {
      throw new Error(`GitHub mutation endpoint not allowed: ${method} ${relativePath}`);
    }
    const headers = {
      Accept: 'application/vnd.github+json',
      'X-GitHub-Api-Version': '2022-11-28',
      'User-Agent': 'fitllm-day0-watch',
      ...(TOKEN ? { Authorization: `Bearer ${TOKEN}` } : {}),
    };
    if (init.body !== undefined) headers['Content-Type'] = 'application/json';
    const response = await fetch(`${base}${relativePath}`, {
      method,
      headers,
      body: init.body === undefined ? undefined : JSON.stringify(init.body),
    });
    if (!response.ok) {
      const error = new Error(`GitHub API ${method} ${relativePath} returned ${response.status}`);
      error.status = response.status;
      throw error;
    }
    const data = response.status === 204 ? null : await response.json();
    return { data, headers: response.headers };
  };
  return {
    request: async (relativePath, init) => (await call(relativePath, init)).data,
    requestWithHeaders: call,
  };
}

async function currentHfModel(modelId) {
  return fetchJsonWithRetry(`https://huggingface.co/api/models/${modelId}`, { fetchImpl: fetch });
}

async function writeGithubOutput(name, value) {
  if (!process.env.GITHUB_OUTPUT) return;
  await appendFile(process.env.GITHUB_OUTPUT, `${name}=${value}\n`, 'utf8');
}

async function main() {
  const apply = args.has('--apply-issues');
  if (apply) {
    if (process.env.GITHUB_EVENT_NAME !== 'workflow_dispatch' || process.env.DAY0_MODE !== 'apply-issues') {
      throw new Error('issue apply requires explicit workflow_dispatch mode=apply-issues');
    }
    const planPath = valueFor('--plan');
    const artifactUrl = process.env.DAY0_ARTIFACT_URL;
    const artifactDigest = process.env.DAY0_ARTIFACT_DIGEST;
    if (!planPath || !artifactUrl || !artifactDigest) throw new Error('apply requires plan and uploaded artifact URL/digest');
    const rawPlan = JSON.parse(await readFile(planPath, 'utf8'));
    const plan = attachArtifactRef(rawPlan, { url: artifactUrl, digest: artifactDigest });
    const result = await applyIssuePlan(plan, githubClient(), currentHfModel);
    console.log(JSON.stringify({ mode: 'apply-issues', ...result }, null, 2));
    return;
  }

  const policy = loadSourcePolicy(await readFile(path.join(SOURCE_ROOT, '.github/day0-sources.json'), 'utf8'));
  const outputDir = valueFor('--output-dir') || await mkdtemp(path.join(tmpdir(), 'fitllm-day0-'));
  const mode = args.has('--plan') ? 'plan' : 'dry-run';
  const result = await runDay0Watch({ policy, fetchImpl: fetch, ghClient: githubClient() }, {
    outputDir,
    sourceRoot: SOURCE_ROOT,
    mode,
    githubRunId: process.env.GITHUB_RUN_ID || null,
    githubRunUrl: process.env.GITHUB_RUN_ID && process.env.GITHUB_SERVER_URL && process.env.GITHUB_REPOSITORY
      ? `${process.env.GITHUB_SERVER_URL}/${process.env.GITHUB_REPOSITORY}/actions/runs/${process.env.GITHUB_RUN_ID}`
      : null,
    verifierGitSha: process.env.GITHUB_SHA || null,
  });
  await writeGithubOutput('mutations', result.issuePlan.mutationCount);
  await writeGithubOutput('output-dir', result.outputDir);
  console.log(JSON.stringify({ ...result.summary, outputDir: result.outputDir }, null, 2));
}

main().catch((error) => {
  console.error(`day0-watch: ${error.message}`);
  process.exitCode = 1;
});
