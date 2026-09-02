import { test } from 'node:test';
import { strict as assert } from 'node:assert';
import { createHash } from 'node:crypto';
import { mkdtemp, readFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { parseHfConfig } from '../engine.js';
import {
  applyIssuePlan,
  attachArtifactRef,
  buildEvidenceManifest,
  buildIssueBotBlock,
  canonicalCandidateId,
  classifyCapability,
  detectCapabilityBlockers,
  discoverCandidates,
  fetchJsonWithRetry,
  loadSourcePolicy,
  mergeIssueBotBlock,
  pinEvidence,
  planIssueMutations,
  runDay0Watch,
  sha256CanonicalManifest,
} from '../.github/scripts/day0-core.mjs';

const FIXTURES = new URL('./fixtures/day0/', import.meta.url);
const REVISION = 'de4b8e4d43b917e7706784d8bb445c9af86a3540';
const CONFIG_SHA = '889658f2508e8c61d409b02e70e0d78d8d4452ec65aaafbe129805d213d2e74b';
const INDEX_SHA = '99e815241ef03325536b0aaa4441deea45174c17fae31e10f0bb456410c590de';

async function fixture(name, encoding = 'utf8') {
  return readFile(new URL(name, FIXTURES), encoding);
}

function jsonResponse(value, { status = 200, headers = {} } = {}) {
  return new Response(JSON.stringify(value), {
    status,
    headers: { 'content-type': 'application/json', ...headers },
  });
}

async function sourcePolicy() {
  return loadSourcePolicy(await readFile(new URL('../.github/day0-sources.json', import.meta.url), 'utf8'));
}

test('공식 namespace release와 global trend를 합치고 pipeline을 로컬 exact filter한다', async () => {
  const feeds = JSON.parse(await fixture('hf-qwen-models.json'));
  const policy = await sourcePolicy();
  const fetchImpl = async (url) => {
    const parsed = new URL(url);
    if (parsed.searchParams.get('author') === 'Qwen') return jsonResponse(feeds.release);
    if (parsed.searchParams.has('author')) return jsonResponse([]);
    return jsonResponse(feeds.trending);
  };

  const result = await discoverCandidates({ policy, fetchImpl, now: new Date('2026-09-03T00:00:00Z') });
  assert.deepEqual(result.candidates.map((candidate) => candidate.id), [
    'Qwen/Qwen3.8-Flash-Next',
    'Qwen/Qwen-Unclassified',
    'Qwen/Qwen3.8-Flash-Next-FP8',
  ]);
  assert.deepEqual(result.candidates[0].discoverySources, [
    'hf_official_namespace_release',
    'hf_official_namespace_trending',
  ]);
  assert.equal(result.candidates[2].checkpointKind, 'quantized_variant');
  assert.ok(!result.candidates.some(({ id }) => id.startsWith('AtomicChat/')));
  assert.ok(!result.candidates.some(({ pipelineTag }) => pipelineTag === 'text-to-speech'));
});

test('candidate identity는 정확히 namespace/repo 두 segment만 허용한다', () => {
  assert.equal(canonicalCandidateId({ id: 'Qwen/Qwen3.8-Flash-Next' }), 'Qwen/Qwen3.8-Flash-Next');
  for (const id of ['Qwen', 'Qwen/a/b', 'Qwen/<script>', 'Qwen//model', '/model']) {
    assert.throws(() => canonicalCandidateId({ id }), /candidate id/i);
  }
});

test('fetchJsonWithRetry는 non-JSON을 재시도하고 JSON array를 반환한다', async () => {
  let calls = 0;
  const fetchImpl = async () => {
    calls += 1;
    return calls === 1
      ? new Response('<html>wait</html>', { headers: { 'content-type': 'text/html' } })
      : jsonResponse([{ id: 'Qwen/ok' }]);
  };
  const value = await fetchJsonWithRetry('https://example.test/feed', {
    fetchImpl,
    attempts: 2,
    timeoutMs: 100,
    retryDelayMs: 0,
  });
  assert.equal(calls, 2);
  assert.equal(value[0].id, 'Qwen/ok');
});

test('Qwen evidence는 revision URL, raw bytes, SHA-256, total_size를 고정한다', async () => {
  const [model, config, index, license] = await Promise.all([
    fixture('qwen3.8-flash-next.model.json').then(JSON.parse),
    fixture('qwen3.8-flash-next.config.json', null),
    fixture('qwen3.8-flash-next.index.json', null),
    fixture('qwen3.8-flash-next.LICENSE', null),
  ]);
  const fetchImpl = async (url) => {
    const name = new URL(url).pathname.split('/').at(-1);
    const bytes = name === 'config.json' ? config : name === 'model.safetensors.index.json' ? index : license;
    return new Response(bytes, {
      headers: {
        'content-type': 'text/plain; charset=utf-8',
        'x-repo-commit': REVISION,
        etag: name === 'config.json' ? 'W/"491017e9980e44ef01afa2c4782f5c7e169b9b26"' : 'W/"sealed"',
      },
    });
  };
  const evidence = await pinEvidence({
    id: model.id,
    namespace: 'Qwen',
    identityEvidenceUrl: 'https://github.com/QwenLM',
    revision: model.sha,
    pipelineTag: model.pipeline_tag,
    createdAt: model.createdAt,
    discoverySources: ['hf_official_namespace_release'],
    modelInfo: model,
  }, { fetchImpl });

  assert.equal(evidence.lifecycleState, 'EVIDENCE_PINNED');
  assert.equal(evidence.config.bytes, 4745);
  assert.equal(evidence.config.sha256, CONFIG_SHA);
  assert.equal(evidence.weightsIndex.bytes, 170726);
  assert.equal(evidence.weightsIndex.sha256, INDEX_SHA);
  assert.equal(evidence.weightsIndex.totalSizeBytes, 359999963128);
  assert.equal(evidence.license.id, 'qwen-community-1.0');
  assert.equal(evidence.license.bytes, 3235);
  for (const slot of [evidence.config, evidence.weightsIndex, evidence.license]) {
    assert.match(slot.url, new RegExp(`/resolve/${REVISION}/`));
    assert.ok(!slot.url.includes('/main/'));
  }
});

test('identity/evidence failure는 typed numeric-null manifest로 남고 parser를 호출하지 않는다', async () => {
  const invalid = await pinEvidence({
    id: 'Qwen/Qwen3.8-Flash-Next', namespace: 'Qwen', revision: 'main',
    pipelineTag: 'image-text-to-text', discoverySources: [], modelInfo: {},
  }, { fetchImpl: () => assert.fail('invalid identity must not fetch') });
  assert.equal(invalid.lifecycleState, 'UNVERIFIED_IDENTITY');

  const changed = await pinEvidence({
    id: 'Qwen/Qwen3.8-Flash-Next', namespace: 'Qwen', revision: REVISION,
    identityEvidenceUrl: 'https://github.com/QwenLM',
    pipelineTag: 'image-text-to-text', discoverySources: [],
    modelInfo: {
      cardData: { license: 'other' },
      siblings: [
        { rfilename: 'config.json' },
        { rfilename: 'model.safetensors.index.json' },
        { rfilename: 'LICENSE' },
      ],
    },
  }, {
    fetchImpl: async () => new Response('{}', {
      headers: { 'content-type': 'application/json', 'x-repo-commit': 'f'.repeat(40), etag: '"x"' },
    }),
  });
  assert.equal(changed.lifecycleState, 'EVIDENCE_CHANGED');
  const manifest = buildEvidenceManifest(changed, null, 'capability-v1');
  assert.equal(manifest.capability, null);
  assert.deepEqual(manifest.failureCodes, ['REVISION_HEADER_MISMATCH']);
  assert.ok(!('verdict' in manifest));
  assert.ok(!('usedGB' in manifest));
});

test('Qwen official config는 five blocker와 numeric-null unsupported로 분류한다', async () => {
  const rawConfig = JSON.parse(await fixture('qwen3.8-flash-next.config.json'));
  const blockers = detectCapabilityBlockers(rawConfig);
  assert.deepEqual(blockers, {
    state: 'UNSUPPORTED_ARCHITECTURE',
    unsupportedComponents: [
      'QSA_INDEXER',
      'NGRAM_RESIDENCY',
      'MTP_RESIDENCY',
      'HC_ACTIVATION',
      'MULTIMODAL_WORKLOAD',
    ],
    numericResult: null,
  });
  assert.equal(createHash('sha256').update(await fixture('qwen3.8-flash-next.config.json', null)).digest('hex'), CONFIG_SHA);
  const capability = classifyCapability({
    lifecycleState: 'EVIDENCE_PINNED', pipelineTag: 'image-text-to-text', rawConfig,
    weightsIndex: { totalSizeBytes: 359999963128 }, id: 'Qwen/Qwen3.8-Flash-Next',
  });
  assert.deepEqual(capability, blockers);
  assert.throws(
    () => parseHfConfig('Qwen/Qwen3.8-Flash-Next', rawConfig, 359999963128),
    /구조를 알 수 없는|linear\/recurrent/,
  );
  for (const key of ['verdict', 'usedGB', 'maxContext']) assert.ok(!(key in capability));
});

test('parser 성공은 supported이되 AWAITING_GOLDEN_VECTOR에서 멈춘다', () => {
  const capability = classifyCapability({
    lifecycleState: 'EVIDENCE_PINNED',
    id: 'meta-llama/Llama-test',
    pipelineTag: 'text-generation',
    rawConfig: {
      model_type: 'llama', num_hidden_layers: 32, num_attention_heads: 32,
      num_key_value_heads: 8, hidden_size: 4096, head_dim: 128,
      intermediate_size: 14336, vocab_size: 128256, torch_dtype: 'bfloat16',
    },
    weightsIndex: { totalSizeBytes: 16e9 },
  });
  assert.equal(capability.state, 'SUPPORTED_BY_CURRENT_ENGINE');
  assert.equal(capability.lifecycleState, 'AWAITING_GOLDEN_VECTOR');
  assert.equal(capability.numericResult, null);
  assert.ok(!('verdict' in capability));
});

test('pipeline null과 generic parser rejection은 CAPABILITY_UNKNOWN이지 unsupported가 아니다', () => {
  const unknownWorkload = classifyCapability({ lifecycleState: 'EVIDENCE_PINNED', pipelineTag: null });
  assert.equal(unknownWorkload.state, 'CAPABILITY_UNKNOWN');
  const parserRejected = classifyCapability({
    lifecycleState: 'EVIDENCE_PINNED', id: 'Qwen/unknown', pipelineTag: 'text-generation',
    rawConfig: { model_type: 'future_attention' }, weightsIndex: { totalSizeBytes: 100 },
  });
  assert.equal(parserRejected.state, 'CAPABILITY_UNKNOWN');
  assert.equal(parserRejected.numericResult, null);
});

function manifestFor(revision = REVISION, lifecycleState = 'CAPABILITY_UNKNOWN') {
  const evidence = {
    lifecycleState: 'EVIDENCE_PINNED', id: 'Qwen/Qwen3.8-Flash-Next', revision,
    identityEvidenceUrl: 'https://github.com/QwenLM', pipelineTag: 'image-text-to-text',
    createdAt: '2026-08-24T08:24:59.000Z', discoverySources: ['hf_official_namespace_release'],
    config: { url: `https://huggingface.co/x/resolve/${revision}/config.json`, bytes: 1, etag: '"c"', sha256: 'a'.repeat(64) },
    weightsIndex: { url: `https://huggingface.co/x/resolve/${revision}/model.safetensors.index.json`, bytes: 2, etag: '"i"', sha256: 'b'.repeat(64), totalSizeBytes: 3 },
    license: { id: 'other', url: `https://huggingface.co/x/resolve/${revision}/LICENSE`, bytes: 4, etag: '"l"', sha256: 'c'.repeat(64) },
    failureCodes: [],
  };
  return buildEvidenceManifest(evidence, { state: lifecycleState, numericResult: null }, 'capability-v1');
}

test('manifest digest는 결정적이고 create/noop/update는 human text를 보존한다', async () => {
  const issues = JSON.parse(await fixture('github-issues.json'));
  const manifest = manifestFor();
  const digest = sha256CanonicalManifest(manifest);
  assert.equal(digest, sha256CanonicalManifest(structuredClone(manifest)));
  const block = buildIssueBotBlock(manifest, { url: 'https://artifact.test/1', digest: 'd'.repeat(64) });
  const merged = mergeIssueBotBlock(issues[0].body, block);
  assert.ok(merged.startsWith('Human triage notes. Keep this exact.'));
  assert.ok(merged.endsWith('Human footer.'));

  const create = planIssueMutations([{ manifest }], []);
  assert.equal(create.mutationCount, 1);
  assert.equal(create.operations[0].action, 'create');

  const sameIssue = [{ ...issues[0], body: `human\n\n${buildIssueBotBlock(manifest, null)}\n\nfooter` }];
  const noop = planIssueMutations([{ manifest }], sameIssue);
  assert.equal(noop.mutationCount, 0);
  assert.equal(noop.operations[0].action, 'noop');

  const nextManifest = manifestFor('e'.repeat(40));
  const update = planIssueMutations([{ manifest: nextManifest }], sameIssue);
  assert.equal(update.operations[0].action, 'update');
  assert.ok(update.operations[0].body.startsWith('human'));
  assert.ok(update.operations[0].body.endsWith('footer'));
  assert.equal(update.operations[0].state, undefined);
});

test('mutation은 최대 3개이며 dropped candidate가 결정적으로 기록된다', () => {
  const records = ['A', 'B', 'C', 'D'].map((name, index) => ({
    manifest: {
      ...manifestFor(String(index + 1).repeat(40)),
      candidateId: `Qwen/${name}@${String(index + 1).repeat(40)}`,
      idempotencyKey: `Qwen/${name}@${String(index + 1).repeat(40)}#capability-v1`,
      officialModelUrl: `https://huggingface.co/Qwen/${name}`,
    },
  }));
  const plan = planIssueMutations(records, [], { maxMutations: 3 });
  assert.equal(plan.mutationCount, 3);
  assert.deepEqual(plan.dropped, [{ candidateId: `Qwen/D@${'4'.repeat(40)}`, reason: 'MUTATION_LIMIT' }]);
});

test('apply-time body/revision conflict는 PATCH 없이 중단된다', async () => {
  const manifest = manifestFor('e'.repeat(40));
  const existing = { number: 77, title: 'day0: Qwen/Qwen3.8-Flash-Next', body: 'old human body' };
  const plan = planIssueMutations([{ manifest }], [existing]);
  const calls = [];
  const ghClient = {
    request: async (path, init = {}) => {
      calls.push([path, init.method || 'GET']);
      if ((init.method || 'GET') === 'GET') return { ...existing, body: 'human edited after planning' };
      assert.fail('conflict must not mutate');
    },
  };
  const result = await applyIssuePlan(plan, ghClient, async () => assert.fail('body conflict comes first'));
  assert.equal(result.results[0].status, 'BODY_CHANGED');
  assert.equal(calls.filter(([, method]) => method === 'PATCH').length, 0);
});

test('apply-time stale HF revision은 POST/PATCH를 막는다', async () => {
  const manifest = manifestFor();
  const plan = planIssueMutations([{ manifest }], []);
  const mutations = [];
  const ghClient = { request: async (path, init = {}) => mutations.push([path, init.method]) };
  const result = await applyIssuePlan(plan, ghClient, async () => ({ sha: 'f'.repeat(40) }));
  assert.equal(result.results[0].status, 'STALE_REVISION');
  assert.equal(mutations.length, 0);
});

test('explicit apply create는 pinned revision 재검사 뒤 issue endpoint만 쓴다', async () => {
  const manifest = manifestFor();
  const plan = attachArtifactRef(planIssueMutations([{ manifest }], []), {
    url: 'https://github.com/example/repo/actions/runs/1/artifacts/2',
    digest: 'a'.repeat(64),
  });
  const calls = [];
  const ghClient = {
    request: async (requestPath, init = {}) => {
      calls.push([requestPath, init.method || 'GET', init.body]);
      if (requestPath === '/issues') return { number: 91 };
      return { name: 'day0-candidate' };
    },
  };
  const result = await applyIssuePlan(plan, ghClient, async (modelId) => ({ id: modelId, sha: REVISION }));
  assert.equal(result.mutationCount, 1);
  assert.deepEqual(calls.map(([requestPath, method]) => [requestPath, method]), [
    ['/labels/day0-candidate', 'GET'],
    ['/issues', 'POST'],
  ]);
  assert.deepEqual(Object.keys(calls[1][2]).sort(), ['body', 'labels', 'title']);
});

test('runDay0Watch dry-run은 GET 외 GitHub 호출 없이 temp evidence만 쓴다', async () => {
  const feeds = JSON.parse(await fixture('hf-qwen-models.json'));
  const [config, index, license] = await Promise.all([
    fixture('qwen3.8-flash-next.config.json', null),
    fixture('qwen3.8-flash-next.index.json', null),
    fixture('qwen3.8-flash-next.LICENSE', null),
  ]);
  const fetchImpl = async (url) => {
    const parsed = new URL(url);
    if (parsed.pathname === '/api/models') {
      if (parsed.searchParams.get('author') === 'Qwen') return jsonResponse(feeds.release);
      if (parsed.searchParams.has('author')) return jsonResponse([]);
      return jsonResponse(feeds.trending);
    }
    const name = parsed.pathname.split('/').at(-1);
    const bytes = name === 'config.json' ? config : name === 'model.safetensors.index.json' ? index : license;
    return new Response(bytes, { headers: { 'x-repo-commit': REVISION, etag: '"fixture"' } });
  };
  const githubCalls = [];
  const ghClient = {
    requestWithHeaders: async (requestPath) => {
      githubCalls.push(['GET', requestPath]);
      return { data: [], headers: new Headers() };
    },
    request: async () => assert.fail('dry-run must not call a GitHub mutation method'),
  };
  const outputDir = await mkdtemp(path.join(tmpdir(), 'fitllm-day0-test-'));
  const result = await runDay0Watch({
    policy: await sourcePolicy(), fetchImpl, ghClient, now: new Date('2026-09-03T00:00:00Z'),
  }, { outputDir, sourceRoot: new URL('..', import.meta.url).pathname, mode: 'dry-run' });
  assert.equal(result.summary.mode, 'dry-run');
  assert.equal(result.records[0].manifest.lifecycleState, 'UNSUPPORTED_ARCHITECTURE');
  assert.deepEqual(githubCalls.map(([method]) => method), ['GET']);
  assert.equal(JSON.parse(await readFile(path.join(outputDir, 'summary.json'), 'utf8')).mode, 'dry-run');
});
