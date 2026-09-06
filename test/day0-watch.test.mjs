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
  fetchAllGitHubIssues,
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
const TRUSTED_ISSUE_AUTHORS = ['click6067-ship-it', 'github-actions[bot]'];

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

function planMutations(records, issues, options = {}) {
  return planIssueMutations(records, issues, { ...options, trustedIssueAuthors: TRUSTED_ISSUE_AUTHORS });
}

test('source policy는 explicit unique trusted issue author allowlist를 요구한다', async () => {
  const raw = JSON.parse(await readFile(new URL('../.github/day0-sources.json', import.meta.url), 'utf8'));
  assert.throws(() => loadSourcePolicy({ ...raw, trustedIssueAuthors: undefined }), /trustedIssueAuthors/);
  assert.throws(
    () => loadSourcePolicy({ ...raw, trustedIssueAuthors: ['github-actions[bot]', 'GITHUB-ACTIONS[BOT]'] }),
    /duplicate trusted issue author/,
  );
});

test('공식 namespace release와 global trend를 합치고 pipeline을 로컬 exact filter한다', async () => {
  const feeds = JSON.parse(await fixture('hf-qwen-models.json'));
  const policy = await sourcePolicy();
  assert.deepEqual(policy.trustedIssueAuthors, TRUSTED_ISSUE_AUTHORS);
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
  assert.ok(!result.candidates.some(({ id }) => /(?:BF16|F16)$/.test(id)));
  assert.deepEqual(
    result.droppedCandidates
      .filter(({ reason }) => reason === 'FULL_PRECISION_VARIANT_EXCLUDED')
      .map(({ candidateId }) => candidateId)
      .sort(),
    ['Qwen/Qwen3.8-Flash-Next-BF16', 'Qwen/Qwen3.8-Flash-Next-F16', 'Qwen/Qwen-Trend-BF16'].sort(),
  );
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

test('discovery fetch는 timeout과 content-length/stream byte cap을 강제한다', async () => {
  await assert.rejects(
    fetchJsonWithRetry('https://example.test/timeout', {
      attempts: 1,
      timeoutMs: 5,
      fetchImpl: async () => new Promise(() => {}),
    }),
    /failed after 1 attempts/,
  );
  await assert.rejects(
    fetchJsonWithRetry('https://example.test/declared-too-large', {
      attempts: 1,
      maxBytes: 16,
      fetchImpl: async () => new Response('[]', {
        headers: { 'content-type': 'application/json', 'content-length': '17' },
      }),
    }),
    /byte limit/i,
  );
  await assert.rejects(
    fetchJsonWithRetry('https://example.test/stream-too-large', {
      attempts: 1,
      maxBytes: 16,
      fetchImpl: async () => new Response(JSON.stringify({ value: 'x'.repeat(32) }), {
        headers: { 'content-type': 'application/json' },
      }),
    }),
    /byte limit/i,
  );
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

test('evidence fetch는 content-length byte cap을 body read 전에 차단한다', async () => {
  let bodyReads = 0;
  const candidate = {
    id: 'Qwen/oversized', namespace: 'Qwen', identityEvidenceUrl: 'https://github.com/QwenLM',
    revision: REVISION, pipelineTag: 'text-generation', discoverySources: ['hf_official_namespace_release'],
    modelInfo: {
      id: 'Qwen/oversized', sha: REVISION, cardData: { license: 'mit' },
      siblings: [
        { rfilename: 'config.json' },
        { rfilename: 'model.safetensors.index.json' },
        { rfilename: 'LICENSE' },
      ],
    },
  };
  const oversized = {
    ok: true,
    status: 200,
    headers: new Headers({
      'content-length': String(2 * 1024 * 1024),
      'x-repo-commit': REVISION,
      etag: '"oversized"',
    }),
    arrayBuffer: async () => { bodyReads += 1; return new ArrayBuffer(0); },
  };
  const evidence = await pinEvidence(candidate, {
    fetchImpl: async (url) => {
      if (url.endsWith('/config.json')) return oversized;
      if (url.endsWith('/model.safetensors.index.json')) {
        return new Response(JSON.stringify({ metadata: { total_size: 1_000_000 } }), {
          headers: { 'x-repo-commit': REVISION, etag: '"index"' },
        });
      }
      return new Response('MIT', { headers: { 'x-repo-commit': REVISION, etag: '"license"' } });
    },
  });
  assert.equal(evidence.lifecycleState, 'INSUFFICIENT_EVIDENCE');
  assert.ok(evidence.failureCodes.includes('CONFIG_PAYLOAD_TOO_LARGE'));
  assert.equal(bodyReads, 0);
});

test('evidence fetch timeout은 non-cooperative fetch도 bounded SOURCE_UNAVAILABLE로 남긴다', async () => {
  const evidence = await pinEvidence({
    id: 'Qwen/timeout', namespace: 'Qwen', identityEvidenceUrl: 'https://github.com/QwenLM',
    revision: REVISION, pipelineTag: 'text-generation', discoverySources: ['hf_official_namespace_release'],
    modelInfo: {
      id: 'Qwen/timeout', sha: REVISION, cardData: { license: 'mit' },
      siblings: ['config.json', 'model.safetensors.index.json', 'LICENSE'].map((rfilename) => ({ rfilename })),
    },
  }, { fetchImpl: async () => new Promise(() => {}), timeoutMs: 5 });
  assert.equal(evidence.lifecycleState, 'SOURCE_UNAVAILABLE');
  assert.ok(evidence.failureCodes.includes('CONFIG_TIMEOUT'));
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

test('inactive blocker fields는 supported 가능하고 ambiguous presence만 CAPABILITY_UNKNOWN이다', () => {
  const base = {
    model_type: 'llama', num_hidden_layers: 2, num_attention_heads: 2,
    num_key_value_heads: 2, hidden_size: 128, head_dim: 64,
    intermediate_size: 256, vocab_size: 1024, torch_dtype: 'bfloat16',
  };
  const inactiveConfigs = [
    { ...base, mtp: false },
    { ...base, mtp_num_hidden_layers: 0 },
    { ...base, hc_count: 0 },
  ];
  const ambiguousConfigs = [
    { ...base, indexer_budget: null },
    { text_config: base, vision_config: {} },
  ];
  for (const rawConfig of inactiveConfigs) {
    const capability = classifyCapability({
      lifecycleState: 'EVIDENCE_PINNED', id: 'Qwen/inactive', pipelineTag: 'text-generation',
      rawConfig, weightsIndex: { totalSizeBytes: 1_000_000 },
    });
    assert.notEqual(capability.state, 'UNSUPPORTED_ARCHITECTURE', JSON.stringify(rawConfig));
    assert.equal(capability.numericResult, null);
  }
  for (const rawConfig of ambiguousConfigs) {
    const capability = classifyCapability({
      lifecycleState: 'EVIDENCE_PINNED', id: 'Qwen/inactive', pipelineTag: 'text-generation',
      rawConfig, weightsIndex: { totalSizeBytes: 1_000_000 },
    });
    assert.equal(capability.state, 'CAPABILITY_UNKNOWN', JSON.stringify(rawConfig));
    assert.equal(capability.numericResult, null);
  }
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

  const create = planMutations([{ manifest }], []);
  assert.equal(create.mutationCount, 1);
  assert.equal(create.operations[0].action, 'create');

  const sameIssue = [{ ...issues[0], body: `human\n\n${buildIssueBotBlock(manifest, null)}\n\nfooter` }];
  const noop = planMutations([{ manifest }], sameIssue);
  assert.equal(noop.mutationCount, 0);
  assert.equal(noop.operations[0].action, 'noop');

  const nextManifest = manifestFor('e'.repeat(40));
  const update = planMutations([{ manifest: nextManifest }], sameIssue);
  assert.equal(update.operations[0].action, 'update');
  assert.ok(update.operations[0].body.startsWith('human'));
  assert.ok(update.operations[0].body.endsWith('footer'));
  assert.equal(update.operations[0].state, undefined);
});

test('noop은 label + 단일 정상 managed block 내부의 exact model/digest만 신뢰한다', () => {
  const manifest = manifestFor();
  const validBlock = buildIssueBotBlock(manifest, null);
  const digest = sha256CanonicalManifest(manifest);
  const baseIssue = {
    number: 80,
    title: 'day0: Qwen/Qwen3.8-Flash-Next',
    user: { login: 'github-actions[bot]' },
    labels: [{ name: 'day0-candidate' }],
  };
  const forgedUnlabelled = {
    ...baseIssue,
    labels: [{ name: 'human-label' }],
    body: `human prefix\n\n${validBlock.replace('pending explicit apply workflow upload', 'https://evil.example/artifact')}\n\nhuman footer`,
  };
  const digestOutsideMalformedBlock = {
    ...baseIssue,
    body: `<!-- fitllm-day0:digest=${digest} -->\n<!-- fitllm-day0:begin -->\n<!-- fitllm-day0:model=Qwen/Qwen3.8-Flash-Next -->\nmissing in-block digest\n<!-- fitllm-day0:end -->`,
  };
  const multipleBlocks = { ...baseIssue, body: `${validBlock}\n\n${validBlock}` };

  for (const issue of [forgedUnlabelled, digestOutsideMalformedBlock, multipleBlocks]) {
    const plan = planMutations([{ manifest }], [issue]);
    assert.equal(plan.operations[0].action, 'update');
  }
  const repaired = planMutations([{ manifest }], [multipleBlocks]).operations[0].body;
  const secondPlan = planMutations([{ manifest }], [{ ...baseIssue, body: repaired }]);
  assert.equal(secondPlan.operations[0].action, 'noop');
});

test('untrusted author의 완벽히 위조된 labelled block은 noop/update/duplicate blocker가 아니다', async () => {
  const manifest = manifestFor();
  const forgedIssue = {
    number: 82,
    title: 'day0: Qwen/Qwen3.8-Flash-Next',
    user: { login: 'public-attacker' },
    labels: [{ name: 'day0-candidate' }],
    body: buildIssueBotBlock(manifest, null).replace(
      '- Run artifact: pending explicit apply workflow upload',
      `- Run artifact: <https://evil.example/forged-artifact>\n- Artifact digest: \`${'e'.repeat(64)}\``,
    ),
  };
  let plan = planMutations([{ manifest }], [forgedIssue]);
  assert.equal(plan.operations[0].action, 'create');
  assert.equal(plan.operations[0].issueNumber, undefined);

  plan = attachArtifactRef(plan, {
    url: 'https://github.com/example/repo/actions/runs/1/artifacts/4', digest: 'd'.repeat(64),
  });
  const calls = [];
  const ghClient = {
    requestWithHeaders: async () => ({ data: [forgedIssue], headers: new Headers() }),
    request: async (requestPath, init = {}) => {
      calls.push([requestPath, init.method || 'GET']);
      if (requestPath === '/labels/day0-candidate') return { name: 'day0-candidate' };
      if (requestPath === '/issues' && init.method === 'POST') return { number: 83 };
      assert.fail(`untrusted issue must never be touched: ${init.method || 'GET'} ${requestPath}`);
    },
  };
  const applied = await applyIssuePlan(plan, ghClient, async (modelId) => ({ id: modelId, sha: REVISION }));
  assert.equal(applied.results[0].status, 'CREATED');
  assert.deepEqual(calls, [
    ['/labels/day0-candidate', 'GET'],
    ['/issues', 'POST'],
  ]);
});

test('unlabelled update apply는 현재 human labels/text/state를 보존하며 managed label을 복원한다', async () => {
  const manifest = manifestFor('e'.repeat(40));
  const currentIssue = {
    number: 81,
    title: 'day0: Qwen/Qwen3.8-Flash-Next',
    user: { login: 'click6067-ship-it' },
    state: 'closed',
    labels: [{ name: 'human-label' }],
    body: 'human prefix\n\nold body\n\nhuman footer',
  };
  const plan = attachArtifactRef(planMutations([{ manifest }], [currentIssue]), {
    url: 'https://github.com/example/repo/actions/runs/1/artifacts/3', digest: 'c'.repeat(64),
  });
  let patchBody;
  const ghClient = {
    request: async (requestPath, init = {}) => {
      if (requestPath === '/issues/81' && !init.method) return currentIssue;
      if (requestPath === '/labels/day0-candidate') return { name: 'day0-candidate' };
      if (requestPath === '/issues/81' && init.method === 'PATCH') { patchBody = init.body; return currentIssue; }
      assert.fail(`unexpected GitHub call: ${init.method || 'GET'} ${requestPath}`);
    },
  };
  const applied = await applyIssuePlan(plan, ghClient, async (modelId) => ({ id: modelId, sha: 'e'.repeat(40) }));
  assert.equal(applied.results[0].status, 'UPDATED');
  assert.deepEqual(patchBody.labels, ['human-label', 'day0-candidate']);
  assert.ok(patchBody.body.startsWith('human prefix'));
  assert.ok(patchBody.body.includes('human footer'));
  assert.ok(!('state' in patchBody));
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
  const plan = planMutations(records, [], { maxMutations: 3 });
  assert.equal(plan.mutationCount, 3);
  assert.deepEqual(plan.dropped, [{ candidateId: `Qwen/D@${'4'.repeat(40)}`, reason: 'MUTATION_LIMIT' }]);
});

test('invalid revision은 blocked이고 같은 plan의 valid create를 막지 않는다', async () => {
  const invalidManifest = buildEvidenceManifest({
    lifecycleState: 'UNVERIFIED_IDENTITY', id: 'Qwen/invalid', revision: null,
    identityEvidenceUrl: 'https://github.com/QwenLM', pipelineTag: 'text-generation',
    discoverySources: ['hf_official_namespace_release'], failureCodes: ['IDENTITY_OR_REVISION_INVALID'],
  }, null, 'capability-v1');
  const validManifest = manifestFor();
  let plan = planMutations([{ manifest: invalidManifest }, { manifest: validManifest }], []);
  assert.deepEqual(plan.operations.map(({ action }) => action), ['blocked', 'create']);
  assert.equal(plan.mutationCount, 1);

  plan = attachArtifactRef(plan, {
    url: 'https://github.com/example/repo/actions/runs/1/artifacts/2', digest: 'b'.repeat(64),
  });
  const mutations = [];
  const ghClient = {
    requestWithHeaders: async () => ({ data: [], headers: new Headers() }),
    request: async (requestPath, init = {}) => {
      if (init.method === 'POST') mutations.push(requestPath);
      return requestPath === '/issues' ? { number: 92 } : { name: 'day0-candidate' };
    },
  };
  const applied = await applyIssuePlan(plan, ghClient, async (modelId) => ({ id: modelId, sha: REVISION }));
  assert.deepEqual(applied.results.map(({ status }) => status), ['BLOCKED', 'CREATED']);
  assert.deepEqual(mutations, ['/issues']);
});

test('apply-time body/revision conflict는 PATCH 없이 중단된다', async () => {
  const manifest = manifestFor('e'.repeat(40));
  const existing = {
    number: 77, title: 'day0: Qwen/Qwen3.8-Flash-Next', body: 'old human body',
    user: { login: 'github-actions[bot]' },
  };
  const plan = planMutations([{ manifest }], [existing]);
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

test('apply update는 planning 뒤 author가 untrusted로 보이면 PATCH하지 않는다', async () => {
  const manifest = manifestFor('e'.repeat(40));
  const existing = {
    number: 84, title: 'day0: Qwen/Qwen3.8-Flash-Next', body: 'trusted planning body',
    user: { login: 'github-actions[bot]' },
  };
  const plan = planMutations([{ manifest }], [existing]);
  const ghClient = {
    request: async (requestPath, init = {}) => {
      if (requestPath === '/issues/84' && !init.method) {
        return { ...existing, user: { login: 'public-attacker' } };
      }
      assert.fail(`untrusted issue must not be mutated: ${init.method || 'GET'} ${requestPath}`);
    },
  };
  const result = await applyIssuePlan(plan, ghClient, async () => assert.fail('author gate comes before HF recheck'));
  assert.equal(result.results[0].status, 'UNTRUSTED_AUTHOR');
});

test('apply-time stale HF revision은 POST/PATCH를 막는다', async () => {
  const manifest = manifestFor();
  const plan = planMutations([{ manifest }], []);
  const mutations = [];
  const ghClient = {
    requestWithHeaders: async () => ({ data: [], headers: new Headers() }),
    request: async (requestPath, init = {}) => mutations.push([requestPath, init.method]),
  };
  const result = await applyIssuePlan(plan, ghClient, async () => ({ sha: 'f'.repeat(40) }));
  assert.equal(result.results[0].status, 'STALE_REVISION');
  assert.equal(mutations.length, 0);
});

test('existing issue scan은 label 제거된 exact marker/title도 찾도록 label filter를 쓰지 않는다', async () => {
  const paths = [];
  const issues = await fetchAllGitHubIssues({
    requestWithHeaders: async (requestPath) => {
      paths.push(requestPath);
      if (paths.length > 1) return { data: [], headers: new Headers() };
      return {
        data: [{ number: 44, title: 'day0: Qwen/unlabelled', body: 'human', labels: [] }],
        headers: new Headers({
          link: '<https://api.github.com/repositories/123/issues?state=all&per_page=100&page=2&after=cursor>; rel="next"',
        }),
      };
    },
  });
  assert.equal(issues.length, 1);
  assert.ok(paths.every((requestPath) => !requestPath.includes('labels=')));
  assert.match(paths[1], /^\/issues\?/);
});

test('create apply는 trusted managed issue를 다시 조회해 concurrent duplicate를 만들지 않는다', async () => {
  const manifest = manifestFor();
  const plan = planMutations([{ manifest }], []);
  const ghClient = {
    requestWithHeaders: async () => ({
      data: [{
        number: 90,
        title: 'day0: Qwen/Qwen3.8-Flash-Next',
        user: { login: 'github-actions[bot]' },
        body: buildIssueBotBlock(manifest, null),
        labels: [{ name: 'day0-candidate' }],
      }],
      headers: new Headers(),
    }),
    request: async () => assert.fail('duplicate recheck must stop all mutation calls'),
  };
  const applied = await applyIssuePlan(plan, ghClient, async () => assert.fail('duplicate check runs before HF recheck'));
  assert.equal(applied.results[0].status, 'ALREADY_EXISTS');
  assert.equal(applied.mutationCount, 0);
});

test('explicit apply create는 pinned revision 재검사 뒤 issue endpoint만 쓴다', async () => {
  const manifest = manifestFor();
  const plan = attachArtifactRef(planMutations([{ manifest }], []), {
    url: 'https://github.com/example/repo/actions/runs/1/artifacts/2',
    digest: 'a'.repeat(64),
  });
  const calls = [];
  const ghClient = {
    requestWithHeaders: async () => ({ data: [], headers: new Headers() }),
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
  // Queue order now puts text-generation and unknown-pipeline candidates first; look the multimodal
  // fixture up by id instead of assuming it is the first evaluated record.
  const flashNext = result.records.find(({ manifest }) => manifest.candidateId.startsWith('Qwen/Qwen3.8-Flash-Next@'));
  assert.equal(flashNext.manifest.lifecycleState, 'UNSUPPORTED_ARCHITECTURE');
  assert.deepEqual(result.records.map(({ manifest }) => manifest.pipelineTag),
    ['text-generation', null, 'image-text-to-text']);
  assert.deepEqual(githubCalls.map(([method]) => method), ['GET']);
  assert.equal(JSON.parse(await readFile(path.join(outputDir, 'summary.json'), 'utf8')).mode, 'dry-run');
});

test('13개 중 선두 12개가 trusted noop이어도 13번째 신규 후보를 12회 cap 안에서 우선 평가한다', async () => {
  const policy = await sourcePolicy();
  policy.officialNamespaces = policy.officialNamespaces.slice(0, 1);
  const baseConfig = {
    model_type: 'llama', num_hidden_layers: 2, num_attention_heads: 2,
    num_key_value_heads: 2, hidden_size: 128, head_dim: 64,
    intermediate_size: 256, vocab_size: 1024, torch_dtype: 'bfloat16',
  };
  const bytesByName = {
    'config.json': Buffer.from(JSON.stringify(baseConfig)),
    'model.safetensors.index.json': Buffer.from(JSON.stringify({ metadata: { total_size: 1_000_000 } })),
    LICENSE: Buffer.from('MIT'),
  };
  const models = 'ABCDEFGHIJKLM'.split('').map((name, index) => ({
    id: `Qwen/${name}`,
    sha: (index + 1).toString(16).repeat(40),
    pipeline_tag: 'text-generation',
    createdAt: new Date(Date.UTC(2026, 8, 20 - index)).toISOString(),
    tags: ['license:mit'],
    siblings: Object.keys(bytesByName).map((rfilename) => ({ rfilename })),
  }));
  const fetchImpl = async (url) => {
    const parsed = new URL(url);
    if (parsed.pathname === '/api/models') return jsonResponse(parsed.searchParams.has('author') ? models : []);
    const [, revision, filename] = parsed.pathname.match(/\/resolve\/([0-9a-f]{40})\/(.+)$/);
    return new Response(bytesByName[filename], { headers: { 'x-repo-commit': revision, etag: `"${filename}"` } });
  };
  const discovery = await discoverCandidates({ policy, fetchImpl, now: new Date('2026-09-21T12:00:00Z') });
  const existingIssues = [];
  for (const candidate of discovery.candidates.slice(0, 12)) {
    const evidence = await pinEvidence(candidate, { fetchImpl });
    const manifest = buildEvidenceManifest(evidence, classifyCapability(evidence), 'capability-v1');
    existingIssues.push({
      number: existingIssues.length + 1,
      title: `day0: ${candidate.id}`,
      body: buildIssueBotBlock(manifest, null),
      user: { login: 'github-actions[bot]' },
      labels: [{ name: 'day0-candidate' }],
    });
  }
  const outputDir = await mkdtemp(path.join(tmpdir(), 'fitllm-day0-starvation-'));
  const result = await runDay0Watch({
    policy, fetchImpl, existingIssues, now: new Date('2026-09-21T12:00:00Z'),
  }, { outputDir, sourceRoot: new URL('..', import.meta.url).pathname, mode: 'dry-run' });
  assert.equal(result.summary.evaluated, 12);
  assert.ok(result.issuePlan.operations.some(({ action, modelId }) => action === 'create' && modelId === 'Qwen/M'));
  assert.equal(result.issuePlan.mutationCount, 1);
  assert.ok(result.issuePlan.dropped.some(({ candidateId, reason }) => candidateId.startsWith('Qwen/L@') && reason === 'EVALUATION_LIMIT'));
});

test('선두 12개 revision이 invalid여도 13번째 valid 후보를 evidence cap 전에 우선 평가한다', async () => {
  const policy = await sourcePolicy();
  policy.officialNamespaces = policy.officialNamespaces.slice(0, 1);
  const bytesByName = {
    'config.json': Buffer.from(JSON.stringify({
      model_type: 'llama', num_hidden_layers: 2, num_attention_heads: 2,
      num_key_value_heads: 2, hidden_size: 128, head_dim: 64,
      intermediate_size: 256, vocab_size: 1024, torch_dtype: 'bfloat16',
    })),
    'model.safetensors.index.json': Buffer.from(JSON.stringify({ metadata: { total_size: 1_000_000 } })),
    LICENSE: Buffer.from('MIT'),
  };
  const models = 'ABCDEFGHIJKLM'.split('').map((name, index) => ({
    id: `Qwen/${name}`,
    sha: index < 12 ? `invalid-${index}` : 'd'.repeat(40),
    pipeline_tag: 'text-generation',
    createdAt: new Date(Date.UTC(2026, 8, 20 - index)).toISOString(),
    tags: ['license:mit'],
    siblings: Object.keys(bytesByName).map((rfilename) => ({ rfilename })),
  }));
  const fetchImpl = async (url) => {
    const parsed = new URL(url);
    if (parsed.pathname === '/api/models') return jsonResponse(parsed.searchParams.has('author') ? models : []);
    const [, revision, filename] = parsed.pathname.match(/\/resolve\/([0-9a-f]{40})\/(.+)$/);
    return new Response(bytesByName[filename], { headers: { 'x-repo-commit': revision, etag: `"${filename}"` } });
  };
  const outputDir = await mkdtemp(path.join(tmpdir(), 'fitllm-day0-invalid-starvation-'));
  const result = await runDay0Watch({
    policy, fetchImpl, existingIssues: [], now: new Date('2026-09-21T12:00:00Z'),
  }, { outputDir, sourceRoot: new URL('..', import.meta.url).pathname, mode: 'dry-run' });
  assert.equal(result.summary.evaluated, 12);
  assert.ok(result.issuePlan.operations.some(({ action, modelId }) => action === 'create' && modelId === 'Qwen/M'));
  assert.equal(result.issuePlan.mutationCount, 1);
  assert.ok(result.issuePlan.dropped.some(({ candidateId, reason }) => candidateId.startsWith('Qwen/L@') && reason === 'EVALUATION_LIMIT'));
});

async function mixedPipelineFixture() {
  const policy = await sourcePolicy();
  policy.officialNamespaces = policy.officialNamespaces.slice(0, 1);
  const bytesByName = {
    'config.json': Buffer.from(JSON.stringify({
      model_type: 'llama', num_hidden_layers: 2, num_attention_heads: 2,
      num_key_value_heads: 2, hidden_size: 128, head_dim: 64,
      intermediate_size: 256, vocab_size: 1024, torch_dtype: 'bfloat16',
    })),
    'model.safetensors.index.json': Buffer.from(JSON.stringify({ metadata: { total_size: 1_000_000 } })),
    LICENSE: Buffer.from('MIT'),
  };
  // Newest first: one unknown-pipeline and three image-text-to-text checkpoints are newer than
  // every text-generation checkpoint, so createdAt order alone would starve the text tier.
  const models = [
    ['Unknown-P', null, 21],
    ['Img-A', 'image-text-to-text', 20],
    ['Img-B', 'image-text-to-text', 19],
    ['Img-C', 'image-text-to-text', 18],
    ['Text-U', 'text-generation', 17],
    ['Text-N1', 'text-generation', 16],
    ['Text-N2', 'text-generation', 15],
  ].map(([name, pipelineTag, day], index) => ({
    id: `Qwen/${name}`,
    sha: (index + 1).toString(16).repeat(40),
    ...(pipelineTag === null ? {} : { pipeline_tag: pipelineTag }),
    createdAt: new Date(Date.UTC(2026, 8, day)).toISOString(),
    tags: ['license:mit'],
    siblings: Object.keys(bytesByName).map((rfilename) => ({ rfilename })),
  }));
  const fetchImpl = async (url) => {
    const parsed = new URL(url);
    if (parsed.pathname === '/api/models') return jsonResponse(parsed.searchParams.has('author') ? models : []);
    const [, revision, filename] = parsed.pathname.match(/\/resolve\/([0-9a-f]{40})\/(.+)$/);
    return new Response(bytesByName[filename], { headers: { 'x-repo-commit': revision, etag: `"${filename}"` } });
  };
  const now = new Date('2026-09-22T12:00:00Z');
  const discovery = await discoverCandidates({ policy, fetchImpl, now });
  const trustedIssueFor = async (modelId, verifierSchemaVersion) => {
    const candidate = discovery.candidates.find(({ id }) => id === modelId);
    const evidence = await pinEvidence(candidate, { fetchImpl });
    const manifest = buildEvidenceManifest(evidence, classifyCapability(evidence), verifierSchemaVersion);
    return {
      number: 100 + discovery.candidates.indexOf(candidate),
      title: `day0: ${modelId}`,
      body: buildIssueBotBlock(manifest, null),
      user: { login: 'github-actions[bot]' },
      labels: [{ name: 'day0-candidate' }],
    };
  };
  const run = async (existingIssues, prefix) => runDay0Watch({ policy, fetchImpl, existingIssues, now }, {
    outputDir: await mkdtemp(path.join(tmpdir(), prefix)),
    sourceRoot: new URL('..', import.meta.url).pathname,
    mode: 'dry-run',
  });
  return { discovery, trustedIssueFor, run };
}

const mutationOps = (plan) => plan.operations
  .filter(({ action }) => action === 'create' || action === 'update')
  .map(({ action, modelId }) => [action, modelId]);

test('mixed pipeline queue는 mutation cap 전에 text-generation candidates를 new-before-update 순으로 먼저 평가한다', async () => {
  const { discovery, trustedIssueFor, run } = await mixedPipelineFixture();
  assert.equal(discovery.candidates.length, 7);
  assert.deepEqual(discovery.candidates.slice(0, 4).map(({ pipelineTag }) => pipelineTag),
    [null, 'image-text-to-text', 'image-text-to-text', 'image-text-to-text']);
  // Text-U already has a trusted managed issue with a stale digest, so it is an update and must
  // queue behind the two issue-less text candidates even though it is newer than both.
  const result = await run([await trustedIssueFor('Qwen/Text-U', 'capability-v0')], 'fitllm-day0-mixed-cap-');
  assert.equal(result.summary.sourceFailures, 0);
  assert.equal(result.issuePlan.mutationCount, 3);
  assert.deepEqual(mutationOps(result.issuePlan), [
    ['create', 'Qwen/Text-N1'],
    ['create', 'Qwen/Text-N2'],
    ['update', 'Qwen/Text-U'],
  ]);
  for (const modelId of ['Qwen/Unknown-P', 'Qwen/Img-A', 'Qwen/Img-B', 'Qwen/Img-C']) {
    assert.ok(
      result.issuePlan.dropped.some(({ candidateId, reason }) => candidateId.startsWith(`${modelId}@`) && reason === 'MUTATION_LIMIT'),
      `${modelId} must stay queued behind the mutation cap, not be excluded`,
    );
  }
});

test('text-generation candidates가 noop으로 capacity를 비우면 unknown pipeline 다음에 image-text-to-text가 남아 있다', async () => {
  const { trustedIssueFor, run } = await mixedPipelineFixture();
  const existingIssues = await Promise.all(
    ['Qwen/Text-U', 'Qwen/Text-N1', 'Qwen/Text-N2'].map((modelId) => trustedIssueFor(modelId, 'capability-v1')),
  );
  const result = await run(existingIssues, 'fitllm-day0-mixed-freed-');
  // The text tier is still evaluated first (deterministic createdAt order inside the tier) ...
  assert.deepEqual(result.issuePlan.operations.slice(0, 3).map(({ action, modelId }) => [action, modelId]), [
    ['noop', 'Qwen/Text-U'],
    ['noop', 'Qwen/Text-N1'],
    ['noop', 'Qwen/Text-N2'],
  ]);
  // ... and the freed slots go to unknown-pipeline before image-text-to-text, in discovery order.
  assert.equal(result.issuePlan.mutationCount, 3);
  assert.deepEqual(mutationOps(result.issuePlan), [
    ['create', 'Qwen/Unknown-P'],
    ['create', 'Qwen/Img-A'],
    ['create', 'Qwen/Img-B'],
  ]);
  assert.ok(result.issuePlan.dropped.some(({ candidateId, reason }) => candidateId.startsWith('Qwen/Img-C@') && reason === 'MUTATION_LIMIT'));
});
