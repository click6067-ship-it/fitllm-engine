// Regression guard for issue #119: a text-only checkpoint that retains a vision_config wrapper.
// Pinned source: https://huggingface.co/Kwaipilot/KAT-Coder-V2.5-Dev/tree/7be56fe773e72b6f5ca93c1ae45d828ddb893922
import { test } from 'node:test';
import { strict as assert } from 'node:assert';
import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { parseHfConfig } from '../engine.js';
import {
  EVIDENCE_BYTE_LIMITS,
  buildEvidenceManifest,
  classifyCapability,
  detectCapabilityBlockers,
  hasShippedVisionTensor,
  pinEvidence,
  sha256CanonicalManifest,
} from '../.github/scripts/day0-core.mjs';

const FIXTURES = new URL('./fixtures/day0/', import.meta.url);
const MODEL_ID = 'Kwaipilot/KAT-Coder-V2.5-Dev';
const REVISION = '7be56fe773e72b6f5ca93c1ae45d828ddb893922';
const CONFIG_SHA = 'e88f4ab90b749a0b544bdcba28d3b1f7e377b47b10f177f9875d3869689f1c4a';
const CONFIG_URL = `https://huggingface.co/${MODEL_ID}/resolve/${REVISION}/config.json`;
const INDEX_URL = `https://huggingface.co/${MODEL_ID}/resolve/${REVISION}/model.safetensors.index.json`;

async function fixture(name, encoding = 'utf8') {
  return readFile(new URL(name, FIXTURES), encoding);
}

// Deterministic expansion of the sealed namespace summary, per the contract stated inside the
// fixture: every template crossed with its layer set (and expert range) reproduces the exact
// upstream tensor-name set. Shard assignment is synthesized — upstream key order and per-name shard
// placement are explicitly not preserved because day0 classification reads namespaces only.
function expandWeightMap(summary) {
  const names = [];
  for (const template of summary.templates) {
    const layers = template.layers === null ? [null] : summary.layerIndexSets[template.layers];
    for (const layer of layers) {
      const withLayer = layer === null ? template.name : template.name.replace('{layer}', String(layer));
      if (!template.experts) {
        names.push(withLayer);
        continue;
      }
      for (let expert = 0; expert < summary.expertCount; expert += 1) {
        names.push(withLayer.replace('{expert}', String(expert)));
      }
    }
  }
  names.sort();
  const blockSize = Math.ceil(names.length / summary.shardFiles.length);
  return Object.fromEntries(names.map((name, index) => [name, summary.shardFiles[Math.floor(index / blockSize)]]));
}

const summary = JSON.parse(await fixture('kat-coder-v2.5-dev.weight-map-summary.json'));
const modelInfo = JSON.parse(await fixture('kat-coder-v2.5-dev.model.json'));
const rawConfig = JSON.parse(await fixture('kat-coder-v2.5-dev.config.json'));
const weightMap = expandWeightMap(summary);

// The evidence day0 pins for this checkpoint. Byte provenance (bytes/etag/sha256) is the upstream
// index metadata sealed in the fixture; rawWeightsIndex is the namespace-lossless expansion of that
// same index, proven equivalent by the tensorNameSetSha256 assertion below.
function katEvidence(overrides = {}) {
  return {
    lifecycleState: 'EVIDENCE_PINNED',
    id: MODEL_ID,
    revision: REVISION,
    pipelineTag: 'text-generation',
    createdAt: '2026-07-23T09:54:08.000Z',
    checkpointKind: 'base',
    discoverySources: ['hf_official_namespace_release'],
    failureCodes: [],
    rawConfig,
    rawWeightsIndex: { metadata: { total_size: summary.metadata.total_size }, weight_map: weightMap },
    config: { url: CONFIG_URL, bytes: 3686, etag: '"babf62b1f94a26983a087e7ce716b80dc112665c"', sha256: CONFIG_SHA },
    weightsIndex: {
      url: INDEX_URL,
      bytes: summary.source.bytes,
      etag: summary.source.etag,
      sha256: summary.source.sha256,
      totalSizeBytes: summary.metadata.total_size,
    },
    license: null,
    ...overrides,
  };
}

function withWeightsIndex(patch) {
  return katEvidence({ weightsIndex: { ...katEvidence().weightsIndex, ...patch } });
}

function withWeightMap(patch) {
  return katEvidence({
    rawWeightsIndex: { metadata: { total_size: summary.metadata.total_size }, weight_map: { ...weightMap, ...patch } },
  });
}

test('sealed weight-map summary는 pinned upstream tensor namespace 집합을 무손실 복원한다', () => {
  assert.equal(summary.source.revision, REVISION);
  assert.equal(summary.source.url, INDEX_URL);
  assert.match(summary.source.sha256, /^[0-9a-f]{64}$/);
  assert.equal(Object.keys(weightMap).length, summary.entryCount);
  assert.equal(
    createHash('sha256').update(Object.keys(weightMap).sort().join('\n')).digest('hex'),
    summary.tensorNameSetSha256,
  );
  assert.equal(summary.templates.reduce((total, { count }) => total + count, 0), summary.entryCount);
  for (const { name, layers, experts, count } of summary.templates) {
    const expected = (layers === null ? 1 : summary.layerIndexSets[layers].length) * (experts ? summary.expertCount : 1);
    assert.equal(count, expected, name);
  }
  const shards = new Set(summary.shardFiles);
  assert.ok(Object.values(weightMap).every((shard) => shards.has(shard)));
  assert.equal(hasShippedVisionTensor(weightMap), false);
});

test('pinned KAT config fixture는 upstream bytes와 digest를 그대로 보존한다', async () => {
  const bytes = await fixture('kat-coder-v2.5-dev.config.json', null);
  assert.equal(bytes.length, 3686);
  assert.equal(createHash('sha256').update(bytes).digest('hex'), CONFIG_SHA);
  assert.equal(modelInfo.sha, REVISION);
  assert.equal(modelInfo.pipeline_tag, 'text-generation');
  assert.ok(Object.prototype.hasOwnProperty.call(rawConfig, 'vision_config'));
  assert.ok(Object.keys(rawConfig.vision_config).length > 0);
});

test('day0 classifier와 parser는 pinned text-only wrapper에서 일치한다', () => {
  assert.doesNotThrow(() => parseHfConfig(MODEL_ID, rawConfig, summary.metadata.total_size));
  const capability = classifyCapability(katEvidence());
  assert.deepEqual(capability, {
    state: 'SUPPORTED_BY_CURRENT_ENGINE',
    lifecycleState: 'AWAITING_GOLDEN_VECTOR',
    numericResult: null,
  });
  for (const key of ['verdict', 'usedGB', 'maxContext', 'unsupportedComponents']) assert.ok(!(key in capability));
});

test('evidence 없이 호출한 blocker 판정은 vision_config를 계속 fail-closed로 본다', () => {
  for (const evidence of [undefined, null, {}, katEvidence({ lifecycleState: 'INSUFFICIENT_EVIDENCE' })]) {
    assert.deepEqual(detectCapabilityBlockers(rawConfig, evidence), {
      state: 'UNSUPPORTED_ARCHITECTURE',
      unsupportedComponents: ['MULTIMODAL_WORKLOAD'],
      numericResult: null,
    });
  }
});

test('vision tensor namespace가 하나라도 있으면 MULTIMODAL_WORKLOAD로 남는다', () => {
  const lookAlikes = [
    'model.visual.blocks.0.attn.qkv.weight',
    'model.vision_tower.encoder.layers.0.self_attn.k_proj.weight',
    'model.vision_model.embeddings.patch_embedding.weight',
    'multi_modal_projector.linear_1.weight',
    'model.mm_projector.0.weight',
    'model.image_newline',
    'visual.merger.mlp.0.weight',
    'model.video_tower.blocks.0.norm1.weight',
    'model.visual.patch_embed.proj.weight',
    'model.audio_tower.layers.0.self_attn.k_proj.weight',
  ];
  for (const name of lookAlikes) {
    const evidence = withWeightMap({ [name]: summary.shardFiles[0] });
    assert.equal(hasShippedVisionTensor(evidence.rawWeightsIndex.weight_map), true, name);
    assert.deepEqual(classifyCapability(evidence), {
      state: 'UNSUPPORTED_ARCHITECTURE',
      unsupportedComponents: ['MULTIMODAL_WORKLOAD'],
      numericResult: null,
    }, name);
  }
});

test('불완전·초과·malformed weight-map evidence는 numeric-null fail-closed로 남는다', () => {
  const shard = summary.shardFiles[0];
  const failClosed = [
    ['weight map 누락', katEvidence({ rawWeightsIndex: undefined })],
    ['weight_map 누락', katEvidence({ rawWeightsIndex: { metadata: { total_size: 1 } } })],
    ['weight_map 빈 객체', katEvidence({ rawWeightsIndex: { weight_map: {} } })],
    ['weight_map 배열', katEvidence({ rawWeightsIndex: { weight_map: ['model.language_model.norm.weight'] } })],
    ['weight_map 값 누락', katEvidence({ rawWeightsIndex: { weight_map: { 'model.language_model.norm.weight': '' } } })],
    ['weight_map 값 비문자열', katEvidence({ rawWeightsIndex: { weight_map: { 'model.language_model.norm.weight': 3 } } })],
    ['weight_map 빈 키', withWeightMap({ '': shard })],
    ['index slot 누락', katEvidence({ weightsIndex: null })],
    ['index byte 한도 초과', withWeightsIndex({ bytes: EVIDENCE_BYTE_LIMITS['model.safetensors.index.json'] + 1 })],
    ['index byte 0', withWeightsIndex({ bytes: 0 })],
    ['index digest 부재', withWeightsIndex({ sha256: null })],
    ['index digest 형식 위반', withWeightsIndex({ sha256: 'A'.repeat(64) })],
    ['index etag 부재', withWeightsIndex({ etag: '' })],
    ['index total_size 0', withWeightsIndex({ totalSizeBytes: 0 })],
    ['index URL이 revision 밖', withWeightsIndex({ url: `https://huggingface.co/${MODEL_ID}/resolve/main/model.safetensors.index.json` })],
    ['index URL이 다른 파일', withWeightsIndex({ url: CONFIG_URL })],
    ['evidence failure code 존재', katEvidence({ failureCodes: ['LICENSE_ID_MISSING'] })],
    ['revision 미고정', katEvidence({ revision: 'main' })],
    ['revision 길이 위반', katEvidence({ revision: REVISION.slice(0, 39) })],
    ['candidate id 위반', katEvidence({ id: 'Kwaipilot/KAT/Coder' })],
    ['pipeline이 image-text-to-text', katEvidence({ pipelineTag: 'image-text-to-text' })],
  ];
  for (const [label, evidence] of failClosed) {
    assert.deepEqual(classifyCapability(evidence), {
      state: 'UNSUPPORTED_ARCHITECTURE',
      unsupportedComponents: ['MULTIMODAL_WORKLOAD'],
      numericResult: null,
    }, label);
  }
  const pipelineUnknown = classifyCapability(katEvidence({ pipelineTag: null }));
  assert.equal(pipelineUnknown.state, 'CAPABILITY_UNKNOWN');
  assert.equal(pipelineUnknown.numericResult, null);
  assert.equal(classifyCapability(katEvidence({ lifecycleState: 'INSUFFICIENT_EVIDENCE' })), null);
});

test('실제 multimodal 대조군인 Qwen3.8-Flash-Next는 완전한 evidence에도 다섯 blocker를 유지한다', async () => {
  const qwenConfig = JSON.parse(await fixture('qwen3.8-flash-next.config.json'));
  const qwenIndex = JSON.parse(await fixture('qwen3.8-flash-next.index.json'));
  assert.equal(hasShippedVisionTensor(qwenIndex.weight_map), true);
  const qwenRevision = 'de4b8e4d43b917e7706784d8bb445c9af86a3540';
  const capability = classifyCapability({
    lifecycleState: 'EVIDENCE_PINNED',
    id: 'Qwen/Qwen3.8-Flash-Next',
    revision: qwenRevision,
    pipelineTag: 'text-generation',
    failureCodes: [],
    rawConfig: qwenConfig,
    rawWeightsIndex: qwenIndex,
    weightsIndex: {
      url: `https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/${qwenRevision}/model.safetensors.index.json`,
      bytes: 170726,
      etag: 'W/"sealed"',
      sha256: '99e815241ef03325536b0aaa4441deea45174c17fae31e10f0bb456410c590de',
      totalSizeBytes: qwenIndex.metadata.total_size,
    },
  });
  assert.deepEqual(capability, {
    state: 'UNSUPPORTED_ARCHITECTURE',
    unsupportedComponents: ['QSA_INDEXER', 'NGRAM_RESIDENCY', 'MTP_RESIDENCY', 'HC_ACTIVATION', 'MULTIMODAL_WORKLOAD'],
    numericResult: null,
  });
});

test('upstream이 LICENSE를 제공하지 않으므로 evidence pinning은 그대로 fail-closed다', async () => {
  const configBytes = await fixture('kat-coder-v2.5-dev.config.json', null);
  const evidence = await pinEvidence({
    id: modelInfo.id,
    namespace: 'Kwaipilot',
    identityEvidenceUrl: 'https://huggingface.co/Kwaipilot',
    revision: modelInfo.sha,
    pipelineTag: modelInfo.pipeline_tag,
    createdAt: modelInfo.createdAt,
    discoverySources: ['hf_official_namespace_release'],
    modelInfo,
  }, {
    fetchImpl: async (url) => {
      assert.ok(url.endsWith('/config.json') || url.endsWith('/model.safetensors.index.json'), url);
      const body = url.endsWith('/config.json')
        ? configBytes
        : Buffer.from(JSON.stringify({ metadata: { total_size: summary.metadata.total_size }, weight_map: weightMap }));
      return new Response(body, { headers: { 'x-repo-commit': REVISION, etag: '"sealed"' } });
    },
  });
  assert.equal(evidence.lifecycleState, 'INSUFFICIENT_EVIDENCE');
  assert.deepEqual(evidence.failureCodes, ['LICENSE_MISSING']);
  assert.equal(classifyCapability(evidence), null);
});

test('supported KAT record의 manifest는 결정적이고 numeric publication이 없다', () => {
  const evidence = katEvidence();
  const manifest = buildEvidenceManifest(evidence, classifyCapability(evidence), 'capability-v1');
  assert.equal(manifest.candidateId, `${MODEL_ID}@${REVISION}`);
  assert.equal(manifest.hfRevision, REVISION);
  assert.equal(manifest.lifecycleState, 'AWAITING_GOLDEN_VECTOR');
  assert.equal(manifest.capability.state, 'SUPPORTED_BY_CURRENT_ENGINE');
  assert.equal(manifest.capability.numericResult, null);
  assert.equal(manifest.weightsIndex.sha256, summary.source.sha256);
  assert.equal(manifest.config.sha256, CONFIG_SHA);
  assert.equal(sha256CanonicalManifest(manifest), sha256CanonicalManifest(structuredClone(manifest)));
  for (const key of ['verdict', 'usedGB', 'maxContext', 'tokensPerSecond']) assert.ok(!(key in manifest));
});
