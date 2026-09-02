import { createHash } from 'node:crypto';
import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import * as fitllmEngine from '../../engine.js';

const HF_ORIGIN = 'https://huggingface.co';
const EVIDENCE_SCHEMA = 'fitllm.day0-evidence.v1';
const ISSUE_BEGIN = '<!-- fitllm-day0:begin -->';
const ISSUE_END = '<!-- fitllm-day0:end -->';
const REVISION_RE = /^[0-9a-f]{40}$/;
const ID_PART_RE = /^[A-Za-z0-9][A-Za-z0-9._-]*$/;
const VARIANT_RE = /(?:^|[-_.])(gguf|awq|gptq|mlx|bnb|exl2|int[48]|nvfp4|mxfp4|fp4|fp8)(?:$|[-_.])/i;

const isPlainObject = (value) => value !== null && typeof value === 'object'
  && !Array.isArray(value) && Object.getPrototypeOf(value) === Object.prototype;
const hasOwn = (object, key) => Object.prototype.hasOwnProperty.call(object, key);
const sha256 = (value) => createHash('sha256').update(value).digest('hex');

export function loadSourcePolicy(input) {
  const policy = typeof input === 'string' || Buffer.isBuffer(input)
    ? JSON.parse(String(input))
    : structuredClone(input);
  if (!isPlainObject(policy) || policy.schemaVersion !== 1) throw new Error('day0 source policy schemaVersion must be 1');
  if (!Array.isArray(policy.officialNamespaces) || policy.officialNamespaces.length === 0) {
    throw new Error('officialNamespaces must be a non-empty array');
  }
  const namespaces = new Set();
  for (const entry of policy.officialNamespaces) {
    if (!isPlainObject(entry) || !ID_PART_RE.test(entry.namespace || '')) throw new Error('invalid official namespace');
    if (namespaces.has(entry.namespace)) throw new Error(`duplicate official namespace: ${entry.namespace}`);
    namespaces.add(entry.namespace);
    let evidenceUrl;
    try { evidenceUrl = new URL(entry.identityEvidenceUrl); } catch { throw new Error(`invalid identity evidence URL: ${entry.namespace}`); }
    if (evidenceUrl.protocol !== 'https:') throw new Error(`identity evidence URL must use https: ${entry.namespace}`);
  }
  if (!Array.isArray(policy.pipelineTags) || policy.pipelineTags.length === 0
      || policy.pipelineTags.some((tag) => !['text-generation', 'image-text-to-text'].includes(tag))) {
    throw new Error('pipelineTags must contain only reviewed local pipeline tags');
  }
  if (new Set(policy.pipelineTags).size !== policy.pipelineTags.length) throw new Error('duplicate pipeline tag');
  if (!Number.isInteger(policy.releaseLookbackDays) || policy.releaseLookbackDays < 1 || policy.releaseLookbackDays > 90) {
    throw new Error('releaseLookbackDays must be an integer from 1 to 90');
  }
  const { likes, downloads } = policy.priorityThresholds || {};
  if (![likes, downloads].every((value) => Number.isInteger(value) && value >= 0)) {
    throw new Error('priority thresholds must be non-negative integers');
  }
  if (!Number.isInteger(policy.maxIssueMutationsPerRun)
      || policy.maxIssueMutationsPerRun < 1 || policy.maxIssueMutationsPerRun > 3) {
    throw new Error('maxIssueMutationsPerRun must be an integer from 1 to 3');
  }
  return policy;
}

const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

export async function fetchJsonWithRetry(url, {
  fetchImpl = fetch,
  attempts = 3,
  timeoutMs = 8000,
  retryDelayMs = 250,
} = {}) {
  let lastError;
  for (let attempt = 1; attempt <= attempts; attempt += 1) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const response = await fetchImpl(url, {
        signal: controller.signal,
        headers: { Accept: 'application/json', 'User-Agent': 'fitllm-day0-watch' },
      });
      if (!response?.ok) throw new Error(`HTTP ${response?.status ?? 'unknown'}`);
      const contentType = response.headers?.get?.('content-type') || '';
      if (!contentType.toLowerCase().includes('json')) throw new Error(`non-JSON content type: ${contentType || 'missing'}`);
      const text = await response.text();
      try { return JSON.parse(text); } catch { throw new Error('invalid JSON response'); }
    } catch (error) {
      lastError = error;
      if (attempt < attempts && retryDelayMs > 0) await delay(retryDelayMs * attempt);
    } finally {
      clearTimeout(timeout);
    }
  }
  throw new Error(`source fetch failed after ${attempts} attempts: ${lastError?.message || 'unknown error'}`);
}

export function canonicalCandidateId(modelInfo) {
  const id = modelInfo?.id ?? modelInfo?.modelId;
  if (typeof id !== 'string') throw new Error('invalid candidate id: missing');
  const parts = id.split('/');
  if (parts.length !== 2 || parts.some((part) => !ID_PART_RE.test(part))) throw new Error(`invalid candidate id: ${id}`);
  return `${parts[0]}/${parts[1]}`;
}

function checkpointKind(id) {
  return VARIANT_RE.test(id.split('/')[1]) ? 'quantized_variant' : 'base';
}

function candidateSort(a, b) {
  const variantDelta = Number(a.checkpointKind !== 'base') - Number(b.checkpointKind !== 'base');
  if (variantDelta) return variantDelta;
  const releaseDelta = Number(!a.discoverySources.includes('hf_official_namespace_release'))
    - Number(!b.discoverySources.includes('hf_official_namespace_release'));
  if (releaseDelta) return releaseDelta;
  const createdDelta = Date.parse(b.createdAt || 0) - Date.parse(a.createdAt || 0);
  if (createdDelta) return createdDelta;
  return a.id.localeCompare(b.id, 'en');
}

function candidateFromModel(modelInfo, namespacePolicy, source) {
  const id = canonicalCandidateId(modelInfo);
  const namespace = id.split('/')[0];
  if (namespace !== namespacePolicy.namespace) return null;
  const revision = typeof modelInfo.sha === 'string' ? modelInfo.sha : null;
  return {
    id,
    namespace,
    identityEvidenceUrl: namespacePolicy.identityEvidenceUrl,
    revision,
    pipelineTag: modelInfo.pipeline_tag ?? null,
    createdAt: modelInfo.createdAt ?? null,
    checkpointKind: checkpointKind(id),
    discoverySources: [source],
    modelInfo,
  };
}

export async function discoverCandidates({ policy, fetchImpl = fetch, now = new Date() }) {
  policy = loadSourcePolicy(policy);
  const allowedPipelines = new Set(policy.pipelineTags);
  const namespaceByName = new Map(policy.officialNamespaces.map((entry) => [entry.namespace, entry]));
  const cutoff = now.getTime() - policy.releaseLookbackDays * 86400000;
  const sourceFailures = [];
  const droppedCandidates = [];
  const discovered = new Map();
  let successfulSources = 0;

  const merge = (candidate) => {
    const existing = discovered.get(candidate.id);
    if (!existing) {
      discovered.set(candidate.id, candidate);
      return;
    }
    const discoverySources = [...new Set([...existing.discoverySources, ...candidate.discoverySources])].sort();
    const candidateIsRelease = candidate.discoverySources.includes('hf_official_namespace_release');
    const existingIsRelease = existing.discoverySources.includes('hf_official_namespace_release');
    if (candidateIsRelease && !existingIsRelease) {
      candidate.discoverySources = discoverySources;
      candidate.discoveryRevisionConflict = Boolean(existing.discoveryRevisionConflict)
        || Boolean(existing.revision && candidate.revision && existing.revision !== candidate.revision);
      discovered.set(candidate.id, candidate);
      return;
    }
    existing.discoverySources = discoverySources;
    if (existing.revision && candidate.revision && existing.revision !== candidate.revision) {
      existing.discoveryRevisionConflict = true;
    }
  };

  const releaseTasks = policy.officialNamespaces.map(async (namespacePolicy) => {
    const url = new URL('/api/models', HF_ORIGIN);
    url.search = new URLSearchParams({
      author: namespacePolicy.namespace,
      sort: 'createdAt',
      direction: '-1',
      limit: '10',
      full: 'true',
    });
    try {
      const models = await fetchJsonWithRetry(url, { fetchImpl });
      if (!Array.isArray(models)) throw new Error('release payload is not an array');
      successfulSources += 1;
      for (const modelInfo of models) {
        let candidate;
        try { candidate = candidateFromModel(modelInfo, namespacePolicy, 'hf_official_namespace_release'); }
        catch { droppedCandidates.push({ candidateId: String(modelInfo?.id || modelInfo?.modelId || ''), reason: 'INVALID_ID' }); continue; }
        if (!candidate) {
          droppedCandidates.push({ candidateId: String(modelInfo?.id || ''), reason: 'NAMESPACE_MISMATCH' });
          continue;
        }
        if (!Number.isFinite(Date.parse(candidate.createdAt)) || Date.parse(candidate.createdAt) < cutoff) {
          droppedCandidates.push({ candidateId: candidate.id, reason: 'OUTSIDE_RELEASE_LOOKBACK' });
          continue;
        }
        if (candidate.pipelineTag !== null && !allowedPipelines.has(candidate.pipelineTag)) {
          droppedCandidates.push({ candidateId: candidate.id, reason: 'PIPELINE_EXCLUDED' });
          continue;
        }
        merge(candidate);
      }
    } catch (error) {
      sourceFailures.push({ source: 'hf_official_namespace_release', namespace: namespacePolicy.namespace, message: error.message });
    }
  });

  const trendTask = (async () => {
    const url = new URL('/api/models', HF_ORIGIN);
    url.search = new URLSearchParams({ sort: 'trendingScore', direction: '-1', limit: '100', full: 'true' });
    try {
      const models = await fetchJsonWithRetry(url, { fetchImpl });
      if (!Array.isArray(models)) throw new Error('trend payload is not an array');
      successfulSources += 1;
      for (const modelInfo of models) {
        let id;
        try { id = canonicalCandidateId(modelInfo); }
        catch { droppedCandidates.push({ candidateId: String(modelInfo?.id || modelInfo?.modelId || ''), reason: 'INVALID_ID' }); continue; }
        const namespacePolicy = namespaceByName.get(id.split('/')[0]);
        if (!namespacePolicy) {
          droppedCandidates.push({ candidateId: id, reason: 'NAMESPACE_NOT_ALLOWLISTED' });
          continue;
        }
        const candidate = candidateFromModel(modelInfo, namespacePolicy, 'hf_official_namespace_trending');
        if (candidate.pipelineTag !== null && !allowedPipelines.has(candidate.pipelineTag)) {
          droppedCandidates.push({ candidateId: id, reason: 'PIPELINE_EXCLUDED' });
          continue;
        }
        const likes = Number.isFinite(modelInfo.likes) ? modelInfo.likes : 0;
        const downloads = Number.isFinite(modelInfo.downloads) ? modelInfo.downloads : 0;
        if (likes < policy.priorityThresholds.likes && downloads < policy.priorityThresholds.downloads) {
          droppedCandidates.push({ candidateId: id, reason: 'BELOW_TREND_THRESHOLD' });
          continue;
        }
        merge(candidate);
      }
    } catch (error) {
      sourceFailures.push({ source: 'hf_official_namespace_trending', namespace: null, message: error.message });
    }
  })();

  await Promise.all([...releaseTasks, trendTask]);
  if (successfulSources === 0) {
    const error = new Error('all Hugging Face discovery sources failed');
    error.code = 'ALL_SOURCES_FAILED';
    error.sourceFailures = sourceFailures;
    throw error;
  }
  return {
    candidates: [...discovered.values()].sort(candidateSort),
    sourceFailures: sourceFailures.sort((a, b) => `${a.source}:${a.namespace}`.localeCompare(`${b.source}:${b.namespace}`)),
    droppedCandidates: droppedCandidates.sort((a, b) => `${a.candidateId}:${a.reason}`.localeCompare(`${b.candidateId}:${b.reason}`)),
  };
}

function evidenceUrl(id, revision, filename) {
  return `${HF_ORIGIN}/${id}/resolve/${revision}/${filename}`;
}

async function fetchEvidenceSlot(url, revision, fetchImpl) {
  let response;
  try {
    response = await fetchImpl(url, { headers: { 'User-Agent': 'fitllm-day0-watch' } });
  } catch {
    return { ok: false, lifecycleState: 'SOURCE_UNAVAILABLE', failureCode: 'SOURCE_UNAVAILABLE' };
  }
  if (!response?.ok) {
    return { ok: false, lifecycleState: 'INSUFFICIENT_EVIDENCE', failureCode: `HTTP_${response?.status ?? 'UNKNOWN'}` };
  }
  const commit = response.headers?.get?.('x-repo-commit');
  if (commit !== revision) {
    return { ok: false, lifecycleState: 'EVIDENCE_CHANGED', failureCode: 'REVISION_HEADER_MISMATCH' };
  }
  const etag = response.headers?.get?.('etag');
  if (!etag) return { ok: false, lifecycleState: 'INSUFFICIENT_EVIDENCE', failureCode: 'ETAG_MISSING' };
  const bytes = Buffer.from(await response.arrayBuffer());
  if (bytes.length === 0) return { ok: false, lifecycleState: 'INSUFFICIENT_EVIDENCE', failureCode: 'EMPTY_EVIDENCE' };
  return {
    ok: true,
    bytes,
    metadata: {
      url,
      bytes: bytes.length,
      etag,
      sha256: sha256(bytes),
    },
  };
}

export async function pinEvidence(candidate, { fetchImpl = fetch } = {}) {
  let id;
  try { id = canonicalCandidateId({ id: candidate?.id }); } catch {
    return {
      lifecycleState: 'UNVERIFIED_IDENTITY', id: candidate?.id ?? null, revision: null,
      pipelineTag: candidate?.pipelineTag ?? null, discoverySources: candidate?.discoverySources || [],
      failureCodes: ['INVALID_CANDIDATE_ID'], config: null, weightsIndex: null, license: null,
    };
  }
  const namespace = id.split('/')[0];
  const revision = candidate.revision;
  let identityEvidenceUrl = null;
  try {
    const parsed = new URL(candidate.identityEvidenceUrl);
    if (parsed.protocol === 'https:') identityEvidenceUrl = parsed.toString();
  } catch {
    // A reviewed HTTPS namespace mapping is required below.
  }
  if (namespace !== candidate.namespace || !REVISION_RE.test(revision || '')
      || !identityEvidenceUrl
      || (candidate.modelInfo?.id && candidate.modelInfo.id !== id)
      || (candidate.modelInfo?.modelId && candidate.modelInfo.modelId !== id)
      || (candidate.modelInfo?.sha && candidate.modelInfo.sha !== revision)) {
    return {
      lifecycleState: 'UNVERIFIED_IDENTITY', id, revision: REVISION_RE.test(revision || '') ? revision : null,
      identityEvidenceUrl,
      pipelineTag: candidate.pipelineTag ?? null, createdAt: candidate.createdAt ?? null,
      discoverySources: candidate.discoverySources || [], failureCodes: ['IDENTITY_OR_REVISION_INVALID'],
      config: null, weightsIndex: null, license: null,
    };
  }
  if (candidate.discoveryRevisionConflict) {
    return {
      lifecycleState: 'EVIDENCE_CHANGED', id, revision, identityEvidenceUrl,
      pipelineTag: candidate.pipelineTag ?? null, createdAt: candidate.createdAt ?? null,
      discoverySources: candidate.discoverySources || [], failureCodes: ['DISCOVERY_REVISION_MISMATCH'],
      config: null, weightsIndex: null, license: null,
    };
  }

  const siblings = new Set((candidate.modelInfo?.siblings || []).map((entry) => entry.rfilename));
  const licenseFilename = ['LICENSE', 'LICENSE.md', 'LICENSE.txt'].find((name) => siblings.has(name));
  const required = [
    ['config', 'config.json'],
    ['weightsIndex', 'model.safetensors.index.json'],
    ['license', licenseFilename],
  ];
  const result = {
    lifecycleState: 'EVIDENCE_PINNED', id, revision,
    identityEvidenceUrl,
    pipelineTag: candidate.pipelineTag ?? null, createdAt: candidate.createdAt ?? null,
    checkpointKind: candidate.checkpointKind ?? 'base',
    discoverySources: [...new Set(candidate.discoverySources || [])].sort(),
    failureCodes: [], config: null, weightsIndex: null, license: null,
  };
  let failureState = null;
  for (const [slot, filename] of required) {
    if (!filename || !siblings.has(filename)) {
      result.failureCodes.push(`${slot.toUpperCase()}_MISSING`);
      failureState ||= 'INSUFFICIENT_EVIDENCE';
      continue;
    }
    const fetched = await fetchEvidenceSlot(evidenceUrl(id, revision, filename), revision, fetchImpl);
    if (!fetched.ok) {
      const code = fetched.failureCode === 'REVISION_HEADER_MISMATCH'
        ? fetched.failureCode
        : `${slot.toUpperCase()}_${fetched.failureCode}`;
      result.failureCodes.push(code);
      if (fetched.lifecycleState === 'EVIDENCE_CHANGED') failureState = 'EVIDENCE_CHANGED';
      else if (failureState !== 'EVIDENCE_CHANGED') failureState ||= fetched.lifecycleState;
      continue;
    }
    if (slot === 'license') {
      const tagLicense = candidate.modelInfo?.tags?.find?.((tag) => String(tag).startsWith('license:'))?.slice(8);
      result.license = {
        id: candidate.modelInfo?.cardData?.license_name ?? candidate.modelInfo?.cardData?.license ?? tagLicense ?? null,
        ...fetched.metadata,
      };
      result.rawLicense = fetched.bytes;
      if (!result.license.id) {
        result.failureCodes.push('LICENSE_ID_MISSING');
        failureState ||= 'INSUFFICIENT_EVIDENCE';
      }
      continue;
    }
    result[slot] = fetched.metadata;
    let parsed;
    try { parsed = JSON.parse(fetched.bytes.toString('utf8')); } catch {
      result.failureCodes.push(`${slot.toUpperCase()}_INVALID_JSON`);
      failureState ||= 'INSUFFICIENT_EVIDENCE';
      continue;
    }
    if (!isPlainObject(parsed)) {
      result.failureCodes.push(`${slot.toUpperCase()}_INVALID_JSON`);
      failureState ||= 'INSUFFICIENT_EVIDENCE';
      continue;
    }
    if (slot === 'config') {
      result.rawConfig = parsed;
    } else {
      const totalSizeBytes = parsed?.metadata?.total_size;
      if (!Number.isSafeInteger(totalSizeBytes) || totalSizeBytes <= 0) {
        result.failureCodes.push('WEIGHTSINDEX_TOTAL_SIZE_INVALID');
        failureState ||= 'INSUFFICIENT_EVIDENCE';
        continue;
      }
      result.weightsIndex = { ...fetched.metadata, totalSizeBytes };
      result.rawWeightsIndex = parsed;
    }
  }
  if (failureState) result.lifecycleState = failureState;
  result.failureCodes.sort();
  return result;
}

const BLOCKER_FIELDS = [
  ['QSA_INDEXER', ['indexer_budget', 'indexer_compress_ratio', 'indexer_head_dim', 'indexer_kv_heads', 'indexer_n_heads']],
  ['NGRAM_RESIDENCY', ['ngram_size', 'ngram_vocab_size_base', 'heads_per_ngram', 'ple_embed_dim', 'ple_layer_ids', 'ple_conv_kernel_size', 'split_ngram_parts']],
  ['MTP_RESIDENCY', ['mtp', 'mtp_num_hidden_layers', 'mtp_use_dedicated_embeddings']],
  ['HC_ACTIVATION', ['hc_count', 'hc_lowrank', 'output_gate_type']],
];

export function detectCapabilityBlockers(raw) {
  if (typeof fitllmEngine.detectCapabilityBlockers === 'function') {
    const classified = fitllmEngine.detectCapabilityBlockers(raw);
    return classified?.unsupportedComponents?.length ? classified : null;
  }
  if (!isPlainObject(raw)) return null;
  const text = isPlainObject(raw.text_config) ? raw.text_config : raw;
  const unsupportedComponents = [];
  for (const [component, fields] of BLOCKER_FIELDS) {
    if (fields.some((field) => hasOwn(text, field))) unsupportedComponents.push(component);
  }
  const multimodalFields = ['image_token_id', 'video_token_id', 'vision_start_token_id', 'vision_end_token_id'];
  if (isPlainObject(raw.vision_config) || multimodalFields.some((field) => hasOwn(raw, field))) {
    unsupportedComponents.push('MULTIMODAL_WORKLOAD');
  }
  if (unsupportedComponents.length === 0) return null;
  return { state: 'UNSUPPORTED_ARCHITECTURE', unsupportedComponents, numericResult: null };
}

export function classifyCapability(evidence) {
  if (!evidence || evidence.lifecycleState !== 'EVIDENCE_PINNED') return null;
  if (evidence.pipelineTag === null || evidence.pipelineTag === undefined) {
    return { state: 'CAPABILITY_UNKNOWN', failureCodes: ['PIPELINE_TAG_UNKNOWN'], numericResult: null };
  }
  const blockers = detectCapabilityBlockers(evidence.rawConfig);
  if (blockers) return blockers;
  try {
    fitllmEngine.parseHfConfig(evidence.id, evidence.rawConfig, evidence.weightsIndex?.totalSizeBytes);
    return {
      state: 'SUPPORTED_BY_CURRENT_ENGINE',
      lifecycleState: 'AWAITING_GOLDEN_VECTOR',
      numericResult: null,
    };
  } catch {
    return { state: 'CAPABILITY_UNKNOWN', failureCodes: ['PARSER_REJECTED'], numericResult: null };
  }
}

function cleanEvidenceSlot(slot, extra = {}) {
  if (!slot) return null;
  return {
    url: slot.url,
    bytes: slot.bytes,
    etag: slot.etag,
    sha256: slot.sha256,
    ...extra,
  };
}

export function buildEvidenceManifest(evidence, capability, verifierSchemaVersion = 'capability-v1') {
  const revision = evidence?.revision && REVISION_RE.test(evidence.revision) ? evidence.revision : null;
  const modelId = evidence?.id ?? null;
  const lifecycleState = evidence?.lifecycleState === 'EVIDENCE_PINNED'
    ? (capability?.lifecycleState || capability?.state || 'CAPABILITY_UNKNOWN')
    : (evidence?.lifecycleState || 'INSUFFICIENT_EVIDENCE');
  const capabilityRecord = capability ? {
    state: capability.state,
    ...(capability.lifecycleState ? { lifecycleState: capability.lifecycleState } : {}),
    ...(capability.unsupportedComponents ? { unsupportedComponents: [...capability.unsupportedComponents] } : {}),
    ...(capability.failureCodes ? { failureCodes: [...capability.failureCodes].sort() } : {}),
    numericResult: null,
  } : null;
  return {
    schemaVersion: EVIDENCE_SCHEMA,
    candidateId: `${modelId}@${revision || 'unverified'}`,
    idempotencyKey: `${modelId}@${revision || 'unverified'}#${verifierSchemaVersion}`,
    verifierSchemaVersion,
    discoverySources: [...new Set(evidence?.discoverySources || [])].sort(),
    identityEvidenceUrl: evidence?.identityEvidenceUrl ?? null,
    officialModelUrl: modelId ? `${HF_ORIGIN}/${modelId}` : null,
    hfCreatedAt: evidence?.createdAt ?? null,
    hfRevision: revision,
    pipelineTag: evidence?.pipelineTag ?? null,
    checkpointKind: evidence?.checkpointKind ?? null,
    announcementUrl: null,
    announcementDate: null,
    lifecycleState,
    failureCodes: [...new Set(evidence?.failureCodes || [])].sort(),
    config: cleanEvidenceSlot(evidence?.config),
    weightsIndex: cleanEvidenceSlot(evidence?.weightsIndex, evidence?.weightsIndex
      ? { totalSizeBytes: evidence.weightsIndex.totalSizeBytes } : {}),
    license: evidence?.license ? {
      id: evidence.license.id,
      ...cleanEvidenceSlot(evidence.license),
    } : null,
    capability: capabilityRecord,
  };
}

function canonicalize(value) {
  if (Array.isArray(value)) return value.map(canonicalize);
  if (!isPlainObject(value)) return value;
  return Object.fromEntries(Object.keys(value).sort().map((key) => [key, canonicalize(value[key])]));
}

export function canonicalJson(value) {
  return JSON.stringify(canonicalize(value));
}

export function sha256CanonicalManifest(manifest) {
  return sha256(Buffer.from(canonicalJson(manifest), 'utf8'));
}

function modelIdFromManifest(manifest) {
  const separator = manifest.candidateId.lastIndexOf('@');
  return manifest.candidateId.slice(0, separator);
}

export function buildIssueBotBlock(manifest, artifactRef = null) {
  const modelId = canonicalCandidateId({ id: modelIdFromManifest(manifest) });
  const digest = sha256CanonicalManifest(manifest);
  let artifactLines = ['- Run artifact: pending explicit apply workflow upload'];
  if (artifactRef?.url) {
    let artifactUrl;
    try { artifactUrl = new URL(artifactRef.url); } catch { throw new Error('invalid artifact URL'); }
    if (artifactUrl.protocol !== 'https:' || artifactUrl.username || artifactUrl.password) throw new Error('invalid artifact URL');
    if (!/^(?:sha256:)?[0-9a-f]{64}$/i.test(artifactRef.digest || '')) throw new Error('invalid artifact digest');
    artifactLines = [`- Run artifact: <${artifactUrl.toString()}>`, `- Artifact digest: \`${artifactRef.digest}\``];
  }
  return [
    ISSUE_BEGIN,
    `<!-- fitllm-day0:model=${modelId} -->`,
    `<!-- fitllm-day0:digest=${digest} -->`,
    '## FitLLM day-0 evidence (bot-managed)',
    '',
    `- Official model: ${manifest.officialModelUrl}`,
    `- Pinned revision: \`${manifest.hfRevision || 'unverified'}\``,
    `- Lifecycle: \`${manifest.lifecycleState}\``,
    `- Manifest SHA-256: \`${digest}\``,
    ...artifactLines,
    '',
    'Canonical evidence manifest (numeric publication is intentionally disabled):',
    '',
    '<pre><code>',
    canonicalJson(manifest).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;'),
    '</code></pre>',
    ISSUE_END,
  ].join('\n');
}

export function mergeIssueBotBlock(existingBody, nextBlock) {
  const body = typeof existingBody === 'string' ? existingBody : '';
  const start = body.indexOf(ISSUE_BEGIN);
  const end = body.indexOf(ISSUE_END, start + ISSUE_BEGIN.length);
  if (start >= 0 && end >= 0) return `${body.slice(0, start)}${nextBlock}${body.slice(end + ISSUE_END.length)}`;
  if (!body) return nextBlock;
  return `${body}\n\n${nextBlock}`;
}

function currentBotBlock(body) {
  const start = String(body || '').indexOf(ISSUE_BEGIN);
  const end = String(body || '').indexOf(ISSUE_END, start + ISSUE_BEGIN.length);
  return start >= 0 && end >= 0 ? String(body).slice(start, end + ISSUE_END.length) : null;
}

function digestFromBody(body) {
  return String(body || '').match(/<!-- fitllm-day0:digest=([0-9a-f]{64}) -->/)?.[1] || null;
}

function issueForModel(existingIssues, modelId) {
  return existingIssues.find((issue) => !issue.pull_request
    && (String(issue.body || '').includes(`<!-- fitllm-day0:model=${modelId} -->`)
      || issue.title === `day0: ${modelId}`));
}

export function planIssueMutations(records, existingIssues, { maxMutations = 3, artifactRef = null } = {}) {
  if (!Number.isInteger(maxMutations) || maxMutations < 0 || maxMutations > 3) throw new Error('maxMutations must be 0..3');
  const operations = [];
  const dropped = [];
  let mutationCount = 0;
  for (const record of records) {
    const manifest = record.manifest || record;
    const modelId = modelIdFromManifest(manifest);
    canonicalCandidateId({ id: modelId });
    const digest = sha256CanonicalManifest(manifest);
    const issue = issueForModel(existingIssues, modelId);
    if (issue && digestFromBody(issue.body) === digest) {
      operations.push({ action: 'noop', modelId, revision: manifest.hfRevision, manifestDigest: digest, issueNumber: issue.number });
      continue;
    }
    if (mutationCount >= maxMutations) {
      dropped.push({ candidateId: manifest.candidateId, reason: 'MUTATION_LIMIT' });
      continue;
    }
    const block = buildIssueBotBlock(manifest, artifactRef);
    if (!issue) {
      operations.push({
        action: 'create', modelId, revision: manifest.hfRevision, manifestDigest: digest, manifest,
        title: `day0: ${modelId}`, body: block, labels: ['day0-candidate'],
      });
    } else {
      operations.push({
        action: 'update', modelId, revision: manifest.hfRevision, manifestDigest: digest, manifest,
        issueNumber: issue.number, beforeBodySha256: sha256(Buffer.from(String(issue.body || ''), 'utf8')),
        previousBotBlock: currentBotBlock(issue.body), beforeBody: String(issue.body || ''),
        body: mergeIssueBotBlock(issue.body, block),
      });
    }
    mutationCount += 1;
  }
  return { operations, mutationCount, dropped };
}

export function attachArtifactRef(plan, artifactRef) {
  return {
    ...plan,
    operations: plan.operations.map((operation) => {
      if (operation.action === 'noop') return operation;
      const block = buildIssueBotBlock(operation.manifest, artifactRef);
      return {
        ...operation,
        body: operation.action === 'update' ? mergeIssueBotBlock(operation.beforeBody, block) : block,
      };
    }),
  };
}

async function readCurrentModel(hfClient, modelId) {
  if (typeof hfClient === 'function') return hfClient(modelId);
  if (typeof hfClient?.getModel === 'function') return hfClient.getModel(modelId);
  throw new Error('hfClient must be a function or expose getModel');
}

async function ensureIssueLabel(ghClient) {
  try {
    await ghClient.request('/labels/day0-candidate');
  } catch (error) {
    if (error?.status !== 404) throw error;
    await ghClient.request('/labels', {
      method: 'POST',
      body: {
        name: 'day0-candidate', color: '2f6f5e',
        description: 'Official HF release awaiting evidence review',
      },
    });
  }
}

export async function applyIssuePlan(plan, ghClient, hfClient) {
  if (!Array.isArray(plan?.operations)) throw new Error('invalid issue plan');
  for (const operation of plan.operations) {
    if (!['create', 'update', 'noop'].includes(operation.action)) throw new Error(`invalid issue action: ${operation.action}`);
    canonicalCandidateId({ id: operation.modelId });
    if (operation.action === 'noop') continue;
    if (!REVISION_RE.test(operation.revision || '')
        || !operation.manifest
        || operation.manifestDigest !== sha256CanonicalManifest(operation.manifest)
        || modelIdFromManifest(operation.manifest) !== operation.modelId
        || operation.manifest.hfRevision !== operation.revision
        || typeof operation.body !== 'string'
        || !operation.body.includes(`<!-- fitllm-day0:model=${operation.modelId} -->`)) {
      throw new Error(`invalid issue plan operation: ${operation.modelId}`);
    }
    if (operation.action === 'create'
        && (operation.title !== `day0: ${operation.modelId}` || canonicalJson(operation.labels) !== '["day0-candidate"]')) {
      throw new Error(`invalid issue create operation: ${operation.modelId}`);
    }
    if (operation.action === 'update'
        && (!Number.isInteger(operation.issueNumber) || operation.issueNumber <= 0
          || !/^[0-9a-f]{64}$/.test(operation.beforeBodySha256 || ''))) {
      throw new Error(`invalid issue update operation: ${operation.modelId}`);
    }
  }
  const mutations = plan.operations.filter((operation) => operation.action === 'create' || operation.action === 'update');
  if (mutations.length > 3) throw new Error('refusing more than three issue mutations');
  const results = [];
  let labelReady = false;
  for (const operation of plan.operations) {
    if (operation.action === 'noop') {
      results.push({ modelId: operation.modelId, action: 'noop', status: 'NOOP' });
      continue;
    }
    if (operation.action === 'update') {
      const currentIssue = await ghClient.request(`/issues/${operation.issueNumber}`);
      const currentSha = sha256(Buffer.from(String(currentIssue?.body || ''), 'utf8'));
      if (currentSha !== operation.beforeBodySha256) {
        results.push({ modelId: operation.modelId, action: operation.action, status: 'BODY_CHANGED' });
        continue;
      }
    }
    const currentModel = await readCurrentModel(hfClient, operation.modelId);
    let currentModelId = null;
    try { currentModelId = canonicalCandidateId(currentModel); } catch { /* stale/malformed response */ }
    if (currentModelId !== operation.modelId || currentModel?.sha !== operation.revision) {
      results.push({ modelId: operation.modelId, action: operation.action, status: 'STALE_REVISION' });
      continue;
    }
    if (operation.action === 'create') {
      if (!labelReady) {
        await ensureIssueLabel(ghClient);
        labelReady = true;
      }
      const issue = await ghClient.request('/issues', {
        method: 'POST',
        body: { title: operation.title, body: operation.body, labels: operation.labels },
      });
      results.push({ modelId: operation.modelId, action: 'create', status: 'CREATED', issueNumber: issue?.number ?? null });
    } else {
      await ghClient.request(`/issues/${operation.issueNumber}`, {
        method: 'PATCH',
        body: { body: operation.body },
      });
      results.push({ modelId: operation.modelId, action: 'update', status: 'UPDATED', issueNumber: operation.issueNumber });
    }
  }
  return { results, mutationCount: results.filter(({ status }) => status === 'CREATED' || status === 'UPDATED').length };
}

export async function fetchAllGitHubIssues(ghClient) {
  const issues = [];
  let pathName = '/issues?labels=day0-candidate&state=all&per_page=100&page=1';
  while (pathName) {
    const response = await ghClient.requestWithHeaders(pathName);
    if (!Array.isArray(response.data)) throw new Error('GitHub issues payload is not an array');
    issues.push(...response.data.filter((issue) => !issue.pull_request));
    const link = response.headers?.get?.('link') || '';
    const next = [...link.matchAll(/<([^>]+)>;\s*rel="([^"]+)"/g)].find((match) => match[2] === 'next')?.[1];
    pathName = next ? `${new URL(next).pathname}${new URL(next).search}`.replace(/^\/repos\/[^/]+\/[^/]+/, '') : null;
  }
  return issues;
}

function safeOutputDirectory(outputDir, sourceRoot) {
  const output = path.resolve(outputDir);
  const source = path.resolve(sourceRoot);
  if (output === source || output.startsWith(`${source}${path.sep}`)) {
    throw new Error('day0 output directory must be outside the source tree');
  }
  return output;
}

function safeCandidateDirectory(manifest) {
  return manifest.candidateId.replaceAll('/', '--').replaceAll('@', '--at--').replace(/[^A-Za-z0-9._-]/g, '_');
}

export async function runDay0Watch(deps, options = {}) {
  const {
    policy,
    fetchImpl = fetch,
    ghClient,
    existingIssues: suppliedIssues,
    now = new Date(),
    verifierSchemaVersion = 'capability-v1',
  } = deps;
  const outputDir = safeOutputDirectory(options.outputDir, options.sourceRoot || process.cwd());
  const discovery = await discoverCandidates({ policy, fetchImpl, now });
  const max = policy.maxIssueMutationsPerRun;
  const selected = discovery.candidates.slice(0, max);
  const droppedCandidates = [
    ...discovery.droppedCandidates,
    ...discovery.candidates.slice(max).map((candidate) => ({ candidateId: `${candidate.id}@${candidate.revision || 'unverified'}`, reason: 'MUTATION_LIMIT' })),
  ];
  const records = [];
  for (const candidate of selected) {
    const evidence = await pinEvidence(candidate, { fetchImpl });
    const capability = classifyCapability(evidence);
    const manifest = buildEvidenceManifest(evidence, capability, verifierSchemaVersion);
    records.push({ manifest, digest: sha256CanonicalManifest(manifest) });
  }
  const existingIssues = suppliedIssues || await fetchAllGitHubIssues(ghClient);
  const issuePlan = planIssueMutations(records, existingIssues, { maxMutations: max });
  issuePlan.dropped.push(...droppedCandidates);

  await mkdir(outputDir, { recursive: true });
  for (const record of records) {
    const directory = path.join(outputDir, 'evidence', safeCandidateDirectory(record.manifest));
    await mkdir(directory, { recursive: true });
    await writeFile(path.join(directory, 'evidence-manifest.json'), `${canonicalJson(record.manifest)}\n`, 'utf8');
  }
  const observation = {
    retrievedAt: now.toISOString(),
    githubRunId: options.githubRunId ?? null,
    githubRunUrl: options.githubRunUrl ?? null,
    verifierGitSha: options.verifierGitSha ?? null,
    sourceFailures: discovery.sourceFailures,
    evidenceDigests: records.map(({ manifest, digest }) => ({ candidateId: manifest.candidateId, digest })),
  };
  const summary = {
    mode: options.mode || 'dry-run',
    discovered: discovery.candidates.length,
    evaluated: records.length,
    sourceFailures: discovery.sourceFailures.length,
    issueMutationsPlanned: issuePlan.mutationCount,
    operations: issuePlan.operations.map(({ action, modelId, manifestDigest }) => ({ action, modelId, manifestDigest })),
    droppedCandidates: issuePlan.dropped,
  };
  await Promise.all([
    writeFile(path.join(outputDir, 'run-observation.json'), `${JSON.stringify(observation, null, 2)}\n`, 'utf8'),
    writeFile(path.join(outputDir, 'issue-plan.json'), `${JSON.stringify(issuePlan, null, 2)}\n`, 'utf8'),
    writeFile(path.join(outputDir, 'summary.json'), `${JSON.stringify(summary, null, 2)}\n`, 'utf8'),
  ]);
  return { summary, issuePlan, records, outputDir };
}
