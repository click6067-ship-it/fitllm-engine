const MAX_JSON_BYTES = 1_000_000;
const MAX_METADATA_BYTES = 5_000_000;
const MAX_INDEX_BYTES = 12_000_000;
const DEFAULT_TIMEOUT_MS = 8_000;
const COMMIT_SHA_RE = /^(?:[a-f0-9]{40}|[a-f0-9]{64})$/i;

function positiveSafeInteger(value) {
  return Number.isSafeInteger(value) && value > 0;
}

export function parseHfId(input) {
  const value = String(input || '').trim();
  let candidate = value;
  if (/^https?:\/\//i.test(value)) {
    try {
      const url = new URL(value);
      if (url.protocol !== 'https:' || url.hostname !== 'huggingface.co') return null;
      const parts = url.pathname.split('/').filter(Boolean);
      if (parts.length < 2) return null;
      candidate = `${parts[0]}/${parts[1]}`;
    } catch {
      return null;
    }
  }
  if (!/^[A-Za-z0-9][A-Za-z0-9._-]{0,95}\/[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/.test(candidate)) return null;
  return candidate;
}

async function withDeadline(promise, controller, timeoutMs) {
  let timer;
  try {
    return await Promise.race([
      promise,
      new Promise((_, reject) => {
        timer = setTimeout(() => {
          controller.abort();
          reject(new Error('Hugging Face request timed out'));
        }, timeoutMs);
      }),
    ]);
  } finally {
    clearTimeout(timer);
  }
}

async function readJsonLimited(response, label, maxBytes = MAX_JSON_BYTES) {
  const stated = Number(response.headers.get('content-length'));
  if (Number.isFinite(stated) && stated > maxBytes) throw new Error(`${label} is too large`);
  const reader = response.body?.getReader?.();
  if (!reader) {
    const buffer = new Uint8Array(await response.arrayBuffer());
    if (buffer.byteLength > maxBytes) throw new Error(`${label} is too large`);
    return JSON.parse(new TextDecoder().decode(buffer));
  }
  const chunks = [];
  let size = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    size += value.byteLength;
    if (size > maxBytes) {
      await reader.cancel();
      throw new Error(`${label} is too large`);
    }
    chunks.push(value);
  }
  const bytes = new Uint8Array(size);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  try {
    return JSON.parse(new TextDecoder().decode(bytes));
  } catch {
    throw new Error(`${label} is not valid JSON`);
  }
}

function checkedResponse(response, label) {
  if ([401, 403, 404].includes(response.status)) throw new Error(`${label} is gated, private, or not found`);
  if (!response.ok) throw new Error(`${label} request failed with HTTP ${response.status}`);
  return response;
}

export async function fetchHfModel(input, { fetchImpl = globalThis.fetch, timeoutMs = DEFAULT_TIMEOUT_MS } = {}) {
  const id = parseHfId(input);
  if (!id) throw new Error('expected a Hugging Face ID like org/model or a huggingface.co model URL');
  if (typeof fetchImpl !== 'function') throw new Error('fetch is unavailable in this Node runtime');
  const encoded = id.split('/').map(encodeURIComponent).join('/');
  const controller = new AbortController();
  const deadline = Date.now() + timeoutMs;
  const bounded = (operation) => {
    const remaining = deadline - Date.now();
    if (remaining <= 0) {
      controller.abort();
      return Promise.reject(new Error('Hugging Face request timed out'));
    }
    return withDeadline(Promise.resolve().then(operation), controller, remaining);
  };
  const request = (root, path, options = {}) => bounded(
    () => fetchImpl(`${root}/${path}`, { redirect: 'follow', ...options, signal: controller.signal }),
  );

  // model metadata를 먼저 읽어 immutable revision을 봉인한다. config·index·shard가 서로 다른
  // main 시점을 가리키면 그럴듯하지만 서로 맞지 않는 증거 묶음이 만들어진다.
  const metaResponse = checkedResponse(
    await bounded(() => fetchImpl(
      `https://huggingface.co/api/models/${encoded}?blobs=true`,
      { redirect: 'follow', signal: controller.signal },
    )),
    'model metadata',
  );
  const metadata = await bounded(() => readJsonLimited(metaResponse, 'model metadata', MAX_METADATA_BYTES));
  const revision = String(metadata?.sha || '');
  if (!COMMIT_SHA_RE.test(revision)) {
    throw new Error('Hugging Face response did not provide an immutable model revision');
  }
  const pinnedBase = `https://huggingface.co/${encoded}/resolve/${revision}`;

  const configResponse = checkedResponse(await request(pinnedBase, 'config.json'), 'config.json');
  const config = await bounded(() => readJsonLimited(configResponse, 'config.json'));

  // 실제 checkpoint bytes는 blob 크기 합계로만 구한다. index의 metadata.total_size와
  // safetensors.total은 레포에 따라 parameter/element 수라서 바이트로 쓰면 2배 과소계산이 난다.
  // https://huggingface.co/docs/hub/api#get-apimodelsrepoid
  const siblingSizes = new Map(
    (Array.isArray(metadata?.siblings) ? metadata.siblings : [])
      .filter((file) => typeof file?.rfilename === 'string' && positiveSafeInteger(file?.size))
      .map((file) => [file.rfilename, file.size]),
  );
  // index가 있으면 그 레포는 샤드 레포다 — 샤드 합계를 못 만들면 크기를 포기하지, 루트
  // model.safetensors로 내려가지 않는다. "original + fp8" 류 레포는 루트에 훨씬 작은 잔여
  // 파일을 두는데, 거기로 폴백하면 30B급 레포를 몇 B로 읽어 다시 거짓 FITS가 된다.
  // (web 프록시 api/hf-config.js와 같은 정책 — 두 표면이 같은 모델에 다른 답을 내면 안 된다.)
  let totalSize = null;
  const indexResponse = await request(pinnedBase, 'model.safetensors.index.json');
  if (indexResponse.ok) {
    const index = await bounded(() => readJsonLimited(indexResponse, 'model.safetensors.index.json', MAX_INDEX_BYTES));
    const shardNames = [...new Set(Object.values(index?.weight_map || {}))];
    if (shardNames.length && shardNames.every((name) => typeof name === 'string' && siblingSizes.has(name))) {
      const sum = shardNames.reduce((acc, name) => acc + siblingSizes.get(name), 0);
      if (positiveSafeInteger(sum)) totalSize = sum;
    }
  } else if (indexResponse.status === 404) {
    if (positiveSafeInteger(siblingSizes.get('model.safetensors'))) {
      totalSize = siblingSizes.get('model.safetensors');
    }
    if (!totalSize) {
      const head = await request(pinnedBase, 'model.safetensors', { method: 'HEAD' });
      if (head.ok) {
        const sz = Number(head.headers.get('x-linked-size') || head.headers.get('content-length'));
        if (positiveSafeInteger(sz)) totalSize = sz;
      }
    }
  } else {
    checkedResponse(indexResponse, 'model.safetensors.index.json');
  }

  const rawParameters = metadata?.safetensors?.parameters;
  const safetensorsParameters = rawParameters
    && typeof rawParameters === 'object'
    && !Array.isArray(rawParameters)
    && Object.keys(rawParameters).length > 0
    ? rawParameters
    : null;
  const safetensorsTotal = metadata?.safetensors?.total ?? null;
  if (!safetensorsParameters && !totalSize) {
    throw new Error('Hugging Face model size is unavailable; refusing to guess');
  }
  const parameterEvidence = safetensorsParameters
    ? { revision, safetensorsParameters, safetensorsTotal }
    : null;
  return { id, revision, config, totalSize, parameterEvidence };
}
