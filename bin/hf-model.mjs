const MAX_JSON_BYTES = 1_000_000;
const DEFAULT_TIMEOUT_MS = 8_000;

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

async function readJsonLimited(response, label) {
  const stated = Number(response.headers.get('content-length'));
  if (Number.isFinite(stated) && stated > MAX_JSON_BYTES) throw new Error(`${label} is too large`);
  const reader = response.body?.getReader?.();
  if (!reader) {
    const buffer = new Uint8Array(await response.arrayBuffer());
    if (buffer.byteLength > MAX_JSON_BYTES) throw new Error(`${label} is too large`);
    return JSON.parse(new TextDecoder().decode(buffer));
  }
  const chunks = [];
  let size = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    size += value.byteLength;
    if (size > MAX_JSON_BYTES) {
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
  const base = `https://huggingface.co/${encoded}/resolve/main`;
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

  const configResponse = checkedResponse(await request(base, 'config.json'), 'config.json');
  const revision = configResponse.headers.get('x-repo-commit');
  if (!/^[a-f0-9]{40}$/i.test(String(revision || ''))) {
    throw new Error('Hugging Face response did not provide an immutable model revision');
  }
  const config = await bounded(() => readJsonLimited(configResponse, 'config.json'));
  const pinnedBase = `https://huggingface.co/${encoded}/resolve/${revision}`;
  let totalSize = null;
  const indexResponse = await request(pinnedBase, 'model.safetensors.index.json');
  if (indexResponse.ok) {
    const index = await bounded(() => readJsonLimited(indexResponse, 'model.safetensors.index.json'));
    totalSize = Number(index?.metadata?.total_size);
  } else if (indexResponse.status === 404) {
    const head = await request(pinnedBase, 'model.safetensors', { method: 'HEAD' });
    if (head.ok) totalSize = Number(head.headers.get('content-length'));
  } else {
    checkedResponse(indexResponse, 'model.safetensors.index.json');
  }
  if (!Number.isFinite(totalSize) || totalSize <= 0) {
    throw new Error('Hugging Face model size is unavailable; refusing to guess');
  }
  return { id, revision, config, totalSize };
}
