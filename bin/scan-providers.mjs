// 설치된 모델을 **읽기만** 하는 provider들. 런타임을 조작하지 않는다 —
// pull·load·unload·설정 쓰기는 전부 범위 밖이고, 목록 조회만 한다.
// 사용자 기계에서 도는 도구가 그 기계의 상태를 바꾸면 신뢰를 잃는다.
//
// provider 계약: listInstalled(deps) -> { provider, ok, models, note }
//   ok:false 는 "런타임이 없거나 꺼져 있다"는 정상 결과다. 에러가 아니다.
//   models[] 는 런타임이 **실제로 알려준 것**만 담는다. 빠진 필드를 추측으로 채우지 않는다.
import { execFileSync } from 'node:child_process';

// 루프백만. provider 가 외부 호스트를 부르면 그건 버그가 아니라 신뢰 위반이다.
const OLLAMA_DEFAULT_HOST = '127.0.0.1:11434';
const TIMEOUT_MS = 1500;

function ollamaBase(env) {
  // OLLAMA_HOST 는 사용자가 이미 쓰는 관례라 존중하되, 루프백이 아니면 거부한다.
  const raw = (env.OLLAMA_HOST || OLLAMA_DEFAULT_HOST).trim();
  const withScheme = /^https?:\/\//i.test(raw) ? raw : `http://${raw}`;
  let url;
  try { url = new URL(withScheme); } catch { return null; }
  const host = url.hostname.toLowerCase();
  if (!['127.0.0.1', 'localhost', '::1', '0.0.0.0'].includes(host)) return null;
  return `${url.protocol}//${url.host}`;
}

// 공식 스키마(GET /api/tags): models[].{name,model,size,digest,details.{family,parameter_size,quantization_level,format}}
// 모양이 다른 항목은 **그 항목만** 버리고 나머지는 계속 처리한다 — 한 줄이 이상하다고
// 사용자의 나머지 모델을 통째로 못 보게 만들 이유가 없다.
function normalizeOllamaModel(entry) {
  if (!entry || typeof entry !== 'object') return null;
  const id = typeof entry.model === 'string' && entry.model
    ? entry.model
    : (typeof entry.name === 'string' ? entry.name : '');
  if (!id) return null;
  const details = (entry.details && typeof entry.details === 'object') ? entry.details : {};
  return {
    id,
    sizeBytes: Number.isFinite(entry.size) && entry.size >= 0 ? entry.size : null,
    quant: typeof details.quantization_level === 'string' ? details.quantization_level : null,
    paramSizeLabel: typeof details.parameter_size === 'string' ? details.parameter_size : null,
    family: typeof details.family === 'string' ? details.family : null,
    format: typeof details.format === 'string' ? details.format : null,
  };
}

export async function listOllama({ fetchImpl = fetch, env = process.env } = {}) {
  const base = ollamaBase(env);
  if (!base) {
    return { provider: 'ollama', ok: false, models: [], note: 'OLLAMA_HOST is not a loopback address; refusing to query it' };
  }
  let payload;
  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);
    try {
      const response = await fetchImpl(`${base}/api/tags`, { method: 'GET', signal: controller.signal });
      if (!response || response.ok !== true) return { provider: 'ollama', ok: false, models: [], note: 'not running' };
      payload = await response.json();
    } finally { clearTimeout(timer); }
  } catch {
    // 연결거부·타임아웃 = 안 돌고 있다. 사용자에게 스택트레이스를 보여줄 일이 아니다.
    return { provider: 'ollama', ok: false, models: [], note: 'not running' };
  }
  const raw = Array.isArray(payload?.models) ? payload.models : [];
  const models = raw.map(normalizeOllamaModel).filter(Boolean);
  return { provider: 'ollama', ok: true, models, note: null };
}

// LM Studio: `lms ls --json`. 이 개발 머신에 LM Studio 가 없어 **실제 실행으로 검증하지 못했다** —
// 픽스처 테스트만 있다. README 와 보고에도 그대로 적는다.
export function listLmStudio({ execImpl = execFileSync } = {}) {
  let stdout;
  try {
    stdout = execImpl('lms', ['ls', '--json'], { encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'], timeout: 5000 });
  } catch {
    return { provider: 'lm-studio', ok: false, models: [], note: 'lms CLI not found' };
  }
  let parsed;
  try { parsed = JSON.parse(stdout); } catch {
    return { provider: 'lm-studio', ok: false, models: [], note: 'lms returned output this version cannot read' };
  }
  const raw = Array.isArray(parsed) ? parsed : (Array.isArray(parsed?.models) ? parsed.models : []);
  const models = raw.map((entry) => {
    if (!entry || typeof entry !== 'object') return null;
    const id = ['modelKey', 'path', 'name'].map((k) => entry[k]).find((v) => typeof v === 'string' && v);
    if (!id) return null;
    return {
      id,
      sizeBytes: Number.isFinite(entry.sizeBytes) ? entry.sizeBytes : (Number.isFinite(entry.size) ? entry.size : null),
      quant: typeof entry.quantization === 'string' ? entry.quantization : null,
      paramSizeLabel: typeof entry.paramsString === 'string' ? entry.paramsString : null,
      family: typeof entry.architecture === 'string' ? entry.architecture : null,
      format: typeof entry.format === 'string' ? entry.format : null,
    };
  }).filter(Boolean);
  return { provider: 'lm-studio', ok: true, models, note: null };
}

export const PROVIDERS = Object.freeze([
  { name: 'ollama', run: (deps) => listOllama(deps) },
  { name: 'lm-studio', run: (deps) => listLmStudio(deps) },
]);

export async function listAllInstalled(deps = {}) {
  const results = [];
  for (const provider of PROVIDERS) results.push(await provider.run(deps));
  return results;
}
