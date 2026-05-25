// ============================================================================
//  FitLLM — Memory Engine
//  검증된 계산 엔진(v1에서 실모델 config.json 대조로 교정한 값/식)을 그대로 이식.
//  데이터·수식은 신뢰의 핵 → 건드리지 않음. v2는 이 위에 직관적 UI만 새로 얹는다.
//  벤치마크: 공개·검증된 GPQA(Diamond) · MMLU-Pro · SWE-Bench(Verified)만 사용.
// ============================================================================

export const MODELS = [
  // === Qwen 3.6 27B (dense, 하이브리드 linear+full attn ~3:1) — HF config.json ===
  {
    name: 'Qwen 3.6 27B',
    group: 'Qwen 3.6',
    tags: ['dense'],
    totalParams: 27.2,
    activeParams: null,
    layerCount: 64,
    fullAttnLayers: 16, // 64레이어 중 16개만 Gated(full) attention, 48개는 Gated DeltaNet(linear) → KV 캐시는 16레이어만
    kvHeads: 4,
    kvHeadDim: 256,
    attnHeads: 24,
    hiddenSize: 5120,
    maxContext: 262144,
    benchmarks: { GPQA: 0.878, 'MMLU-Pro': 0.817, 'SWE-Bench': 0.772 },
    desc: 'Dense · 64레이어(풀어텐션 16 + linear 48, 1:3) · KV는 16레이어만 · 최대 256K',
  },
  // === Qwen 3.6 35B-A3B (MoE) — HF config.json (qwen3_5_moe) ===
  {
    name: 'Qwen 3.6 35B-A3B',
    group: 'Qwen 3.6',
    tags: ['moe'],
    totalParams: 35.0,
    activeParams: 3.0,
    layerCount: 40,
    fullAttnLayers: 10, // 40레이어 중 10개만 full attention(4개마다 1개), 30개는 linear attention → KV 캐시는 10레이어만
    kvHeads: 2,
    kvHeadDim: 256,
    attnHeads: 16,
    hiddenSize: 2048,
    numExperts: 256,
    expertsPerToken: 8,
    maxContext: 262144,
    benchmarks: { GPQA: 0.86, 'MMLU-Pro': 0.852, 'SWE-Bench': null },
    desc: 'MoE · ~35B total / ~3B active · 256 experts · 풀어텐션 10/40(linear 30) · 최대 256K',
  },
  // === Gemma 4 E2B (Dense + Per-Layer-Embeddings, NOT MoE) — HF config.json ===
  {
    name: 'Gemma 4 e2b',
    group: 'Gemma 4',
    tags: ['dense', 'ple'],
    totalParams: 5.1,
    activeParams: 2.3,
    layerCount: 35,
    kvHeads: 1,
    kvHeadDim: 256,
    attnHeads: 8,
    hiddenSize: 1536,
    maxContext: 131072,
    slidingWindow: 512,
    slidingPattern: '5:1',
    benchmarks: { GPQA: 0.3, 'MMLU-Pro': 0.68, 'SWE-Bench': null },
    desc: 'Dense+PLE · 5.1B raw / 2.3B 유효 · 슬라이딩윈도우 512(5:1) · 최대 128K · 벤치 근사치',
  },
  // === Gemma 4 E4B (Dense + PLE) — HF config.json ===
  {
    name: 'Gemma 4 e4b',
    group: 'Gemma 4',
    tags: ['dense', 'ple'],
    totalParams: 8,
    activeParams: 4.5,
    layerCount: 42,
    kvHeads: 2,
    kvHeadDim: 256,
    attnHeads: 8,
    hiddenSize: 2560,
    maxContext: 131072,
    slidingWindow: 512,
    slidingPattern: '5:1',
    benchmarks: { GPQA: 0.586, 'MMLU-Pro': 0.694, 'SWE-Bench': null },
    desc: 'Dense+PLE · 8B raw / 4.5B 유효 · 슬라이딩윈도우 512(5:1) · 최대 128K',
  },
  // === Gemma 4 26B A4B (true MoE) — HF config.json ===
  {
    name: 'Gemma 4 26b A4B',
    group: 'Gemma 4',
    tags: ['moe'],
    totalParams: 25.5,
    activeParams: 4.0,
    layerCount: 30,
    kvHeads: 8, // 슬라이딩 레이어: KV헤드 8 × head_dim 256
    kvHeadDim: 256,
    globalKvHeads: 2, // 글로벌(풀어텐션) 5레이어: KV헤드 2 × head_dim 512 (이종)
    globalHeadDim: 512,
    attnHeads: 16,
    hiddenSize: 2816,
    numExperts: 128,
    expertsPerToken: 8,
    maxContext: 262144,
    slidingWindow: 1024,
    slidingPattern: '5:1',
    benchmarks: { GPQA: 0.823, 'MMLU-Pro': 0.826, 'SWE-Bench': null },
    desc: 'MoE · ~26B total / ~4B active · 128 experts · 슬라이딩윈도우 1024(5:1, 글로벌 head_dim 512) · 최대 256K',
  },
  // === Gemma 4 31B (dense) — HF config.json ===
  {
    name: 'Gemma 4 31b',
    group: 'Gemma 4',
    tags: ['dense'],
    totalParams: 30.7,
    activeParams: 30.7,
    layerCount: 60,
    kvHeads: 16, // 슬라이딩 50레이어: KV헤드 16 × head_dim 256
    kvHeadDim: 256,
    globalKvHeads: 4, // 글로벌(풀어텐션) 10레이어: KV헤드 4 × head_dim 512 (이종)
    globalHeadDim: 512,
    attnHeads: 32,
    hiddenSize: 5376,
    maxContext: 262144,
    slidingWindow: 1024,
    slidingPattern: '5:1',
    benchmarks: { GPQA: 0.843, 'MMLU-Pro': 0.852, 'SWE-Bench': null },
    desc: 'Dense · 30.7B · 60레이어 · 슬라이딩윈도우 1024(5:1, 글로벌 10레이어 head_dim 512) · 최대 256K',
  },
  // === Claude Opus 4.7 — Cloud (벤치마크 기준점, 메모리 시뮬 제외) ===
  {
    name: 'Claude Opus 4.7',
    group: 'Claude (Cloud)',
    tags: ['cloud', 'dense'],
    totalParams: null,
    activeParams: null,
    layerCount: null,
    kvHeads: null,
    kvHeadDim: null,
    attnHeads: null,
    hiddenSize: null,
    maxContext: 1000000,
    isCloud: true,
    benchmarks: { GPQA: 0.942, 'MMLU-Pro': 0.899, 'SWE-Bench': 0.876 },
    contextLimit: '1M',
    desc: 'Cloud 모델 — 벤치마크 기준점 (로컬 설치 불가, 비교용)',
  },
];

// 로컬에서 돌릴 수 있는(시뮬 대상) 모델만
export const LOCAL_MODELS = MODELS.filter((m) => !m.isCloud);

// ===== 맥북 RAM 옵션 =====
export const MACBOOK_RAM_GROUPS = {
  'M5 Pro': [24, 48, 64],
  'M5 Max': [36, 48, 64, 128],
};

// ===== 정밀도(양자화) 옵션 =====
export const QUANT_OPTIONS = [
  { bits: 4, label: '4bit', sub: 'NVFP4 / Q4 — 가장 작음' },
  { bits: 8, label: '8bit', sub: 'MXFP8 / Q8 — 권장 균형' },
  { bits: 16, label: '16bit', sub: 'BF16 — 원본 정밀도' },
];

// ===== macOS 기본 메모리(통합 메모리, Apple Silicon) =====
// macOS + 기본 프로세스 + 로컬 LLM 데몬 ≈ 6~7GB. RAM이 적을수록 비중↑.
export function getOsOverhead(ram) {
  if (ram <= 24) return 7;
  if (ram <= 36) return 6.5;
  if (ram <= 48) return 6;
  return 6;
}

// 5:1 인터리브(슬라이딩 윈도우): 로컬 레이어(5/6)는 윈도우만, 글로벌(1/6)은 전체 컨텍스트 KV 유지.
export function slidingSplit(model) {
  let globalLayers;
  if (model.globalAttnLayers) {
    globalLayers = model.globalAttnLayers; // HF config의 full_attention 레이어 수(명시)
  } else {
    const ratio = model.slidingPattern ? parseInt(model.slidingPattern.split(':')[0]) : 5;
    globalLayers = Math.max(1, Math.round(model.layerCount / (ratio + 1)));
  }
  return { globalLayers, localLayers: model.layerCount - globalLayers };
}

export function calcKVCache(model, ctx, bits) {
  if (!model.kvHeads || !model.kvHeadDim || !model.layerCount) {
    return { totalGB: 0, perTokenKB: 0, kvPerToken: 0, totalBytes: 0, effectiveCtx: 0 };
  }
  const bpe = bits / 8;
  // 레이어·토큰당 바이트. Gemma 4는 슬라이딩 레이어(kvHeads×kvHeadDim)와
  // 글로벌 레이어(globalKvHeads×globalHeadDim)의 헤드 구성이 다름(이종).
  const perLocal = 2 * model.kvHeads * model.kvHeadDim * bpe;
  const perGlobal = 2 * (model.globalKvHeads || model.kvHeads) * (model.globalHeadDim || model.kvHeadDim) * bpe;

  let totalBytes, marginalPerToken;
  if ((model.slidingWindow || 0) > 0) {
    // 슬라이딩 윈도우(Gemma): 로컬 레이어는 윈도우만큼(head_dim 256), 글로벌 레이어는 전체 ctx(head_dim 512 가능)
    const { globalLayers, localLayers } = slidingSplit(model);
    totalBytes = perGlobal * globalLayers * ctx + perLocal * localLayers * Math.min(ctx, model.slidingWindow);
    marginalPerToken = perGlobal * globalLayers; // 윈도우 초과 후 1토큰 추가 비용(글로벌 레이어만 증가)
  } else {
    // 하이브리드(Qwen 3.6): linear attention 레이어는 ctx 비례 KV가 없음 → 풀어텐션 레이어만
    const attnLayers = model.fullAttnLayers || model.layerCount;
    totalBytes = perLocal * attnLayers * ctx;
    marginalPerToken = perLocal * attnLayers;
  }
  return {
    totalGB: totalBytes / 1024 ** 3,
    perTokenKB: marginalPerToken / 1024, // 1토큰 추가 시 증가량(관련 레이어 합산)
    kvPerToken: marginalPerToken,
    totalBytes,
    effectiveCtx: ctx,
  };
}

// 벤치마크 평균 — 숫자값만(null/미공개 건너뜀)
export function benchAvg(benchmarks) {
  if (!benchmarks) return null;
  const vals = Object.values(benchmarks).filter((v) => typeof v === 'number' && !isNaN(v));
  if (!vals.length) return null;
  return vals.reduce((s, v) => s + v, 0) / vals.length;
}

// 양자화별 메모리 보정(공개 벤치 기준 — 작은 모델일수록 오버헤드↑)
const quantAdjust = {
  'Gemma 4 e2b': { 16: 0.94, 8: 0.9, 4: 1.26 },
  'Gemma 4 e4b': { 16: 0.94, 8: 0.94, 4: 1.25 },
  'Gemma 4 31b': { 16: 0.95, 8: 0.99, 4: 1.13 },
  'Gemma 4 26b A4B': { 16: 0.95, 8: 0.99, 4: 1.24 },
};

export function calcParamMemory(model, bits) {
  if (!model.totalParams) return { totalGB: 0, activeGB: null };
  const bpe = bits / 8;
  const baseTotalGB = (model.totalParams * 1e9 * bpe) / 1024 ** 3;
  const baseActiveGB = model.activeParams ? (model.activeParams * 1e9 * bpe) / 1024 ** 3 : null;
  const multiplier = (quantAdjust[model.name] && quantAdjust[model.name][bits]) || 1.0;
  return { totalGB: baseTotalGB * multiplier, activeGB: baseActiveGB ? baseActiveGB * multiplier : null };
}

// 런타임 오버헤드: 양자화 메타(12%) + KV 블록 padding(15%) + 활성화 버퍼 + 고정 2GB
// 검증: Qwen3.6 35B @130K @8bit → 이론 43GB, 실제 ~54GB (오버헤드 ~11GB)
export function calcRuntimeOverhead(model, ctx, bits) {
  const paramMem = calcParamMemory(model, bits).totalGB;
  const kvMem = calcKVCache(model, ctx, bits).totalGB;
  const paramOverhead = paramMem * 0.12;
  const kvOverhead = kvMem * 0.15;
  const activationOverhead = ctx * 0.00003;
  const fixedOverhead = 2.0;
  return {
    paramOverheadGB: paramOverhead,
    kvOverheadGB: kvOverhead,
    activationOverheadGB: activationOverhead,
    fixedOverheadGB: fixedOverhead,
    totalGB: paramOverhead + kvOverhead + activationOverhead + fixedOverhead,
  };
}

export function calcMaxContext(model, ram, bits) {
  if (!model.kvHeads || !model.kvHeadDim || !model.layerCount || !model.totalParams) return 0;
  const attnLayers = model.fullAttnLayers || model.layerCount; // 하이브리드: 풀어텐션 레이어만 KV
  const os = getOsOverhead(ram);
  const bpe = bits / 8;
  const quantMultiplier = (quantAdjust[model.name] && quantAdjust[model.name][bits]) || 1.0;
  const paramBytes = model.totalParams * 1e9 * bpe * quantMultiplier;
  const budget =
    ram * 1024 ** 3 * 0.8 - paramBytes - paramBytes * 0.12 - os * 1024 ** 3 - 2.0 * 1024 ** 3;
  if (budget <= 0) return 0;
  const overhead = 1.15; // KV 블록 할당 padding
  const perLocal = 2 * model.kvHeads * model.kvHeadDim * bpe * overhead;
  const perGlobal = 2 * (model.globalKvHeads || model.kvHeads) * (model.globalHeadDim || model.kvHeadDim) * bpe * overhead;
  if (perLocal <= 0) return 0;

  if ((model.slidingWindow || 0) > 0) {
    const { globalLayers, localLayers } = slidingSplit(model);
    const w = model.slidingWindow;
    const perTokWithin = perLocal * localLayers + perGlobal * globalLayers; // ctx ≤ window: 전 레이어 증가
    const baseKV = perTokWithin * w;
    if (budget <= baseKV) {
      return Math.min(Math.floor(budget / perTokWithin), model.maxContext);
    }
    // window 초과: 글로벌 레이어만 증가
    const ctx = w + Math.floor((budget - baseKV) / (perGlobal * globalLayers));
    return Math.min(ctx, model.maxContext);
  }
  return Math.min(Math.floor(budget / (perLocal * attnLayers)), model.maxContext);
}

// ============================================================================
//  고수준 시뮬레이션 — UI가 쓰는 단일 진입점
// ============================================================================

export const HEADROOM_RATIO = 0.2; // RAM의 20%는 앱/스파이크용 여유로 남겨두는 게 안전

// verdict: 'yes'(넉넉) | 'tight'(빠듯) | 'no'(초과)
export function simulate(model, ram, ctx, bits) {
  const os = getOsOverhead(ram);
  const param = calcParamMemory(model, bits).totalGB;
  const kv = calcKVCache(model, ctx, bits).totalGB;
  const rt = calcRuntimeOverhead(model, ctx, bits).totalGB;

  const system = os + rt; // 비전공자용 묶음: 'macOS + 실행 엔진'
  const used = system + param + kv;
  const free = ram - used;
  const headroom = ram * HEADROOM_RATIO;

  let verdict;
  if (free < 0) verdict = 'no';
  else if (free < headroom) verdict = 'tight';
  else verdict = 'yes';

  return {
    model,
    ram,
    ctx,
    bits,
    os,
    param,
    kv,
    rt,
    system,
    used,
    free,
    headroom,
    verdict,
    pct: used / ram,
    maxContext: calcMaxContext(model, ram, bits),
  };
}

// 안 들어갈 때(또는 빠듯할 때) "이렇게 하면 들어가요" 한 가지 제안을 찾는다.
// 우선순위: 정밀도 낮추기 → 대화 길이 줄이기 → 더 큰 RAM.
export function suggestFix(model, ram, ctx, bits, L) {
  const t = L || ((ko) => ko);
  // 1) 더 낮은 정밀도로 현재 길이가 들어가나?
  const lowerBits = [8, 4].filter((b) => b < bits);
  for (const b of lowerBits) {
    if (simulate(model, ram, ctx, b).verdict !== 'no') {
      return { kind: 'bits', bits: b, text: t(`${b}bit로 양자화하면 들어가요.`, `Quantize to ${b}-bit and it fits.`) };
    }
  }
  // 2) 현재 정밀도에서 들어가는 최대 대화 길이
  const maxCtx = calcMaxContext(model, ram, bits);
  if (maxCtx >= 1024) {
    return { kind: 'ctx', ctx: maxCtx, text: t(`컨텍스트를 ${formatTokens(maxCtx, L)}까지 줄이면 들어가요.`, `Shorten context to ${formatTokens(maxCtx, L)} and it fits.`) };
  }
  // 3) 더 큰 RAM이 필요
  const allRam = [...new Set(Object.values(MACBOOK_RAM_GROUPS).flat())].sort((a, b) => a - b);
  const bigger = allRam.find((r) => r > ram && simulate(model, r, ctx, bits).verdict !== 'no');
  if (bigger) return { kind: 'ram', ram: bigger, text: t(`${bigger}GB 이상 맥북이면 들어가요.`, `A ${bigger}GB+ Mac would fit.`) };
  return { kind: 'none', text: t('더 작은 모델이나 더 강한 양자화가 필요해요.', 'You need a smaller model or stronger quantization.') };
}

// ===== 사람이 읽는 단위 변환 =====
// 한국어 1글자 ≈ 1.5토큰. 책 1쪽 ≈ 500자.
export function tokensToKoreanChars(tokens) {
  return Math.round(tokens / 1.5);
}
export function tokensToPages(tokens) {
  return tokensToKoreanChars(tokens) / 500;
}
export function formatTokens(t, L) {
  const unit = L ? L('토큰', 'tokens') : '토큰';
  if (t >= 1000) return `${Math.round(t / 1000)}K ${unit}`;
  return `${t} ${unit}`;
}
export function humanContext(tokens, L) {
  const pages = tokensToPages(tokens);
  if (!L) {
    if (pages >= 1) return `책 약 ${Math.round(pages)}쪽 분량`;
    return `한국어 약 ${tokensToKoreanChars(tokens).toLocaleString()}자`;
  }
  if (pages >= 1) return L(`책 약 ${Math.round(pages)}쪽 분량`, `~${Math.round(pages)} pages`);
  const words = Math.round(tokens * 0.75);
  return L(`한국어 약 ${tokensToKoreanChars(tokens).toLocaleString()}자`, `~${words.toLocaleString()} words`);
}
export function fmtGB(gb) {
  if (gb == null) return '—';
  if (gb >= 100) return gb.toFixed(0);
  if (gb >= 10) return gb.toFixed(1);
  return gb.toFixed(1);
}

// ===== 성능(벤치) — Opus 4.7을 100으로 둔 상대 점수 =====
export const OPUS = MODELS.find((m) => m.name.includes('Opus 4.7'));
const OPUS_AVG = benchAvg(OPUS.benchmarks) || 1;

// 선택 모델이 Opus 4.7 대비 몇 %인지 (공개 벤치 평균 기준)
export function opusPct(model) {
  const a = benchAvg(model.benchmarks);
  if (a == null || OPUS_AVG <= 0) return null;
  return (a / OPUS_AVG) * 100;
}

// 컨텍스트 길이 → 용도 등급
export function classifyTier(tokens) {
  if (tokens < 50000) return { key: 'basic', label: '기본', desc: '대부분의 작업에 충분 (~30K)' };
  if (tokens < 130000)
    return { key: 'mid', label: '적정선', desc: '복잡도 높은 작업 (60~130K)' };
  return { key: 'heavy', label: '특수 목적', desc: '긴 문서·코드베이스 전체 (130K+)' };
}


// ============================================================================
//  생성 속도 추정 (Apple Silicon 디코드는 대체로 메모리 대역폭 바운드)
// ============================================================================

// 칩 메모리 대역폭 (GB/s) — Apple 공식 사양 (M5 Pro/Max, 2026-03 출시).
// M5 Pro 307, M5 Max는 GPU 코어수별 32코어 460 / 40코어 614.
export function chipBandwidth(chip, gpuCores = 40) {
  if (chip === 'M5 Max') return gpuCores === 32 ? 460 : 614;
  return 307; // M5 Pro (16·20코어 모두 307)
}

// 예상 토큰 생성 속도(tok/s): 디코드 1토큰마다 활성 파라미터를 메모리에서 읽음
// → tok/s ≈ 대역폭 ÷ (활성파라미터 × 바이트) × 실현효율
export function estimateSpeed(model, chip, bits, gpuCores = 40) {
  if (model.isCloud || !model.totalParams) return null;
  const bw = chipBandwidth(chip, gpuCores) * 1e9; // bytes/s
  const activeB = (model.activeParams || model.totalParams) * 1e9; // 활성 파라미터 수
  const bytesPerToken = activeB * (bits / 8);
  if (bytesPerToken <= 0) return null;
  return (bw / bytesPerToken) * 0.75; // 0.75 = KV·오버헤드 감안한 실현 효율
}

// ============================================================================
//  HuggingFace config.json → 모델 객체 (임의 모델 지원)
// ============================================================================

// 이름에서 파라미터 수 추정 (safetensors 크기를 못 구했을 때 최후 수단): "7B", "30b-a3b" 등
function paramsFromName(id) {
  const m = id.match(/(\d+(?:\.\d+)?)\s*[bB](?![a-zA-Z])/);
  return m ? parseFloat(m[1]) : null;
}

export function parseHfConfig(id, raw, totalSize) {
  const c = raw.text_config || raw; // 멀티모달은 text_config에 본체
  const layerCount = c.num_hidden_layers;
  if (!layerCount) throw new Error('config에 num_hidden_layers 없음');

  const attnHeads = c.num_attention_heads;
  const kvHeads = c.num_key_value_heads ?? attnHeads ?? 1;
  const headDim = c.head_dim ?? (c.hidden_size && attnHeads ? Math.round(c.hidden_size / attnHeads) : 128);

  // 슬라이딩 윈도우가 "실제로" 쓰이는지 판별 (sliding_window 값만 있고 미사용인 모델 오탐 방지)
  const hasSlidingLayers = Array.isArray(c.layer_types) && c.layer_types.some((t) => String(t).includes('sliding'));
  const slidingActive =
    (c.sliding_window || 0) > 0 &&
    c.use_sliding_window !== false &&
    (hasSlidingLayers || (c.sliding_window < (c.max_position_embeddings || Infinity)));
  const sliding = slidingActive ? c.sliding_window : 0;

  // layer_types로 full attention 레이어 수 파악 (하이브리드/슬라이딩 정확도)
  let fullAttnLayers, globalAttnLayers;
  if (Array.isArray(c.layer_types)) {
    const full = c.layer_types.filter((t) => String(t).includes('full')).length;
    if (full > 0 && full < layerCount) {
      if (sliding > 0) globalAttnLayers = full; // 슬라이딩: full = 글로벌 레이어
      else fullAttnLayers = full; // 하이브리드 linear: full만 KV 보유
    }
  }

  const numExperts = c.num_local_experts || c.num_experts;
  const expertsPerToken = c.num_experts_per_tok;
  const isMoe = !!numExperts;

  // 파라미터 수: safetensors total_size(저장 dtype 바이트)에서 역산, 없으면 이름 추정
  let totalParams = null;
  if (totalSize) {
    const dt = String(c.torch_dtype || '').toLowerCase();
    const dtypeBytes = dt.includes('float32') || dt.includes('fp32') ? 4 : dt.includes('fp8') || dt.includes('int8') ? 1 : 2;
    totalParams = totalSize / dtypeBytes / 1e9;
  }
  if (!totalParams) totalParams = paramsFromName(id);

  return {
    name: id.split('/').pop(),
    group: 'HuggingFace',
    custom: true,
    sourceId: id,
    tags: isMoe ? ['moe'] : ['dense'],
    totalParams: totalParams ? +totalParams.toFixed(1) : null,
    activeParams: null, // MoE 활성 파라미터는 config로 정확 산출 어려움 → 속도만 근사
    layerCount,
    fullAttnLayers,
    globalAttnLayers,
    kvHeads,
    kvHeadDim: headDim,
    globalHeadDim: c.global_head_dim || undefined,
    attnHeads,
    hiddenSize: c.hidden_size,
    numExperts,
    expertsPerToken,
    maxContext: c.max_position_embeddings || 131072,
    slidingWindow: sliding || undefined,
    slidingPattern: sliding ? '5:1' : undefined,
    benchmarks: null,
    desc: id,
  };
}

// ============================================================================
//  "흔한 단순 계산기"의 KV 추정 — 정확도 비교용
//  거의 모든 기초 계산기/튜토리얼이 쓰는 식: 2 × 전체레이어 × KV헤드 × head_dim
//  × 전체컨텍스트 × 바이트. 슬라이딩 윈도우·하이브리드 linear·글로벌 head_dim을
//  무시하므로 최신 모델에서 KV를 크게 부풀린다.
// ============================================================================
export function naiveKVCache(model, ctx, bits) {
  if (!model.kvHeads || !model.kvHeadDim || !model.layerCount) return 0;
  const bytes = 2 * model.kvHeads * model.kvHeadDim * (bits / 8) * model.layerCount * ctx;
  return bytes / 1024 ** 3;
}

// 내장 모델 데이터 기준일 (신선도 표시용). HF 붙여넣기는 항상 실시간이라 무관.
export const DATA_UPDATED = '2026-05';
