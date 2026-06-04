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

// ============================================================================
//  NVIDIA GPU 모드 (로드맵 #7) — 컨슈머 단일 RTX 카드 fit + 온-GPU tok/s
//  ⚠️ 정확도 규율: 모든 수치엔 출처 ≥2 (CLAUDE.md). 데이터 무결성 테스트(engine.gpu.test.js)가
//     출처<2 / CONFLICT 행을 fail시킨다. 아래는 이중트랙(Claude hw-crawler ∥ Codex) 검증 데이터.
// ============================================================================

// GPU 레지스트리 — vramGB(공식), bandwidthGBs(메모리 대역폭, 속도 추정용).
// VRAM은 NVIDIA 공식, 대역폭은 NVIDIA 백서/Wikipedia 스펙표/리뷰 교차검증(전부 ≥2 출처 일치 VERIFIED).
// 출처 = Track A(Claude hw-crawler) 1개 + Track B(Codex 독립) 1개 — 두 독립 트랙이 값 일치 확인(2026-06-04 리콘실, CONFLICT 0).
const NV_BLACKWELL = 'https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf'; // Track B 1차
const WIKI40 = 'https://en.wikipedia.org/wiki/GeForce_RTX_40_series';
const WIKI30 = 'https://en.wikipedia.org/wiki/GeForce_30_series';
// 모든 행 2026-06-04 이중트랙(Claude hw-crawler ∥ Codex 독립) 리콘실 검증, CONFLICT 0.
const GPU_VERIFIED_AT = '2026-06-04';
const GPU_TRACKS = ['claude-hw-crawler', 'codex-independent']; // 독립 검증 트랙 2종
const _GPUS = [
  { name: 'RTX 5090', series: '50', vramGB: 32, bandwidthGBs: 1792, status: 'VERIFIED', sources: { vramGB: ['https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/', NV_BLACKWELL], bandwidthGBs: ['https://www.notebookcheck.net/NVIDIA-GeForce-RTX-5090-Benchmarks-and-Specs.935680.0.html', NV_BLACKWELL] } },
  { name: 'RTX 5080', series: '50', vramGB: 16, bandwidthGBs: 960, status: 'VERIFIED', sources: { vramGB: ['https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5080/', 'https://www.techspot.com/specs/gpu/303555-nvidia-geforce-rtx-5080.html'], bandwidthGBs: ['https://www.tomshardware.com/pc-components/gpus/nvidia-rtx-5080-allegedly-adopts-faster-30-gbps-gddr7-modules-delivering-960-gb-s-of-bandwidth-the-remaining-blackwell-lineup-is-expected-to-stick-with-slower-28-gbps-memory', 'https://www.techspot.com/specs/gpu/303555-nvidia-geforce-rtx-5080.html'] } },
  { name: 'RTX 5070 Ti', series: '50', vramGB: 16, bandwidthGBs: 896, status: 'VERIFIED', sources: { vramGB: ['https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5070-family/', 'https://www.notebookcheck.net/NVIDIA-GeForce-RTX-5070-Ti-Benchmarks-and-Specs.935685.0.html'], bandwidthGBs: ['https://www.notebookcheck.net/NVIDIA-GeForce-RTX-5070-Ti-Benchmarks-and-Specs.935685.0.html', 'https://www.guru3d.com/story/nvidia-rtx-5070-ti-specs-include-256bit-memory-bus-and-350w-tbp/'] } },
  { name: 'RTX 5070', series: '50', vramGB: 12, bandwidthGBs: 672, status: 'VERIFIED', sources: { vramGB: ['https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5070-family/', 'https://www.pcgamesn.com/nvidia/geforce-rtx-5070-review'], bandwidthGBs: ['https://www.notebookcheck.net/NVIDIA-GeForce-RTX-5070-Benchmarks-and-Specs.935682.0.html', 'https://www.pcgamesn.com/nvidia/geforce-rtx-5070-review'] } },
  { name: 'RTX 4090', series: '40', vramGB: 24, bandwidthGBs: 1008, status: 'VERIFIED', sources: { vramGB: ['https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/', 'https://www.techspot.com/products/graphics-cards/nvidia-geforce-rtx-4090.252744/'], bandwidthGBs: [WIKI40, 'https://www.techspot.com/products/graphics-cards/nvidia-geforce-rtx-4090.252744/'] } },
  { name: 'RTX 4080 SUPER', series: '40', vramGB: 16, bandwidthGBs: 736, status: 'VERIFIED', sources: { vramGB: ['https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4080-family/', 'https://www.techpowerup.com/gpu-specs/geforce-rtx-4080-super.c4182'], bandwidthGBs: ['https://www.notebookcheck.net/NVIDIA-GeForce-RTX-4080-Super-Benchmarks-and-Specs.799497.0.html', 'https://www.techpowerup.com/gpu-specs/geforce-rtx-4080-super.c4182'] } },
  { name: 'RTX 4070 Ti SUPER', series: '40', vramGB: 16, bandwidthGBs: 672, status: 'VERIFIED', sources: { vramGB: ['https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4070-family/', 'https://www.techspot.com/specs/gpu/290250-nvidia-geforce-rtx-4070-ti-super.html'], bandwidthGBs: [WIKI40, 'https://www.tomshardware.com/pc-components/gpus/nvidia-geforce-rtx-4070-ti-super-review'] } },
  { name: 'RTX 4070', series: '40', vramGB: 12, bandwidthGBs: 504, status: 'VERIFIED', sources: { vramGB: ['https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4070-family/', 'https://www.techspot.com/specs/gpu/254404-nvidia-geforce-rtx-4070.html'], bandwidthGBs: [WIKI40, 'https://www.techspot.com/specs/gpu/254404-nvidia-geforce-rtx-4070.html'] } },
  { name: 'RTX 4060 Ti 16GB', series: '40', vramGB: 16, bandwidthGBs: 288, status: 'VERIFIED', sources: { vramGB: [WIKI40, 'https://www.tomshardware.com/reviews/nvidia-geforce-rtx-4060-ti-16gb-review'], bandwidthGBs: [WIKI40, 'https://www.tomshardware.com/reviews/nvidia-geforce-rtx-4060-ti-16gb-review'] } },
  { name: 'RTX 3090', series: '30', vramGB: 24, bandwidthGBs: 936, status: 'VERIFIED', sources: { vramGB: [WIKI30, 'https://www.techspot.com/specs/gpu/224809-nvidia-geforce-rtx-3090.html'], bandwidthGBs: [WIKI30, 'https://www.techspot.com/specs/gpu/224809-nvidia-geforce-rtx-3090.html'] } },
  { name: 'RTX 3090 Ti', series: '30', vramGB: 24, bandwidthGBs: 1008, status: 'VERIFIED', sources: { vramGB: [WIKI30, 'https://www.guru3d.com/review/asus-geforce-rtx-3090-ti-tuf-gaming-review/page-4/'], bandwidthGBs: [WIKI30, 'https://www.guru3d.com/review/msi-geforce-rtx-3090-ti-suprim-x-review/page-4/'] } },
  { name: 'RTX 3080 10GB', series: '30', vramGB: 10, bandwidthGBs: 760, status: 'VERIFIED', sources: { vramGB: [WIKI30, 'https://www.techspot.com/specs/gpu/223293-nvidia-geforce-rtx-3080.html'], bandwidthGBs: [WIKI30, 'https://www.techspot.com/specs/gpu/223293-nvidia-geforce-rtx-3080.html'] } },
  { name: 'RTX 3080 12GB', series: '30', vramGB: 12, bandwidthGBs: 912, status: 'VERIFIED', sources: { vramGB: [WIKI30, 'https://www.notebookcheck.net/NVIDIA-GeForce-RTX-3080-12-GB-GPU-Benchmarks-and-Specs.635433.0.html'], bandwidthGBs: ['https://www.techspot.com/specs/gpu/247309-nvidia-geforce-rtx-3080-12gb.html', 'https://www.notebookcheck.net/NVIDIA-GeForce-RTX-3080-12-GB-GPU-Benchmarks-and-Specs.635433.0.html'] } },
  { name: 'RTX 3060 12GB', series: '30', vramGB: 12, bandwidthGBs: 360, status: 'VERIFIED', sources: { vramGB: [WIKI30, 'https://www.asus.com/motherboards-components/graphics-cards/dual/dual-rtx3060-12g/techspec/'], bandwidthGBs: [WIKI30, 'https://www.techpowerup.com/gpu-specs/geforce-rtx-3060-12-gb.c3682'] } },
];
// 검증 메타(이중트랙·검증일)를 전 행에 주입 — 데이터 무결성 테스트가 강제.
export const GPUS = _GPUS.map((g) => ({ verifiedAt: GPU_VERIFIED_AT, tracks: GPU_TRACKS, ...g }));

// GGUF Q-tier 유효 bits-per-weight — llama.cpp 공식 README(참조모델 Llama-3.1-8B 실측).
// 출처: https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md
// ⚠️ weight 양자화 전용(파라미터 메모리). KV 캐시는 별도 kvBits(기본 F16=16) — 혼동 시 KV 3.27× 과소.
export const GPU_QUANTS = [
  { tier: 'Q4_K_M', bpw: 4.8944, label: 'Q4_K_M', sub: '4-bit — 가장 인기(가성비)' },
  { tier: 'Q5_K_M', bpw: 5.7036, label: 'Q5_K_M', sub: '5-bit — 품질↑' },
  { tier: 'Q6_K', bpw: 6.5633, label: 'Q6_K', sub: '6-bit — 고품질' },
  { tier: 'Q8_0', bpw: 8.5008, label: 'Q8_0', sub: '8-bit — 거의 무손실' },
  { tier: 'FP16', bpw: 16.0005, label: 'FP16', sub: '원본 정밀도' },
];
export const GGUF_BPW_SOURCE = 'https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md';

// 환경(OS) 프리셋 — usable VRAM 차감용 reserve(OS+디스플레이+CUDA 컨텍스트 통합).
// ⚠️ reserve는 측정 분포의 대표값(추정) — 앵커 케이스로 캘리브레이션 예정(spec §5).
// 단일 셀렉터로 차원 폭발 방지. headless < linux+display < windows+display.
export const ENV_PRESETS = {
  'linux-headless': { key: 'linux-headless', label: ['Linux 헤드리스(세컨드 카드)', 'Linux headless (2nd card)'], reserveGB: 0.6, note: ['CUDA 컨텍스트만, 디스플레이 0', 'CUDA context only, no display'] },
  'linux-display': { key: 'linux-display', label: ['Linux + 디스플레이', 'Linux + display'], reserveGB: 1.2, note: ['컴포지터 경량', 'lightweight compositor'] },
  'windows-display': { key: 'windows-display', label: ['Windows + 디스플레이', 'Windows + display'], reserveGB: 2.0, note: ['WDDM + 디스플레이 (보수적·기본)', 'WDDM + display (conservative, default)'] },
};
export const DEFAULT_ENV = 'windows-display';
export const GPU_HEADROOM_RATIO = 0.05; // GPU는 전용 메모리 — 통합메모리 20%보다 훨씬 작음(전용이라 풀에 가깝게 사용)

// ===== 디바이스 추상화 — simulate/calcMaxContext/estimateSpeed가 받는 공통 descriptor =====
// Apple 경로 보존: reserveGB = getOsOverhead(ram) + 2.0 (기존 os + fixed 2.0 = 단일 reserve).
export function appleDevice(ram) {
  return { type: 'apple', memoryGB: ram, reserveGB: getOsOverhead(ram) + 2.0, headroomRatio: HEADROOM_RATIO, _os: getOsOverhead(ram) };
}
export function gpuDevice(gpu, envKey = DEFAULT_ENV) {
  const env = ENV_PRESETS[envKey] || ENV_PRESETS[DEFAULT_ENV];
  return { type: 'gpu', gpu, env: env.key, memoryGB: gpu.vramGB, bandwidthGBs: gpu.bandwidthGBs, reserveGB: env.reserveGB, headroomRatio: GPU_HEADROOM_RATIO, _os: 0 };
}
// 인자 정규화: device(number=ram→appleDevice / object=그대로), quant(number=weight·kv동일 / {weightBpw,kvBits})
function toDevice(d) { return typeof d === 'number' ? appleDevice(d) : d; }
function toQuant(q) { return typeof q === 'number' ? { weightBpw: q, kvBits: q } : { weightBpw: q.weightBpw, kvBits: q.kvBits ?? 16 }; }

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
  // quantAdjust는 정수 bits(Apple NVFP4/MXFP8 4/8/16) 보정용. GGUF는 weightBpw가 소수(4.8944 등)라
  // 매치 안 돼 multiplier=1.0 — 이는 *의도*다: GGUF bpw는 블록 스케일 등 실측 오버헤드를 이미 포함하므로
  // Apple 보정을 또 곱하면 이중계상. ⚠️ 단 기준 bpw는 Llama-3.1-8B 실측이라 소형모델은 임베딩 비중↑로
  // 약간 과소추정될 수 있음(v1 근사 — 모델별 .gguf 실크기 보정은 v2, spec §6).
  const multiplier = (quantAdjust[model.name] && quantAdjust[model.name][bits]) || 1.0;
  return { totalGB: baseTotalGB * multiplier, activeGB: baseActiveGB ? baseActiveGB * multiplier : null };
}

// 런타임 오버헤드: 양자화 메타(12%) + KV 블록 padding(15%) + 활성화 버퍼 + 고정 2GB
// 검증: Qwen3.6 35B @130K @8bit → 이론 43GB, 실제 ~54GB (오버헤드 ~11GB)
export function calcRuntimeOverhead(model, ctx, bitsOrQuant, device) {
  const { weightBpw, kvBits } = toQuant(bitsOrQuant); // weight↔KV 분리(GGUF: KV padding이 weight bpw 오염 금지)
  const paramMem = calcParamMemory(model, weightBpw).totalGB;
  const kvMem = calcKVCache(model, ctx, kvBits).totalGB;
  const paramOverhead = paramMem * 0.12;
  const kvOverhead = kvMem * 0.15;
  const activationOverhead = ctx * 0.00003;
  // GPU는 고정 reserve가 device.envReserveGB로 분리됨 → 런타임 fixed=0 (simulate의 rtDyn과 일치, 이중차감 금지)
  const fixedOverhead = device && device.type === 'gpu' ? 0 : 2.0;
  return {
    paramOverheadGB: paramOverhead,
    kvOverheadGB: kvOverhead,
    activationOverheadGB: activationOverhead,
    fixedOverheadGB: fixedOverhead,
    totalGB: paramOverhead + kvOverhead + activationOverhead + fixedOverhead,
  };
}

export function calcMaxContext(model, deviceOrRam, bitsOrQuant) {
  if (!model.kvHeads || !model.kvHeadDim || !model.layerCount || !model.totalParams) return 0;
  const device = toDevice(deviceOrRam);        // number(ram)→appleDevice / device 객체→그대로
  const { weightBpw, kvBits } = toQuant(bitsOrQuant);
  const attnLayers = model.fullAttnLayers || model.layerCount; // 하이브리드: 풀어텐션 레이어만 KV
  const wbpe = weightBpw / 8; // 파라미터(weight) 바이트
  const kbpe = kvBits / 8;    // KV 원소 바이트(GPU 기본 F16=16)
  const quantMultiplier = (quantAdjust[model.name] && quantAdjust[model.name][weightBpw]) || 1.0;
  const paramBytes = model.totalParams * 1e9 * wbpe * quantMultiplier;
  const budget =
    device.memoryGB * 1024 ** 3 * (1 - device.headroomRatio) - paramBytes - paramBytes * 0.12 - device.reserveGB * 1024 ** 3;
  if (budget <= 0) return 0;
  const overhead = 1.15; // KV 블록 할당 padding
  const perLocal = 2 * model.kvHeads * model.kvHeadDim * kbpe * overhead;
  const perGlobal = 2 * (model.globalKvHeads || model.kvHeads) * (model.globalHeadDim || model.kvHeadDim) * kbpe * overhead;
  if (perLocal <= 0) return 0;
  const actPerTok = 0.00003 * 1024 ** 3; // 활성화 버퍼(simulate와 동일) — ctx 비례라 토큰당 비용에 포함

  if ((model.slidingWindow || 0) > 0) {
    const { globalLayers, localLayers } = slidingSplit(model);
    const w = model.slidingWindow;
    const perTokWithin = perLocal * localLayers + perGlobal * globalLayers; // ctx ≤ window: 전 레이어 증가
    const baseWithAct = (perTokWithin + actPerTok) * w; // 윈도우 내 KV + 활성화
    if (budget <= baseWithAct) {
      return Math.min(Math.floor(budget / (perTokWithin + actPerTok)), model.maxContext);
    }
    // window 초과: 글로벌 레이어 KV + 활성화만 증가
    const ctx = w + Math.floor((budget - baseWithAct) / (perGlobal * globalLayers + actPerTok));
    return Math.min(ctx, model.maxContext);
  }
  return Math.min(Math.floor(budget / (perLocal * attnLayers + actPerTok)), model.maxContext);
}

// ============================================================================
//  고수준 시뮬레이션 — UI가 쓰는 단일 진입점
// ============================================================================

export const HEADROOM_RATIO = 0.2; // RAM의 20%는 앱/스파이크용 여유로 남겨두는 게 안전

// verdict: 'yes'(넉넉) | 'tight'(빠듯) | 'no'(초과)
export function simulate(model, deviceOrRam, ctx, bitsOrQuant) {
  const device = toDevice(deviceOrRam);              // number(ram)→appleDevice / device 객체→그대로
  const { weightBpw, kvBits } = toQuant(bitsOrQuant); // weight(파라미터) ↔ KV 비트 분리
  const param = calcParamMemory(model, weightBpw).totalGB;
  const kv = calcKVCache(model, ctx, kvBits).totalGB;

  // 단일 reserve 방정식(Codex council): used = param + kv + rtDyn + reserve. reserve는 1회만.
  const rtDyn = param * 0.12 + kv * 0.15 + ctx * 0.00003; // 동적 런타임(고정 reserve 미포함)
  const reserve = device.reserveGB;                       // OS/CUDA/디스플레이 통합 reserve
  const used = param + kv + rtDyn + reserve;
  const free = device.memoryGB - used;
  const headroom = device.memoryGB * device.headroomRatio;

  let verdict;
  if (free < 0) verdict = 'no';
  else if (free < headroom) verdict = 'tight';
  else verdict = 'yes';

  const os = device._os ?? 0; // Apple 표시 호환
  const rt = rtDyn + (device.type === 'apple' ? 2.0 : 0); // 기존 rt = 동적 + 고정2.0 (Apple 분해 보존)

  return {
    model,
    device,
    ram: device.memoryGB, // 하위호환 별칭
    memoryGB: device.memoryGB,
    ctx,
    weightBpw,
    kvBits,
    quant: { weightBpw, kvBits }, // 컴포넌트가 재시뮬레이트 시 그대로 전달(Apple/GPU 공통)
    bits: weightBpw, // 하위호환
    os,
    param,
    kv,
    rt,
    rtDyn,
    reserve,
    system: os + rt, // 비전공자용 묶음
    used,
    free,
    headroom,
    verdict,
    pct: used / device.memoryGB,
    maxContext: calcMaxContext(model, device, { weightBpw, kvBits }),
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
export function estimateSpeed(model, chipOrDevice, bitsOrWeightBpw, gpuCores = 40) {
  if (model.isCloud || !model.totalParams) return null;
  // chip 문자열(Apple) → chipBandwidth / device 객체(GPU) → device.bandwidthGBs
  const bwGBs = (chipOrDevice && typeof chipOrDevice === 'object' && chipOrDevice.bandwidthGBs != null)
    ? chipOrDevice.bandwidthGBs
    : chipBandwidth(chipOrDevice, gpuCores);
  const bw = bwGBs * 1e9; // bytes/s
  const activeB = (model.activeParams || model.totalParams) * 1e9; // 활성 파라미터 수
  const bytesPerToken = activeB * (bitsOrWeightBpw / 8); // 디코드: weight를 메모리에서 1회 read
  if (bytesPerToken <= 0) return null;
  return (bw / bytesPerToken) * 0.75; // 0.75 = KV·오버헤드 감안한 실현 효율(GPU 재앵커링 대상)
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
