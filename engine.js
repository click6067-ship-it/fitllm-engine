// ============================================================================
//  FitLLM — Memory Engine
//  검증된 계산 엔진(v1에서 실모델 config.json 대조로 교정한 값/식)을 그대로 이식.
//  데이터·수식은 신뢰의 핵 → 건드리지 않음. v2는 이 위에 직관적 UI만 새로 얹는다.
//  벤치마크: 공개·검증된 GPQA(Diamond) · MMLU-Pro · SWE-Bench(Verified)만 사용.
// ============================================================================

export const MODELS = [
  // 정렬: 최신·화제 순 (GLM → gpt-oss → Qwen 3.6 → Qwen3.5 → Gemma 4 → Llama → Cloud) — 2026-07-09 재정렬·구세대(Qwen3/2.5) 정리.
  // 확장분은 Track A(Claude) ∥ Track B(Codex) 블라인드 이중검증(CONFLICT=0). 구조=공식 HF config.json, 파라미터=safetensors index.
  // ⚠️ ?m= 공유링크는 이 배열 인덱스 기준 — 순서 변경 시 기존 링크가 다른 모델을 가리킴(2026-07-09 사용자 승인 하에 변경).

  // --- GLM (zAI, MLA) — 압축 KV(kv_lora_rank 512 + qk_rope 64 = 576 elem/tok/layer, ×1) ---
  { name: 'GLM-4.7-Flash', group: 'GLM', tags: ['moe', 'mla'],
    totalParams: 30, activeParams: 3, layerCount: 47, kvHeads: 20, kvHeadDim: 256, attnHeads: 20, hiddenSize: 2048,
    numExperts: 64, expertsPerToken: 4, mlaKvLoraRank: 512, mlaRopeDim: 64, maxContext: 202752, benchmarks: null,
    desc: 'MoE · MLA · ~30B / ~3B active · 64 experts(top-4) · 압축 KV(576/tok/layer) · 최대 202K' }, // config.json: zai-org/GLM-4.7-Flash (fp8 체크포인트)
  { name: 'GLM-5.2', group: 'GLM', tags: ['moe', 'mla'],
    totalParams: 753, activeParams: 40, layerCount: 78, kvHeads: 64, kvHeadDim: 256, attnHeads: 64, hiddenSize: 6144,
    numExperts: 256, expertsPerToken: 8, mlaKvLoraRank: 512, mlaRopeDim: 64, maxContext: 1048576, benchmarks: null,
    desc: 'MoE · MLA · 753B / ~40B active · 256 experts(top-8) · 압축 KV · 최대 1M (4bit도 512GB급만 fit)' }, // config.json+index(1.5TB÷2 bf16): zai-org/GLM-5.2

  // --- gpt-oss (OpenAI, MoE + 슬라이딩 128, 절반이 full-attn) — MXFP4 네이티브(파라미터 카운트는 카드 기준) ---
  { name: 'gpt-oss-20b', group: 'gpt-oss', tags: ['moe'],
    totalParams: 21, activeParams: 3.6, layerCount: 24, kvHeads: 8, kvHeadDim: 64, attnHeads: 64, hiddenSize: 2880,
    numExperts: 32, expertsPerToken: 4, maxContext: 131072, slidingWindow: 128, globalAttnLayers: 12, benchmarks: null,
    desc: 'MoE · 21B / 3.6B active · 32 experts(top-4) · 슬라이딩128(full 12/24) · 최대 128K' }, // config.json: openai/gpt-oss-20b
  { name: 'gpt-oss-120b', group: 'gpt-oss', tags: ['moe'],
    totalParams: 117, activeParams: 5.1, layerCount: 36, kvHeads: 8, kvHeadDim: 64, attnHeads: 64, hiddenSize: 2880,
    numExperts: 128, expertsPerToken: 4, maxContext: 131072, slidingWindow: 128, globalAttnLayers: 18, benchmarks: null,
    desc: 'MoE · 117B / 5.1B active · 128 experts(top-4) · 슬라이딩128(full 18/36) · 최대 128K' }, // config.json: openai/gpt-oss-120b

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
  // --- Qwen3.5 하이브리드 linear (AgentWorld) — 40레이어 중 full 10 + linear 30(DeltaNet), KV는 full 10만 ---
  { name: 'Qwen-AgentWorld-35B-A3B', group: 'Qwen3.5', tags: ['moe'],
    totalParams: 34.7, activeParams: 3.0, layerCount: 40, fullAttnLayers: 10, kvHeads: 2, kvHeadDim: 256, attnHeads: 16, hiddenSize: 2048,
    numExperts: 256, expertsPerToken: 8, maxContext: 262144, benchmarks: null,
    desc: 'MoE · ~34.7B / ~3B active · 하이브리드(full 10 + linear 30) · KV는 10레이어만 · 최대 256K' }, // config.json+index: Qwen/Qwen-AgentWorld-35B-A3B

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
    slidingPattern: '4:1', // layer_types 실측: sliding 28 : full 7 (google/gemma-4-E2B-it config.json — 구 '5:1'은 글로벌 6층 오계산, 2026-07-11 정정)
    globalAttnLayers: 7,
    benchmarks: { GPQA: 0.3, 'MMLU-Pro': 0.68, 'SWE-Bench': null },
    desc: 'Dense+PLE · 5.1B raw / 2.3B 유효 · 슬라이딩윈도우 512(4:1) · 최대 128K · 벤치 근사치',
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
    globalAttnLayers: 7, // layer_types 실측: sliding 35 : full 7 (google/gemma-4-E4B-it config.json — 패턴 유도값과 일치, 명시 고정 2026-07-11)
    benchmarks: { GPQA: 0.586, 'MMLU-Pro': 0.694, 'SWE-Bench': null },
    desc: 'Dense+PLE · 8B raw / 4.5B 유효 · 슬라이딩윈도우 512(5:1) · 최대 128K',
  },
  // === Gemma 4 12B Unified (dense) — https://huggingface.co/google/gemma-4-12B-it/blob/main/config.json ===
  {
    name: 'Gemma 4 12b',
    group: 'Gemma 4',
    tags: ['dense'],
    totalParams: 11.95, // 모델카드 "Total Parameters: 11.95B" (encoder-free Unified, 별도 비전/오디오 인코더 없음)
    activeParams: 11.95,
    layerCount: 48,
    kvHeads: 8, // 슬라이딩 40레이어: KV헤드 8 × head_dim 256 (config num_key_value_heads/head_dim)
    kvHeadDim: 256,
    globalKvHeads: 1, // 글로벌(풀어텐션) 8레이어: KV헤드 1 × head_dim 512 (config num_global_key_value_heads/global_head_dim — 이종)
    globalHeadDim: 512,
    attnHeads: 16,
    hiddenSize: 3840,
    maxContext: 262144,
    slidingWindow: 1024,
    slidingPattern: '5:1', // layer_types 실측 카운트: sliding_attention 40 : full_attention 8 (full @ 5,11,…,47)
    benchmarks: { GPQA: 0.788, 'MMLU-Pro': 0.772, 'SWE-Bench': null }, // README 공식 벤치표(GPQA Diamond·MMLU Pro)
    desc: 'Dense · 11.95B · 48레이어 · 슬라이딩윈도우 1024(5:1, 글로벌 8레이어 head_dim 512×1KV) · Unified 멀티모달 · 최대 256K',
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
  // --- Llama (Meta) — 별개 생태계 기준점(유지). 공식 config gated(401), unsloth·NousResearch 미러 교차검증(동일) ---
  { name: 'Llama-3.2-3B-Instruct', group: 'Llama', tags: ['dense'],
    totalParams: 3.2, activeParams: 3.2, layerCount: 28, kvHeads: 8, kvHeadDim: 128, attnHeads: 24, hiddenSize: 3072,
    maxContext: 131072, benchmarks: null,
    desc: 'Dense · 3.2B · 28레이어 · GQA(24/8) · 최대 128K' }, // meta-llama/Llama-3.2-3B-Instruct (gated) ↔ unsloth 미러
  { name: 'Llama-3.1-8B-Instruct', group: 'Llama', tags: ['dense'],
    totalParams: 8.0, activeParams: 8.0, layerCount: 32, kvHeads: 8, kvHeadDim: 128, attnHeads: 32, hiddenSize: 4096,
    maxContext: 131072, benchmarks: null,
    desc: 'Dense · 8.0B · 32레이어 · GQA(32/8) · 최대 128K' }, // meta-llama/Llama-3.1-8B-Instruct (gated) ↔ unsloth/Meta-Llama-3.1-8B-Instruct 미러

  // --- Draft 소형모델 — Stack 탭(speculative decoding draft+target · IDE 자동완성) 페어용. 플래그십 아님(카탈로그 최신순 정리와 별개 기능군) ---
  // ⚠️ Qwen3 2종 totalParams = GGUF/디스크 기준(임베딩 1회): safetensors 카운트(751.6M/2031.7M)는 tied lm_head 중복 저장 포함 — 사용 금지.
  //    검증(2026-07-09 verifier): Qwen3-0.6B-Q8_0.gguf 실측 639,446,688B = 596.0M×Q8_0+메타 ✓ / 1.7B GGUF 1,834,426,016B ✓ (2출처+GGUF 실측)
  { name: 'Qwen3-0.6B', group: 'Draft', tags: ['dense', 'draft'],
    totalParams: 0.596, activeParams: 0.596, layerCount: 28, kvHeads: 8, kvHeadDim: 128, attnHeads: 16, hiddenSize: 1024,
    maxContext: 40960, benchmarks: null, // config max_position_embeddings 40960 (모델카드 공표 native 32K — config 기준 원칙)
    desc: 'Dense · 0.6B · vLLM 표준 draft — Qwen 계열 타깃 페어 (vocab 151936 호환)' }, // Qwen/Qwen3-0.6B config.json + HF API + GGUF 실측
  { name: 'Qwen3-1.7B', group: 'Draft', tags: ['dense', 'draft'],
    totalParams: 1.721, activeParams: 1.721, layerCount: 28, kvHeads: 8, kvHeadDim: 128, attnHeads: 16, hiddenSize: 2048,
    maxContext: 40960, benchmarks: null,
    desc: 'Dense · 1.7B · 30B급 타깃엔 0.6B보다 스루풋 우위(vLLM 벤치) — Qwen 계열 draft' }, // Qwen/Qwen3-1.7B config.json + HF API + GGUF 실측
  { name: 'Llama-3.2-1B-Instruct', group: 'Draft', tags: ['dense', 'draft'],
    totalParams: 1.236, activeParams: 1.236, layerCount: 16, kvHeads: 8, kvHeadDim: 64, attnHeads: 32, hiddenSize: 2048,
    maxContext: 131072, benchmarks: null, // params 1,235,814,400 = 분석 카운트 바이트단위 일치(tied·lm_head 미저장 — 그대로 디스크 기준)
    desc: 'Dense · 1.2B · Llama 3.x 타깃 표준 draft (vocab 128256 호환) · 최대 128K' }, // meta-llama(gated) ↔ unsloth·NousResearch 미러 2종 일치
  { name: 'Gemma-3-1B-it', group: 'Draft', tags: ['dense', 'draft'],
    totalParams: 1.0, activeParams: 1.0, layerCount: 26, kvHeads: 1, kvHeadDim: 256, attnHeads: 4, hiddenSize: 1152,
    maxContext: 32768, slidingWindow: 512, globalAttnLayers: 4, // transformers gemma3 + google/gemma_pytorch 2출처: 26층 중 full-attention 4(레이어 6·12·18·24), 나머지 22층 window 512
    benchmarks: null,
    desc: 'Dense · 1.0B · MQA(4/1) · 슬라이딩 512(글로벌 4레이어) — Gemma 계열 draft · 최대 32K' }, // google/gemma-3-1b-it(gated) ↔ unsloth·mlx 미러 + gemma_pytorch

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

// ===== 맥북/맥 RAM 옵션 (통합메모리 GB) =====
// M1–M4 전세대 + Ultra/Studio. fit엔 통합메모리 GB만 영향(대역폭은 chipBandwidth = dormant speed용).
// Track A(Apple newsroom+Wikipedia) ∥ Track B(Codex) 이중검증(2026-07-09, 일치). M4 Ultra 미존재(M3 Ultra가 최신).
// 출처: Apple Newsroom 각 세대 발표 + support.apple.com 스펙 + Wikipedia Apple_M1..M4.
export const MACBOOK_RAM_GROUPS = {
  'M5': [16, 24, 32], // MacBook Air/Pro 14" (Wikipedia Apple_M5 ∥ Codex 트랙 교차확인 2026-07-09)
  'M5 Pro': [24, 48, 64],
  'M5 Max': [36, 48, 64, 128],
  'M4': [16, 24, 32],
  'M4 Pro': [24, 48, 64],
  'M4 Max': [36, 48, 64, 128],
  'M3': [8, 16, 24],
  'M3 Pro': [18, 36],
  'M3 Max': [36, 48, 64, 96, 128],
  'M3 Ultra': [96, 256, 512], // Mac Studio — 최대 512GB(개인용 최대)
  'M2': [8, 16, 24],
  'M2 Pro': [16, 32],
  'M2 Max': [32, 64, 96],
  'M2 Ultra': [64, 128, 192], // Mac Studio/Pro — 최대 192GB
  'M1': [8, 16],
  'M1 Pro': [16, 32],
  'M1 Max': [32, 64],
  'M1 Ultra': [64, 128],
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

  // ===== 확장 (2026-07-09, Track A[claude hw-crawler] ∥ Track B[codex] 블라인드 리콘실, CONFLICT=0) =====
  // NVIDIA consumer 추가
  { name: 'RTX 5060 Ti 16GB', series: '50', vramGB: 16, bandwidthGBs: 448, status: 'VERIFIED', verifiedAt: '2026-07-09', sources: { vramGB: ['https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5060-family/', 'https://www.asus.com/us/motherboards-components/graphics-cards/prime/prime-rtx5060ti-16g/techspec/'], bandwidthGBs: ['https://www.tomshardware.com/pc-components/gpus/nvidia-geforce-rtx-5060-ti-16gb-review', 'https://www.techpowerup.com/gpu-specs/geforce-rtx-5060-ti.c4246'] } },
  { name: 'RTX 4070 Ti', series: '40', vramGB: 12, bandwidthGBs: 504, status: 'VERIFIED', verifiedAt: '2026-07-09', sources: { vramGB: ['https://www.techspot.com/specs/gpu/258458-nvidia-geforce-rtx-4070-ti.html', 'https://www.thefpsreview.com/gpu-family/nvidia-geforce-rtx-4070-ti-gpu-family-specifications/'], bandwidthGBs: ['https://www.pcworld.com/article/1444726/nvidia-geforce-rtx-4070-ti-review.html', 'https://www.thefpsreview.com/gpu-family/nvidia-geforce-rtx-4070-ti-gpu-family-specifications/'] } }, // 504 (NOT Wiki 554.4 아웃라이어): 192b×21Gbps÷8
  { name: 'RTX 4080', series: '40', vramGB: 16, bandwidthGBs: 717, status: 'VERIFIED', verifiedAt: '2026-07-09', sources: { vramGB: ['https://www.custompc.com/nvidia-geforce-rtx-4080-review', 'https://www.notebookcheck.net/NVIDIA-GeForce-RTX-4080-GPU-Benchmarks-and-Specs.674575.0.html'], bandwidthGBs: [WIKI40, 'https://www.custompc.com/nvidia-geforce-rtx-4080-review'] } },
  { name: 'RTX 4060 Ti 8GB', series: '40', vramGB: 8, bandwidthGBs: 288, status: 'VERIFIED', verifiedAt: '2026-07-09', sources: { vramGB: ['https://www.tomshardware.com/reviews/nvidia-geforce-rtx-4060-ti-review', 'https://us.msi.com/Graphics-Card/GeForce-RTX-4060-Ti-GAMING-X-8G/Specification'], bandwidthGBs: ['https://www.tomshardware.com/reviews/nvidia-geforce-rtx-4060-ti-review', 'https://www.techspot.com/specs/gpu/280961-nvidia-geforce-rtx-4060-ti-16gb.html'] } },
  { name: 'RTX 3060 8GB', series: '30', vramGB: 8, bandwidthGBs: 240, status: 'VERIFIED', verifiedAt: '2026-07-09', sources: { vramGB: ['https://www.tomshardware.com/news/nvidia-geforce-rtx-3060-8gb-with-128-bit-memory-bus', 'https://videocardz.com/newz/nvidia-geforce-rtx-3060-with-8gb-memory-released-features-128-bit-memory-bus'], bandwidthGBs: ['https://www.tomshardware.com/news/nvidia-geforce-rtx-3060-8gb-with-128-bit-memory-bus', 'https://www.guru3d.com/story/geforce-rtx-3060-with-8gb-128-bit-memory-bus-memory-released'] } }, // 12GB(360)과 별개 128-bit SKU
  { name: 'RTX 2080 Ti', series: '20', vramGB: 11, bandwidthGBs: 616, status: 'VERIFIED', verifiedAt: '2026-07-09', sources: { vramGB: ['https://www.nvidia.com/en-us/geforce/graphics-cards/rtx-2080-ti/', 'https://www.techpowerup.com/gpu-specs/geforce-rtx-2080-ti.c3305'], bandwidthGBs: ['https://www.techpowerup.com/gpu-specs/geforce-rtx-2080-ti.c3305', 'https://videocardz.net/nvidia-geforce-rtx-2080ti'] } },
  // NVIDIA workstation
  { name: 'RTX 6000 Ada', series: 'workstation', vramGB: 48, bandwidthGBs: 960, status: 'VERIFIED', verifiedAt: '2026-07-09', sources: { vramGB: ['https://www.nvidia.com/en-us/products/workstations/rtx-6000/', 'https://www.leadtek.com/eng/products/workstation_graphics(2)/NVIDIA_RTX_6000_Ada_Generation(40949)/detail'], bandwidthGBs: ['https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/rtx-6000/proviz-print-rtx6000-datasheet-web-2504660.pdf', 'https://www.techpowerup.com/gpu-specs/rtx-6000-ada-generation.c3933'] } },
  { name: 'RTX PRO 6000 Blackwell', series: 'workstation', vramGB: 96, bandwidthGBs: 1792, status: 'VERIFIED', verifiedAt: '2026-07-09', sources: { vramGB: ['https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000/', 'https://www.storagereview.com/review/nvidia-rtx-pro-6000-workstation-gpu-review-blackwell-architecture-and-96-gb-for-pro-workflows'], bandwidthGBs: ['https://www.nvidia.com/content/dam/en-zz/Solutions/data-center/rtx-pro-6000-blackwell-workstation-edition/workstation-blackwell-rtx-pro-6000-workstation-edition-nvidia-us-3519208-web.pdf', 'https://www.techpowerup.com/gpu-specs/rtx-pro-6000-blackwell.c4272'] } }, // Workstation Edition
  // AMD Radeon (GDDR 대역폭 — Infinity Cache effective 제외)
  { name: 'RX 7900 XTX', series: 'amd', vramGB: 24, bandwidthGBs: 960, status: 'VERIFIED', verifiedAt: '2026-07-09', sources: { vramGB: ['https://www.amd.com/en/products/graphics/desktops/radeon/7000-series/amd-radeon-rx-7900xtx.html', 'https://www.techpowerup.com/gpu-specs/radeon-rx-7900-xtx.c3941'], bandwidthGBs: ['https://www.amd.com/en/products/graphics/desktops/radeon/7000-series/amd-radeon-rx-7900xtx.html', 'https://www.notebookcheck.net/AMD-Radeon-RX-7900-XTX-GPU-Benchmarks-and-Specs.674159.0.html'] } },
  { name: 'RX 7900 XT', series: 'amd', vramGB: 20, bandwidthGBs: 800, status: 'VERIFIED', verifiedAt: '2026-07-09', sources: { vramGB: ['https://www.amd.com/en/products/graphics/desktops/radeon/7000-series/amd-radeon-rx-7900xt.html', 'https://www.techpowerup.com/gpu-specs/radeon-rx-7900-xt.c3912'], bandwidthGBs: ['https://www.amd.com/en/products/graphics/desktops/radeon/7000-series/amd-radeon-rx-7900xt.html', 'https://www.notebookcheck.net/AMD-Radeon-RX-7900-XT-GPU-Benchmarks-and-Specs.674155.0.html'] } },
  { name: 'RX 7800 XT', series: 'amd', vramGB: 16, bandwidthGBs: 624, status: 'VERIFIED', verifiedAt: '2026-07-09', sources: { vramGB: ['https://www.amd.com/en/products/graphics/desktops/radeon/7000-series/amd-radeon-rx-7800-xt.html', 'https://www.techpowerup.com/gpu-specs/radeon-rx-7800-xt.c3839'], bandwidthGBs: ['https://www.amd.com/en/products/graphics/desktops/radeon/7000-series/amd-radeon-rx-7800-xt.html', 'https://videocardz.com/amd/radeon-rx-7000/radeon-rx-7800-xt'] } },
  { name: 'RX 9070 XT', series: 'amd', vramGB: 16, bandwidthGBs: 640, status: 'VERIFIED', verifiedAt: '2026-07-09', sources: { vramGB: ['https://www.amd.com/en/products/graphics/desktops/radeon/9000-series/amd-radeon-rx-9070xt.html', 'https://www.techpowerup.com/gpu-specs/radeon-rx-9070-xt.c4229'], bandwidthGBs: ['https://www.amd.com/en/products/graphics/desktops/radeon/9000-series/amd-radeon-rx-9070xt.html', 'https://www.tomshardware.com/pc-components/gpus/amd-radeon-rx-9070-xt-review'] } },
  { name: 'RX 9070', series: 'amd', vramGB: 16, bandwidthGBs: 640, status: 'VERIFIED', verifiedAt: '2026-07-09', sources: { vramGB: ['https://www.amd.com/en/products/graphics/desktops/radeon/9000-series/amd-radeon-rx-9070.html', 'https://www.techpowerup.com/gpu-specs/radeon-rx-9070.c4227'], bandwidthGBs: ['https://www.amd.com/en/products/graphics/desktops/radeon/9000-series/amd-radeon-rx-9070.html', 'https://www.tomshardware.com/pc-components/gpus/amd-radeon-rx-9070-xt-review'] } },
  { name: 'Radeon PRO W7900', series: 'amd', vramGB: 48, bandwidthGBs: 864, status: 'VERIFIED', verifiedAt: '2026-07-09', sources: { vramGB: ['https://www.amd.com/en/products/graphics/workstations/radeon-pro/w7900.html', 'https://www.techpowerup.com/gpu-specs/radeon-pro-w7900.c4147'], bandwidthGBs: ['https://www.amd.com/content/dam/amd/en/documents/products/graphics/workstation/radeon-pro-w7900-datasheet.pdf', 'https://www.techpowerup.com/gpu-specs/radeon-pro-w7900.c4147'] } },
  // 멀티GPU 프리셋 — vramGB=합산, bandwidthGBs=per-card(합산 금지, PCIe/NVLink 바운드). count/perCardVramGB로 표기.
  { name: '2× RTX 3090', series: 'multi', vramGB: 48, bandwidthGBs: 936, count: 2, perCardVramGB: 24, status: 'VERIFIED', verifiedAt: '2026-07-09', sources: { vramGB: [WIKI30, 'https://www.techspot.com/specs/gpu/224809-nvidia-geforce-rtx-3090.html'], bandwidthGBs: [WIKI30, 'https://www.techspot.com/specs/gpu/224809-nvidia-geforce-rtx-3090.html'] } },
  { name: '2× RTX 4090', series: 'multi', vramGB: 48, bandwidthGBs: 1008, count: 2, perCardVramGB: 24, status: 'VERIFIED', verifiedAt: '2026-07-09', sources: { vramGB: ['https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/', 'https://www.techspot.com/products/graphics-cards/nvidia-geforce-rtx-4090.252744/'], bandwidthGBs: [WIKI40, 'https://www.techspot.com/products/graphics-cards/nvidia-geforce-rtx-4090.252744/'] } },
  { name: '4× RTX 3090', series: 'multi', vramGB: 96, bandwidthGBs: 936, count: 4, perCardVramGB: 24, status: 'VERIFIED', verifiedAt: '2026-07-09', sources: { vramGB: [WIKI30, 'https://www.techspot.com/specs/gpu/224809-nvidia-geforce-rtx-3090.html'], bandwidthGBs: [WIKI30, 'https://www.techspot.com/specs/gpu/224809-nvidia-geforce-rtx-3090.html'] } },
  // 데이터센터 GPU (A100/H100/H200/B200) — 셀프호스트·클라우드 렌탈 대상. VRAM=fit 핵심, 대역폭=dormant.
  { name: 'A100 40GB', series: 'datacenter', vramGB: 40, bandwidthGBs: 1555, status: 'VERIFIED', verifiedAt: '2026-07-09', sources: { vramGB: ['https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf', 'https://en.wikipedia.org/wiki/Ampere_(microarchitecture)'], bandwidthGBs: ['https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf', 'https://en.wikipedia.org/wiki/Ampere_(microarchitecture)'] } }, // HBM2 1.55TB/s
  { name: 'A100 80GB', series: 'datacenter', vramGB: 80, bandwidthGBs: 1935, status: 'VERIFIED', verifiedAt: '2026-07-09', sources: { vramGB: ['https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/a100-80gb-datasheet-update-nvidia-us-1521051-r2-web.pdf', 'https://en.wikipedia.org/wiki/Ampere_(microarchitecture)'], bandwidthGBs: ['https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/a100-80gb-datasheet-update-nvidia-us-1521051-r2-web.pdf', 'https://www.techpowerup.com/gpu-specs/a100-pcie-80-gb.c3821'] } }, // PCIe HBM2e 1.94TB/s
  { name: 'H100 80GB', series: 'datacenter', vramGB: 80, bandwidthGBs: 3350, status: 'VERIFIED', verifiedAt: '2026-07-09', sources: { vramGB: ['https://www.nvidia.com/en-us/data-center/h100/', 'https://en.wikipedia.org/wiki/Hopper_(microarchitecture)'], bandwidthGBs: ['https://www.nvidia.com/en-us/data-center/h100/', 'https://en.wikipedia.org/wiki/Hopper_(microarchitecture)'] } }, // SXM5 HBM3 3.35TB/s
  { name: 'H200 141GB', series: 'datacenter', vramGB: 141, bandwidthGBs: 4800, status: 'VERIFIED', verifiedAt: '2026-07-09', sources: { vramGB: ['https://www.nvidia.com/en-us/data-center/h200/', 'https://en.wikipedia.org/wiki/Hopper_(microarchitecture)'], bandwidthGBs: ['https://www.nvidia.com/en-us/data-center/h200/', 'https://en.wikipedia.org/wiki/Hopper_(microarchitecture)'] } }, // HBM3e 4.8TB/s
  { name: 'B200', series: 'datacenter', vramGB: 180, bandwidthGBs: 7700, status: 'VERIFIED', verifiedAt: '2026-07-11', sources: { vramGB: ['https://www.nvidia.com/en-us/data-center/dgx-b200/', 'https://lenovopress.lenovo.com/lp2226-thinksystem-nvidia-b200-180gb-1000w-gpu'], bandwidthGBs: ['https://lenovopress.lenovo.com/lp2226-thinksystem-nvidia-b200-180gb-1000w-gpu', 'https://www.nvidia.com/en-us/data-center/dgx-b200/'] } }, // Blackwell HBM3e — 출하 구성 180GB/7.7TB/s(DGX 1440GB÷8·Lenovo) 또는 186GB/8TB/s. 192GB는 GTC 발표치로 출하 스펙 아님(2026-07-11 이중검증 정정, 구 192/8000은 거짓-fits 방향 오류)
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
  const count = gpu.count || 1; // 멀티GPU: reserve는 카드당 발생(OS/CUDA 컨텍스트 ×count). vramGB는 이미 합산값.
  return { type: 'gpu', gpu, env: env.key, memoryGB: gpu.vramGB, bandwidthGBs: gpu.bandwidthGBs, reserveGB: env.reserveGB * count, headroomRatio: GPU_HEADROOM_RATIO, _os: 0, gpuCount: count };
}

// 임의 멀티GPU 조합 — 이종 포함 (예: RTX 5090 + RTX 3090 = 56GB 풀). llama.cpp 레이어 분할 기준의 풀링 근사:
// VRAM=합산 · reserve=카드당 env reserve 합(드라이버/CUDA 컨텍스트는 카드마다 발생) · 대역폭=최소 카드(디코드는 느린 구간 바운드 — dormant).
// 한계(Methodology 고지와 동일): 단일 레이어는 한 카드에 들어가야 하며, 분할 오버헤드·PCIe 통신은 미반영(낙관적 근사).
export function combineGpus(gpus, envKey = DEFAULT_ENV) {
  if (!Array.isArray(gpus) || gpus.length === 0) return null;
  if (gpus.length === 1) return gpuDevice(gpus[0], envKey);
  const env = ENV_PRESETS[envKey] || ENV_PRESETS[DEFAULT_ENV];
  const combo = {
    name: gpus.map((g) => g.name).join(' + '),
    vramGB: gpus.reduce((s, g) => s + g.vramGB, 0),
    bandwidthGBs: Math.min(...gpus.map((g) => g.bandwidthGBs)),
    series: 'multi',
  };
  const cards = gpus.reduce((s, g) => s + (g.count || 1), 0); // 프리셋(count:2)이 조합에 끼어도 카드 수·reserve 정확히
  return { type: 'gpu', gpu: combo, env: env.key, memoryGB: combo.vramGB, bandwidthGBs: combo.bandwidthGBs, reserveGB: env.reserveGB * cards, headroomRatio: GPU_HEADROOM_RATIO, _os: 0, gpuCount: cards };
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

  // MLA(Multi-head Latent Attention) — GLM-5.2/GLM-4.7-Flash 등: K/V를 저차원 latent로 압축.
  // 캐시 = 압축 latent(dim=kv_lora_rank) + decoupled RoPE key(dim=qk_rope_head_dim), 전 헤드 공유(×1, K/V 분리 없음).
  // 출처: DeepSeek-V2 논문 arXiv:2405.04434 §2.1 "cache (d_c + d_h^R)·l elements" + 공식 DeepSeek-V3 추론코드
  //       (github.com/deepseek-ai/DeepSeek-V3 inference/model.py: kv_cache[dim=kv_lora_rank] + pe_cache[dim=qk_rope_head_dim]).
  //       absorb 모드(vLLM/SGLang 실사용)가 ×1 — 표준 GQA 대비 ~5× 작음.
  if (model.mlaKvLoraRank) {
    const perLayer = (model.mlaKvLoraRank + (model.mlaRopeDim || 0)) * bpe; // ×1, 전 레이어 균일
    const totalBytes = perLayer * model.layerCount * ctx;
    const marginalPerToken = perLayer * model.layerCount;
    return {
      totalGB: totalBytes / 1024 ** 3,
      perTokenKB: marginalPerToken / 1024,
      kvPerToken: marginalPerToken,
      totalBytes,
      effectiveCtx: ctx,
    };
  }

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

// 양자화별 메모리 보정 — MLX 공개 산출물 실크기 ÷ (totalParams × bits/8), safetensors 바이트합 기준 (HF API blobs, 2026-07-11 재유도).
// 구값(16bit 0.94~0.95, 8bit 0.9~0.99)은 출처 불명 + 방향 오류(실제는 base보다 큼 → 7~17% 과소, 거짓-fits) → 전면 교체.
// 작은 모델일수록 4bit 배율↑(임베딩 등 비양자 텐서 비중): e2b 1.39 > e4b 1.29 > 26b/31b 1.20. GGUF는 소수 bpw 경로(multiplier 1.0, 이중계상 금지).
// 출처: huggingface.co/mlx-community/gemma-4-{e2b,e4b,26b-a4b,31b}-it-{4bit,8bit,bf16} (+nvfp4 4bit 동률 1.20 확인)
const quantAdjust = {
  'Gemma 4 e2b': { 16: 1.0, 8: 1.15, 4: 1.39 },   // 실측 1.001 / 1.150 / 1.392
  'Gemma 4 e4b': { 16: 1.0, 8: 1.11, 4: 1.29 },   // 실측 0.993 / 1.110 / 1.287
  'Gemma 4 31b': { 16: 1.02, 8: 1.1, 4: 1.2 },    // 실측 1.019 / 1.100 / 1.199
  'Gemma 4 26b A4B': { 16: 1.01, 8: 1.1, 4: 1.2 }, // 실측 1.012 / 1.096 / 1.203
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

  // MLA: 압축 latent+rope, 전 레이어 균일 (calcKVCache MLA 분기와 동일 식)
  if (model.mlaKvLoraRank) {
    const perTokMla = (model.mlaKvLoraRank + (model.mlaRopeDim || 0)) * kbpe * overhead * model.layerCount;
    return Math.min(Math.floor(budget / (perTokMla + actPerTok)), model.maxContext);
  }

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
  // 오버헤드 공식의 단일 진실원 = calcRuntimeOverhead (KvDeepDive와 같은 소스 — 인라인 중복 금지, 드리프트 방지)
  const ov = calcRuntimeOverhead(model, ctx, { weightBpw, kvBits }, device);
  const rtDyn = ov.paramOverheadGB + ov.kvOverheadGB + ov.activationOverheadGB; // 동적 런타임(고정 reserve 미포함)
  const reserve = device.reserveGB;                       // OS/CUDA/디스플레이 통합 reserve
  const used = param + kv + rtDyn + reserve;
  const free = device.memoryGB - used;
  const headroom = device.memoryGB * device.headroomRatio;

  let verdict;
  if (free < 0) verdict = 'no';
  else if (free < headroom) verdict = 'tight';
  else verdict = 'yes';

  const os = device._os ?? 0; // Apple 표시 호환
  const rt = rtDyn + ov.fixedOverheadGB; // 기존 rt = 동적 + 고정(Apple 2.0 / GPU 0 — calcRuntimeOverhead가 결정)

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

// 멀티모델 스택 fit — 여러 모델이 같은 디바이스에 "동시 상주"할 때 (speculative decoding draft+target,
// IDE 자동완성+챗 등). 모델별 weights+KV+동적오버헤드를 각각 계산해 합산, reserve는 디바이스당 1회.
// 근거: llama.cpp speculative.md "need enough VRAM to load both models simultaneously" — 합산이 정확한 수학.
// entries: [{ model, ctx, weightBpw, kvBits }...] · deviceOrRam: gpu device 객체 또는 Mac RAM(GB)
export function simulateStack(entries, deviceOrRam) {
  const device = toDevice(deviceOrRam);
  const parts = entries.map(({ model, ctx, weightBpw, kvBits }) => {
    const c = Math.min(ctx, model.maxContext);
    const kb = kvBits ?? 16;
    const param = calcParamMemory(model, weightBpw).totalGB;
    const kv = calcKVCache(model, c, kb).totalGB;
    const ov = calcRuntimeOverhead(model, c, { weightBpw, kvBits: kb }, device);
    const rtDyn = ov.paramOverheadGB + ov.kvOverheadGB + ov.activationOverheadGB;
    return { model, ctx: c, weightBpw, kvBits: kb, param, kv, rtDyn, subtotal: param + kv + rtDyn };
  });
  const reserve = device.reserveGB;
  const used = parts.reduce((s, p) => s + p.subtotal, 0) + reserve;
  const free = device.memoryGB - used;
  const headroom = device.memoryGB * device.headroomRatio;
  const verdict = free < 0 ? 'no' : free < headroom ? 'tight' : 'yes';
  return {
    parts, device, memoryGB: device.memoryGB, reserve, used, free, headroom, verdict,
    pct: used / device.memoryGB,
    param: parts.reduce((s, p) => s + p.param, 0),
    kv: parts.reduce((s, p) => s + p.kv, 0),
    rt: parts.reduce((s, p) => s + p.rtDyn, 0),
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
      return { kind: 'bits', bits: b, text: t(`${b}bit로 양자화하면 들어가요 (품질 소폭 손실).`, `Quantize to ${b}-bit and it fits (small quality cost).`) };
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
  if (bigger) return { kind: 'ram', ram: bigger, text: t(`${bigger}GB 이상 맥이면 들어가요.`, `A ${bigger}GB+ Mac would fit.`) };
  return { kind: 'none', text: t('더 작은 모델이나 더 강한 양자화가 필요해요.', 'You need a smaller model or stronger quantization.') };
}

// GPU 탭용 suggestFix — Apple과 대칭인 "계산된" 제안 (하드코딩 문구 대체).
// 우선순위: 낮은 GGUF 티어(최소 변화부터) → KV 캐시 양자화 → 컨텍스트 축소 → 더 큰 VRAM 단일 카드.
export function suggestFixGpu(model, device, ctx, quant, L) {
  const t = L || ((ko) => ko);
  const kvBits = quant.kvBits ?? 16;
  // 1) 더 낮은 GGUF weight 티어 — bpw 내림차순(품질 손실 최소인 것부터)
  const lower = GPU_QUANTS.filter((q) => q.bpw < quant.weightBpw).sort((a, b) => b.bpw - a.bpw);
  for (const q of lower) {
    if (simulate(model, device, ctx, { weightBpw: q.bpw, kvBits }).verdict !== 'no') {
      return { kind: 'gguf', tier: q.tier, text: t(`${q.label}로 양자화하면 들어가요 (품질 소폭 손실).`, `Quantize to ${q.label} and it fits (small quality cost).`) };
    }
  }
  // 2) KV 캐시 양자화 (F16 → Q8 → Q4, llama.cpp -ctk/-ctv)
  for (const kb of [8, 4].filter((b) => b < kvBits)) {
    if (simulate(model, device, ctx, { weightBpw: quant.weightBpw, kvBits: kb }).verdict !== 'no') {
      return { kind: 'kv', kvBits: kb, text: t(`KV 캐시를 Q${kb}로 낮추면 들어가요 (-ctk/-ctv).`, `Drop the KV cache to Q${kb} (-ctk/-ctv) and it fits.`) };
    }
  }
  // 3) 현재 양자화에서 들어가는 최대 컨텍스트
  const maxCtx = calcMaxContext(model, device, quant);
  if (maxCtx >= 1024) {
    return { kind: 'ctx', ctx: maxCtx, text: t(`컨텍스트를 ${formatTokens(maxCtx, L)}까지 줄이면 들어가요.`, `Shorten context to ${formatTokens(maxCtx, L)} and it fits.`) };
  }
  // 4) 더 큰 VRAM 단일 카드 (멀티GPU 프리셋 제외 — "카드를 늘려라"는 별개 결정)
  const bigger = GPUS.filter((g) => !g.count && g.vramGB > device.memoryGB)
    .sort((a, b) => a.vramGB - b.vramGB)
    .find((g) => simulate(model, gpuDevice(g, device.env), ctx, quant).verdict !== 'no');
  if (bigger) {
    // 이름에 이미 용량이 있으면(예: 'A100 80GB') 괄호 중복 방지
    const label = bigger.name.includes('GB') ? bigger.name : `${bigger.name} (${bigger.vramGB}GB)`;
    return { kind: 'gpu', gpu: bigger.name, text: t(`${label}급 카드면 들어가요.`, `${label} would fit.`) };
  }
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
  if (t >= 1000000) return `${Math.round(t / 100000) / 10}M ${unit}`; // 1,048,576 → "1M" (1049K 방지)
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

// 칩 메모리 대역폭 (GB/s) — Apple 마케팅/풀다이 기준(Track A∥B 이중검증 2026-07-09).
// ⚠ binned Max는 저GPU코어 변종이 더 낮음(M3 Max 300, M4 Max 410) — 여기선 풀다이 값 저장.
//   대역폭은 estimateSpeed(dormant)만 사용, fit엔 무관하므로 칩당 단일 대표값으로 충분.
const CHIP_BANDWIDTH = {
  'M1': 68, 'M1 Pro': 200, 'M1 Max': 400, 'M1 Ultra': 800,
  'M2': 100, 'M2 Pro': 200, 'M2 Max': 400, 'M2 Ultra': 800,
  'M3': 100, 'M3 Pro': 150, 'M3 Max': 400, 'M3 Ultra': 819,
  'M4': 120, 'M4 Pro': 273, 'M4 Max': 546,
  'M5': 153, 'M5 Pro': 307, 'M5 Max': 614, // M5 base 153.6 (Wikipedia ∥ Codex 이중확인)
};
export function chipBandwidth(chip, gpuCores = 40) {
  if (chip === 'M5 Max') return gpuCores === 32 ? 460 : 614; // M5 Max GPU 코어수별
  return CHIP_BANDWIDTH[chip] || 307;
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

  const numExperts = c.num_local_experts || c.num_experts || c.n_routed_experts;
  const expertsPerToken = c.num_experts_per_tok;
  const isMoe = !!numExperts;

  // MLA(Multi-head Latent Attention) 감지 — kv_lora_rank 있으면 압축 KV 경로(GLM-5.2/GLM-4.7-Flash 등).
  // ⚠ DeepSeek-V4류(kv_lora_rank 부재 + MQA/compressor)는 MLA 아님 → 표준 경로 유지.
  const mlaKvLoraRank = c.kv_lora_rank || undefined;
  const mlaRopeDim = mlaKvLoraRank ? (c.qk_rope_head_dim || 0) : undefined;

  // 파라미터 수: safetensors total_size(저장 dtype 바이트)에서 역산, 없으면 이름 추정
  let totalParams = null;
  if (totalSize) {
    // 선-양자화 레포(MLX/AWQ/bnb): 저장 비트폭의 진실은 quantization(.bits) — torch_dtype은 원본 정밀도라
    // ÷2 과소계산 → 거짓 "fits" (issue #2). 합성 재현: 8bit 레포에 qbits 무시 시 params 절반 (test/parsehf.test.mjs).
    // 혼합 정밀도(일부 레이어 상위 bit)는 params 과대 방향으로만 틀림 — 보수적이라 허용.
    const qc = c.quantization_config;
    // bitsandbytes는 bits 필드 없이 load_in_4bit/8bit 불리언만 씀 — 누락 시 torch_dtype(2B) 경로로 ÷2~4 과소계산(거짓 fits)
    const qbits = c.quantization?.bits ?? qc?.bits ?? (qc?.load_in_4bit ? 4 : qc?.load_in_8bit ? 8 : undefined);
    const dt = String(c.torch_dtype || '').toLowerCase();
    const dtypeBytes = qbits ? qbits / 8
      : dt.includes('float32') || dt.includes('fp32') ? 4 : dt.includes('fp8') || dt.includes('int8') ? 1 : 2;
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
    mlaKvLoraRank,
    mlaRopeDim,
    maxContext: c.max_position_embeddings || 131072,
    // MLA가 우선 경로 → MLA 모델엔 sliding 필드 미설정(계산은 MLA 먼저 타지만 dead data 방지, correct-by-construction)
    slidingWindow: mlaKvLoraRank ? undefined : sliding || undefined,
    slidingPattern: mlaKvLoraRank ? undefined : sliding ? '5:1' : undefined,
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
export const DATA_UPDATED = '2026-07';
