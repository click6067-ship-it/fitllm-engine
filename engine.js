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
  // added = 행이 카탈로그에 처음 들어온 날(KST). 계산에 쓰지 않는 메타데이터 — fitllm.run의 '새 모델' 띠가 최근 순으로 뽑는다(2.21.2).
  //   값은 fitllm-v2 src/lib/engine.js · fitllm-engine engine.js git 기록에서 그 이름 행이 처음 생긴 커밋 날짜 중 이른 쪽이다.
  //   2026-05-25 = v2 첫 커밋(v1에서 이식한 행). 새 행은 추가하는 날로 채운다 — 비면 test/catalog-added.test.mjs가 막는다.
  // officialParams = 모델 카드가 공표한 총 파라미터(B, 출처 URL은 그 줄 주석). 표시용 메타데이터 — 메모리 계산은 totalParams(저장 텐서 실측)를 쓴다.
  //   카드가 '백본'만 세거나(DeepSeek-V4.1 Engram 별도) 비전·오디오를 빼면 두 값이 다르다. null = 카드 미공표(2.21.3).

  // --- GLM (zAI, MLA) — 압축 KV(kv_lora_rank 512 + qk_rope 64 = 576 elem/tok/layer, ×1) ---
  { name: 'GLM-4.7-Flash', group: 'GLM', added: '2026-07-09', tags: ['moe', 'mla'],
    // Exact tensor count: HF API at pinned revision; 30B-A3B class corroboration: official model card.
    // https://huggingface.co/api/models/zai-org/GLM-4.7-Flash/revision/7dd20894a642a0aa287e9827cb1a1f7f91386b67
    // https://huggingface.co/zai-org/GLM-4.7-Flash
    totalParams: 31.2, activeParams: 3, layerCount: 47, kvHeads: 20, kvHeadDim: 256, attnHeads: 20, hiddenSize: 2048,
    numExperts: 64, expertsPerToken: 4, mlaKvLoraRank: 512, mlaRopeDim: 64, maxContext: 202752, benchmarks: null,
    desc: 'MoE · MLA · ~31B / ~3B active · 64 experts(top-4) · 압축 KV(576/tok/layer) · 최대 202K' }, // config.json: zai-org/GLM-4.7-Flash
  { name: 'GLM-5.2', group: 'GLM', added: '2026-07-09', tags: ['moe', 'mla'],
    totalParams: 753, activeParams: 40, layerCount: 78, kvHeads: 64, kvHeadDim: 256, attnHeads: 64, hiddenSize: 6144,
    numExperts: 256, expertsPerToken: 8, mlaKvLoraRank: 512, mlaRopeDim: 64, maxContext: 1048576, benchmarks: null,
    desc: 'MoE · MLA · 753B / ~40B active · 256 experts(top-8) · 압축 KV · 최대 1M (4bit도 512GB급만 fit)' }, // config.json+index(1.5TB÷2 bf16): zai-org/GLM-5.2

  // --- gpt-oss (OpenAI, MoE + 슬라이딩 128, 절반이 full-attn) — MXFP4 네이티브 ---
  // totalParams는 모델카드 반올림(21B/117B)이 아니라 HF API safetensors.parameters(dtype별 *논리* 파라미터 수)의
  // immutable revision 합계를 parseHfConfig와 같은 0.1B 정밀도로 옮긴 값이다 — 카탈로그 행과 같은 모델의 HF 즉석 파싱이
  // 같은 판정을 내야 한다(#98 잔여: 21↔20.9, 117↔116.8 차이로 no↔tight 경계 케이스가 갈렸다).
  // BF16 = attention·router·embedding, U8 = MXFP4 packed expert(논리 수). 검증: src/lib/gpt-oss-hf-parity.test.js
  //   gpt-oss-20b  @ 6cee5e81ee83917806bbde320786a8fb61efebee — BF16 1,804,459,584 + U8 19,110,297,600 = 20,914,757,184 → 20.9
  //     https://huggingface.co/api/models/openai/gpt-oss-20b/revision/6cee5e81ee83917806bbde320786a8fb61efebee
  //     https://huggingface.co/openai/gpt-oss-20b/resolve/6cee5e81ee83917806bbde320786a8fb61efebee/config.json
  //   gpt-oss-120b @ b5c939de8f754692c1647ca79fbf85e8c1e70f8a — BF16 2,167,371,072 + U8 114,661,785,600 = 116,829,156,672 → 116.8
  //     https://huggingface.co/api/models/openai/gpt-oss-120b/revision/b5c939de8f754692c1647ca79fbf85e8c1e70f8a
  //     https://huggingface.co/openai/gpt-oss-120b/resolve/b5c939de8f754692c1647ca79fbf85e8c1e70f8a/config.json
  // 활성 파라미터 3.6B/5.1B는 OpenAI 모델카드 표기를 유지한다(https://huggingface.co/openai/gpt-oss-20b · https://huggingface.co/openai/gpt-oss-120b).
  // nativeFp4Params = 위 U8(MXFP4 전문가) 논리 수 — 어느 정밀도를 골라도 이 몫은 출시 크기(4.25 bit)로 센다(FP4_STORAGE_BPW 주석).
  //  공개 파일 바이트(HF 파일 목록, 2026-10-23 조회 · 단계 폴더/파일 합) — 출시 원본보다 큰 GGUF 단계는 F16뿐이다(F32 노름·메타데이터):
  //   20b  원본 13,761,316,904 · unsloth/gpt-oss-20b-GGUF@d449b42 Q4_K_M 11,624,759,488 · Q5_K_M 11,717,357,248 · Q6_K 12,041,000,128
  //        · Q8_0 12,109,567,168 · F16 13,792,639,168 · ggml-org/gpt-oss-20b-GGUF@15dcfc7 MXFP4 12,109,566,624
  //        · MLX mlx-community/gpt-oss-20b-MXFP4-Q4@f356f27 11,178,569,159 · -MXFP4-Q8@773a7da 12,076,207,568
  //   120b 원본 65,248,893,184 · unsloth/gpt-oss-120b-GGUF@ff1a82d Q4_K_M 62,768,723,552 · Q5_K_M 62,889,522,272 · Q6_K 63,284,496,960
  //        · Q8_0 63,387,347,520 · F16 65,369,017,728 · ggml-org/gpt-oss-120b-GGUF@7200b74 MXFP4 63,387,346,208
  //        · MLX mlx-community/gpt-oss-120b-MXFP4-Q4@bce781b 62,330,057,589 · -MXFP4-Q8@81e5ac3 63,386,948,506 · -mxfp4-bf16@a0ed0ff 65,248,907,820
  //   https://huggingface.co/api/models/unsloth/gpt-oss-120b-GGUF/revision/ff1a82da6ad466e32284fa3d2b86694db3204789?blobs=true
  //   https://huggingface.co/api/models/ggml-org/gpt-oss-120b-GGUF/revision/7200b74bea78b0a2964ef9f3897eefa401e890ff?blobs=true
  //   https://huggingface.co/api/models/unsloth/gpt-oss-20b-GGUF/revision/d449b42d93e1c2c7bda5312f5c25c8fb91dfa9b4?blobs=true
  //   https://huggingface.co/api/models/ggml-org/gpt-oss-20b-GGUF/revision/15dcfc7e7c608c6161507211ed43c5b590801b4b?blobs=true
  //   https://huggingface.co/api/models/bartowski/openai_gpt-oss-120b-GGUF/revision/8d64014efc314685c68ad448590f905fcb20e534?blobs=true
  //   https://huggingface.co/api/models/bartowski/openai_gpt-oss-20b-GGUF/revision/e39ba3aa000c47c83dacdc9e1ca2c9dd0808c205?blobs=true
  //  ggufFileBytes = 단계별 공개 GGUF 중 가장 큰 파일(unsloth·bartowski·ggml-org) — 가중치는 이 크기보다 작게 세지 않는다(weightBytes).
  //  ggufFileSource = 그 파일들이 나온 텐서 목록(HF dtype별 정확한 수 + 텐서 형상을 정하는 config 값). HF 즉석 파싱은 이것이 정확히 같을 때만
  //   크기를 물려받는다 — 반올림한 총 파라미터만 보면 어휘를 늘린 파생 모델이 원본 파일 크기를 받는다(Codex 2차 검수).
  //   hidden 2880이 K-quant 블록 256의 배수가 아니라 llama.cpp가 2880폭 텐서를 Q8_0 등으로 대체해, 비전문가 몫은 단계 평균보다 크다
  //   (bartowski Q4_K_M 120b 62,841,713,344 · 20b 11,673,418,816 · Q5_K_M 20b 11,728,415,296; bf16 120b 65,369,016,768 · 20b 13,792,637,760).
  //   F16 GGUF는 1차원 텐서·바이어스를 F32로 담아 원본 safetensors보다 120 MB(120b)·31 MB(20b) 크다 — 실제로 내려받는 파일이라 그대로 쓴다.
  //  2.21.4까지는 전부 선택 단계 평균 bit로 세서 120b Q8_0이 115.6 GiB(실파일 59.0), Q4_K_M이 66.6 GiB(실파일 58.5)였다.
  { name: 'gpt-oss-20b', group: 'gpt-oss', added: '2026-07-09', tags: ['moe'],
    totalParams: 20.9, activeParams: 3.6, layerCount: 24, kvHeads: 8, kvHeadDim: 64, attnHeads: 64, hiddenSize: 2880,
    numExperts: 32, expertsPerToken: 4, maxContext: 131072, slidingWindow: 128, globalAttnLayers: 12, benchmarks: null,
    nativeFp4Params: 19.1102976, // 19,110,297,600 = 24층 × 32 전문가 × (2880×5760 + 2880×2880)
    fp4ReleaseBytes: 13761316904, // 출시 safetensors 샤드 합(original/ 제외)
    ggufFileBytes: { Q4_K_M: 11673418816, Q5_K_M: 11728415296, Q6_K: 12041000128, Q8_0: 12109567168, FP16: 13792639168 },
    ggufFileSource: { parameters: { BF16: 1804459584, U8: 19110297600 }, config: { num_hidden_layers: 24, hidden_size: 2880, intermediate_size: 2880, num_local_experts: 32, num_attention_heads: 64, num_key_value_heads: 8, head_dim: 64, vocab_size: 201088, tie_word_embeddings: false } },
    desc: 'MoE · 20.9B / 3.6B active · 32 experts(top-4) · 슬라이딩128(full 12/24) · 최대 128K' }, // config.json: openai/gpt-oss-20b @ 6cee5e8
  { name: 'gpt-oss-120b', group: 'gpt-oss', added: '2026-07-09', tags: ['moe'],
    totalParams: 116.8, activeParams: 5.1, layerCount: 36, kvHeads: 8, kvHeadDim: 64, attnHeads: 64, hiddenSize: 2880,
    numExperts: 128, expertsPerToken: 4, maxContext: 131072, slidingWindow: 128, globalAttnLayers: 18, benchmarks: null,
    nativeFp4Params: 114.6617856, // 114,661,785,600 = 36층 × 128 전문가 × (2880×5760 + 2880×2880)
    fp4ReleaseBytes: 65248893184, // 출시 safetensors 샤드 합(original/ 제외)
    ggufFileBytes: { Q4_K_M: 62841713344, Q5_K_M: 62889522272, Q6_K: 63284496960, Q8_0: 63387347520, FP16: 65369017728 },
    ggufFileSource: { parameters: { BF16: 2167371072, U8: 114661785600 }, config: { num_hidden_layers: 36, hidden_size: 2880, intermediate_size: 2880, num_local_experts: 128, num_attention_heads: 64, num_key_value_heads: 8, head_dim: 64, vocab_size: 201088, tie_word_embeddings: false } },
    desc: 'MoE · 116.8B / 5.1B active · 128 experts(top-4) · 슬라이딩128(full 18/36) · 최대 128K' }, // config.json: openai/gpt-oss-120b @ b5c939d

  // === Qwen 3.6 27B (dense, 하이브리드 linear+full attn ~3:1) — HF config.json ===
  {
    name: 'Qwen 3.6 27B',
    group: 'Qwen 3.6',
    added: '2026-05-25',
    tags: ['dense'],
    totalParams: 27.2,
    activeParams: null,
    layerCount: 64,
    fullAttnLayers: 16, // 64레이어 중 16개만 Gated(full) attention, 48개는 Gated DeltaNet(linear) → KV 캐시는 16레이어만
    // Gated DeltaNet 고정 상태(ctx 무관) — Qwen/Qwen3.6-27B config.json linear_* 필드
    linearAttn: { layers: 48, numKHeads: 16, numVHeads: 48, headKDim: 128, headVDim: 128, convKernel: 4 },
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
    added: '2026-05-25',
    tags: ['moe'],
    totalParams: 35.0,
    activeParams: 3.0,
    layerCount: 40,
    fullAttnLayers: 10, // 40레이어 중 10개만 full attention(4개마다 1개), 30개는 linear attention → KV 캐시는 10레이어만
    // Gated DeltaNet 고정 상태 — Qwen/Qwen3.6-35B-A3B config.json linear_* 필드
    linearAttn: { layers: 30, numKHeads: 16, numVHeads: 32, headKDim: 128, headVDim: 128, convKernel: 4 },
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
  { name: 'Qwen-AgentWorld-35B-A3B', group: 'Qwen3.5', added: '2026-07-09', tags: ['moe'],
    totalParams: 34.7, activeParams: 3.0, layerCount: 40, fullAttnLayers: 10, kvHeads: 2, kvHeadDim: 256, attnHeads: 16, hiddenSize: 2048,
    linearAttn: { layers: 30, numKHeads: 16, numVHeads: 32, headKDim: 128, headVDim: 128, convKernel: 4 }, // config.json linear_* 필드
    numExperts: 256, expertsPerToken: 8, maxContext: 262144, benchmarks: null,
    desc: 'MoE · ~34.7B / ~3B active · 하이브리드(full 10 + linear 30) · KV는 10레이어만 · 최대 256K' }, // config.json+index: Qwen/Qwen-AgentWorld-35B-A3B

  // === Gemma 4 E2B (Dense + Per-Layer-Embeddings, NOT MoE) — HF config.json ===
  {
    name: 'Gemma 4 e2b',
    group: 'Gemma 4',
    added: '2026-05-25',
    tags: ['dense', 'ple'],
    totalParams: 5.1,
    activeParams: 2.3,
    // PLE 텐서 = vocab_size_per_layer_input 262144 × hidden_size_per_layer_input 256 × 35L = 2,348,810,240
    // 출처: google/gemma-4-E2B-it config.json(text_config) + GGUF per_layer_token_embd 실측 산술 일치
    // (unsloth GGUF discussion: 1,837MiB = 262144×256×35 × Q6_K 6.5625bpw 정확 재현). GPU 상주 제외 근거는 residentParamsB 참조.
    pleParams: 2.349,
    pleOffloadVerified: true,
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
    added: '2026-05-25',
    tags: ['dense', 'ple'],
    totalParams: 8,
    activeParams: 4.5,
    // PLE 텐서 = 262144 × 256 × 42L = 2,818,572,288 — google/gemma-4-E4B-it config.json(text_config).
    // 공식 유효치 역산 정합: raw 7,996.2M − PLE 2,818.6M − token_embd 671.1M ≈ 4,506M = "4.5B effective" (모델카드·HF 블로그)
    pleParams: 2.819,
    pleOffloadVerified: true,
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
    added: '2026-06-05',
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
    added: '2026-05-25',
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
    added: '2026-05-25',
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
  { name: 'Llama-3.2-3B-Instruct', group: 'Llama', added: '2026-07-09', tags: ['dense'],
    totalParams: 3.2, activeParams: 3.2, layerCount: 28, kvHeads: 8, kvHeadDim: 128, attnHeads: 24, hiddenSize: 3072,
    maxContext: 131072, benchmarks: null,
    desc: 'Dense · 3.2B · 28레이어 · GQA(24/8) · 최대 128K' }, // meta-llama/Llama-3.2-3B-Instruct (gated) ↔ unsloth 미러
  { name: 'Llama-3.1-8B-Instruct', group: 'Llama', added: '2026-07-09', tags: ['dense'],
    totalParams: 8.0, activeParams: 8.0, layerCount: 32, kvHeads: 8, kvHeadDim: 128, attnHeads: 32, hiddenSize: 4096,
    maxContext: 131072, benchmarks: null,
    desc: 'Dense · 8.0B · 32레이어 · GQA(32/8) · 최대 128K' }, // meta-llama/Llama-3.1-8B-Instruct (gated) ↔ unsloth/Meta-Llama-3.1-8B-Instruct 미러

  // --- MiniCPM (openbmb) — on-device/edge 계열 ---
  { name: 'MiniCPM5-1B', group: 'MiniCPM', added: '2026-07-16', tags: ['dense'],
    totalParams: 1.081, activeParams: 1.081, layerCount: 24, kvHeads: 2, kvHeadDim: 128, attnHeads: 16, hiddenSize: 1536,
    maxContext: 131072, benchmarks: null,
    // openbmb/MiniCPM5-1B config.json(model_type=llama·tie_word_embeddings=false). 2독립출처 파라미터: safetensors.index total_size 2,161,265,664B ÷ 2(bf16) = 1,080,632,832 == config-dim 손계산(embed 130560×1536 + 미tied lm_head + 24층×(attn 7,077,888 + mlp 21,233,664)) = 1,080,632,832 정확 일치
    desc: 'Dense · 1.1B · GQA(16/2) · on-device/edge · 최대 128K' },

  // --- Draft 소형모델 — Stack 탭(speculative decoding draft+target · IDE 자동완성) 페어용. 플래그십 아님(카탈로그 최신순 정리와 별개 기능군) ---
  // ⚠️ Qwen3 2종 totalParams = GGUF/디스크 기준(임베딩 1회): safetensors 카운트(751.6M/2031.7M)는 tied lm_head 중복 저장 포함 — 사용 금지.
  //    검증(2026-07-09 verifier): Qwen3-0.6B-Q8_0.gguf 실측 639,446,688B = 596.0M×Q8_0+메타 ✓ / 1.7B GGUF 1,834,426,016B ✓ (2출처+GGUF 실측)
  { name: 'Qwen3-0.6B', group: 'Draft', added: '2026-07-09', tags: ['dense', 'draft'],
    totalParams: 0.596, activeParams: 0.596, layerCount: 28, kvHeads: 8, kvHeadDim: 128, attnHeads: 16, hiddenSize: 1024,
    maxContext: 40960, benchmarks: null, // config max_position_embeddings 40960 (모델카드 공표 native 32K — config 기준 원칙)
    desc: 'Dense · 0.6B · vLLM 표준 draft — Qwen 계열 타깃 페어 (vocab 151936 호환)' }, // Qwen/Qwen3-0.6B config.json + HF API + GGUF 실측
  { name: 'Qwen3-1.7B', group: 'Draft', added: '2026-07-09', tags: ['dense', 'draft'],
    totalParams: 1.721, activeParams: 1.721, layerCount: 28, kvHeads: 8, kvHeadDim: 128, attnHeads: 16, hiddenSize: 2048,
    maxContext: 40960, benchmarks: null,
    desc: 'Dense · 1.7B · 30B급 타깃엔 0.6B보다 스루풋 우위(vLLM 벤치) — Qwen 계열 draft' }, // Qwen/Qwen3-1.7B config.json + HF API + GGUF 실측
  { name: 'Llama-3.2-1B-Instruct', group: 'Draft', added: '2026-07-09', tags: ['dense', 'draft'],
    totalParams: 1.236, activeParams: 1.236, layerCount: 16, kvHeads: 8, kvHeadDim: 64, attnHeads: 32, hiddenSize: 2048,
    maxContext: 131072, benchmarks: null, // params 1,235,814,400 = 분석 카운트 바이트단위 일치(tied·lm_head 미저장 — 그대로 디스크 기준)
    desc: 'Dense · 1.2B · Llama 3.x 타깃 표준 draft (vocab 128256 호환) · 최대 128K' }, // meta-llama(gated) ↔ unsloth·NousResearch 미러 2종 일치
  { name: 'Gemma-3-1B-it', group: 'Draft', added: '2026-07-09', tags: ['dense', 'draft'],
    totalParams: 1.0, activeParams: 1.0, layerCount: 26, kvHeads: 1, kvHeadDim: 256, attnHeads: 4, hiddenSize: 1152,
    maxContext: 32768, slidingWindow: 512, globalAttnLayers: 4, // transformers gemma3 + google/gemma_pytorch 2출처: 26층 중 full-attention 4(레이어 6·12·18·24), 나머지 22층 window 512
    benchmarks: null,
    desc: 'Dense · 1.0B · MQA(4/1) · 슬라이딩 512(글로벌 4레이어) — Gemma 계열 draft · 최대 32K' }, // google/gemma-3-1b-it(gated) ↔ unsloth·mlx 미러 + gemma_pytorch

  // === Claude Opus 4.7 — Cloud (벤치마크 기준점, 메모리 시뮬 제외) ===
  // 출처: Claude Opus 4.7 System Card, Table 8.1.A(p.190-191) 및 §8.2 — 2026-09-14 원문 직접 확인.
  //   https://www-cdn.anthropic.com/037f06850df7fbe871e206dad004c3db5fd50340/Claude%20Opus%204.7%20System%20Card.pdf
  //   SWE-bench Verified 87.6% · GPQA Diamond 94.2% (둘 다 adaptive thinking max effort, 5회 평균)
  // ⚠️ MMLU-Pro = null: Anthropic은 이 모델에 MMLU-Pro를 공표한 적이 없다(공표하는 건 MMMLU 91.5% §8.5,
  //   GMMLU 평균 89.9% §8.12.1). 2026-09-14 이전까지 이 자리에 있던 0.899 는 그 GMMLU 89.9% 와 정확히
  //   일치해 다른 벤치가 MMLU-Pro 칸에 들어간 것으로 판단하고 제거했다 — 출처 없는 숫자는 남기지 않는다.
  {
    name: 'Claude Opus 4.7',
    group: 'Claude (Cloud)',
    added: '2026-05-25',
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
    benchmarks: { GPQA: 0.942, 'MMLU-Pro': null, 'SWE-Bench': 0.876 },
    contextLimit: '1M',
    desc: 'Cloud 모델 — 벤치마크 기준점 (로컬 설치 불가, 비교용)',
  },

  // === Hy3 (Tencent, MoE 192E top-8 + shared 1) — Day-0 2026-07-13, https://huggingface.co/tencent/Hy3 config.json ===
  // ⚠️ 배열 끝 append 고정(?m= 링크 보존). BF16 확정: HF API safetensors BF16 298,786,140,416 + F32 15,360
  //    → ×2B(+×4B) = index total 597,572,342,272 B 바이트 정확 일치. FP8은 별도 레포(tencent/Hy3-FP8).
  {
    name: 'Hy3',
    group: 'Hunyuan',
    added: '2026-07-13',
    tags: ['moe'],
    // 298.8 = 디스크/로드 풋프린트(엔진 관례) — 모델카드 "295B 본체 + MTP 3.8B". ⚠️ 커뮤니티 GGUF가 MTP를 포함
    //    ("80 layers + 1 MTP", Q4_K_M 181GB = 298.8B×4.8944/8 ±1.5% — vcruz305/Hy3-GGUF 실측, verifier 2026-07-13)
    //    → nextn 탈락 가정(295) 기각, 실파일 기준 298.8 채택.
    totalParams: 298.8,
    activeParams: 21, // 모델카드 "Activated: 21B" (산술 재구성 ≈20.6B 부합: attn 6.04 + MoE 79L×(8+1shared)×18.87M + dense L0 + embed×2 untied)
    layerCount: 80,
    kvHeads: 8,
    kvHeadDim: 128,
    attnHeads: 64,
    hiddenSize: 4096,
    numExperts: 192,
    expertsPerToken: 8,
    maxContext: 262144, // config max_position_embeddings, rope_type default(스케일링 없음) — 모델카드 "256K"
    benchmarks: { GPQA: 0.904, 'MMLU-Pro': null, 'SWE-Bench': 0.78 }, // 모델카드 공식: GPQA Diamond 90.4 / SWE-Bench Verified 78 / MMLU-Pro 미공표
    desc: 'MoE · 295B(+3.8B MTP) / 21B active · 표준 GQA(8KV×128) 전층 풀어텐션 · 최대 256K · Apache 2.0',
  },

  // ==========================================================================
  //  Qwen 3.8 (Alibaba) — 2026-08 Day-0. qwen3_5 아키(Gated DeltaNet 3 : Gated Attention 1).
  //  ⚠️ 배열 끝 append 고정(?m= 링크 보존) — Hy3와 동일 규칙.
  //  3중 교차검증: ① config.json layer_types 카운트 ② safetensors index total_size÷2
  //               ③ 모델카드 "Hidden Layout" 서술 — 셋이 완전 일치.
  // ==========================================================================
  {
    name: 'Qwen 3.8 27B',
    group: 'Qwen 3.8',
    added: '2026-08-24',
    tags: ['dense', 'vlm'],
    // 27,781,427,952 = HF API safetensors BF16 == index total_size 55,562,855,904 ÷ 2 (바이트 정확 일치)
    // 비전 인코더(model.visual 333텐서, SigLIP depth 27) 포함한 실제 체크포인트 전체 — 사용자 확정 2026-08-24.
    totalParams: 27.781,
    activeParams: null, // dense
    layerCount: 64,
    fullAttnLayers: 16, // layer_types: full_attention 16 / linear_attention 48 (모델카드 "16 × (3×DeltaNet → 1×Gated Attention)")
    kvHeads: 4,         // num_key_value_heads (모델카드 "4 for KV")
    kvHeadDim: 256,     // head_dim (모델카드 "Head Dimension: 256")
    attnHeads: 24,      // 모델카드 "24 for Q"
    hiddenSize: 5120,
    linearAttn: { layers: 48, numKHeads: 16, numVHeads: 48, headKDim: 128, headVDim: 128, convKernel: 4 }, // 모델카드 "48 for V, 16 for QK, Head Dim 128"
    maxContext: 262144, // config max_position_embeddings — 모델카드 "262,144 natively"(1M은 호스티드 확장)
    benchmarks: { GPQA: 0.892, 'MMLU-Pro': null, 'SWE-Bench': null }, // 모델카드 GPQA Diamond 89.2. MMLU-Pro 미공표, SWE는 Pro만 공표(Verified 아님 → 미기입)
    desc: 'Dense · 27.8B(비전 인코더 포함) · 64레이어(풀어텐션 16 + DeltaNet 48) · KV는 16레이어만 · 최대 256K',
  },
  {
    name: 'Qwen 3.8 2.4T-A95B',
    group: 'Qwen 3.8',
    added: '2026-08-24',
    tags: ['moe'],
    // 2,446,182,725,504 = HF API safetensors BF16 == index total_size 4,892,365,451,008 ÷ 2 (바이트 정확 일치)
    totalParams: 2446.183,
    activeParams: 95, // 모델카드 "2.4T in total and 95B activated"
    layerCount: 92,
    fullAttnLayers: 23, // layer_types: full 23 / linear 69 (모델카드 "23 × (3×DeltaNet → 1×Gated Attention)")
    kvHeads: 4,
    kvHeadDim: 256,
    attnHeads: 64,
    hiddenSize: 8192,
    linearAttn: { layers: 69, numKHeads: 16, numVHeads: 128, headKDim: 128, headVDim: 128, convKernel: 4 }, // 모델카드 "128 for V, 16 for QK, Head Dim 128"
    numExperts: 512,
    expertsPerToken: 10, // 모델카드 "10 Routed + 1 Shared"
    maxContext: 262144,  // 모델카드 "262,144 natively"
    benchmarks: null,    // 카드의 벤치 열 헤더가 호스티드 변형 "Qwen3.8-Max" — 오픈 체크포인트 귀속 불확실 → 미기입
    desc: 'MoE · 2.4T / 95B active · 512 experts(top-10+shared) · 풀어텐션 23/92(DeltaNet 69) · 최대 256K',
  },

  // ==========================================================================
  //  Laguna 2.1 (poolside) — 2026-07 릴리스. 표준 GQA + full/SWA(window 512) 인터리브.
  //  현 엔진의 슬라이딩 경로로 그대로 계산 가능(신규 수학 불필요).
  //  교차검증: config.json layer_types 카운트 ↔ 모델카드 서술 일치.
  // ==========================================================================
  {
    name: 'Laguna XS 2.1',
    group: 'Laguna',
    added: '2026-08-24',
    tags: ['moe'],
    // 33,442,617,088 = HF API safetensors BF16 == index total_size 66,885,234,176 ÷ 2
    totalParams: 33.443,
    activeParams: 3, // 모델카드 "33B total ... with 3B activated parameters per token"
    layerCount: 40,
    globalAttnLayers: 10, // layer_types: full_attention 10 / sliding_attention 30
    slidingWindow: 512,   // config sliding_window
    kvHeads: 8,
    kvHeadDim: 128,
    attnHeads: 48,
    hiddenSize: 2048,
    numExperts: 256,
    expertsPerToken: 8,
    maxContext: 262144,
    benchmarks: { GPQA: null, 'MMLU-Pro': null, 'SWE-Bench': 0.709 }, // 모델카드 SWE-bench Verified 70.9%(pass@1 4회 평균). GPQA·MMLU-Pro 미공표
    desc: 'MoE · 33.4B / 3B active · 256 experts(top-8) · 슬라이딩512(full 10/40) · 최대 256K · OpenMDW-1.1',
  },
  {
    name: 'Laguna S 2.1',
    group: 'Laguna',
    added: '2026-08-24',
    tags: ['moe'],
    // 117,561,977,600 = HF API safetensors BF16 == index total_size 235,123,955,200 ÷ 2
    totalParams: 117.562,
    activeParams: 8, // 모델카드 "118B total ... with 8B activated"
    layerCount: 48,
    globalAttnLayers: 12, // 모델카드 "48 layers in a 1:3 global-to-SWA ratio (12 global, 36 sliding, window 512)" — config layer_types와 일치
    slidingWindow: 512,
    kvHeads: 8,
    kvHeadDim: 128,
    attnHeads: 48,
    hiddenSize: 3072,
    numExperts: 256,
    expertsPerToken: 10,
    maxContext: 1048576, // config max_position_embeddings
    benchmarks: null,    // 카드 벤치표에 SWE-bench Verified·GPQA·MMLU-Pro 없음(Multilingual/Pro/Terminal만) → 미기입
    desc: 'MoE · 117.6B / 8B active · 256 experts(top-10) · 슬라이딩512(full 12/48) · 최대 1M · OpenMDW-1.1',
  },

  // ==========================================================================
  //  Spark-X2.5 (XHToken) — 2026-09 Day-0 (#103). model_type spark2_5: 표준 GQA(16/4, head_dim 256) + SWA(512) 3:1 인터리브.
  //  ⚠️ 배열 끝 append 고정(?m= 링크 보존). 1차 출처(전부 pinned revision 5e10fcc0286756aebf7c41dc52c1e42d95c70281):
  //   config.json  https://huggingface.co/XHToken/Spark-X2.5-4B/blob/5e10fcc0286756aebf7c41dc52c1e42d95c70281/config.json
  //   modeling     https://huggingface.co/XHToken/Spark-X2.5-4B/blob/5e10fcc0286756aebf7c41dc52c1e42d95c70281/modeling_spark.py
  //   index        https://huggingface.co/XHToken/Spark-X2.5-4B/blob/5e10fcc0286756aebf7c41dc52c1e42d95c70281/model.safetensors.index.json
  //   HF API       https://huggingface.co/api/models/XHToken/Spark-X2.5-4B/revision/5e10fcc0286756aebf7c41dc52c1e42d95c70281
  //  3중 교차검증(바이트 정확 일치): ① HF API safetensors BF16 4,112,079,360 == index total_parameters == total_size 8,224,158,720 ÷ 2
  //   ② 5개 shard 헤더의 텐서 shape 곱 합계 == 4,112,079,360
  //   ③ config 치수 손계산 == 4,112,079,360: 36 × [q_k_v_proj 2560×6144 + out_proj 4096×2560 + g_proj 2560×16
  //      + MLP 3×2560×10240 + norm 2×2560] + embedding 131072×2560(tied) + 최종 norm 2560
  //  g_proj(hidden→num_heads)는 어텐션 출력에 곱하는 헤드별 게이트(modeling_spark.py Spark2_5Attention) — 가중치 40,960/layer만
  //  늘고 K/V 캐시 shape(4 kvh × 256)는 불변. 라이선스 Apache-2.0(모델카드·LICENSE). 1.7B 형제 모델은 이번 변경에 미포함.
  // ==========================================================================
  {
    name: 'Spark-X2.5-4B',
    group: 'Spark',
    added: '2026-09-05',
    tags: ['dense'],
    totalParams: 4.112, // 4,112,079,360
    activeParams: 4.112, // dense — 전 파라미터 활성
    layerCount: 36,
    globalAttnLayers: 9,   // layer_types: full_attention 9 (idx 3,7,…,35) / sliding_attention 27
    slidingWindow: 512,    // config sliding_window
    slidingPattern: '3:1', // 27:9 실카운트 — 모델카드 "one full-attention layer with three sliding-window attention layers"
    kvHeads: 4,
    kvHeadDim: 256,
    attnHeads: 16,
    hiddenSize: 2560,
    maxContext: 1048576, // config max_position_embeddings — 모델카드 "native context windows of up to 1M tokens"
    benchmarks: null,    // 카드 표의 "GPQA"는 Diamond 명시·체크포인트 귀속이 없고 MMLU-Pro 미공표 → 미기입(엔진 원칙: 공개·검증된 것만)
    desc: 'Dense · 4.1B · 36레이어 · GQA(16/4, head_dim 256) · 슬라이딩512(3:1, full 9/36) · 최대 1M · Apache 2.0',
  },
  // ==========================================================================
  //  Granite-4.2-30B (IBM, model_type granite / GraniteForCausalLM) — 2026-09-05 Day-0 (#93)
  //  구조 = 표준 dense GQA(32 heads / 8 kv-heads, head_dim = hidden 4096 / 32 = 128) · 64 레이어 전부 full attention
  //  (sliding·layer_types·linear·MoE·MLA·PLE·MTP 키 전무). Granite 고유 스칼라(attention_multiplier 0.0078125 = 1/128 ·
  //  embedding_multiplier · residual_multiplier · logits_scaling)는 값 스케일링만 하고 텐서 치수·KV 레이아웃을 바꾸지 않는다
  //  (config-gate BENIGN). trust_remote_code/auto_map 없음(transformers 4.57.1 네이티브); 레포의 granite_thinking_parser.py는
  //  vLLM 추론 텍스트(<think>) 파서 플러그인, model.sig는 서명 파일 — 둘 다 메모리와 무관.
  //  1차 출처(전부 pinned revision 9e668ce1c538387ef24d3644e9b0606647762636 — 2026-09-04 모델카드 갱신 커밋.
  //   config/generation_config/index blob은 이슈 #93 코멘트의 8b445a5c315f32da0f89e1f648bfec0cd601b154 와 byte-identical):
  //   config.json  https://huggingface.co/ibm-granite/granite-4.2-30b/blob/9e668ce1c538387ef24d3644e9b0606647762636/config.json
  //   index        https://huggingface.co/ibm-granite/granite-4.2-30b/blob/9e668ce1c538387ef24d3644e9b0606647762636/model.safetensors.index.json
  //   HF API       https://huggingface.co/api/models/ibm-granite/granite-4.2-30b/revision/9e668ce1c538387ef24d3644e9b0606647762636
  //   모델카드     https://huggingface.co/ibm-granite/granite-4.2-30b/blob/9e668ce1c538387ef24d3644e9b0606647762636/README.md
  //  3중 교차검증(바이트 정확 일치): ① HF API safetensors BF16 29,276,770,304 == index total_size 58,553,540,608 ÷ 2
  //   ② 11개 shard 헤더의 텐서 shape 곱 합계 == 29,276,770,304 (579 텐서 전부 BF16; shard 합 58,553,607,904 − 텐서 = 헤더 67,296 B)
  //   ③ config 치수 손계산 == 29,276,770,304: 64 × [q 4096×4096 + k 1024×4096 + v 1024×4096 + o 4096×4096
  //      + MLP 3×4096×32768 + norm 2×4096] + embed 100352×4096 + lm_head 100352×4096(untied) + 최종 norm 4096
  //  maxContext = config max_position_embeddings 131072(모델카드 "Natively Supports 128K"). 카드의 "512K 확장"은 config에 없어
  //  미반영(config 기준 원칙). 라이선스 Apache-2.0(모델카드·HF cardData). 3B/8B 형제·base 모델은 이번 변경에 미포함.
  // ==========================================================================
  {
    name: 'Granite-4.2-30B',
    group: 'Granite',
    added: '2026-09-05',
    tags: ['dense'],
    totalParams: 29.277, // 29,276,770,304
    activeParams: 29.277, // dense — 전 파라미터 활성
    layerCount: 64,
    kvHeads: 8,
    kvHeadDim: 128, // config에 head_dim 키 없음 → 4096/32. shard 헤더 k_proj.weight [1024, 4096] = 8 × 128 로 확인
    attnHeads: 32,
    hiddenSize: 4096,
    maxContext: 131072,
    benchmarks: null, // 카드 표의 "GPQA"는 Diamond 명시가 없고 체크포인트(thinking 모드) 귀속이 불명 → 미기입(엔진 원칙: 공개·검증된 것만)
    desc: 'Dense · 29.3B · 64레이어 · GQA(32/8, head_dim 128) · 최대 128K · Apache 2.0',
  },

  // ==========================================================================
  //  GLM-5.3 (zai-org) — 2026-09-05 사설 준비(트래픽 실험 gate 이전, 배포 전). HF revision aca966e4e02791568aa6a4ced368624b3d897f42.
  //  공식 모델카드: "GLM-5.3 uses the same base model as GLM-5.2 — every gain comes from post-training."
  //   https://huggingface.co/zai-org/GLM-5.3/blob/aca966e4e02791568aa6a4ced368624b3d897f42/README.md
  //  config.json은 GLM-5.2(cf457fa734ab149ffef225f80893eb38c6ff5cdc)와 quantization_config(fp8·e4m3·block 128×128)·transformers_version만 다르다:
  //   https://huggingface.co/zai-org/GLM-5.3/blob/aca966e4e02791568aa6a4ced368624b3d897f42/config.json
  //   https://huggingface.co/zai-org/GLM-5.2/blob/cf457fa734ab149ffef225f80893eb38c6ff5cdc/config.json
  //  파라미터: HF API safetensors.parameters(논리 수) BF16 2,103,729,152 + F8_E4M3 751,226,191,872 + F32 19,456 = 753,329,940,480
  //   — GLM-5.2(BF16 753,329,921,024 + F32 19,456)와 total 정확 일치. 카탈로그는 GLM-5.2와 같은 정수 관례 753 (parseHfConfig는
  //   두 config 모두 DSA index_topk로 fail-closed 거부하므로 카탈로그↔즉석 파싱 parity 경로가 없다).
  //   https://huggingface.co/api/models/zai-org/GLM-5.3/revision/aca966e4e02791568aa6a4ced368624b3d897f42
  //  형상: 78 layers · hidden 6144 · 64/64 heads · kv_lora_rank 512 · qk_rope_head_dim 64 · 256 experts(top-8) · max 1,048,576.
  //   kvHeadDim 256 = config v_head_dim/qk_head_dim(qk_nope 192 + rope 64); config head_dim 192는 qk_nope_head_dim이다.
  //   MLA 경로(calcKVCache/calcMaxContext)는 mlaKvLoraRank+mlaRopeDim만 쓰므로 kvHeadDim은 비작동(naiveKVCache 비교값·MCP 원시 노출뿐)
  //   — GLM-5.2 행과 같은 값으로 parity 유지. activeParams 40은 GLM-5.2 행에서 상속(같은 base) — 새 측정 아님.
  //  핀 리비전의 저장 정밀도는 FP8이지만 판정 입력은 사용자가 고르는 양자화라 행은 논리 파라미터 수만 갖는다(Hy3-FP8 관례).
  //  벤치마크 미기입(엔진 원칙). ?m= 인덱스 26 — append-only, 표시 순서는 MODEL_GROUP_ORDER('GLM').
  // ==========================================================================
  { name: 'GLM-5.3', group: 'GLM', added: '2026-09-06', tags: ['moe', 'mla'],
    totalParams: 753, activeParams: 40, layerCount: 78, kvHeads: 64, kvHeadDim: 256, attnHeads: 64, hiddenSize: 6144,
    officialParams: 744, // HF 카드(@aca966e)엔 미기재 · 공식 GitHub 모델 표 "GLM-5.3 | 744B-A40B" — https://github.com/zai-org/GLM-5/blob/83c03d317378f86f0a012f7c4cd48b6243c7e69b/README.md · AWS Bedrock 모델 카드 "744B total parameters" — https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-zai-glm-5-3.html (저장 텐서 753B는 MTP 층 포함)
    numExperts: 256, expertsPerToken: 8, mlaKvLoraRank: 512, mlaRopeDim: 64, maxContext: 1048576, benchmarks: null,
    desc: 'MoE · MLA · 753B / ~40B active · 256 experts(top-8) · 압축 KV · 최대 1M (4bit도 512GB급만 fit) · GLM-5.2와 같은 base(post-training만 차이)' },

  // ==========================================================================
  //  MiniCPM5-2B (openbmb) — 2026-09-14 Day-0. 기존 MiniCPM5-1B와 같은 표준 llama GQA 경로(신규 수학 없음).
  //  ⚠️ 배열 끝 append 고정(?m= 링크 보존). 1차 출처(전부 pinned revision 12a3808a956f869c767195e9266b59c4d21d92e2):
  //   config.json  https://huggingface.co/openbmb/MiniCPM5-2B/blob/12a3808a956f869c767195e9266b59c4d21d92e2/config.json
  //   index        https://huggingface.co/openbmb/MiniCPM5-2B/blob/12a3808a956f869c767195e9266b59c4d21d92e2/model.safetensors.index.json
  //   HF API       https://huggingface.co/api/models/openbmb/MiniCPM5-2B
  //  3중 교차검증(바이트 정확 일치, 2,516,756,480): ① HF API safetensors BF16 == index total_size 5,033,512,960 ÷ 2
  //   ② config 치수 손계산 == 2,516,756,480: 42 × [q 2048×2048 + k 2048×256 + v 2048×256 + o 2048×2048
  //      + MLP 3×2048×6144 + norm 2×2048] + embed 130560×2048 + lm_head 130560×2048(untied) + 최종 norm 2048
  //   ③ 단일 shard(model-00000-of-00001.safetensors) 파일 하나라 index total_size가 곧 실파일 바이트다.
  //  maxContext = config max_position_embeddings 131072(rope_scaling null — 확장 없음). 라이선스 Apache-2.0(HF cardData).
  //  벤치 미기입: 모델카드 표에 GPQA Diamond·MMLU-Pro·SWE-Bench Verified 귀속 표기가 없다(엔진 원칙).
  // ==========================================================================
  { name: 'MiniCPM5-2B', group: 'MiniCPM', added: '2026-09-14', tags: ['dense'],
    totalParams: 2.517, activeParams: 2.517, layerCount: 42, kvHeads: 2, kvHeadDim: 128, attnHeads: 16, hiddenSize: 2048,
    officialParams: 2.517, // 모델카드 "Number of Parameters: 2,516,756,480" — https://huggingface.co/openbmb/MiniCPM5-2B/blob/12a3808a956f869c767195e9266b59c4d21d92e2/README.md (= 위 3중 교차검증 값)
    maxContext: 131072, benchmarks: null,
    desc: 'Dense · 2.5B · 42레이어 · GQA(16/2) · on-device/edge · 최대 128K · Apache 2.0' },

  // ==========================================================================
  //  Claude Opus 5 — Cloud (2026-07-24 공개). 2026-09-14 추가.
  //  ⚠️ 배열 끝 append 고정(?m= 링크 보존). 벤치 기준점(OPUS·opusPct)은 **4.7 그대로 둔다** —
  //     Opus 5 공식 발표는 GPQA Diamond·MMLU-Pro·SWE-bench Verified 중 **어느 것도 공표하지 않는다**
  //     (Frontier-Bench v0.1 · CursorBench 3.2 · ARC-AGI 3 · OSWorld 2.0 · GDPval-AA v2 · HLE 등 에이전틱 평가만).
  //     https://www.anthropic.com/news/claude-opus-5
  //     엔진 원칙상 그 세 가지 외의 벤치는 기입하지 않으므로 benchmarks = null 이다. 카탈로그에 최신 모델이
  //     보이되 비교 숫자는 지어내지 않는 상태 — "틀린 숫자 > 없는 숫자" 규칙 그대로.
  //  maxContext 1,000,000 = 공식 모델 사양(컨텍스트 1M). 구조 필드는 비공개라 전부 null(메모리 시뮬 제외).
  // ==========================================================================
  {
    name: 'Claude Opus 5',
    group: 'Claude (Cloud)',
    added: '2026-09-14',
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
    benchmarks: null,
    contextLimit: '1M',
    desc: 'Cloud 모델 — 최신 Claude (공식 GPQA/MMLU-Pro/SWE-V 미공표라 비교 수치 없음)',
  },

  // ==========================================================================
  //  Nex-N2.5 (nex-agi) — 2026-09-14 Day-0. model_type qwen3_5_moe(Gated DeltaNet 3 : Gated Attention 1).
  //  현 엔진의 linearAttn + fullAttnLayers 경로로 그대로 계산된다(신규 수학 불필요).
  //  ⚠️ 배열 끝 append 고정(?m= 링크 보존).
  //  ⚠️ totalParams 는 비전 타워(vision_config depth 27)를 **포함한 체크포인트 전체**다 — Qwen 3.8 27B 행의
  //     확정 관례(2026-08-24)와 같다. 텍스트 전용 가중치만 재려면 별도 근거가 필요하므로 하지 않는다.
  //  ⚠️ activeParams = null: 모델카드가 활성 파라미터를 공표하지 않는다(Hy3 "Activated: 21B", Laguna
  //     "3B activated" 같은 표기가 없음). 산술 추정치를 카드값처럼 적지 않는다.
  //  라이선스 Apache-2.0(HF cardData). 형제 Nex-N2.5-Max 는 model_type deepseek_v4(compressed_sparse_attention
  //  + index_topk) 라 미모델링 — 추가하지 않는다.
  // ==========================================================================
  {
    // 35,107,181,936 = HF API safetensors BF16 == index total_size 70,214,363,872 ÷ 2 (바이트 정확 일치)
    // https://huggingface.co/nex-agi/Nex-N2.5-mini/blob/87420286149d9cce9bd46cd335ef9bda33c37c1b/config.json
    name: 'Nex-N2.5-mini',
    group: 'Nex',
    added: '2026-09-14',
    tags: ['moe', 'vlm'],
    totalParams: 35.107,
    officialParams: null, // 모델카드(@8742028) 총 파라미터 미공표(공식 X 글 "Mini (35B)"는 단일 출처·미검증) — 표시는 totalParams(실측)
    activeParams: null, // 모델카드 미공표
    layerCount: 40,
    fullAttnLayers: 10, // layer_types: full_attention 10 / linear_attention 30
    kvHeads: 2,
    kvHeadDim: 256,
    attnHeads: 16,
    hiddenSize: 2048,
    linearAttn: { layers: 30, numKHeads: 16, numVHeads: 32, headKDim: 128, headVDim: 128, convKernel: 4 }, // config linear_* 필드
    numExperts: 256,
    expertsPerToken: 8,
    maxContext: 262144,
    benchmarks: null, // 카드 벤치표에 GPQA Diamond·MMLU-Pro·SWE-Bench Verified 귀속 표기 없음
    desc: 'MoE · 35.1B(비전 인코더 포함) · 256 experts(top-8) · 풀어텐션 10/40(DeltaNet 30) · 최대 256K · Apache 2.0',
  },
  {
    // 396,802,360,816 = HF API safetensors 논리 파라미터 합(BF16 10,255,304,176 + F8_E4M3 386,547,056,640).
    // 핀 리비전은 FP8 블록 양자화 배포본이지만 판정 입력은 사용자가 고르는 양자화라 행은 논리 수만 갖는다
    // (GLM-5.3 · Hy3-FP8 관례). F8 텐서 수는 config 손계산과 정확 일치: 60 × 512 × 3 × 4096 × 1024.
    // index total_size 407,104,850,912 B 와의 차이는 FP8 블록 스케일 텐서(파라미터 아님)로 설명된다.
    // https://huggingface.co/nex-agi/Nex-N2.5-Pro/blob/0f389abfca976dcbab5c7638268d01cfe8ace4fe/config.json
    name: 'Nex-N2.5-Pro',
    group: 'Nex',
    added: '2026-09-14',
    tags: ['moe', 'vlm'],
    totalParams: 396.802,
    officialParams: null, // 모델카드(@0f389ab) 총 파라미터 미공표(공식 X 글 "Pro (397B)"는 단일 출처·미검증) — 표시는 totalParams(실측)
    activeParams: null, // 모델카드 미공표
    layerCount: 60,
    fullAttnLayers: 15, // layer_types: full_attention 15 / linear_attention 45
    kvHeads: 2,
    kvHeadDim: 256,
    attnHeads: 32,
    hiddenSize: 4096,
    linearAttn: { layers: 45, numKHeads: 16, numVHeads: 64, headKDim: 128, headVDim: 128, convKernel: 4 },
    numExperts: 512,
    expertsPerToken: 10,
    maxContext: 262144,
    benchmarks: null,
    desc: 'MoE · 396.8B(FP8 배포·논리 파라미터) · 512 experts(top-10) · 풀어텐션 15/60(DeltaNet 45) · 최대 256K · Apache 2.0',
  },

  // ==========================================================================
  //  DeepSeek-V4 계열 (deepseek-ai, MIT) — 2026-10-18 추가(2.21.0). 8월 #40·#101에서 fail-closed로 닫았던 압축·희소 어텐션을
  //  compressedKv로 모델링한다(compressedKvBytes 주석 = 공식 추론 코드의 버퍼 목록). ⚠️ 배열 끝 append 고정(?m= 링크 보존).
  //  형상은 pinned config를 parseHfConfig가 읽은 값과 같다 — test/deepseek-v4.test.mjs가 카탈로그 행 == 즉석 파싱을 고정한다.
  //  totalParams = HF API safetensors.parameters(dtype별 논리 수) 합 — FP4 전문가(I8로 보고, 논리 수 = 손계산과 정확 일치)·FP8·BF16·F32.
  //   모델카드의 284B·1.6T·552B는 MTP·DSpark·Engram·비전 타워를 뺀 표기라 이보다 작다. 행은 체크포인트 전체(Nex·Qwen 3.8 관례).
  //  activeParams = 모델카드 표기(V4-Flash 13B · V4-Pro 49B · V4.1 decode 16B, prefill 8B) — 속도 근사용, 판정 비관여.
  //  벤치 미기입: GPQA Diamond·MMLU-Pro·SWE-bench Verified가 추론 모드별로만 공표돼 체크포인트 하나에 귀속되지 않는다(1017은 미공표).
  // ==========================================================================
  {
    // 763,205,315,794 = BF16 1,976,441,856 + F32 42,307,282 + F8_E4M3 204,015,223,296 + I8(FP4 전문가) 557,171,343,360.
    //   I8 = (40층 × 384 + DSpark 3층 × 128) 전문가 × 3 × 5120 × 2304 — 정확 일치. F8에 Engram 표 196.6B(384,006,168 + 384,016,682 행 × 256) 포함.
    // https://huggingface.co/api/models/deepseek-ai/DeepSeek-V4.1-Flash-1017/revision/1bae0e1088993b9b82de08f59de5be5a34e42170
    // https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/1bae0e1088993b9b82de08f59de5be5a34e42170/config.json
    // KV: 소스 4층(2·8·14 ratio 2, 20 ratio 1)만 캐시 — 512·128 차원 → 토큰당 1,600원소(FP4면 공식 890B/tok과 정확 일치).
    name: 'DeepSeek-V4.1-Flash-1017', group: 'DeepSeek', added: '2026-10-18', tags: ['moe', 'vlm'],
    // 공식 HF ID는 이 고정 리비전 행으로 해석한다(2.21.1) — CLI·REST·MCP·배지가 같은 답을 네트워크 없이 낸다. V4.1-Flash(@314113f)는
    // config.json이 바이트 동일하고 safetensors 논리 수(dtype별)도 같아 메모리 판정이 같다. 파생·미러 레포는 별칭이 아니라 HF 즉석 파싱으로 간다.
    aliases: ['deepseek-ai/DeepSeek-V4.1-Flash-1017', 'deepseek-ai/DeepSeek-V4.1-Flash', 'DeepSeek-V4.1-Flash'],
    totalParams: 763.2, activeParams: 16, layerCount: 40, kvHeads: 1, kvHeadDim: 512, attnHeads: 64, hiddenSize: 5120,
    officialParams: 552, // DeepSeek 공식 표기는 백본 552B(Engram 196B 별도 표기, 공식 합계 없음 · 저장 763.2B = 백본+Engram+비전·MTP) — V4.1-Flash 카드 "552B backbone parameters"·표 # Backbone Params 552B — https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/314113f8a7dadb388215396ba5fbee2c610a04d6/README.md · 1017 카드 = 그 minor update: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/1bae0e1088993b9b82de08f59de5be5a34e42170/README.md
    numExperts: 384, expertsPerToken: 6, mtpLayerCount: 3, maxContext: 1048576, benchmarks: null,
    compressedKv: {
      family: 'deepseek_v41', windowSize: 128, windowLayers: 43, headDim: 512,
      caches: [
        { ratio: 1, dim: 512, layers: 1 }, { ratio: 1, dim: 128, layers: 1 },
        { ratio: 2, dim: 512, layers: 3 }, { ratio: 2, dim: 128, layers: 3 },
      ],
      stateBytes: 24576, tokenStateBytes: 8, workspaceBytes: 536870912, kvLayers: 4, oGroups: 8, indexHeads: 32,
      vocabSize: 129280, routedExperts: [384, 128], // 본체 384 · DSpark 초안 블록 128
    },
    engramParams: 196.6138496, // (384,006,168 + 384,016,682) 행 × 256 — FP8 저장, ENGRAM_MIN_BPW 하한으로 센다
    // 위 I8 = FP4 전문가 — 출시 크기(4.25 bit) 위로 세지 않는다(FP4_STORAGE_BPW). 바이트 확인: I8 × 4.25/8 + BF16×2 + F32×4 + F8×1
    //   + Engram E8M0 스케일(196.6138496e9 / 32) = 510,278,795,096 B vs 샤드 합 510,296,708,312 B(차 17.9 MB = 헤더·FP8 블록 스케일).
    //   공개 GGUF unsloth/DeepSeek-V4.1-Flash-1017-GGUF@60ad365(2026-10-23 조회): UD-Q4_K_XL 500,707,576,573 · UD-Q8_K_XL 508,376,763,197
    //   (+ DSpark Q8_0 7,976,914,240 · mmproj BF16 972,208,288 별도 파일) — FP4 전문가를 그대로 둔다. UD-IQ1_S 232,979,608,475는 다시 양자화한 빌드.
    //   같은 이름 단계(Q4_K_M~Q8_0)의 공개 파일은 없어 4.25 bit 위 단계는 출시본(fp4ReleaseBytes) 아래로 세지 않는다(fp4FloorBytes).
    nativeFp4Params: 557.17134336,
    fp4ReleaseBytes: 510296708312, // 출시 safetensors 샤드 합
    desc: 'MoE · 763.2B(Engram 196.6B·비전·DSpark 포함) / 16B active · 384 experts(top-6) · CSA2 KV 공유(소스 4/40층) · 최대 1M · MIT',
  },
  {
    // 290,944,616,402 = BF16 1,415,259,264 + I64 2,327,040 + F32 36,168,018 + F8_E4M3 6,023,020,544 + I8(FP4 전문가) 283,467,841,536.
    // https://huggingface.co/api/models/deepseek-ai/DeepSeek-V4-Flash/revision/60d8d70770c6776ff598c94bb586a859a38244f1
    // https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/config.json
    // KV: CSA(ratio 4) 21층 512 + 인덱서 128 · HCA(ratio 128) 20층 512 · 앞 2층은 창만 → 토큰당 3,440원소.
    // nativeFp4Params = I8(FP4 전문가). I8 × 4.25/8 + 나머지 dtype 바이트 = 159,609,118,280 B vs 샤드 합 159,617,149,040 B(차 8.0 MB).
    //   공개 GGUF unsloth/DeepSeek-V4-Flash-GGUF@e3aa0d6(2026-10-23 조회): UD-Q4_K_XL 155,095,240,320 · UD-Q8_K_XL 161,869,614,720 B
    //   (bartowski/DeepSeek-V4-Flash-GGUF@f9815d7 MXFP4 156,004,658,464) — 같은 이름 단계 파일이 없어 출시본이 하한이다(fp4FloorBytes).
    name: 'DeepSeek-V4-Flash', group: 'DeepSeek', added: '2026-10-18', tags: ['moe'],
    aliases: ['deepseek-ai/DeepSeek-V4-Flash'],
    totalParams: 290.9, activeParams: 13, layerCount: 43, kvHeads: 1, kvHeadDim: 512, attnHeads: 64, hiddenSize: 4096,
    officialParams: 284, // 모델카드 "284B parameters (13B activated)" — https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/README.md · V4.1-Flash 카드 표도 284B
    numExperts: 256, expertsPerToken: 6, mtpLayerCount: 1, maxContext: 1048576, benchmarks: null,
    nativeFp4Params: 283.467841536, // 44층(본체 43 + MTP 1) × 256 × 3 × 4096 × 2048
    fp4ReleaseBytes: 159617149040, // 출시 safetensors 샤드 합
    compressedKv: {
      family: 'deepseek_v4', windowSize: 128, windowLayers: 44, headDim: 512,
      caches: [{ ratio: 4, dim: 512, layers: 21 }, { ratio: 4, dim: 128, layers: 21 }, { ratio: 128, dim: 512, layers: 20 }],
      stateBytes: 12206080, tokenStateBytes: 0, workspaceBytes: 536870912, kvLayers: 41, oGroups: 8, indexHeads: 64,
      vocabSize: 129280, routedExperts: [256],
    },
    desc: 'MoE · 290.9B(MTP 포함) / 13B active · 256 experts(top-6) · CSA 21 + HCA 20층(창 128) · 최대 1M · MIT',
  },
  {
    // 1,598,839,674,782 = BF16 2,816,899,328 + I64 2,327,040 + F32 87,776,414 + F8_E4M3 23,169,335,296 + I8(FP4 전문가) 1,572,763,336,704.
    // https://huggingface.co/api/models/deepseek-ai/DeepSeek-V4-Pro/revision/b5968e9190ef611bbf34a7229255be88a0e937c1
    // https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/b5968e9190ef611bbf34a7229255be88a0e937c1/config.json
    // KV: CSA 30층 + HCA 31층(앞 2층 포함) → 토큰당 4,924원소.
    // nativeFp4Params = I8(FP4 전문가). I8 × 4.25/8 + 나머지 dtype 바이트 = 864,703,378,552 B vs 샤드 합 864,721,029,744 B(차 17.7 MB).
    //   공개 GGUF는 후속 체크포인트 빌드만 있다: unsloth/DeepSeek-V4-Pro-0813-GGUF@6d05361 UD-Q4_K_XL 849,683,927,055 · UD-Q8_K_XL 873,445,601,295 B.
    name: 'DeepSeek-V4-Pro', group: 'DeepSeek', added: '2026-10-18', tags: ['moe'],
    aliases: ['deepseek-ai/DeepSeek-V4-Pro'],
    totalParams: 1598.8, activeParams: 49, layerCount: 61, kvHeads: 1, kvHeadDim: 512, attnHeads: 128, hiddenSize: 7168,
    officialParams: 1600, // 모델카드 "1.6T parameters (49B activated)" — https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/b5968e9190ef611bbf34a7229255be88a0e937c1/README.md · V4.1-Flash 카드 표도 1.6T
    numExperts: 384, expertsPerToken: 6, mtpLayerCount: 1, maxContext: 1048576, benchmarks: null,
    nativeFp4Params: 1572.763336704,
    fp4ReleaseBytes: 864721029744, // 출시 safetensors 샤드 합
    compressedKv: {
      family: 'deepseek_v4', windowSize: 128, windowLayers: 62, headDim: 512,
      caches: [{ ratio: 4, dim: 512, layers: 30 }, { ratio: 4, dim: 128, layers: 30 }, { ratio: 128, dim: 512, layers: 31 }],
      stateBytes: 18710528, tokenStateBytes: 0, workspaceBytes: 536870912, kvLayers: 61, oGroups: 16, indexHeads: 64,
      vocabSize: 129280, routedExperts: [384],
    },
    desc: 'MoE · 1.6T(MTP 포함) / 49B active · 384 experts(top-6) · CSA 30 + HCA 31층(창 128) · 최대 1M · MIT',
  },

  // ==========================================================================
  //  Muse Spark Open (Meta Superintelligence Labs, Muse Model License) — 2026-10-15 공개, 2026-10-18 추가(2.21.0).
  //  표준 GQA(128 Q / 16 KV × 64) + 슬라이딩 2048 46층 · 전체 16층(1:3 반복, 전체 층은 NoPE) — 기존 슬라이딩 경로 그대로(신규 수학 없음).
  //   https://github.com/huggingface/transformers/blob/487dcfa6ab1c294ce2695005108fdf19a63363eb/src/transformers/models/muse_spark/modeling_muse_spark.py
  //  원본 meta-models/Muse-Spark-Open-816B-A42B(@0fb02478b02565b2ce7fc57e6051c91172774ac9)는 gated라 config를 직접 못 받는다.
  //  공개 미러 unsloth/Muse-Spark-Open-816B-A42B(@49cade7b0c6ed297f0434229ac67eea5f9ae1220)의 config.json은 HF API가 보고하는
  //  원본의 git blob e6df3a846ebfcb0d55c350c1e79e7886c65b7399와 같은 blob이다(바이트 동일 — README 말고 78개 중 77개 파일 동일).
  //   https://huggingface.co/api/models/meta-models/Muse-Spark-Open-816B-A42B?blobs=true
  //   https://huggingface.co/unsloth/Muse-Spark-Open-816B-A42B/blob/49cade7b0c6ed297f0434229ac67eea5f9ae1220/config.json
  //  원본 HF ID는 gated라 CLI의 HF 경로가 401로 끝난다 — 그래서 원본 ID를 별칭으로 둬 이 행으로 해석되게 한다.
  //  totalParams 822.0 = safetensors BF16 821,906,490,368 + F32 130,039,296(비전 50층·오디오 타워 포함) — 카드 816B는 텍스트 본체.
  //  activeParams 42 = 모델카드 "816B total / 42B active". 벤치 미기입: 원본 README가 gated라 미러 사본으로만 확인된다.
  // ==========================================================================
  {
    name: 'Muse Spark Open', group: 'Muse', added: '2026-10-18', tags: ['moe', 'vlm'],
    aliases: ['Muse-Spark-Open-816B-A42B', 'meta-models/Muse-Spark-Open-816B-A42B'],
    totalParams: 822.0, activeParams: 42, layerCount: 62, kvHeads: 16, kvHeadDim: 64, attnHeads: 128, hiddenSize: 8192,
    officialParams: 816, // 모델카드 "816B total / 42B active"(원본 gated, 미러 사본) — https://huggingface.co/unsloth/Muse-Spark-Open-816B-A42B/blob/49cade7b0c6ed297f0434229ac67eea5f9ae1220/README.md · 공식 레포 ID …-816B-A42B
    numExperts: 256, expertsPerToken: 8, maxContext: 1048576,
    slidingWindow: 2048, slidingPattern: '23:8', globalAttnLayers: 16, // layer_types: sliding 46 : full 16
    benchmarks: null,
    desc: 'MoE · 822B(비전·오디오 포함) / 42B active · 256 experts(top-8) · GQA 128/16 · 슬라이딩 2048(full 16/62) · 최대 1M',
  },
];

// 카탈로그 표시 순서(최신·화제순). MODELS 배열은 ?m= 공유링크 때문에 append-only라
// 배열 순서 == 표시 순서가 더 이상 성립하지 않는다. 이 목록이 표시 순서의 단일 출처다.
// 여기 없는 그룹은 뒤에 배열 순서대로 붙는다(신규 그룹 추가를 잊어도 사라지지 않게).
export const MODEL_GROUP_ORDER = [
  'DeepSeek', 'Muse', 'Nex', 'Granite', 'Spark', 'Qwen 3.8', 'Laguna', 'GLM', 'gpt-oss', 'Qwen 3.6', 'Qwen3.5',
  'Hunyuan', 'Gemma 4', 'Llama', 'MiniCPM', 'Draft',
];

// 그룹 단위로 묶어 표시 순서대로 반환 — 드롭다운(ControlBar/GpuControlBar) 공용.
export function groupedForDisplay(models) {
  const groups = [];
  for (const m of models) {
    let g = groups.find((x) => x.group === m.group);
    if (!g) { g = { group: m.group, items: [] }; groups.push(g); }
    g.items.push(m);
  }
  const rank = (g) => { const i = MODEL_GROUP_ORDER.indexOf(g); return i === -1 ? MODEL_GROUP_ORDER.length : i; };
  return groups.sort((a, b) => rank(a.group) - rank(b.group) || groups.indexOf(a) - groups.indexOf(b));
}

// 로컬에서 돌릴 수 있는(시뮬 대상) 모델만
export const LOCAL_MODELS = MODELS.filter((m) => !m.isCloud);

// ===== 맥북/맥 RAM 옵션 (통합메모리 GB) =====
// M1–M6 전세대 + Ultra/Studio. fit엔 통합메모리 GB만 영향(대역폭은 chipBandwidth = dormant speed용).
// Track A(Apple newsroom+Wikipedia) ∥ Track B(Codex) 이중검증(2026-07-09, 일치). M4 Ultra는 미출시 — 2026-08-25 기준 최신 Ultra는 M5 Ultra.
// 출처: Apple Newsroom 각 세대 발표 + support.apple.com 스펙 + Wikipedia Apple_M1..M4.
// 2026-08-27 추가(M6 · M5 Ultra) — 2026-08-25 Apple 발표, 출하 2026-09-22. 1차 출처 직접 확인:
//   뉴스룸 https://www.apple.com/newsroom/2026/08/apple-introduces-m6-and-m5-ultra-for-a-big-leap-in-performance-and-ai-compute/
//   기술사양 https://www.apple.com/mac-mini/specs/ · https://www.apple.com/mac-studio/specs/
// ⚠️ 최대 통합메모리는 안 늘었다: M6 32GB = M4·M5와 동일 / M5 Ultra 512GB = M3 Ultra와 동일 → fit 판정 불변, 바뀐 건 대역폭뿐.
export const MACBOOK_RAM_GROUPS = {
  'M6': [16, 24, 32], // Mac mini(2026). 최대 32GB — M4·M5와 동일. 대역폭만 용량 연동(16GB 153 / 24·32GB 170) → chipBandwidth 참조
  'M5': [16, 24, 32], // MacBook Air/Pro 14" (Wikipedia Apple_M5 ∥ Codex 트랙 교차확인 2026-07-09)
  'M5 Pro': [24, 48, 64], // Mac mini 기술사양 재확인 2026-08-27 (24 base / 48 / 64)
  'M5 Max': [36, 48, 64, 128], // Mac Studio 기술사양 재확인 2026-08-27 (36=32코어GPU / 48·64·128=40코어GPU)
  // ⚠️ 512GB 의도적 제외 — 2026-08-27 apple.com/shop/buy-mac/mac-studio 실측: M5 Ultra는 96GB·256GB만 예약 가능하고
  //    512GB는 예약조차 안 됨. Apple 뉴스룸 원문(1차 출처) "Mac Studio with 512GB of unified memory is coming in late October."
  //    가격 미공개. 실제 판매 시작되면 512 추가할 것. ⚠️ Sol 크로스리뷰는 "공식 카탈로그면 512를 남기고 가용성 메타데이터를 붙이라"며 반대함 — 사용자 결정으로 제외 유지.
  //    (512GB 티어 자체는 M3 Ultra로 이미 선택 가능 — fit 수학은 동일)
  'M5 Ultra': [96, 256], // Mac Studio(2026). 128GB 옵션 없음(96 다음이 256) — 예약 가능 구성만 등재
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
  // 통합 메모리 reserve — 맥 경로(OS 몫 + 런타임 고정 2GB)와 같은 구조를 GB10 실측으로 채운다.
  //   reserve = (128 − OS가 보는 총량) + DGX OS 유휴 사용량 + 런타임 고정분
  // NVIDIA는 GPU 전용 고정 할당이 없다고 명시한다("share the same physical memory space without a fixed carve-out",
  //   https://docs.nvidia.com/dgx/dgx-spark-porting-guide/overview.html) — 맥의 GPU wired limit 같은 상한은 없고, OS가 보는 총량이 곧 CUDA 총량이다.
const GB10_128GB_UNIFIED = Object.freeze({
    environment: 'dgx-os-unified',
    // OS·CUDA가 보는 총량(GiB). /proc/meminfo MemTotal = cuDeviceTotalMem = torch mem_get_info total, DGX OS 7.6 / 커널 6.17 실측:
    //   127,599,376 kB·127,535,276 kB(기기 2대, MemTotal×1024 = cuDeviceTotalMem) — https://github.com/midagedev/bloomery/issues/13 (2026-10-10)
    //   127,600,524 kB — https://raw.githubusercontent.com/vllm-project/vllm-omni/main/recipes/MiniMaxAI/MiniMax-H3-Spark-GB10.md
    //   torch.cuda.mem_get_info total 121.69 GiB — https://forums.developer.nvidia.com/t/driver-issue-with-580-178-04/383859 (2026-09-21)
    //   "121.7 GiB visible to the OS and to CUDA"(GIGABYTE ATOM, DGX OS 7.6.0) — https://www.back2gaming.com/review/gigabyte-ai-top-atom-128gb-dgx-spark/
    // ⚠ 더 낮은 관측: 초기 DGX OS·4K 페이지 커널·일부 OEM 초기 펌웨어는 119.5–119.7 GiB(약 2 GiB 적음 —
    //   https://simonwillison.net/2025/Oct/14/nvidia-dgx-spark/ · https://github.com/emjotde/spark-ram-reclaim). 전제 문구로 고지한다.
    osVisibleGB: 121.69,
    // DGX OS 유휴 시 새 프로세스가 못 쓰는 양(MemTotal − MemAvailable, GiB): 헤드리스 2대 4.8–5.1(bloomery #13) ·
    //   free -h 121Gi total / 117Gi available ≈ 4(https://aaif.io/blog/running-goose-fully-offline-on-a-dgx-spark) ·
    //   GNOME 데스크톱 ~116 available vs 서버 ~118(https://technotim.com/posts/ubuntu-gb10/). 대표값 5.0 — 데스크톱(기본)과 헤드리스 사이.
    osIdleGB: 5.0,
    // 추론 프로세스 고정분(CUDA 컨텍스트 + 프로세스 host 메모리 + 고정 버퍼). 통합 메모리에서는 host 몫도 같은 풀이라
    // 맥 경로의 고정 런타임 2GB(calcRuntimeOverhead fixed)를 그대로 쓴다 — GB10 전용 실측이 아닌 엔진 추정치다.
    //   방향 확인: torch CUDA 컨텍스트가 떠 있는 유휴 상태에서 available 115.52 / 121.69 GiB(차 6.17 GiB, 위 forum 383859) — OS 유휴 ~4–5 GiB보다 1–2 GiB 큼.
    runtimeFixedGB: 2.0,
    sources: {
      osVisibleGB: [
        'https://github.com/midagedev/bloomery/issues/13',
        'https://forums.developer.nvidia.com/t/driver-issue-with-580-178-04/383859',
        'https://www.back2gaming.com/review/gigabyte-ai-top-atom-128gb-dgx-spark/',
      ],
      osIdleGB: [
        'https://github.com/midagedev/bloomery/issues/13',
        'https://aaif.io/blog/running-goose-fully-offline-on-a-dgx-spark',
        'https://technotim.com/posts/ubuntu-gb10/',
      ],
    },
    estimates: { runtimeFixedGB: 'engine runtime constant shared with the Apple unified-memory path; not a GB10 measurement' },
  });

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
  // 별칭(2026-10-16): Max-Q Workstation Edition과 Server Edition도 96GB GDDR7 — fit에 쓰는 VRAM이 같다.
  //   Max-Q 96GB·1792GB/s(300W) https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/workstation-datasheet-blackwell-rtx-pro-6000-max-q-nvidia-3519233.pdf · https://lenovopress.lenovo.com/lp2364-thinksystem-nvidia-rtx-pro-6000-blackwell-max-q-workstation-edition-gpu
  //   Server 96GB·1597GB/s     https://www.nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/ · https://lenovopress.lenovo.com/lp2263-thinksystem-nvidia-rtx-pro-6000-blackwell-server-edition-pcie-gen5-gpu
  //   ⚠ Server Edition 대역폭(1597)은 이 행(1792, WE)과 다르다 — 대역폭은 fit에 안 쓰는 dormant 값이고 표시는 정본 이름으로만 나간다.
  //   실측 nvidia-smi memory.total 97887 MiB — 세 에디션·ECC 켜짐/꺼짐 모두 같다(GDDR7 ECC는 용량을 깎지 않음):
  //   https://github.com/gpustack/gpustack-operator/pull/755 (Server, ECC ENABLED) · https://github.com/ollama/ollama/issues/14287 · https://github.com/MeiGen-AI/InfiniteTalk/issues/159 (WE, ECC Off)
  { name: 'RTX PRO 6000 Blackwell', series: 'workstation', vramGB: 96, bandwidthGBs: 1792, status: 'VERIFIED', verifiedAt: '2026-07-09',
    aliases: [
      'RTX PRO 6000 Blackwell Workstation Edition', 'RTX PRO 6000 Workstation Edition',
      'RTX PRO 6000 Blackwell Max-Q Workstation Edition', 'RTX PRO 6000 Blackwell Max-Q', 'RTX PRO 6000 Max-Q',
      'RTX PRO 6000 Blackwell Server Edition', 'RTX PRO 6000 Server Edition',
    ], sources: { vramGB: ['https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000/', 'https://www.storagereview.com/review/nvidia-rtx-pro-6000-workstation-gpu-review-blackwell-architecture-and-96-gb-for-pro-workflows'], bandwidthGBs: ['https://www.nvidia.com/content/dam/en-zz/Solutions/data-center/rtx-pro-6000-blackwell-workstation-edition/workstation-blackwell-rtx-pro-6000-workstation-edition-nvidia-us-3519208-web.pdf', 'https://www.techpowerup.com/gpu-specs/rtx-pro-6000-blackwell.c4272'] } }, // Workstation Edition
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

  // ===== 확장 (2026-10-16, Track A[claude hw-crawler] ∥ Track B[codex] 블라인드 리콘실 — fit 값 CONFLICT 0) =====
  // 기준: 2026-10-16까지 실제 판매된 제품만. RTX 50 SUPER는 미발표·미출시(NVIDIA 50 시리즈 페이지에 없음,
  //   https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/ · https://www.pcgamer.com/hardware/graphics-cards/schrodingers-gpu-set-for-an-early-2027-release-as-the-dead-alive-24-gb-rtx-5080-super-box-is-shaken-once-more/)
  //   → 넣지 않는다. 중국 전용 5090 D / D v2도 제외.
  // NVIDIA GeForce RTX 50 하위 SKU (출시: 5060 Ti 8GB 2025-04-16 · 5060 2025-05-19 · 5050 2025-07)
  { name: 'RTX 5060 Ti 8GB', series: '50', vramGB: 8, bandwidthGBs: 448, status: 'VERIFIED', verifiedAt: '2026-10-16', matchRequires: ['5060', 'ti'], sources: { vramGB: ['https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5060-family/', 'https://www.pcgamer.com/hardware/graphics-cards/nvidia-rtx-5060-ti-8-gb-review-palit-dual/'], bandwidthGBs: ['https://www.nvidia.com/en-us/geforce/graphics-cards/compare/', 'https://www.pcgamer.com/hardware/graphics-cards/nvidia-rtx-5060-ti-8-gb-review-palit-dual/'] } }, // 128-bit GDDR7 28Gbps — 16GB판과 대역폭 동일
  // 'RTX 5060 8GB'(판매점 표기)는 이 카드다 — 'ti' 없이 '5060 8gb'가 5060 Ti 8GB로 끌려가지 않게 Ti 행은 matchRequires ['ti'].
  { name: 'RTX 5060', series: '50', vramGB: 8, bandwidthGBs: 448, status: 'VERIFIED', verifiedAt: '2026-10-16', aliases: ['RTX 5060 8GB'], sources: { vramGB: ['https://www.nvidia.com/en-us/geforce/graphics-cards/compare/', 'https://www.asus.com/motherboards-components/graphics-cards/prime/prime-rtx5060-o8g/techspec/'], bandwidthGBs: ['https://www.nvidia.com/en-us/geforce/graphics-cards/compare/', 'https://www.guru3d.com/review/geforce-rtx-5060-8gb-review/page-2/'] } }, // 데스크톱 128-bit GDDR7 28Gbps
  { name: 'RTX 5050', series: '50', vramGB: 8, bandwidthGBs: 320, status: 'VERIFIED', verifiedAt: '2026-10-16', sources: { vramGB: ['https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5050/', 'https://www.asus.com/motherboards-components/graphics-cards/prime/prime-rtx5050-o8g/techspec/'], bandwidthGBs: ['https://www.nvidia.com/en-us/geforce/graphics-cards/compare/', 'https://www.techspot.com/review/3011-nvidia-geforce-rtx-5050/'] } }, // 데스크톱은 GDDR6 20Gbps 128-bit(노트북 5050은 GDDR7 — 섞지 말 것)
  // NVIDIA RTX PRO Blackwell 워크스테이션 (출시 확인: 5000 72GB GA 2025-12-18 https://blogs.nvidia.com/blog/rtx-pro-5000-72gb-blackwell-gpu/ ·
  //   5000 48GB/4500/4000/4000 SFF/2000 판매 https://premioinc.com/blogs/press-kits/premio-validates-nvidia-rtx-pro-blackwell-gpu-support-for-accelerated-ai-across-its-edge-computing-solutions ·
  //   실물 리뷰 https://www.pugetsystems.com/labs/articles/2025-professional-gpu-content-creation-roundup/). RTX PRO 5500(84GB)은 NVIDIA "Coming Soon" → 제외.
  // GDDR7 "with ECC"는 사이드밴드/온다이라 용량을 깎지 않는다(https://www.leadtek.com/eng/products/workstation_graphics(2)/nvidia_rtx_pro_6000_blackwell_max-q_workstation_edition(51029)/detail
  //   "without any performance penalties or capacity overhead") → 명목 용량 그대로. (GDDR6 세대 RTX 6000 Ada는 ECC를 켜면 1/16 줄어든다 — 별건)
  // 5000은 48GB/72GB 두 행: 이름에 용량을 넣어 'pro 5000' 질의가 A100처럼 모호로 남게 한다(잘못 고르면 판정이 뒤집힘).
  // 신규 PRO 행은 matchRequires ['pro', 번호]: 'RTX 4500'·'RTX 4000'·'RTX 2000'·'RTX 5000'은 Ada(24/20/16/32GB)·Quadro 카드 이름이고,
  // 'pro 48gb'처럼 번호 없는 질의는 같은 용량의 다른 PRO 구성(예: PRO 4000 두 장)과 구분되지 않는다.
  { name: 'RTX PRO 5000 Blackwell 48GB', series: 'workstation', matchRequires: ['pro', '5000'], vramGB: 48, bandwidthGBs: 1344, status: 'VERIFIED', verifiedAt: '2026-10-16', sources: { vramGB: ['https://www.nvidia.com/content/dam/en-zz/Solutions/products/workstations/professional-desktop-gpus/rtx-pro-5000-blackwell/workstation-datasheet-blackwell-rtx-pro-5000-5488550-nvidia.pdf', 'https://www.leadtek.com/eng/products/workstation_graphics(2)/NVIDIA_RTX_PRO_5000_Blackwell(51030)/detail'], bandwidthGBs: ['https://www.nvidia.com/content/dam/en-zz/Solutions/products/workstations/professional-desktop-gpus/rtx-pro-5000-blackwell/workstation-datasheet-blackwell-rtx-pro-5000-5488550-nvidia.pdf', 'https://www.leadtek.com/eng/products/workstation_graphics(2)/NVIDIA_RTX_PRO_5000_Blackwell(51030)/detail'] } }, // 384-bit GDDR7 ECC 28Gbps. 초기 보도자료의 "45GB"는 확정 전 오기(실측 48935 MiB)
  { name: 'RTX PRO 5000 Blackwell 72GB', series: 'workstation', matchRequires: ['pro', '5000'], vramGB: 72, bandwidthGBs: 1344, status: 'VERIFIED', verifiedAt: '2026-10-16', sources: { vramGB: ['https://www.nvidia.com/content/dam/en-zz/Solutions/products/workstations/professional-desktop-gpus/rtx-pro-5000-blackwell/workstation-datasheet-blackwell-rtx-pro-5000-5488550-nvidia.pdf', 'https://www.pny.com/nvidia-rtx-pro-5000-72gb-blackwell'], bandwidthGBs: ['https://www.nvidia.com/content/dam/en-zz/Solutions/products/workstations/professional-desktop-gpus/rtx-pro-5000-blackwell/workstation-datasheet-blackwell-rtx-pro-5000-5488550-nvidia.pdf', 'https://www.pny.com/nvidia-rtx-pro-5000-72gb-blackwell'] } }, // 48GB와 같은 1344GB/s
  // 4500: Server Edition도 32GB → 별칭(대역폭은 800GB/s로 다름 — https://www.nvidia.com/en-us/data-center/rtx-pro-4500-blackwell-server-edition/ · https://lenovopress.lenovo.com/lp2391-thinksystem-nvidia-rtx-pro-4500-blackwell-server-edition-pcie-gen5-gpu).
  //   ⚠ WE 대역폭 충돌 해소: Lenovo lp2443의 "Up to 800"은 Server 값 오기 — NVIDIA·Leadtek·PNY 모두 896(256-bit × 28Gbps ÷ 8).
  { name: 'RTX PRO 4500 Blackwell', series: 'workstation', matchRequires: ['pro', '4500'], vramGB: 32, bandwidthGBs: 896, status: 'VERIFIED', verifiedAt: '2026-10-16', aliases: ['RTX PRO 4500 Blackwell Server Edition', 'RTX PRO 4500 Server Edition'], sources: { vramGB: ['https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-4500/', 'https://www.leadtek.com/eng/products/workstation_graphics(2)/NVIDIA_RTX_PRO_4500_Blackwell(51031)/detail'], bandwidthGBs: ['https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-4500/', 'https://www.leadtek.com/eng/products/workstation_graphics(2)/NVIDIA_RTX_PRO_4500_Blackwell(51031)/detail'] } },
  // 4000: SFF Edition도 24GB → 별칭(대역폭 432GB/s·70W — https://images.nvidia.com/aem-dam/Solutions/design-visualization/quadro-product-literature/workstation-datasheet-blackwell-rtx-pro-4000-sff-nvidia-us-4016700.pdf · https://www.pny.com/nvidia-rtx-pro-4000-sff-blackwell).
  //   실측 memory.total 24467 MiB(두 에디션) — https://github.com/ggml-org/llama.cpp/issues/30224 · https://computingforgeeks.com/rtx-pro-4000-vs-rtx-pro-5000-blackwell-local-ai-benchmarks/
  { name: 'RTX PRO 4000 Blackwell', series: 'workstation', matchRequires: ['pro', '4000'], vramGB: 24, bandwidthGBs: 672, status: 'VERIFIED', verifiedAt: '2026-10-16', aliases: ['RTX PRO 4000 Blackwell SFF Edition', 'RTX PRO 4000 Blackwell SFF', 'RTX PRO 4000 SFF'], sources: { vramGB: ['https://www.nvidia.com/content/dam/en-zz/Solutions/products/workstations/professional-desktop-gpus/rtx-pro-4000/workstation-datasheet-rtx-pro-4000-nvidia-us-web.pdf', 'https://www.leadtek.com/eng/products/workstation_graphics(2)/nvidia_rtx_pro_4000_blackwell(51032)/detail'], bandwidthGBs: ['https://www.nvidia.com/content/dam/en-zz/Solutions/products/workstations/professional-desktop-gpus/rtx-pro-4000/workstation-datasheet-rtx-pro-4000-nvidia-us-web.pdf', 'https://www.leadtek.com/eng/products/workstation_graphics(2)/nvidia_rtx_pro_4000_blackwell(51032)/detail'] } }, // 192-bit GDDR7 ECC 28Gbps
  { name: 'RTX PRO 2000 Blackwell', series: 'workstation', matchRequires: ['pro', '2000'], vramGB: 16, bandwidthGBs: 288, status: 'VERIFIED', verifiedAt: '2026-10-16', sources: { vramGB: ['https://images.nvidia.com/aem-dam/Solutions/design-visualization/quadro-product-literature/workstation-datasheet-blackwell-rtx-pro-2000-nvidia-us-4016661.pdf', 'https://www.pny.com/nvidia-rtx-pro-2000-blackwell'], bandwidthGBs: ['https://images.nvidia.com/aem-dam/Solutions/design-visualization/quadro-product-literature/workstation-datasheet-blackwell-rtx-pro-2000-nvidia-us-4016661.pdf', 'https://www.leadtek.com/eng/products/workstation_graphics(2)/nvidia_rtx_pro_2000_blackwell(51042)/detail'] } }, // 128-bit GDDR7 ECC 18Gbps · 70W
  // NVIDIA GB10 통합 메모리 시스템 — 디스크리트 카드가 아니다(unifiedMemory → gpuDevice가 env 프리셋 대신 이 profile로 reserve를 만든다).
  // 128GB LPDDR5x 256-bit 8533MT/s 273GB/s · 출시 2025-10-15 (https://nvidianews.nvidia.com/news/nvidia-dgx-spark-arrives-for-worlds-ai-developers)
  // ⚠ 이름에 용량을 붙인다: 64GB 구성이 2026-10-02 발표, 2026-10-23 OEM 판매 예정(https://blogs.nvidia.com/blog/local-ai-dgx-spark-64gb-sync/ ·
  //   https://www.tomshardware.com/pc-components/gpus/nvidia-introduces-64gb-dgx-spark-to-throw-local-ai-fans-a-lifeline-amid-the-rampocalypse-new-gb10-config-starts-at-usd4999-for-those-who-can-work-with-less)
  //   — 아직 판매 전이고 64GB 기기 실측이 없어 넣지 않는다. 64GB 행이 들어오면 'dgx spark'·'gx10' 질의는 A100처럼 모호가 된다(의도).
  // 별칭 = 같은 GB10·같은 128GB LPDDR5x·같은 273GB/s로 출시된 파트너 완제품(두 트랙 모두 제조사 사양 + 독립 리뷰로 확인):
  //   ASUS Ascent GX10   https://www.asus.com/networking-iot-servers/desktop-ai-supercomputer/ultra-small-ai-supercomputers/asus-ascent-gx10/techspec/ · https://press.asus.com/news/press-releases/asus-ascent-gx10-ai-supercomputer/
  //   Dell Pro Max GB10  https://www.dell.com/en-us/shop/desktop-computers/dell-pro-max-with-gb10/spd/dell-pro-max-fcm1253-micro/xcto_fcm1253_usx · https://www.storagereview.com/review/dell-pro-max-with-gb10-review
  //   HP ZGX Nano G1n    https://www.hp.com/us-en/workstations/ai-stations.html · https://www.storagereview.com/review/hp-zgx-nano-g1n-ai-station-review-a-secure-sustainable-desk-side-ai-node
  //   Lenovo PGX         https://lenovopress.lenovo.com/lp2321-thinkstation-pgx · https://www.servethehome.com/lenovo-thinkstation-pgx-review-the-nvidia-gb10-128gb-ai-workstation-goes-corporate/
  //   MSI EdgeXpert      https://download-2.msi.com/archive/mnu_exe/ipc/EdgeXpert_DM_dm.pdf · https://linuxgizmos.com/msi-edgexpert-compact-ai-supercomputer-based-on-nvidia-dgx-spark/
  //   Acer Veriton GN100 https://news.acer.com/acer-unveils-the-veriton-gn100-ai-mini-workstation-built-on-the-nvidia-gb10-superchip · https://www.storagereview.com/review/acer-veriton-gn100-review-a-standout-in-the-nvidia-spark-ecosystem
  //   GIGABYTE AI TOP ATOM https://www.gigabyte.com/FileUpload/Global/KeyFeature/4102/innergigabyte/GIGABYTE_AI_TOP_ATOM_Data_sheet.pdf · https://www.storagereview.com/review/gigabyte-ai-top-atom-review
  //   (PNY 'NVIDIA DGX Spark'는 Founders Edition 유통 — 같은 기기)
  {
    name: 'DGX Spark 128GB', series: 'unified', vramGB: 128, bandwidthGBs: 273, status: 'VERIFIED', verifiedAt: '2026-10-16',
    matchRequires: ['spark'], // 'dgx'만으로는 확정하지 않는다(DGX Station·DGX B200 등 다른 DGX가 있다)
    // 별칭은 전부 용량(128GB)을 품는다 — 이름 전체 일치로만 해석되므로, 64GB 기기가 'ASUS Ascent GX10'처럼 용량 없이 적혀도
    // 128GB로 끌려오지 않는다(64GB 구성 판매 2026-10-23). 용량 없는 'DGX Spark'·'spark'는 정본 이름 토큰 규칙으로 해석되고,
    // 64GB 행이 들어오는 순간 A100처럼 자동으로 모호가 된다.
    aliases: [
      'NVIDIA DGX Spark 128GB', 'GB10 128GB', 'NVIDIA GB10 128GB',
      'ASUS Ascent GX10 128GB', 'Ascent GX10 128GB',
      'Dell Pro Max with GB10 128GB', 'Dell Pro Max GB10 128GB',
      'HP ZGX Nano G1n 128GB', 'HP ZGX Nano 128GB',
      'Lenovo ThinkStation PGX 128GB', 'ThinkStation PGX 128GB',
      'MSI EdgeXpert 128GB', 'MSI EdgeXpert MS-C931 128GB',
      'Acer Veriton GN100 128GB',
      'GIGABYTE AI TOP ATOM 128GB',
    ],
    sources: {
      vramGB: ['https://www.nvidia.com/en-us/products/workstations/dgx-spark/', 'https://www.storagereview.com/review/nvidia-dgx-spark-review-the-ai-appliance-bringing-datacenter-capabilities-to-desktops'],
      bandwidthGBs: ['https://docs.nvidia.com/dgx/dgx-spark/hardware.html', 'https://www.storagereview.com/review/nvidia-dgx-spark-review-the-ai-appliance-bringing-datacenter-capabilities-to-desktops'],
    },
    unifiedMemory: GB10_128GB_UNIFIED, // reserve 구성·출처는 위 상수(2×·4× 클러스터 행과 공유)
  },
  // GB10 클러스터(2026-10-16 사용자 승인) — 한 모델이 여러 대의 메모리를 같이 쓰는 경우만. 숫자는 DGX Spark 128GB 행 × 대수이고,
  // 합산이 성립하는 조건은 전제 'gb10-cluster-tensor-parallel'(아래 STRUCTURAL_ASSUMPTIONS)로 판정과 함께 나간다.
  // 연결(NVIDIA 공식): ConnectX-7 QSFP 2포트, 포트당 200Gb/s. 2대 = QSFP 케이블 직결, 3대 = 링, 4대 = 스위치 필수,
  //   4대 지원은 2026-03-16부터(DGX OS 2026년 4월 이후). 4대 초과는 NVIDIA 미검증 → 행 없음(조합도 거부).
  //   https://docs.nvidia.com/dgx/dgx-spark/spark-clustering.html ("up to three DGX Spark systems connected directly through cables, and up to four systems when using a switch")
  //   https://developer.nvidia.com/blog/scaling-autonomous-ai-agents-and-workloads-with-nvidia-dgx-spark/ (2026-03-16, "increased to up to four DGX Spark nodes")
  //   https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/playbook-connect-multiple-sparks/README.md · https://www.servethehome.com/big-cluster-little-power-the-8x-nvidia-gb10-cluster-marvell-cisco-ubiquiti-qnap-arm/
  // 3대(링)는 행을 두지 않는다: 텐서 병렬 3은 대부분 모델의 헤드 수로 나눠지지 않아 파이프라인(층) 분할이 되고, 층 분할의 대당 몫은
  //   모델마다 고르지 않다 — 그 계산은 아직 없다(fail closed).
  // NVIDIA 판매 문구 "2대 최대 405B(현재 제품 페이지 400B)·4대 700B"는 가중치만의 파라미터 수다 — 2026-03 데이터시트 각주 "Using FP4 precision models."
  //   (https://webobjects2.cdw.com/is/content/CDW/cdw/on-domain-cdw/brands/nvidia/workstation-datasheet-dgx-spark-update-partner-us-4699700.pdf). NVIDIA 자신의 405B 예시는
  //   INT4 + 컨텍스트 64토큰이고 "insufficient memory headroom for production use"라고 적는다
  //   (https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/vllm/README.md) — 이 행들은 그 문구를 재현하지 않고 실제 컨텍스트로 계산한다.
  {
    name: '2× DGX Spark 128GB', series: 'unified', vramGB: 256, bandwidthGBs: 273, count: 2, perCardVramGB: 128, clusterOf: 'DGX Spark 128GB',
    status: 'VERIFIED', verifiedAt: '2026-10-16', matchRequires: ['spark'], aliases: ['2x DGX Spark 128GB', 'DGX Spark 128GB x2'],
    sources: {
      vramGB: ['https://www.nvidia.com/en-us/products/workstations/dgx-spark/', 'https://www.storagereview.com/review/nvidia-dgx-spark-cluster-review-distributed-inference-on-dell-gigabyte-and-hp'],
      bandwidthGBs: ['https://docs.nvidia.com/dgx/dgx-spark/hardware.html', 'https://www.storagereview.com/review/nvidia-dgx-spark-review-the-ai-appliance-bringing-datacenter-capabilities-to-desktops'],
      interconnect: ['https://docs.nvidia.com/dgx/dgx-spark/spark-clustering.html', 'https://build.nvidia.com/spark/connect-two-sparks', 'https://www.lmsys.org/blog/2025-10-13-nvidia-dgx-spark/'],
    },
    unifiedMemory: GB10_128GB_UNIFIED, // 대당 reserve는 같고 대수만큼 생긴다(각 대가 자기 DGX OS·런타임을 돌린다)
  },
  {
    name: '4× DGX Spark 128GB', series: 'unified', vramGB: 512, bandwidthGBs: 273, count: 4, perCardVramGB: 128, clusterOf: 'DGX Spark 128GB',
    status: 'VERIFIED', verifiedAt: '2026-10-16', matchRequires: ['spark'], aliases: ['4x DGX Spark 128GB', 'DGX Spark 128GB x4'],
    sources: {
      vramGB: ['https://www.nvidia.com/en-us/products/workstations/dgx-spark/', 'https://www.servethehome.com/big-cluster-little-power-the-8x-nvidia-gb10-cluster-marvell-cisco-ubiquiti-qnap-arm/'],
      bandwidthGBs: ['https://docs.nvidia.com/dgx/dgx-spark/hardware.html', 'https://www.storagereview.com/review/nvidia-dgx-spark-review-the-ai-appliance-bringing-datacenter-capabilities-to-desktops'],
      interconnect: ['https://docs.nvidia.com/dgx/dgx-spark/spark-clustering.html', 'https://developer.nvidia.com/blog/scaling-autonomous-ai-agents-and-workloads-with-nvidia-dgx-spark/', 'https://www.servethehome.com/big-cluster-little-power-the-8x-nvidia-gb10-cluster-marvell-cisco-ubiquiti-qnap-arm/'],
    },
    unifiedMemory: GB10_128GB_UNIFIED,
  },
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
  if (gpu && gpu.unifiedMemory) return unifiedGpuDevice(gpu); // GB10: env 프리셋(디스크리트 카드용)은 적용하지 않는다
  const env = ENV_PRESETS[envKey] || ENV_PRESETS[DEFAULT_ENV];
  const count = gpu.count || 1; // 멀티GPU: reserve는 카드당 발생(OS/CUDA 컨텍스트 ×count). vramGB는 이미 합산값.
  return { type: 'gpu', gpu, env: env.key, memoryGB: gpu.vramGB, bandwidthGBs: gpu.bandwidthGBs, reserveGB: env.reserveGB * count, headroomRatio: GPU_HEADROOM_RATIO, _os: 0, gpuCount: count };
}

// 통합 메모리 시스템(GB10) 디바이스 — 타입은 'gpu'(CUDA·GGUF 티어 경로 그대로)지만 reserve·headroom은 공유 풀 기준.
//   reserve  = (총 메모리 − OS가 보는 총량) + OS 유휴 사용량 + 런타임 고정분  — 각 항목 근거는 GPUS 행의 unifiedMemory
//   headroom = 맥과 같은 HEADROOM_RATIO(20%) — 같은 풀을 앱·페이지 캐시·스파이크가 나눠 쓰므로 전용 VRAM의 5%가 아니다.
// 고정 런타임 2GB는 reserve에 들어 있으므로 calcRuntimeOverhead는 GPU 규칙대로 fixed=0(이중 차감 없음).
// 클러스터 행(count>1)은 대당 reserve × 대수 — 각 대가 자기 DGX OS·런타임을 돌린다.
export function unifiedReserveGB(gpu) {
  const u = gpu.unifiedMemory;
  const units = gpu.count || 1;
  return ((gpu.perCardVramGB ?? gpu.vramGB) - u.osVisibleGB + u.osIdleGB + u.runtimeFixedGB) * units;
}
function unifiedGpuDevice(gpu) {
  const units = gpu.count || 1;
  return {
    type: 'gpu', gpu, env: gpu.unifiedMemory.environment, memoryGB: gpu.vramGB, bandwidthGBs: gpu.bandwidthGBs,
    reserveGB: unifiedReserveGB(gpu), headroomRatio: HEADROOM_RATIO, _os: 0, gpuCount: units, unifiedMemory: true,
    ...(units > 1 ? { clusterUnits: units } : {}),
  };
}
const UNIFIED_COMBINE_ERROR = (name) => `${name} is a complete unified-memory system — pooling it with other GPUs, or linking a number of units other than 2 or 4 identical ones, is not modeled`;
// 같은 GB10 행 N대(2·4) → 클러스터 행. 그 밖(다른 카드와 섞기·3대·5대 이상·클러스터 행끼리 묶기)은 null.
function unifiedCluster(gpus) {
  const base = gpus[0];
  if (!base || !base.unifiedMemory || base.count) return null;
  if (!gpus.every((g) => g && g.name === base.name && !g.count)) return null;
  return GPUS.find((g) => g.clusterOf === base.name && g.count === gpus.length) || null;
}

// 클러스터에서 KV가 몇 배로 상주하나 — NVIDIA 플레이북 경로(vLLM+Ray·TensorRT-LLM, tensor parallel = 대수) 기준.
// vLLM: "Number of KV heads is less than TP size, so we replicate the KV heads"(llama.py) · MLA 잠재 KV는 TP 순위마다 통째로 복제
// ("replicated in full on every TP rank", https://vllm.ai/blog/2026-08-07-decode-context-parallelism ·
//  https://nvidia.github.io/TensorRT-LLM/features/parallel-strategy.html). 층(파이프라인) 분할이면 복제가 없으니 보수 쪽 근사다.
// 순위당 KV 헤드 = max(1, 헤드 ÷ 대수) → 전체 = max(헤드, 대수). 층 종류별(슬라이딩 로컬·전역)로 따로 적용한다
// (Codex 리뷰: Gemma 4 12b 로컬 8헤드는 나뉘고 전역 1헤드만 복제된다 — 한 배율로 묶으면 4대에서 KV 1.75 vs 실제 0.81 GiB).
// 반환값은 KV 계산(calcKVCache·calcMaxContext)에만 쓰는 모델 사본이다 — 가중치·이름은 그대로.
export function clusterKvModel(model, device) {
  const units = device?.clusterUnits || 1;
  if (units <= 1) return model;
  // DeepSeek-V4 계열: 단일 KV 헤드(MQA)의 창·압축 캐시·인덱서 K를 순위마다 통째로 둔다(참조 코드 wkv·Compressor·Indexer.wk가 비분할 Linear).
  if (model.compressedKv) {
    return { ...model, compressedKv: { ...model.compressedKv, replicas: (model.compressedKv.replicas || 1) * units } };
  }
  if (model.mlaKvLoraRank) {
    return { ...model, mlaKvLoraRank: model.mlaKvLoraRank * units, mlaRopeDim: (model.mlaRopeDim || 0) * units };
  }
  return {
    ...model,
    kvHeads: Math.max(model.kvHeads, units),
    ...(model.globalKvHeads ? { globalKvHeads: Math.max(model.globalKvHeads, units) } : {}),
  };
}
// tensor parallel = 대수가 성립하나 — vLLM/TensorRT-LLM이 거부하는 구성(assert)을 숫자로 내지 않는다(Codex 리뷰: Qwen2.5-0.5B
// 14헤드 × 4대가 'yes'로 나왔다). 어텐션 헤드는 대수로 나눠져야 하고, KV 헤드는 대수로 나눠지거나 대수를 나눠야 하며,
// 선형 어텐션 헤드도 나눠져야 한다. 헤드 수를 모르면 성립한다고 하지 않는다(fail closed).
// https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/llama.py (total_num_heads % tp_size, tp_size % total_num_kv_heads)
export function clusterTensorParallelOk(model, device) {
  const units = device?.clusterUnits || 1;
  if (units <= 1) return true;
  const divides = (n) => Number.isInteger(n) && n > 0 && n % units === 0;
  const kvOk = (kv) => Number.isInteger(kv) && kv > 0 && (kv >= units ? kv % units === 0 : units % kv === 0);
  if (!divides(model.attnHeads)) return false;
  if (!model.mlaKvLoraRank) {
    if (!kvOk(model.kvHeads)) return false;
    if (model.globalKvHeads && !kvOk(model.globalKvHeads)) return false;
  }
  const la = model.linearAttn;
  if (la && la.layers && !(divides(la.numKHeads) && divides(la.numVHeads))) return false;
  // DeepSeek-V4 계열 참조 코드: 출력 그룹(n_groups // world_size)과 인덱서 헤드(index_n_heads // world_size)도 순위로 나눈다.
  // 전문가 수(DSpark 초안 블록 포함)와 어휘 수도 assert로 나눈다(MoE: n_routed_experts % world_size, ParallelEmbedding: vocab_size % world_size).
  const ck = model.compressedKv;
  if (ck && !(divides(ck.oGroups) && divides(ck.indexHeads) && divides(ck.vocabSize) &&
      Array.isArray(ck.routedExperts) && ck.routedExperts.length > 0 && ck.routedExperts.every(divides))) return false;
  return true;
}
const clusterUnitsOf = (device) => device?.clusterUnits || 1;

// 임의 멀티GPU 조합 — 이종 포함 (예: RTX 5090 + RTX 3090 = 56GB 풀). llama.cpp 레이어 분할 기준의 풀링 근사:
// VRAM=합산 · reserve=카드당 env reserve 합(드라이버/CUDA 컨텍스트는 카드마다 발생) · 대역폭=최소 카드(디코드는 느린 구간 바운드 — dormant).
// 한계(Methodology 고지와 동일): 단일 레이어는 한 카드에 들어가야 하며, 분할 오버헤드·PCIe 통신은 미반영(낙관적 근사).
export function combineGpus(gpus, envKey = DEFAULT_ENV) {
  if (!Array.isArray(gpus) || gpus.length === 0) return null;
  if (gpus.length === 1) return gpuDevice(gpus[0], envKey);
  // 통합 메모리 시스템은 카드가 아니라 완제품 컴퓨터다 — 다른 카드와의 풀링이나 여러 대 연결(네트워크 분산 추론)은
  // 레이어 분할 근사로 모델링하지 않는다. 숫자를 지어내지 않고 실패한다(fail closed).
  const unified = gpus.find((g) => g && g.unifiedMemory);
  if (unified) {
    const cluster = unifiedCluster(gpus);
    if (!cluster) throw new Error(UNIFIED_COMBINE_ERROR(unified.name));
    return gpuDevice(cluster);
  }
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

// DeepSeek-V4 계열 압축·희소 어텐션(V4: CSA+HCA, V4.1: CSA2) — 공식 추론 코드가 실제로 잡는 시퀀스 버퍼만 센다.
// 1차 출처(pinned):
//   V4   https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/model.py
//        Attention.kv_cache = (window_size + max_seq_len // compress_ratio, head_dim) · Indexer.kv_cache(ratio 4만) =
//        (max_seq_len // 4, index_head_dim) · Compressor kv_state/score_state = fp32 (coff·ratio, coff·head_dim), coff = 2 iff ratio 4
//   V4.1 https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/1bae0e1088993b9b82de08f59de5be5a34e42170/inference/model.py
//        모든 레이어 window_kv_cache (window_size, head_dim) · kv_source_layers만 compress_kv_cache (max_seq_len // ratio, head_dim)
//        와 Indexer.k_cache (max_seq_len // ratio, index_head_dim) — 나머지 레이어는 앞 소스의 캐시를 그대로 읽는다(Reindex/Reuse).
//        engram.py NgramHashState.cache = (max_seq_len,) int64 — 토큰 기록, KV 정밀도와 무관한 토큰당 8B.
//   교차 확인: V4.1 기술 보고서·모델카드의 "global KV 890 bytes per token" = 소스 4층의 주 KV 1280원소 × FP4(E2M1 + 16채널당
//   E4M3 스케일 = 0.5625B) 720 + 인덱서 K 320원소 × MXFP4(0.53125B) 170 — 아래 원소 수와 정확히 같다.
// 참조 코드의 버퍼는 bf16이라 kvBits 16이 그 값 그대로다. 압축 캐시는 ctx/ratio로 센다(floor보다 레이어당 최대 1항목 많다).
// replicas: GB10 클러스터에서 텐서 병렬 순위마다 통째로 복제되는 몫(참조 코드의 wkv·Compressor·Indexer.wk는 분할하지 않는 Linear).
// workspaceBytes: 인덱서 프리필 점수 버퍼. 참조 generate.py는 프롬프트 전체를 한 번에 넣어 (프롬프트 × 인덱서 헤드 × ctx/ratio)
//   텐서를 만든다(131K에서 순위당 128 GiB) — 그 경로는 모델링하지 않는다. vLLM은 질의를 잘라 이 버퍼를 512 MiB로 묶고
//   메모리 프로파일에서 같은 크기를 미리 잡는다(순위마다):
//   https://github.com/vllm-project/vllm/blob/2067297b2f15f0776be6d2a7e1b435bd20af60c1/vllm/envs.py (VLLM_SPARSE_INDEXER_MAX_LOGITS_MB = 512)
//   https://github.com/vllm-project/vllm/blob/2067297b2f15f0776be6d2a7e1b435bd20af60c1/vllm/v1/attention/backends/mla/indexer.py (max_logits_bytes 분할)
//   https://github.com/vllm-project/vllm/blob/2067297b2f15f0776be6d2a7e1b435bd20af60c1/vllm/model_executor/layers/sparse_attn_indexer.py (프로파일 더미 할당)
function compressedKvBytes(ck, bpe) {
  const reps = ck.replicas || 1;
  const perTokenElems = ck.caches.reduce((s, x) => s + (x.layers * x.dim) / x.ratio, 0);
  const perToken = (perTokenElems * bpe + (ck.tokenStateBytes || 0)) * reps;
  const fixed = (ck.windowSize * ck.headDim * ck.windowLayers * bpe + (ck.stateBytes || 0) + (ck.workspaceBytes || 0)) * reps;
  return { perToken, fixed };
}

export function calcKVCache(model, ctx, bits) {
  ctx = Math.max(0, Math.floor(Number(ctx)) || 0); // 음수/NaN 가드
  if (!model.kvHeads || !model.kvHeadDim || !model.layerCount) {
    return { totalGB: 0, perTokenKB: 0, kvPerToken: 0, totalBytes: 0, effectiveCtx: 0 };
  }
  const bpe = bits / 8;

  // DeepSeek-V4 계열: 슬라이딩 창(고정) + 압축 캐시(ctx/ratio) + 인덱서 K. 균일 KV 공식이 성립하지 않아 MLA보다 먼저 탄다.
  if (model.compressedKv) {
    const { perToken, fixed } = compressedKvBytes(model.compressedKv, bpe);
    const totalBytes = perToken * ctx + fixed;
    return {
      totalGB: totalBytes / 1024 ** 3,
      perTokenKB: perToken / 1024,
      kvPerToken: perToken,
      totalBytes,
      effectiveCtx: ctx,
    };
  }

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

// 하이브리드 선형 어텐션(Gated DeltaNet 등)의 *고정* 순환 상태.
// KV 캐시와 근본적으로 다르다 — KV는 ctx에 비례하지만 이 상태는 시퀀스당 상수다.
// 그래서 KV에 합치지 않고 별도 항목으로 둔다(기존 KV 기준값·컨포먼스 벡터 불변).
// 형태 출처(1차): huggingface/transformers src/transformers/models/qwen3_next/modeling_qwen3_next.py
//   key_dim   = linear_num_key_heads   × linear_key_head_dim
//   value_dim = linear_num_value_heads × linear_value_head_dim
//   conv_dim  = key_dim * 2 + value_dim                              (L529 self.conv_dim)
//   conv_states      : (batch, conv_dim, linear_conv_kernel_dim)     — 활성화 dtype(bf16 2B)
//   recurrent_states : (batch, num_v_heads, head_k_dim, head_v_dim)  (L428/L485 torch.zeros)
//                      config mamba_ssm_dtype=float32 → 4B
// 가중치·KV 양자화와 무관한 런타임 상태 텐서라 고정 비용으로 계상한다(batch=1 로컬 추론 기준).
export function calcLinearState(model) {
  const la = model && model.linearAttn;
  if (!la || !la.layers) return { totalBytes: 0, totalGB: 0, perLayerBytes: 0, convBytes: 0, recurrentBytes: 0 };
  const keyDim = la.numKHeads * la.headKDim;
  const valueDim = la.numVHeads * la.headVDim;
  const convDim = keyDim * 2 + valueDim;
  const convBytes = convDim * la.convKernel * (la.convDtypeBytes || 2);
  const recurrentBytes = la.numVHeads * la.headKDim * la.headVDim * (la.stateDtypeBytes || 4);
  const perLayerBytes = convBytes + recurrentBytes;
  const totalBytes = perLayerBytes * la.layers;
  return { totalBytes, totalGB: totalBytes / 1024 ** 3, perLayerBytes, convBytes, recurrentBytes };
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

// Exact Gemma text-family identity:
// https://huggingface.co/google/gemma-4-E2B-it/blob/main/config.json
// Pinned llama.cpp/GGUF placement evidence (primary): per_layer_token_embd is classified as an
// input-layer tensor, and the input layer is always assigned to the CPU device / host buffer list
// regardless of -ngl, so it is never allocated in accelerator memory. The deduction rests on that
// placement alone. The loader's lazy read (TENSOR_READ_LAZY; default auto mode engages it only
// above 4 GiB) also resolves to the CPU buffer type but is a mode-dependent detail, not the
// justification — on-demand file reads are never counted as capacity, and unconditional host-RAM
// residency is not claimed. The host memory this tensor needs is not budgeted by the GPU verdict;
// a runtime that loads PLE onto the accelerator invalidates the deduction.
// https://github.com/ggml-org/llama.cpp/blob/8b4b3558f1459c13e4aa38d5c94d306a00dc6acd/src/llama-model.cpp
// https://github.com/ggml-org/llama.cpp/blob/8b4b3558f1459c13e4aa38d5c94d306a00dc6acd/src/llama-arch.cpp
// https://github.com/ggml-org/llama.cpp/blob/8b4b3558f1459c13e4aa38d5c94d306a00dc6acd/src/models/gemma4.cpp
// https://github.com/ggml-org/llama.cpp/blob/8b4b3558f1459c13e4aa38d5c94d306a00dc6acd/src/llama-model-loader.h
// https://github.com/ggml-org/llama.cpp/blob/8b4b3558f1459c13e4aa38d5c94d306a00dc6acd/src/llama-model-loader.cpp
const PLE_OFFLOAD_FAMILIES = new Set(['gemma4_text']);
// FitLLM이 GGUF per_layer_token_embd 바이트로 재현해 검증한 PLE 텐서 형상 — pinned 공식 text_config 두 개가 전부다.
// 계열 허용만으로는 붙여넣은 config가 hidden_size_per_layer_input을 부풀려 GPU 가중치를 최대 99% 차감하고도
// "verified"를 받았다(2026-09-05 감사 §5.4). PLE 치수 두 개만 대조해도 곱의 세 번째 인자 num_hidden_layers가
// 열려 있어 공식 E2B 체크포인트에 45층을 선언하면 잔여 하한을 통과하고 GPU 가중치 59%를 차감했다(2026-09-05
// 독립 리뷰 P1). 그래서 검증 단위는 스칼라가 아니라 공식 config에 결합된 완전 profile — 층 수·hidden_size·
// intermediate_size·PLE 두 치수 — 이고, 어느 profile과도 완전히 일치하지 않는 조합은 검증되지 않은 것이므로 fail-closed.
// https://huggingface.co/google/gemma-4-E2B-it/blob/3e22461f65e89153144f8adb70e3b8c2cc9845a7/config.json
//   (sha256 1b28f3d2c3100f6c594754b81107428bd7b822a7f48272ca681dae9d2ec38330 · safetensors BF16 5,123,178,051)
// https://huggingface.co/google/gemma-4-E4B-it/blob/ee0ef6023621cff504d758262d4e04895a5af4a2/config.json
//   (sha256 33b10c02df3c2e8536cf323d29d53262aaa2f4d11dbe19bc729373fbe90295d4 · safetensors BF16 7,996,156,490)
// 텐서 형상 [vocab_size_per_layer_input, num_hidden_layers × hidden_size_per_layer_input]:
// https://github.com/huggingface/transformers/blob/4177486a9f199bd7be520eff14431071d5d41ec5/src/transformers/models/gemma4/configuration_gemma4.py
const PLE_VERIFIED_PROFILES = Object.freeze({
  gemma4_text: Object.freeze([
    // E2B: 35L × 1536 × 6144 · PLE 262144 × (35 × 256) = 2,348,810,240
    Object.freeze({ layerCount: 35, hiddenSize: 1536, intermediateSize: 6144, vocabSizePerLayerInput: 262144, hiddenSizePerLayerInput: 256 }),
    // E4B: 42L × 2560 × 10240 · PLE 262144 × (42 × 256) = 2,818,572,288
    Object.freeze({ layerCount: 42, hiddenSize: 2560, intermediateSize: 10240, vocabSizePerLayerInput: 262144, hiddenSizePerLayerInput: 256 }),
  ]),
});

// MLA cache structure:
// https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py
// https://github.com/ggml-org/llama.cpp/blob/master/src/llama-kv-cache.cpp
// MTP trunk/draft separation:
// https://github.com/ggml-org/llama.cpp/blob/master/src/llama-hparams.cpp
// https://github.com/ggml-org/llama.cpp/blob/master/src/llama-model.cpp
const STRUCTURAL_ASSUMPTIONS = Object.freeze({
  mla: Object.freeze({
    id: 'mla-compressed-latent-cache',
    statement: 'KV memory assumes a compressed-latent MLA artifact or mode; legacy non-MLA GGUF or an explicitly uncompressed mode invalidates this estimate.',
  }),
  ple: Object.freeze({
    id: 'ple-llamacpp-non-gpu-residency',
    statement: 'GPU weight memory excludes the verified Gemma 4 PLE tensors only because the pinned llama.cpp/GGUF path assigns the per_layer_token_embd input-layer tensor to CPU/host buffers instead of accelerator memory; that host memory is not budgeted here, and a runtime that loads PLE onto the accelerator invalidates this estimate.',
  }),
  compressedKv: Object.freeze({
    id: 'deepseek-compressed-sparse-kv',
    statement: "KV memory follows the cache buffers of DeepSeek's reference inference code: every layer keeps a sliding-window cache, and only the layers that compress their own KV keep a compressed cache (one entry per compress_ratio tokens) plus indexer keys; layers that reuse an earlier layer's cache add nothing. Elements are counted at the chosen KV precision (16-bit matches the reference code's bf16 buffers; FP8/FP4 block scales are not counted separately), plus the compressors' fp32 state, V4.1's Engram token history and a 512 MiB indexer prefill workspace (vLLM's default bound, VLLM_SPARSE_INDEXER_MAX_LOGITS_MB); on linked GB10 units all of it is counted on every unit. Whole-prompt prefill as in DeepSeek's reference generate.py needs far more workspace and is not modeled, nor is a runtime that caches uncompressed KV. V4.1's Engram tables are counted at no less than their FP8 storage (8.25 bits with scales) whatever the weight quantization; the vision tower and MTP/DSpark draft weights count as resident weights.",
  }),
  nativeFp4: Object.freeze({
    id: 'native-fp4-expert-weights',
    statement: 'This release ships its expert weights in FP4 (4-bit E2M1 values plus one E8M0 scale per 32, 4.25 bits — the MXFP4 layout). At any weight precision above 4.25 bits they are counted at that shipped size, because llama.cpp GGUF builds (native MXFP4 type) and MLX MXFP4 builds keep them as released; only the other weights follow the chosen quantization. The weights are never counted below the largest public GGUF file of the chosen tier where those are known (gpt-oss), and otherwise never below the released checkpoint itself. Builds that re-expand the experts to 8 or 16 bits are larger without adding precision and are not what this counts; below 4.25 bits the experts are counted at the chosen precision.',
  }),
  mtp: Object.freeze({
    id: 'mtp-ordinary-generation',
    statement: 'KV memory assumes ordinary non-speculative generation; an MTP draft context is not included.',
  }),
  clusterSplit: Object.freeze({
    id: 'gb10-cluster-tensor-parallel-incompatible',
    statement: "This model's attention or KV head counts do not divide across the units, so vLLM and TensorRT-LLM refuse a tensor-parallel split of this size; no cluster fit is claimed (a layer split, not modeled here, might still run it).",
  }),
  cluster: Object.freeze({
    id: 'gb10-cluster-tensor-parallel',
    statement: 'Several GB10 units add up for one model only under a runtime that splits it across them over ConnectX-7 — vLLM with Ray or TensorRT-LLM, tensor parallel equal to the unit count (NVIDIA supports two units by a direct QSFP cable and four through a RoCE switch, on DGX OS from April 2026). Ollama and LM Studio use one unit. Each unit keeps its own DGX OS and runtime reserve and its own activation buffers, and the KV cache is counted as replicated when the model has fewer KV heads than units or uses an MLA latent cache; a layer (pipeline) split would need less. Network and NCCL buffers and uneven splits are not budgeted.',
  }),
  unified: Object.freeze({
    id: 'gb10-unified-memory-reserve',
    statement: 'GB10 systems share one 128 GB LPDDR5x pool between DGX OS, CPU processes and the GPU. The reserve subtracts the 6.3 GiB the OS cannot see on current DGX OS (121.7 GiB visible; early firmware shows about 2 GiB less), about 5 GiB of idle DGX OS use and a 2 GiB runtime allowance. On one unit the GGUF weights count at their file size with no extra percentage, plus 15% of the KV cache and an activation allowance that grows with context. That is calibrated on public llama.cpp and Ollama allocation logs on GB10 with flash attention on — the default in both (--flash-attn auto) — whose compute buffers were 0.5–1.7 GiB on top of the weights and KV cache (fixtures/measured.json). With flash attention off, llama.cpp holds the full attention scores (context × batch × heads × 4 bytes, about 24 GiB for 48 heads at 262K tokens and batch 512), and vLLM adds CUDA-graph memory and pre-allocates its KV pool; neither is budgeted. Other running apps, the file page cache and llama-server\'s host-side prompt cache (--cache-ram, 8 GiB by default) and context checkpoints are not budgeted, and weights are assumed resident once: a loader that keeps an mmap page-cache copy can need up to the weight size again while loading.',
  }),
});

function guardedPleParamsB(model) {
  return model.pleOffloadVerified === true && model.pleParams && model.pleParams < model.totalParams
    ? model.pleParams
    : 0;
}

// PLE 차감은 "CPU/host 버퍼는 accelerator 메모리가 아니다"가 성립하는 디스크리트 GPU에서만 쓴다.
// 통합 메모리(GB10)에서는 host 버퍼도 같은 풀이라 차감하면 그만큼 거짓 fits가 된다.
function pleDeductionApplies(device) {
  return Boolean(device && device.type === 'gpu' && !device.unifiedMemory);
}

export function structuralAssumptions(model, device) {
  const result = [];
  if (model.mlaKvLoraRank) result.push(STRUCTURAL_ASSUMPTIONS.mla);
  if (model.compressedKv) result.push(STRUCTURAL_ASSUMPTIONS.compressedKv);
  if (pleDeductionApplies(device) && guardedPleParamsB(model)) result.push(STRUCTURAL_ASSUMPTIONS.ple);
  if (nativeFp4ParamsB(model)) result.push(STRUCTURAL_ASSUMPTIONS.nativeFp4);
  if (model.mtpLayerCount > 0) result.push(STRUCTURAL_ASSUMPTIONS.mtp);
  if (device?.unifiedMemory) result.push(STRUCTURAL_ASSUMPTIONS.unified);
  if (device?.clusterUnits > 1) result.push(STRUCTURAL_ASSUMPTIONS.cluster);
  if (device?.clusterUnits > 1 && !clusterTensorParallelOk(model, device)) result.push(STRUCTURAL_ASSUMPTIONS.clusterSplit);
  return result.map((item) => ({ ...item }));
}

function residentParamsB(model, device) {
  const ple = guardedPleParamsB(model);
  return pleDeductionApplies(device) && ple ? model.totalParams - ple : model.totalParams;
}

// 저장 정밀도 아래로 내려가는 경로가 검증되지 않은 가중치 — 선택한 양자화가 더 낮아도 이 bpw 밑으로 세지 않는다(보수적 하한).
// DeepSeek-V4.1 Engram n-gram 표: 공식 체크포인트·참조 코드 모두 FP8(E4M3) + 32원소당 E8M0 스케일 1B = 8.25 bit로 둔다
// ("the table stays fp8 as stored: rows are dequantized with `scale` on lookup"). 4bit 변환을 검증한 런타임이 없다.
//   https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/1bae0e1088993b9b82de08f59de5be5a34e42170/inference/model.py (ParallelEngramEmbedding)
const ENGRAM_MIN_BPW = 8 + 8 / 32;
function precisionFloorBytes(model, bits) {
  return model.engramParams ? model.engramParams * 1e9 * Math.max(0, ENGRAM_MIN_BPW - bits) / 8 : 0;
}

// 출시 체크포인트가 FP4로 내놓은 가중치(nativeFp4Params, B) — 반대 방향의 하한이다: 이 텐서는 선택한 정밀도가 더 높아도
// 출시 저장 크기 위로 세지 않는다. 형식은 E2M1 4bit + 32원소당 E8M0 스케일 1B = 4.25 bit(MXFP4 배치) — 바이트로 확인:
//   gpt-oss-120b 114,661,785,600 × 4.25/8 + BF16 2,167,371,072 × 2 = 65,248,815,744 B vs 샤드 합 65,248,893,184 B(차 77 KB = 헤더).
// GGUF는 MXFP4를 자체 타입으로 담고(ggml.h GGML_TYPE_MXFP4), llama-quantize는 이미 양자화된 텐서를 --allow-requantize 없이
// 다시 양자화하지 않는다(llama-quant.cpp "requantizing from type %s is disabled"):
//   https://github.com/ggml-org/llama.cpp/blob/970b51ce7c9ba5fff0726c5e9994f1c2f0e4eedc/ggml/include/ggml.h
//   https://github.com/ggml-org/llama.cpp/blob/970b51ce7c9ba5fff0726c5e9994f1c2f0e4eedc/src/llama-quant.cpp
// 공개 GGUF도 Q4_K_M~F16 어느 단계든 이 텐서를 그대로 둔다(HF 파일 목록 2026-10-23 — 행 주석에 단계별 바이트).
// 그래서 4.25 bit보다 높은 정밀도에서는 이 몫만 4.25 bit로 세고 나머지는 선택한 정밀도 그대로 센다. 8·16bit로 다시 편 빌드
// (lmstudio-community/gpt-oss-120b-MLX-8bit 124,168,924,424 B = 115.6 GiB)는 품질 이득 없이 크기만 키우므로 기준으로 삼지 않는다.
// 4.25 bit 미만은 다시 양자화한 빌드라 선택한 정밀도로 센다(종래와 같음 — DeepSeek-V4 UD-IQ1·Q2·Q3 GGUF가 실제로 그렇다).
// FP8(E4M3)은 이 부류가 아니다: GGUF에 FP8 타입이 없어 Q8_0(8.5 bit)·BF16으로 다시 쓴다 — GLM-5.3 FP8 출시 755,632,050,320 B인데
// unsloth/GLM-5.3-GGUF Q8_0 801,357,677,216 B · BF16 1,507,988,027,936 B(@346b359)라 종래 단계별 계산이 공개 파일과 맞는다.
const FP4_STORAGE_BPW = 4 + 8 / 32;
// 검증된 PLE 차감과는 함께 쓰지 않는다 — 두 몫이 겹치지 않는다는 증거가 없으면 이중 차감으로 가중치가 음수까지 내려간다
// (Codex 검수: E4B 형상 + mxfp4 선언 + U8 7.8B 합성 입력에서 FP16 −1.0 GiB). 카탈로그에 둘 다 가진 행은 없다.
function nativeFp4ParamsB(model) {
  const p = model.nativeFp4Params;
  return p > 0 && p < model.totalParams && !guardedPleParamsB(model) ? p : 0;
}
// FP4 할인을 쓰는 단계의 하한 — 위 공식은 비전문가 몫을 단계 평균으로 세서 실제 파일보다 작을 수 있다(gpt-oss 0.1–3.7%:
// hidden 2880이 K-quant 블록 256으로 나눠지지 않아 llama.cpp가 일부 텐서를 Q8_0으로 둔다). 그래서
//  ① 그 단계의 공개 GGUF 크기가 알려져 있으면(ggufFileBytes, 행 주석에 출처) 그보다 작게 세지 않고,
//  ② 모르면 출시 체크포인트 크기(fp4ReleaseBytes)보다 작게 세지 않는다 — 공개 파일 없이는 출시본이 확인된 유일한 바이트다.
//     GGUF F16 단계는 출시본보다 조금 크다: 1차원 텐서·바이어스를 F32로 담는다(gpt-oss F16 GGUF / 출시본 = 1.0018 · 1.0023,
//     행 주석의 바이트). 그래서 F16 단계의 ②에는 0.25%를 더한다.
// 키는 GPU_QUANTS 단계 이름이고, 그 단계의 bpw와 정확히 같은 요청에만 쓴다(Mac 4·8·16bit는 ②만 해당).
const GGUF_F16_SMALL_TENSOR_ALLOWANCE = 1.0025;
function fp4FloorBytes(model, bits) {
  const q = GPU_QUANTS.find((x) => x.bpw === bits);
  const published = q && model.ggufFileBytes ? model.ggufFileBytes[q.tier] : 0;
  if (Number.isSafeInteger(published) && published > 0) return published;
  const release = model.fp4ReleaseBytes;
  if (!(Number.isSafeInteger(release) && release > 0)) return 0;
  return q?.tier === 'FP16' ? release * GGUF_F16_SMALL_TENSOR_ALLOWANCE : release;
}
// 가중치 바이트의 단일 진실원 — calcParamMemory·calcMaxContext가 같은 식을 쓴다(인라인 중복 금지).
function weightBytes(model, bits, device) {
  const resident = residentParamsB(model, device);
  const fp4 = bits > FP4_STORAGE_BPW ? Math.min(nativeFp4ParamsB(model), resident) : 0;
  if (!fp4) return resident * 1e9 * (bits / 8) + precisionFloorBytes(model, bits);
  const bytes = (resident - fp4) * 1e9 * (bits / 8) + fp4 * 1e9 * (FP4_STORAGE_BPW / 8) + precisionFloorBytes(model, bits);
  return Math.max(bytes, fp4FloorBytes(model, bits));
}

export function calcParamMemory(model, bits, device) {
  if (!model.totalParams) return { totalGB: 0, activeGB: null };
  const bpe = bits / 8;
  const baseTotalGB = weightBytes(model, bits, device) / 1024 ** 3;
  const baseActiveGB = model.activeParams ? (model.activeParams * 1e9 * bpe) / 1024 ** 3 : null;
  // quantAdjust는 정수 bits(Apple NVFP4/MXFP8 4/8/16) 보정용. GGUF는 weightBpw가 소수(4.8944 등)라
  // 매치 안 돼 multiplier=1.0 — 이는 *의도*다: GGUF bpw는 블록 스케일 등 실측 오버헤드를 이미 포함하므로
  // Apple 보정을 또 곱하면 이중계상. ⚠️ 단 기준 bpw는 Llama-3.1-8B 실측이라 소형모델은 임베딩 비중↑로
  // 약간 과소추정될 수 있음(v1 근사 — 모델별 .gguf 실크기 보정은 v2, spec §6).
  const multiplier = (quantAdjust[model.name] && quantAdjust[model.name][bits]) || 1.0;
  return { totalGB: baseTotalGB * multiplier, activeGB: baseActiveGB ? baseActiveGB * multiplier : null };
}

// 런타임 오버헤드: 양자화 메타(12%) + KV 블록 padding(15%) + 활성화 버퍼 + 고정 2GB
// ⚠ 이 상수들을 특정 숫자에 맞추려 하지 말 것. 예전 주석은 'Qwen3.6 35B @130K @8bit →
// 이론 43GB, 실제 ~54GB'라고 적었으나 그 '실제 54GB'를 뒷받침하는 측정 픽스처가 없고
// (system_total_peak 0행), 엔진은 그 시나리오에서 49.94 GiB를 낸다. 2026-09-03 조사: 54는
// 주장이 도입된 커밋 시점 엔진에서도 나오지 않았다(49.88). total-to-run 캘리브레이션은 아직 없다.
//
// 가중치 12% 몫의 예외 — GB10 한 대(llama.cpp·Ollama, GGUF): 0. GGUF tier bpw는 블록 스케일·비양자 텐서를 이미 포함한
// 파일 평균이라 12%를 또 얹으면 이중 계상이다. 공개 실측(fixtures/measured.json GB10 4건, 2026-10-20 검증)이 근거:
//   ASUS Ascent GX10 · Qwen3.6-35B-A3B BF16 — CUDA0 model buffer 66,221.98 MiB + CUDA_Host 970.00 MiB = 파일 65.62 GiB 그대로,
//   가중치 외 계산 버퍼는 0.49 GiB(65K)·0.70 GiB(102K)인데 12% 몫은 7.82 GiB였다(https://github.com/ollama/ollama/issues/16610 첨부 로그).
//   DGX Spark · Gemma 4 31B F16 65K: nvidia-smi 64,107 MiB vs 계산 69.34 GiB(https://forums.developer.nvidia.com/t/366162).
// KV padding 15%·활성화 몫·reserve의 런타임 2 GiB는 그대로 둔다(호스트 쪽 버퍼·CUDA 컨텍스트는 위 로그에 안 잡힌다).
// 근거 범위 = flash attention을 켠 llama.cpp·Ollama(둘 다 기본값 auto: llama.cpp common/common.h flash_attn_type,
//   ollama llm/llama_server.go LlamaServerFlashAttention). flash attention을 끈 경로(KQ 점수 전체 보유)·vLLM은 전제 문구로 밝힌다.
// 디스크리트 GPU·맥·GB10 클러스터(vLLM·TensorRT-LLM)는 실측이 없어 12%를 유지한다.
const WEIGHT_OVERHEAD_RATIO = 0.12;
function weightOverheadRatio(device) {
  return device?.unifiedMemory && clusterUnitsOf(device) === 1 ? 0 : WEIGHT_OVERHEAD_RATIO;
}
export function calcRuntimeOverhead(model, ctx, bitsOrQuant, device) {
  ctx = Math.max(0, Math.floor(Number(ctx)) || 0); // 음수/NaN 가드 (활성화 버퍼 ctx 비례)
  const { weightBpw, kvBits } = toQuant(bitsOrQuant); // weight↔KV 분리(GGUF: KV padding이 weight bpw 오염 금지)
  const paramMem = calcParamMemory(model, weightBpw, device).totalGB; // PLE: GPU면 상주분만(오버헤드도 상주 가중치 비례)
  const kvMem = calcKVCache(clusterKvModel(model, device), ctx, kvBits).totalGB; // 클러스터: 복제된 KV까지
  const paramOverhead = paramMem * weightOverheadRatio(device);
  const kvOverhead = kvMem * 0.15;
  const activationOverhead = ctx * 0.00003 * clusterUnitsOf(device); // 활성화 버퍼는 대마다 생긴다
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
  if (!clusterTensorParallelOk(model, device)) return 0; // 대수로 나눌 수 없는 모델 — 클러스터 수치를 내지 않는다
  const { weightBpw, kvBits } = toQuant(bitsOrQuant);
  const attnLayers = model.fullAttnLayers || model.layerCount; // 하이브리드: 풀어텐션 레이어만 KV
  const kbpe = kvBits / 8;    // KV 원소 바이트(GPU 기본 F16=16)
  const quantMultiplier = (quantAdjust[model.name] && quantAdjust[model.name][weightBpw]) || 1.0;
  const paramBytes = weightBytes(model, weightBpw, device) * quantMultiplier; // PLE: GPU면 상주분만 · Engram FP8 하한 · 출시 FP4 상한
  const linearStateBytes = calcLinearState(model).totalBytes; // ctx 무관 고정 → 예산에서 먼저 뺀다
  const budget =
    device.memoryGB * 1024 ** 3 * (1 - device.headroomRatio) - paramBytes - paramBytes * weightOverheadRatio(device) - device.reserveGB * 1024 ** 3 - linearStateBytes;
  if (budget <= 0) return 0;
  const overhead = 1.15; // KV 블록 할당 padding
  const km = clusterKvModel(model, device); // GB10 클러스터: TP 복제된 KV 헤드·MLA 잠재 차원(그 밖 원래 모델)
  const perLocal = 2 * km.kvHeads * km.kvHeadDim * kbpe * overhead;
  const perGlobal = 2 * (km.globalKvHeads || km.kvHeads) * (km.globalHeadDim || km.kvHeadDim) * kbpe * overhead;
  if (perLocal <= 0) return 0;
  const actPerTok = 0.00003 * 1024 ** 3 * clusterUnitsOf(device); // 활성화 버퍼(simulate와 동일) — ctx 비례라 토큰당 비용에 포함, 클러스터는 대마다

  // DeepSeek-V4 계열: 고정분(창·압축기 상태)을 먼저 빼고 토큰당 비용으로 나눈다 (calcKVCache 압축 분기와 동일 식)
  if (model.compressedKv) {
    const { perToken, fixed } = compressedKvBytes(km.compressedKv, kbpe);
    const rest = budget - fixed * overhead;
    if (rest <= 0) return 0;
    return Math.min(Math.floor(rest / (perToken * overhead + actPerTok)), model.maxContext);
  }

  // MLA: 압축 latent+rope, 전 레이어 균일 (calcKVCache MLA 분기와 동일 식)
  if (model.mlaKvLoraRank) {
    const perTokMla = (km.mlaKvLoraRank + (km.mlaRopeDim || 0)) * kbpe * overhead * model.layerCount;
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
  ctx = Math.min(Math.max(1, Math.floor(Number(ctx)) || 1), model.maxContext || Infinity); // 음수/NaN ctx 가드 — 음수 KV가 판정을 fits로 뒤집는 것 방지 (2026-07-11 감사)
  const { weightBpw, kvBits } = toQuant(bitsOrQuant); // weight(파라미터) ↔ KV 비트 분리
  const param = calcParamMemory(model, weightBpw, device).totalGB; // PLE 모델은 GPU에서 상주분만 (residentParamsB)
  const kv = calcKVCache(clusterKvModel(model, device), ctx, kvBits).totalGB; // GB10 클러스터: TP 복제분 포함(그 밖 원래 값)
  // pinned llama.cpp/GGUF 경로가 입력층 텐서(per_layer_token_embd)를 accelerator가 아닌 CPU/host 버퍼에 배치하므로
  // GPU에 상주하지 않는 PLE 근사 크기. host 메모리는 예산에 넣지 않으며, PLE를 accelerator에 적재하는 런타임에서는
  // 이 추정과 GPU weight 차감이 모두 무효다.
  const pleOffloadGB = pleDeductionApplies(device) ? (guardedPleParamsB(model) * 1e9 * (weightBpw / 8)) / 1024 ** 3 : 0;

  // 단일 reserve 방정식(Codex council): used = param + kv + rtDyn + reserve. reserve는 1회만.
  // 오버헤드 공식의 단일 진실원 = calcRuntimeOverhead (KvDeepDive와 같은 소스 — 인라인 중복 금지, 드리프트 방지)
  const ov = calcRuntimeOverhead(model, ctx, { weightBpw, kvBits }, device);
  const rtDyn = ov.paramOverheadGB + ov.kvOverheadGB + ov.activationOverheadGB; // 동적 런타임(고정 reserve 미포함)
  const reserve = device.reserveGB;                       // OS/CUDA/디스플레이 통합 reserve
  const linearState = calcLinearState(model).totalGB;     // 하이브리드 선형 어텐션 고정 상태(ctx 무관, 없으면 0)
  const used = param + kv + linearState + rtDyn + reserve;
  const free = device.memoryGB - used;
  const headroom = device.memoryGB * device.headroomRatio;

  let verdict;
  // 최종 방어선: NaN·Infinity는 `free < 0`도 `free < headroom`도 false라 그대로 'yes'로 떨어진다.
  // 상류 게이트가 다 뚫려도 "모르는데 된다"고 답하지는 않게 한다.
  if (!Number.isFinite(used) || !Number.isFinite(free)) verdict = 'no';
  else if (!clusterTensorParallelOk(model, device)) verdict = 'no'; // TP로 나눌 수 없으면 이 클러스터에서는 돌지 않는다(전제로 이유 고지)
  else if (free < 0) verdict = 'no';
  else if (free < headroom) verdict = 'tight';
  else verdict = 'yes';

  const os = device._os ?? 0; // Apple 표시 호환
  const rt = rtDyn + ov.fixedOverheadGB; // 기존 rt = 동적 + 고정(Apple 2.0 / GPU 0 — calcRuntimeOverhead가 결정)
  const assumptions = structuralAssumptions(model, device);

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
    linearState, // 선형 어텐션 순환 상태(GB) — ctx 무관 고정. 표준 어텐션 모델은 0
    rt,
    rtDyn,
    reserve,
    system: os + rt, // 비전공자용 묶음
    pleOffloadGB, // >0이면 pinned llama.cpp/GGUF 입력층 host 배치 경로에서 GPU에 상주하지 않는 PLE 근사분 — UI 각주용(판정 미포함)
    used,
    free,
    headroom,
    verdict,
    pct: used / device.memoryGB,
    maxContext: calcMaxContext(model, device, { weightBpw, kvBits }),
    ...(assumptions.length ? { structuralAssumptions: assumptions } : {}),
  };
}

// 멀티모델 스택 fit — 여러 모델이 같은 디바이스에 "동시 상주"할 때 (speculative decoding draft+target,
// IDE 자동완성+챗 등). 모델별 weights+KV+동적오버헤드를 각각 계산해 합산, reserve는 디바이스당 1회.
// 근거: llama.cpp speculative.md "need enough VRAM to load both models simultaneously" — 합산이 정확한 수학.
// entries: [{ model, ctx, weightBpw, kvBits }...] · deviceOrRam: gpu device 객체 또는 Mac RAM(GB)
export function simulateStack(entries, deviceOrRam) {
  const device = toDevice(deviceOrRam);
  const parts = entries.map(({ model, ctx, weightBpw, kvBits }) => {
    const c = Math.min(Math.max(1, Math.floor(Number(ctx)) || 1), model.maxContext || Infinity); // simulate와 같은 ctx 가드
    const kb = kvBits ?? 16;
    const param = calcParamMemory(model, weightBpw, device).totalGB;
    const kv = calcKVCache(clusterKvModel(model, device), c, kb).totalGB;
    const ov = calcRuntimeOverhead(model, c, { weightBpw, kvBits: kb }, device);
    const rtDyn = ov.paramOverheadGB + ov.kvOverheadGB + ov.activationOverheadGB;
    const linearState = calcLinearState(model).totalGB; // simulate와 같은 항목 — 하이브리드 선형 어텐션 고정 상태(없으면 0)
    const assumptions = structuralAssumptions(model, device);
    return {
      model, ctx: c, weightBpw, kvBits: kb, param, kv, linearState, rtDyn,
      subtotal: param + kv + linearState + rtDyn,
      ...(assumptions.length ? { structuralAssumptions: assumptions } : {}),
    };
  });
  const reserve = device.reserveGB;
  const used = parts.reduce((s, p) => s + p.subtotal, 0) + reserve;
  const free = device.memoryGB - used;
  const headroom = device.memoryGB * device.headroomRatio;
  const tpOk = parts.every((p) => clusterTensorParallelOk(p.model, device));
  const verdict = !Number.isFinite(used) || !Number.isFinite(free) || !tpOk || free < 0 ? 'no' : free < headroom ? 'tight' : 'yes';
  // 중복 제거한 뒤 다시 복제한다 — Map은 part가 들고 있는 *바로 그* 객체를 담으므로,
  // 복제하지 않으면 스택 레벨 item을 만진 소비자가 part의 값까지 바꾼다(순수 결과 규율).
  const assumptions = [...new Map(
    parts.flatMap((part) => part.structuralAssumptions || []).map((item) => [item.id, item])
  ).values()].map((item) => ({ ...item }));
  return {
    parts, device, memoryGB: device.memoryGB, reserve, used, free, headroom, verdict,
    pct: used / device.memoryGB,
    param: parts.reduce((s, p) => s + p.param, 0),
    kv: parts.reduce((s, p) => s + p.kv, 0),
    linearState: parts.reduce((s, p) => s + p.linearState, 0),
    rt: parts.reduce((s, p) => s + p.rtDyn, 0),
    ...(assumptions.length ? { structuralAssumptions: assumptions } : {}),
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
  const bigger = GPUS.filter((g) => !g.count && !g.unifiedMemory && g.vramGB > device.memoryGB) // 통합 메모리 완제품은 '카드 교체'가 아니다
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
  'M5': 153, 'M5 Pro': 307, 'M5 Max': 614, 'M5 Ultra': 1200, // M5 base 153GB/s (Apple 공식 — apple.com/mac-mini/specs M6 16GB 구성과 동일값) · M5 Ultra 1.2TB/s (Apple 뉴스룸/Mac Studio 기술사양 2026-08-27)
  'M6': 170, // ⚠️ 대표값(마케팅 "up to"). 실제는 용량 연동 — 아래 chipBandwidth의 ramGB 분기 참조
};
// ⚠️ M6는 "같은 칩인데 통합메모리 용량에 따라 대역폭이 다르다":
//    16GB = 153GB/s (M5와 동일) · 24GB·32GB = 170GB/s  — 출처 https://www.apple.com/mac-mini/specs/ (2026-08-27 직접 확인)
//    ⚠️ "베이스 M시리즈 사상 최초"라고는 쓰지 않는다 — M2~M5 공식 사양은 용량 무관 단일 대역폭이지만
//       M1 공식 사양은 8/16GB 대역폭 자체를 게재하지 않아 반증 불가(Sol 크로스리뷰 2026-08-27 지적, 수용).
//       검증 가능한 표현은 "확인된 M2~M5 범위에서는 선례 없음"까지다.
//    ramGB를 안 넘기면 풀스펙 값(170)을 반환한다 — chipBandwidth('M5 Max')가 기본 614(고사양)를 주는 기존 관례와 동일.
export function chipBandwidth(chip, gpuCores = 40, ramGB = null) {
  if (chip === 'M6') return (ramGB != null && ramGB <= 16) ? 153 : 170; // M6 용량별
  if (chip === 'M5 Max') return gpuCores === 32 ? 460 : 614; // M5 Max GPU 코어수별
  return CHIP_BANDWIDTH[chip] || 307;
}

// ⚠️ @deprecated — tok/s 예측은 unwinnable 판정으로 제품에서 삭제됨(2026-06-05 결정, README·사이트 미노출).
// 하위호환(기존 임포터 breaking 방지)용으로만 export 유지 — 신규 사용 금지. fit만이 검증 가능한 주장이다.
// 예상 토큰 생성 속도(tok/s): 디코드 1토큰마다 활성 파라미터를 메모리에서 읽음
// → tok/s ≈ 대역폭 ÷ (활성파라미터 × 바이트) × 실현효율
export function estimateSpeed(model, chipOrDevice, bitsOrWeightBpw, gpuCores = 40, ramGB = null) {
  if (model.isCloud || !model.totalParams) return null;
  // chip 문자열(Apple) → chipBandwidth / device 객체(GPU) → device.bandwidthGBs
  const bwGBs = (chipOrDevice && typeof chipOrDevice === 'object' && chipOrDevice.bandwidthGBs != null)
    ? chipOrDevice.bandwidthGBs
    : chipBandwidth(chipOrDevice, gpuCores, ramGB); // ramGB는 M6 용량별 대역폭에만 쓰임(미전달 시 기존 동작 그대로)
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

// 슬라이딩:글로벌 레이어 수를 최대공약수로 약분한 라벨 — 카탈로그 표기('5:1'·'4:1'·'3:1')와 같은 규약.
function slidingPatternLabel(slidingLayers, globalLayers) {
  const gcd = (a, b) => (b ? gcd(b, a % b) : a);
  const g = gcd(slidingLayers, globalLayers);
  return `${slidingLayers / g}:${globalLayers / g}`;
}

// 모델링이 끝난 아키텍처만 좁게 허용한다(allowlist). qwen3_5 계열 = Gated DeltaNet 하이브리드로,
// full/linear 레이어 분리와 고정 순환 상태를 엔진이 실제로 계산한다(calcLinearState + fullAttnLayers).
// 파생 finetune이 매우 많아(Qwen3.8-27B 계열) 이 계열만 열어두는 가치가 크다.
const HYBRID_LINEAR_TYPES = new Set(['qwen3_5', 'qwen3_5_text', 'qwen3_5_moe', 'qwen3_5_moe_text']);

// 메모리 거동을 실제로 모델링한 레이어 타입만 인정한다(허용목록). 모르는 타입이 하나라도 있으면 거부 —
// 예: lfm2의 'conv'는 어텐션이 아니라 고정 conv 캐시라 KV 공식이 성립하지 않는다.
const KNOWN_LAYER_TYPES = new Set(['full_attention', 'sliding_attention', 'linear_attention']);

// ── 미지 구조 필드 게이트 (issue #87) ────────────────────────────────────────
// 종래 게이트는 "알려진 나쁜 필드"를 하나씩 막는 블랙리스트였다. 그래서 처음 보는 architectures라도
// num_hidden_layers/num_key_value_heads/head_dim만 표준처럼 생겼으면 통과했다 — 필드 *이름*만 보고
// 아키텍처 *정체*는 안 본 것이다. 실증(2026-08-31 day0 28건 전수): TTS 모델(Breeze-TTS-2)이
// 3.5B LLM으로 통과했고, MHC+MLA+sliding 혼합(Motif-3)이 MLA 균일 경로로 통과했다.
// → 화이트리스트로 뒤집는다. 구조에 영향 줄 수 있는 이름의 키가 있는데 우리가 그 의미를 모르면 거부.
//
// 판정 원칙: "이 키가 KV 레이아웃 또는 파라미터 산출을 바꿀 수 있는가?"
// 파라미터는 safetensors total_size(실측)에서 나오므로, 가중치만 늘리는 키는 무해로 본다.
const STRUCTURAL_KEY_RE =
  /attn|attention|kv|head|sliding|window|state|conv|expert|latent|lora|compress|sparse|index|mhc|hybrid|linear|block|mamba|ssm|nope|layer/i;

// 엔진이 실제로 읽어서 계산에 쓰는 키 — 의미를 아는 것들이다.
// (cross_attention_layers·layers_block_type·index_topk 등 "보면 즉시 거부"하는 키는 위쪽 검사에서
//  이미 throw되므로 여기 넣지 않는다. 넣으면 검사 순서가 바뀔 때 조용히 통과할 수 있다.)
const ENGINE_READ_KEYS = new Set([
  'num_hidden_layers', 'num_attention_heads', 'num_key_value_heads', 'head_dim', 'global_head_dim',
  'sliding_window', 'use_sliding_window', 'layer_types',
  'num_local_experts', 'num_experts', 'n_routed_experts',
  'kv_lora_rank', 'qk_rope_head_dim',
  'vocab_size_per_layer_input', 'hidden_size_per_layer_input',
  'linear_num_key_heads', 'linear_num_value_heads', 'linear_key_head_dim',
  'linear_value_head_dim', 'linear_conv_kernel_dim',
]);

// 계열 무관 무해 — 계산 방식은 바꾸지만 메모리 레이아웃은 안 바꾼다.
const BENIGN_STRUCTURAL_KEYS = new Set([
  // RoPE 계열: 위치 인코딩이라 KV *텐서 크기*와 무관 (θ·스케일링·인터리브 모두)
  'rope_theta', 'rope_scaling', 'rope_parameters', 'rope_interleave', 'rope_factor',
  'partial_rotary_factor',
  // 정규화·드롭아웃·바이어스: 학습/수치 안정화용, 캐시 크기 불변
  'attention_dropout', 'attention_bias', 'qk_layernorm', 'layer_norm_epsilon',
  // Granite: 어텐션 스코어에 곱하는 스칼라. 텐서 치수 불변
  'attention_multiplier',
  // FFN/MoE 배치: 어텐션이 아니라 MLP 쪽 구조라 KV 불변. 가중치 증감은 total_size에 이미 반영됨
  'moe_intermediate_size', 'moe_layer_freq', 'shared_expert_intermediate_size',
  'n_shared_experts', 'num_shared_experts', 'norm_topk_prob', 'output_router_logits',
  'router_aux_loss_coef', 'first_k_dense_replace', 'interleave_moe_layer_step',
  'moe_router_enable_expert_bias', 'num_experts_per_tok', 'experts_top_k',
  'experts_per_token', 'decoder_sparse_step', 'mlp_only_layers',
  // MTP(Multi-Token Prediction) 헤드 레이어 — 디코더 본체 밖이라 KV 불변.
  // 가중치는 total_size에 포함돼 파라미터가 소폭 과대되지만 보수적(과대) 방향이라 허용.
  'num_nextn_predict_layers',
  // Qwen 계열 레거시 필드. 슬라이딩이 실제로 쓰이는지는 slidingActive가 따로 판정하므로
  // 이 값 자체는 캐시 크기에 영향이 없다.
  'max_window_layers',
]);

// MLA(kv_lora_rank) 경로에서만 무해한 키 — 압축 latent가 캐시되므로 아래 치수들은 캐시 크기와 무관하다.
// MLA가 아닐 때는 같은 이름이라도 K/V 텐서 치수를 직접 바꿀 수 있어 허용하지 않는다.
// (예: v_head_dim은 비-MLA에서 V의 head_dim이 K와 다르다는 뜻이 되어 2×kvHeads×headDim 공식이 깨진다.)
const MLA_ONLY_KEYS = new Set([
  'q_lora_rank',      // 쿼리 압축 — 캐시되는 건 KV latent라 무관
  'qk_nope_head_dim', // 비-RoPE 성분. 캐시는 kv_lora_rank + qk_rope_head_dim
  'v_head_dim',       // MLA에선 V도 latent에서 복원 — 별도 캐시 없음
]);

// 계열 전용 — 해당 model_type을 실제로 모델링·테스트했을 때만 허용한다.
// 같은 키라도 미검증 계열에서 나오면 의미가 같다는 보장이 없으므로 거부한다(이게 #87의 요지).
const FAMILY_STRUCTURAL_KEYS = new Map([
  // qwen3_5 = Gated DeltaNet 하이브리드. full/linear 분리와 고정 순환 상태를 엔진이 계산하고
  // 벡터로 검증돼 있다(v-qwen38-*). 아래 키들은 그 계열에서 의미가 확인된 것들.
  ['qwen3_5', new Set([
    'attn_output_gate',        // 출력 게이팅 — 가중치만 늘고 KV 불변
    'full_attention_interval', // layer_types와 중복 정보. 엔진은 layer_types를 쓴다
    'mlp_only_layers',         // MoE 대신 dense MLP인 레이어 — FFN 구조, KV 불변
    'mtp_num_hidden_layers',   // Multi-Token Prediction 추가 레이어. 디코더 본체 밖(가중치만)
    'mamba_ssm_dtype',         // dtype 힌트. 실제 SSM 스케줄은 layers_block_type 게이트가 따로 본다
  ])],
  // spark2_5 = 표준 GQA + SWA(512) 3:1 인터리브(XHToken/Spark-X2.5). 두 키는 어텐션 *출력*에 곱하는 헤드별 게이트다:
  //   g_proj = nn.Linear(hidden_size, num_heads) → sigmoid|silu(gate) × attn_output → out_proj. K/V는 그 앞에서
  //   num_key_value_heads × head_dim 그대로 캐시된다 — 가중치(레이어당 hidden×heads)만 늘고 KV 레이아웃 불변.
  //   pinned 1차 출처(revision 5e10fcc0286756aebf7c41dc52c1e42d95c70281):
  //   https://huggingface.co/XHToken/Spark-X2.5-4B/blob/5e10fcc0286756aebf7c41dc52c1e42d95c70281/modeling_spark.py
  //   https://huggingface.co/XHToken/Spark-X2.5-4B/blob/5e10fcc0286756aebf7c41dc52c1e42d95c70281/config.json
  //   카탈로그 Spark-X2.5-4B + 벡터(spark-x25-4b-kv-1m-f16)로 검증. 정확히 'spark2_5'만 — 파생 model_type은 fail-closed.
  ['spark2_5', new Set([
    'gate_attn_act_mode',        // 게이트 활성함수(sigmoid|silu) — 활성화 경로만, 텐서 치수 무관
    'headwise_attn_output_gate', // g_proj 유무 — 가중치만 늘고 KV 불변
  ])],
  // deepseek_v4 = DeepSeek-V4-Flash/-Pro(+Base). 압축 KV 치수는 deepseekCompressedKv가 읽고 검증한다(calcKVCache 압축 분기).
  //   https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/config.json
  //   https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/b5968e9190ef611bbf34a7229255be88a0e937c1/config.json
  ['deepseek_v4', new Set([
    'compress_ratios', 'index_head_dim', // 압축 캐시·인덱서 K 치수 — deepseekCompressedKv가 읽는다
    'index_n_heads',       // 인덱서 쿼리 헤드 — 캐시 없음(K는 1개 공유), 텐서 병렬 분할 검사에만 쓴다
    'index_topk',          // 어텐션이 고르는 압축 항목 수 — 캐시 크기와 무관
    'compress_rope_theta', // 압축 캐시의 RoPE θ — 위치 인코딩, 텐서 치수 불변
    'num_hash_layers',     // 앞 N층 MoE의 해시 라우팅(tid2eid 가중치) — KV 불변
    'q_lora_rank', 'o_lora_rank', // 쿼리·출력 저랭크 사영 — 가중치만, 캐시는 head_dim 잠재 하나
    'expert_dtype',        // 전문가 가중치 저장 형식(fp4) — 파라미터는 safetensors 논리 수에서 온다
  ])],
  // deepseek_v41 = DeepSeek-V4.1-Flash(-1017) 텍스트 본체. 레이어 간 KV·인덱스 공유(CSA2)는 deepseekCompressedKv가 검증한다.
  //   https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/1bae0e1088993b9b82de08f59de5be5a34e42170/config.json
  ['deepseek_v41', new Set([
    'compress_ratios', 'index_head_dim', 'kv_source_layer_ids', 'index_source_layer_ids', // CSA2 캐시 배치 — 읽고 검증
    'index_n_heads', 'index_topk', 'compress_rope_theta', 'q_lora_rank', 'o_lora_rank', // deepseek_v4와 같은 뜻
    // 계층형 인덱서: 첫 Full 층이 매 forward에 만드는 후보 마스크 — 지속 캐시가 아니다(SharedAttentionRuntime.candidates)
    'candidate_source_layer_id', 'candidate_topk_blocks', 'candidate_block_size',
    // Engram n-gram 표: 가중치(safetensors에 포함). 토큰 기록 버퍼는 tokenStateBytes로 따로 센다
    'engram_layer_ids', 'engram_n_heads', 'engram_head_dim', 'engram_compressed_vocab_size',
    // DSpark 초안 블록: 가중치 + 창 캐시(windowLayers에 포함)뿐 — ratio 0만 허용
    'dspark_block_size', 'dspark_target_layer_ids', 'dspark_n_routed_experts', 'dspark_num_experts_per_tok',
  ])],
  // muse_spark = Meta Muse Spark Open 텍스트 본체 — 표준 GQA + 슬라이딩/전체 layer_types(엔진 슬라이딩 경로 그대로).
  //   https://github.com/huggingface/transformers/blob/487dcfa6ab1c294ce2695005108fdf19a63363eb/src/transformers/models/muse_spark/modeling_muse_spark.py
  //   MuseSparkTextAttention: K/V = num_key_value_heads × head_dim를 DynamicCache에 그대로 둔다(sliding 층은 창만).
  ['muse_spark', new Set([
    'layer_rope_theta', // 층별 RoPE θ(0 = NoPE) — 위치 인코딩만, K/V 텐서 치수 불변
    'mlp_layer_types',  // 층별 sparse(MoE)/dense MLP — FFN 구조라 KV 불변, 가중치는 safetensors 논리 수
  ])],
]);

// model_type을 계열 키로 정규화 — qwen3_5 / qwen3_5_moe / qwen3_5_moe_text 를 한 계열로 본다.
function familyOf(modelType) {
  const t = String(modelType || '');
  if (HYBRID_LINEAR_TYPES.has(t)) return 'qwen3_5';
  if (t === 'deepseek_v41' || t === 'deepseek_v41_text') return 'deepseek_v41';
  if (t === 'muse_spark' || t === 'muse_spark_text') return 'muse_spark';
  return t;
}

// DeepSeek-V4 계열 config → compressedKv(calcKVCache 압축 분기의 입력). 공식 추론 코드의 버퍼 배치를 그대로 옮기고,
// 그 코드가 전제하는 배치가 아니면 계산하지 않는다(fail closed). 출처·버퍼 목록은 compressedKvBytes 주석.
//   V4  : 0 = 창만 · 4 = CSA(압축기 overlap + 인덱서) · 128 = HCA(압축기만). 압축 레이어는 전부 자기 캐시를 갖는다.
//   V4.1: 0 = 창만 · 1·2 = CSA2. kv_source_layer_ids만 압축 KV·인덱서 K를 갖고, 나머지는 앞 소스를 읽는다.
//         기술 보고서 §2.3.1: Full(소스 = KV + 인덱싱) · Reindex(앞 KV + 자기 인덱싱) · Reuse(앞 KV + 앞 Top-K).
const DEEPSEEK_V4_RATIOS = Object.freeze({ deepseek_v4: new Set([0, 4, 128]), deepseek_v41: new Set([0, 1, 2]) });
// 인덱서 프리필 점수 버퍼 — vLLM 기본 상한 VLLM_SPARSE_INDEXER_MAX_LOGITS_MB = 512 MiB (compressedKvBytes 주석의 출처).
const DEEPSEEK_INDEXER_WORKSPACE_BYTES = 512 * 1024 * 1024;
function deepseekCompressedKv(c, family, layerCount) {
  const posInt = (v) => Number.isInteger(v) && v > 0;
  const fail = (why) => { throw new Error(`DeepSeek-V4 계열 config의 ${why} — 계산하지 않습니다`); };
  for (const k of ['head_dim', 'index_head_dim', 'index_n_heads', 'index_topk', 'sliding_window', 'o_groups', 'vocab_size', 'n_routed_experts']) {
    if (!posInt(c[k])) fail(`${k} 값이 양의 정수가 아니에요`);
  }
  const routedExperts = [c.n_routed_experts];
  // 참조 코드는 KV 헤드 하나(wkv: dim → head_dim)를 전 쿼리 헤드가 공유한다. 다른 값이면 같은 코드가 아니다.
  if (c.num_key_value_heads !== 1) fail('num_key_value_heads가 1이 아니에요');
  if (c.kv_lora_rank != null) fail('kv_lora_rank(MLA)가 함께 선언돼 있어요');
  const nextn = c.num_nextn_predict_layers ?? 0;
  const ratios = c.compress_ratios;
  if (!Array.isArray(ratios) || ratios.length !== layerCount + nextn) {
    fail('compress_ratios 길이가 num_hidden_layers + num_nextn_predict_layers와 달라요');
  }
  const allowed = DEEPSEEK_V4_RATIOS[family];
  if (!ratios.every((r) => allowed.has(r))) fail(`compress_ratios에 검증된 값(${[...allowed].join('·')}) 밖의 값이 있어요`);
  // MTP/DSpark 층은 창만 쓴다(V4.1 DSparkAttention: assert compress_ratio == 0).
  if (ratios.slice(layerCount).some((r) => r !== 0)) fail('MTP/DSpark 층의 compress_ratios가 0이 아니에요');
  const backbone = ratios.slice(0, layerCount);
  const headDim = c.head_dim;
  const indexDim = c.index_head_dim;
  const byRatio = new Map(); // ratio → 자기 캐시를 가진 층 수
  let stateBytes = 0;
  let windowLayers = ratios.length;
  let tokenStateBytes = 0;
  let engramParams = 0;
  if (family === 'deepseek_v4') {
    if (c.num_hash_layers != null && !(Number.isInteger(c.num_hash_layers) && c.num_hash_layers >= 0 && c.num_hash_layers <= layerCount)) {
      fail('num_hash_layers 값이 0..num_hidden_layers 정수가 아니에요');
    }
    for (const r of backbone) {
      if (!r) continue;
      byRatio.set(r, (byRatio.get(r) || 0) + 1);
      const coff = r === 4 ? 2 : 1; // ratio 4만 overlap 압축(상태 폭 2배)
      stateBytes += 2 * (coff * r) * (coff * headDim) * 4; // kv_state + score_state, fp32
      if (r === 4) stateBytes += 2 * (2 * r) * (2 * indexDim) * 4; // 인덱서 자체 압축기(overlap, index_head_dim)
    }
  } else {
    const ids = (k) => {
      const v = c[k];
      if (!Array.isArray(v) || !v.length || !v.every((i, n) => Number.isInteger(i) && i >= 0 && i < layerCount && (n === 0 || i > v[n - 1]))) {
        fail(`${k}가 0..num_hidden_layers-1의 오름차순 정수 배열이 아니에요`);
      }
      return v;
    };
    const kvSources = ids('kv_source_layer_ids');
    const indexSources = ids('index_source_layer_ids');
    // Full 모드 = KV 소스이자 인덱스 소스. 소스가 인덱싱하지 않으면 뒤 층이 옛 KV 기준 Top-K를 재사용하게 된다.
    if (!kvSources.every((s) => indexSources.includes(s) && backbone[s] > 0)) fail('KV 소스 층이 인덱스 소스가 아니거나 압축 층이 아니에요');
    if (!indexSources.every((s) => backbone[s] > 0)) fail('인덱스 소스 층이 압축 층이 아니에요');
    for (let i = 0; i < layerCount; i++) {
      if (!backbone[i]) continue;
      const kv = kvSources.filter((s) => s <= i).pop();
      const ix = indexSources.filter((s) => s <= i).pop();
      // 압축 층은 같은 비율의 앞 소스 캐시를 읽고, 그 캐시를 기준으로 고른 Top-K를 써야 한다.
      if (kv === undefined || backbone[kv] !== backbone[i] || ix === undefined || ix < kv) {
        fail(`${i}층이 읽을 같은 비율의 KV 소스·인덱스 소스가 없어요`);
      }
    }
    const cand = c.candidate_source_layer_id ?? -1;
    if (!(cand === -1 || (indexSources.includes(cand) && posInt(c.candidate_topk_blocks) && posInt(c.candidate_block_size)))) {
      fail('candidate_source_layer_id가 -1이나 인덱스 소스 층이 아니에요');
    }
    // 후보 마스크는 소스 층의 압축 좌표(폭 ctx/ratio)로 만들어지고 그 뒤 모든 인덱서가 그대로 쓴다(uses_candidates) — 비율이 같아야 한다.
    if (cand !== -1 && indexSources.some((s) => s > cand && backbone[s] !== backbone[cand])) {
      fail('candidate_source_layer_id 뒤 인덱스 층의 압축 비율이 후보 마스크와 달라요');
    }
    for (const s of kvSources) {
      const r = backbone[s];
      byRatio.set(r, (byRatio.get(r) || 0) + 1);
      if (r > 1) stateBytes += 2 * r * headDim * 4; // Compressor kv_state + score_state (ratio, head_dim) fp32 — ratio 1은 상태 없음
    }
    const engram = c.engram_layer_ids ?? [];
    if (!Array.isArray(engram) || !engram.every((i) => Number.isInteger(i) && i >= 0 && i < layerCount)) {
      fail('engram_layer_ids가 0..num_hidden_layers-1 정수 배열이 아니에요');
    }
    if (engram.length) {
      tokenStateBytes = 8; // NgramHashState.cache (max_seq_len,) int64
      const rows = c.engram_num_embeddings;
      if (!Array.isArray(rows) || rows.length !== engram.length || !rows.every((r) => Number.isSafeInteger(r) && r > 0) || !posInt(c.engram_head_dim)) {
        fail('engram_num_embeddings·engram_head_dim이 engram 층마다의 양의 정수가 아니에요');
      }
      engramParams = rows.reduce((a, r) => a + r * c.engram_head_dim, 0); // ParallelEngramEmbedding(rows, engram_head_dim)
    }
    if (!(c.dspark_block_size == null || (Number.isInteger(c.dspark_block_size) && c.dspark_block_size >= 0))) {
      fail('dspark_block_size 값이 비음수 정수가 아니에요');
    }
    // DSpark 블록은 dspark_block_size가 있을 때만 만들어진다(Transformer.__init__) — 없으면 MTP 층 창도 없다.
    if (!c.dspark_block_size) windowLayers = layerCount;
    else if (nextn > 0) routedExperts.push(c.dspark_n_routed_experts || c.n_routed_experts); // ModelArgs.get_moe_config
    if (!routedExperts.every(posInt)) fail('dspark_n_routed_experts 값이 양의 정수가 아니에요');
  }
  const caches = [];
  for (const [ratio, layers] of [...byRatio].sort((a, b) => a[0] - b[0])) {
    caches.push({ ratio, dim: headDim, layers }); // 주 KV(잠재 head_dim, K=V 공유)
    // 인덱서 K: V4는 ratio 4 층만, V4.1은 모든 KV 소스(Indexer.owns_k)
    if (family === 'deepseek_v41' || ratio === 4) caches.push({ ratio, dim: indexDim, layers });
  }
  return {
    compressedKv: {
      family, windowSize: c.sliding_window, windowLayers, headDim, caches, stateBytes, tokenStateBytes,
      workspaceBytes: DEEPSEEK_INDEXER_WORKSPACE_BYTES,
      kvLayers: [...byRatio.values()].reduce((s, n) => s + n, 0), oGroups: c.o_groups, indexHeads: c.index_n_heads,
      vocabSize: c.vocab_size, routedExperts,
    },
    engramParams: engramParams ? engramParams / 1e9 : undefined,
  };
}

// config 치수로 파라미터 수의 자릿수를 재구성한다(이름 추정 교차검증용, 정밀 산출 아님).
function paramsFromDims(c, layerCount) {
  const h = c.hidden_size;
  if (!h || !layerCount) return null;
  const attnHeads = c.num_attention_heads || 0;
  const kvHeads = c.num_key_value_heads ?? attnHeads;
  const hd = c.head_dim ?? (attnHeads ? Math.round(h / attnHeads) : 128);
  const qDim = attnHeads * hd;
  const kvDim = kvHeads * hd;
  const attn = h * qDim + 2 * h * kvDim + qDim * h; // q + k,v + o
  const nExp = c.num_local_experts || c.num_experts || c.n_routed_experts || 0;
  const inter = nExp
    ? (c.moe_intermediate_size || 0) * (nExp + (c.n_shared_experts || c.num_shared_experts || 0))
    : c.intermediate_size || 0;
  const ffn = 3 * h * inter; // gate + up + down
  const embed = c.vocab_size ? c.vocab_size * h * (c.tie_word_embeddings ? 1 : 2) : 0;
  return (layerCount * (attn + ffn) + embed) / 1e9;
}

// checkpoint/evidence 자릿수 검증에 쓸 수 있는 config에서만 구조 추정치를 만든다.
// 전문가 배치가 일부 레이어에만 적용되는 MoE는 현재 근사식이 그 배치를 모델링하지 못한다.
function checkpointSanityParams(c, layerCount) {
  const nExp = c.num_local_experts || c.num_experts || c.n_routed_experts || 0;
  const hasMoePlacement = nExp && (
    c.moe_layer_freq != null || c.decoder_sparse_step != null ||
    c.mlp_only_layers != null || c.first_k_dense_replace != null ||
    c.interleave_moe_layer_step != null || c.n_dense_first_layers != null
  );
  const usable = !hasMoePlacement &&
    c.hidden_size && c.vocab_size && (nExp ? c.moe_intermediate_size : c.intermediate_size);
  return usable ? paramsFromDims(c, layerCount) : null;
}

// 저장 element와 논리 파라미터가 1:1이고 폭이 확정된 dtype만 여기 넣는다 — 이 집합에 있으면
// shard byte 등식이라는 강한 검증을 받는다. FP8은 packing 컨테이너가 아니라 1 element = 1 byte라
// 포함한다(DeepSeek-V3-FP8 류가 약한 밴드 대신 등식 검증을 받게 된다).
// ⚠ I8/U8/I32/U32는 제외 — GPTQ(I32)·NF4(U8)가 여러 값을 packing하는 컨테이너라 폭이 확정되지
// 않는다. 이들은 bytes/parameter 밴드로만 검증한다.
const ONE_TO_ONE_SAFETENSORS_DTYPES = new Map([
  ['F64', 8], ['F32', 4], ['F16', 2], ['BF16', 2],
  ['F8_E4M3', 1], ['F8_E4M3FN', 1], ['F8_E5M2', 1], ['F8_E8M0', 1],
  ['I64', 8], ['U64', 8], ['I16', 2], ['U16', 2], ['BOOL', 1],
]);
const COMMIT_SHA_RE = /^(?:[a-f0-9]{40}|[a-f0-9]{64})$/i;

// HF model API의 safetensors.parameters는 dtype별 *논리 파라미터 수*다. GPTQ/AWQ처럼
// I32 컨테이너에 여러 값을 packing한 레포도 원래 shape의 논리 수를 보고한다. 모든 dtype의
// 합계를 쓰되, float-only는 shard bytes 등식으로, 그 밖은 bytes/parameter 안전 밴드로 검증한다.
// https://huggingface.co/docs/huggingface_hub/package_reference/hf_api#huggingface_hub.hf_api.HfApi.get_safetensors_metadata
// https://huggingface.co/api/models/Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4/revision/e9c932ac1893a49ae0fc497ad6e1e86e2e39af20
export function resolveParameterCount({
  checkpointBytes,
  safetensorsParameters,
  safetensorsTotal,
  revision,
} = {}) {
  if (safetensorsParameters == null) return null;
  if (!COMMIT_SHA_RE.test(String(revision || ''))) {
    throw new Error('Hugging Face parameter evidence의 revision이 immutable commit SHA가 아니에요');
  }
  if (!safetensorsParameters || typeof safetensorsParameters !== 'object' || Array.isArray(safetensorsParameters)) {
    throw new Error('Hugging Face safetensors 파라미터 증거가 객체가 아니에요');
  }
  const entries = Object.entries(safetensorsParameters);
  if (!entries.length) throw new Error('Hugging Face safetensors 파라미터 증거가 비어 있어요');

  let totalParams = 0;
  let tensorBytes = 0;
  let nonOneToOneParams = 0;
  let allOneToOne = true;
  for (const [rawDtype, count] of entries) {
    const dtype = String(rawDtype).toUpperCase();
    const bytes = ONE_TO_ONE_SAFETENSORS_DTYPES.get(dtype);
    if (!(Number.isSafeInteger(count) && count >= 0)) {
      throw new Error(`safetensors 파라미터 수(${rawDtype})가 유효한 비음수 정수가 아니에요`);
    }
    totalParams += count;
    if (bytes) tensorBytes += count * bytes;
    else {
      nonOneToOneParams += count;
      allOneToOne = false;
    }
    if (!Number.isSafeInteger(totalParams) || !Number.isSafeInteger(tensorBytes)) {
      throw new Error('safetensors 파라미터 증거가 안전한 정수 범위를 벗어났어요');
    }
  }
  if (!(totalParams > 0)) throw new Error('safetensors 파라미터 합계가 양수가 아니에요');
  if (safetensorsTotal != null &&
      (!(Number.isSafeInteger(safetensorsTotal) && safetensorsTotal > 0) || safetensorsTotal !== totalParams)) {
    throw new Error('safetensors 파라미터 dtype별 합계와 total 합계가 서로 달라요');
  }

  if (checkpointBytes != null) {
    if (!(Number.isSafeInteger(checkpointBytes) && checkpointBytes > 0)) {
      throw new Error('체크포인트 shard byte 합계가 유효한 양의 정수가 아니에요');
    }
    if (allOneToOne) {
      // safetensors 파일은 tensor data 외에 JSON header를 가진다. header slack은 최소 1MB,
      // 데이터의 1%, 절대 64MB 중 가장 엄격한 상한으로 제한한다.
      const headerSlack = checkpointBytes - tensorBytes;
      const maxHeaderSlack = Math.min(Math.max(1_000_000, tensorBytes * 0.01), 64_000_000);
      if (headerSlack < 0 || headerSlack > maxHeaderSlack) {
        throw new Error('safetensors 파라미터와 실제 checkpoint shard byte 합계가 서로 설명되지 않아요');
      }
    } else {
      // packed/혼합 레포는 auxiliary scale·zero-point 때문에 등식 검증이 불가능하다. 대신
      // API가 저장 element 수로 의미를 바꿀 때 생기는 조용한 2–8배 과소계산을 넓은 밴드로 차단한다.
      const minCheckpointBytes = tensorBytes + nonOneToOneParams * 0.1;
      const maxDataBytes = tensorBytes + nonOneToOneParams * 2.5;
      const maxCheckpointBytes = maxDataBytes + Math.max(64_000_000, maxDataBytes * 0.1);
      if (checkpointBytes < minCheckpointBytes || checkpointBytes > maxCheckpointBytes) {
        throw new Error('packed safetensors 파라미터 수와 checkpoint bytes/parameter 비율이 설명되지 않아요');
      }
    }
  }

  return {
    totalParamsB: totalParams / 1e9,
    tensorBytes: allOneToOne ? tensorBytes : null,
    source: 'hf-safetensors-parameters',
    confidence: 'exact',
    validation: checkpointBytes == null ? 'config-required' : allOneToOne ? 'byte-equality' : 'byte-band',
  };
}

// HF 즉석 파싱도 카탈로그 행과 같은 FP4 상한(weightBytes)을 쓴다 — 출시 FP4 전문가 수를 safetensors 증거에서 읽되,
// 샤드 바이트가 그 수로 정확히 설명될 때만 받는다. 선언: quant_method 'mxfp4'(gpt-oss — U8 컨테이너) ·
// expert_dtype 'fp4'(DeepSeek-V4 계열 — I8 컨테이너). 등식: FP4 수 × 4.25/8 + 1:1 dtype 바이트(+ V4.1 Engram의 32원소당
// E8M0 스케일) ≤ 샤드 합 ≤ 그 값 + 헤더 여유(resolveParameterCount의 1:1 상한과 같다). 실측: gpt-oss-120b 77 KB · V4.1 17.9 MB.
// 확인되지 않으면 상한을 두지 않는다 — 전부 선택한 정밀도로 세는 종래 계산(크게 세는 쪽)이 남는다.
function nativeFp4FromEvidence({ container, safetensorsParameters, checkpointBytes, extraBytes = 0 }) {
  if (!container || !safetensorsParameters || !(Number.isSafeInteger(checkpointBytes) && checkpointBytes > 0)) return undefined;
  let fp4 = 0;
  let bytes = extraBytes;
  for (const [rawDtype, count] of Object.entries(safetensorsParameters)) {
    const dtype = String(rawDtype).toUpperCase();
    if (dtype === container) { fp4 += count; continue; }
    const width = ONE_TO_ONE_SAFETENSORS_DTYPES.get(dtype);
    if (!width) return undefined; // 다른 packed 컨테이너가 섞이면 어느 수가 FP4인지 확정할 수 없다
    bytes += count * width;
  }
  if (!(fp4 > 0)) return undefined;
  const asOneByte = bytes + fp4; // 같은 컨테이너를 원소당 1바이트(8bit 정수)로 읽은 크기
  bytes += fp4 * (FP4_STORAGE_BPW / 8);
  const slack = checkpointBytes - bytes;
  const maxSlack = Math.min(Math.max(1_000_000, bytes * 0.01), 64_000_000);
  // 1바이트 해석도 헤더 여유 안에서 맞으면 증거가 FP4를 가려내지 못한다 — 할인하지 않는다(Codex 2차 검수: U8 100M + BF16 6.9B)
  if (asOneByte - checkpointBytes <= maxSlack) return undefined;
  return slack >= 0 && slack <= maxSlack ? fp4 / 1e9 : undefined;
}

export function parseHfConfig(id, raw, totalSize, parameterEvidence) {
  // 일반 파싱은 의도적으로 fail-closed다. 익숙한 필드명을 쓰면서 메모리 레이아웃이 다른 config
  // (반복 레이어·혼합 linear/GQA·압축/인덱스 어텐션·멀티모달 프로젝터)는 그럴듯한 오답을 만든다.
  // 숫자를 내지 않는 편이 틀린 fit보다 안전하다.
  const inner = raw.text_config || raw;
  // 계열 판정은 *텍스트 본체*(inner)를 우선한다. 종래엔 OR라서 래퍼만 qwen3_5면 본체가 미지 계열이어도
  // linear 상태 공식이 허용됐다. 양쪽이 다 선언돼 있고 계열이 갈리면 어느 쪽이 참인지 알 수 없어 거부한다.
  if (raw !== inner && raw.model_type && inner.model_type &&
      HYBRID_LINEAR_TYPES.has(String(raw.model_type)) !== HYBRID_LINEAR_TYPES.has(String(inner.model_type))) {
    throw new Error('최상위와 text_config의 model_type 계열이 서로 달라요 — 계산하지 않습니다');
  }
  const isHybridLinear = HYBRID_LINEAR_TYPES.has(String(inner.model_type ?? raw.model_type ?? ''));

  // 멀티모달 래퍼(text_config) 자체는 막지 않는다 — 판정 대상은 *텍스트 본체의 메모리 레이아웃*이고,
  // 가중치는 safetensors 전체 크기(비전 타워 포함)로 잡히므로 전체 체크포인트 기준으로 정합하다.
  // 대신 텍스트 본체가 아래 게이트를 전부 통과해야 하고, 융합(cross-attention) 계열은 KV가
  // 이미지 토큰 쪽으로도 자라기 때문에 별도로 거부한다.
  if (raw.cross_attention_layers || raw.cross_attention_config || inner.cross_attention_layers || inner.cross_attention_config) {
    throw new Error('cross-attention 융합 멀티모달은 아직 지원하지 않아요');
  }
  const c = inner;
  const modelType = familyOf(c.model_type || raw.model_type);
  // MTP는 디코더 본체 밖이라 KV·가중치 계산에 들어가지 않는 메타데이터지만, 공시(structuralAssumptions)의
  // 입력이다. 래퍼(top-level)에만 이 키가 있는 config은 구조 키 정책이 이미 통과시키므로
  // (num_nextn_predict_layers=BENIGN, mtp_num_hidden_layers=qwen3_5 family key) text body만 읽으면
  // 공시가 조용히 사라졌다. text body를 우선하고 래퍼를 fallback으로 정규화한다.
  // 읽는 스코프가 늘었으므로 값 검증도 같은 스코프에 적용한다 — 검증 없는 새 입력 경로는 만들지 않는다.
  const mtpScopes = raw === inner ? [c] : [c, raw];
  for (const key of ['num_nextn_predict_layers', 'mtp_num_hidden_layers']) {
    for (const scope of mtpScopes) {
      if (scope[key] != null && !(Number.isInteger(scope[key]) && scope[key] >= 0)) {
        throw new Error(`config의 ${key} 값이 비음수 정수가 아니에요 — 계산하지 않습니다`);
      }
    }
  }
  const mtpField = (key) => (c[key] != null ? c[key] : (raw !== inner ? raw[key] : undefined)) || 0;
  const mtpLayerCount = Math.max(mtpField('num_nextn_predict_layers'), mtpField('mtp_num_hidden_layers')) || undefined;
  if ((c.num_loops || 1) !== 1) {
    throw new Error('반복 레이어(num_loops) 아키텍처는 아직 지원하지 않아요');
  }
  // DeepSeek-V4 계열(deepseek_v4 · deepseek_v41)만 압축·인덱스 키를 deepseekCompressedKv로 모델링한다. 같은 키를 쓰는
  // 다른 계열(GLM DSA 등)은 그대로 거부한다 — 키 이름이 같아도 캐시 배치가 같다는 보장이 없다.
  const dsFamily = modelType === 'deepseek_v4' || modelType === 'deepseek_v41' ? modelType : null;
  if (c.block_configs || c.gqa_layers || c.linear_attn_config ||
      (!dsFamily && (c.compress_ratios || c.num_hash_layers || c.index_topk))) {
    throw new Error('비표준 압축·희소·혼합 어텐션 아키텍처는 아직 지원하지 않아요');
  }
  // 공식 DeepSeek config에는 layer_types가 없다(참조 코드는 compress_ratios만 읽는다). 둘 다 있으면 다른 구현용 config다.
  if (dsFamily && c.layer_types != null) {
    throw new Error('DeepSeek-V4 계열 config에 layer_types가 함께 있어 어느 배치가 참인지 알 수 없어요 — 계산하지 않습니다');
  }
  // Mamba/SSM 블록 스케줄(Nemotron-H류)은 layer_types가 아니라 layers_block_type을 쓴다.
  // 이걸 못 보면 전 레이어를 full attention으로 계산해 KV를 수 배 과대 산정한다(실증: Nemotron-3.5-Lightning 52L 중 attention 6L).
  if (Array.isArray(c.layers_block_type) || Array.isArray(c.block_types) || c.ssm_state_size || c.mamba_num_heads) {
    throw new Error('Mamba/SSM 블록 스케줄 아키텍처는 아직 지원하지 않아요');
  }
  // MLA + 선형/그룹 하이브리드(bailing_hybrid류) — MLA 균일 경로로 계산하면 틀린다.
  if (c.kv_lora_rank && (c.num_kv_heads_for_linear_attn != null || c.layer_group_size || c.short_conv_kernel_size)) {
    throw new Error('MLA + 선형 어텐션 혼합 아키텍처는 아직 지원하지 않아요');
  }
  // 모르는 레이어 타입이 하나라도 있으면 거부(허용목록) — 'conv' 등.
  if (Array.isArray(c.layer_types)) {
    const unknown = [...new Set(c.layer_types.map(String))].filter((t) => !KNOWN_LAYER_TYPES.has(t));
    if (unknown.length) {
      throw new Error(`알 수 없는 어텐션 레이어 타입(${unknown.join(', ')})은 아직 지원하지 않아요`);
    }
  }
  // 레이어별로 *다른* 윈도우 크기(sliding_windows 복수형, ExaoneMoe류). 단수 sliding_window 하나로
  // 표현할 수 없다. 이걸 못 보면 sliding=0이 되어 아래 분기가 슬라이딩 레이어를 'KV 없는 linear'로
  // 취급한다 → KV 과소계산 = 거짓 "fits"(실증: K-EXAONE-2.0 78레이어 중 20레이어만 KV 보유로 산출).
  if (c.sliding_windows || c.mtp_sliding_windows) {
    throw new Error('레이어별로 다른 슬라이딩 윈도우(sliding_windows)는 아직 지원하지 않아요');
  }
  // 미지 구조 필드 게이트(#87) — 구조에 영향 줄 수 있는 이름인데 의미를 모르는 키가 있으면 거부.
  // 특정 실패 패턴을 막는 위 검사들보다 뒤에 둔다(그쪽이 더 정확한 메시지를 준다).
  const familyKeys = FAMILY_STRUCTURAL_KEYS.get(modelType);
  // 두 스코프를 모두 스캔한다 — 종래엔 text_config만 봐서 최상위에 구조 필드를 두면 그대로 통과했다.
  const scanScopes = raw === inner ? [inner] : [inner, raw];
  const unknownStructural = [...new Set(scanScopes.flatMap((s) => Object.keys(s)))].filter(
    (k) =>
      STRUCTURAL_KEY_RE.test(k) &&
      !ENGINE_READ_KEYS.has(k) &&
      !BENIGN_STRUCTURAL_KEYS.has(k) &&
      !(c.kv_lora_rank && MLA_ONLY_KEYS.has(k)) &&
      !(familyKeys && familyKeys.has(k))
  );
  if (unknownStructural.length) {
    throw new Error(
      `구조를 알 수 없는 config 필드(${unknownStructural.slice(0, 4).join(', ')}` +
        `${unknownStructural.length > 4 ? ` 외 ${unknownStructural.length - 4}개` : ''})가 있어 계산하지 않아요`
    );
  }
  const layerCount = c.num_hidden_layers;
  if (!layerCount) throw new Error('config에 num_hidden_layers 없음');

  // 값 검증 — 종래엔 필드가 "있는지"만 보고 값은 안 봤다. 그래서 0·음수·소수·문자열이 그대로
  // 계산에 들어갔다. num_key_value_heads: 0 이면 calcKVCache의 falsy 가드가 KV를 0으로 돌려주고,
  // num_attention_heads가 아예 없으면 kvHeads가 1로 떨어져 32-head MHA를 32배 과소계산한다.
  // 둘 다 거짓 "fits" 방향이다.
  const posInt = (v) => Number.isInteger(v) && v > 0;
  if (!posInt(layerCount)) throw new Error('num_hidden_layers 값이 양의 정수가 아니에요');
  if (!posInt(c.num_attention_heads)) {
    throw new Error('config에 num_attention_heads가 없거나 값이 양의 정수가 아니에요 — KV 치수를 추정하지 않습니다');
  }
  // 계산에 산술로 들어가는 값은 전부 검증한다 — 하나라도 빠지면 음수·NaN이 그대로 흘러가
  // KV가 음수가 되거나(예: global_head_dim: -1024 → KV -63.9 GiB) 상태가 NaN이 되어 'yes'가 나온다.
  for (const k of [
    'num_key_value_heads', 'head_dim', 'hidden_size', 'kv_lora_rank', 'qk_rope_head_dim',
    'global_head_dim', 'max_position_embeddings',
    'linear_num_key_heads', 'linear_num_value_heads', 'linear_key_head_dim',
    'linear_value_head_dim', 'linear_conv_kernel_dim',
  ]) {
    if (c[k] != null && !posInt(c[k])) {
      throw new Error(`config의 ${k} 값이 양의 정수가 아니에요 — 계산하지 않습니다`);
    }
  }
  // layer_types가 배열이 아니면 타입·길이 검사를 모두 우회한다(문자열이면 .filter가 없어 다른 경로로 샌다).
  if (c.layer_types != null && !Array.isArray(c.layer_types)) {
    throw new Error('layer_types가 배열이 아니에요 — 계산하지 않습니다');
  }
  // sliding_window만은 0을 허용한다 — "슬라이딩 비활성"의 관례적 표기다(아래 slidingActive가 판정).
  if (c.sliding_window != null && c.sliding_window !== 0 && !posInt(c.sliding_window)) {
    throw new Error('config의 sliding_window 값이 양의 정수가 아니에요 — 계산하지 않습니다');
  }
  // 두 메타데이터가 어긋나면 어느 쪽이 참인지 알 수 없다.
  if (Array.isArray(c.layer_types) && c.layer_types.length !== layerCount) {
    throw new Error('layer_types 길이가 num_hidden_layers와 달라요 — 계산하지 않습니다');
  }

  const attnHeads = c.num_attention_heads;
  const kvHeads = c.num_key_value_heads ?? attnHeads ?? 1;
  const headDim = c.head_dim ?? (c.hidden_size && attnHeads ? Math.round(c.hidden_size / attnHeads) : 128);

  // muse_spark: 층별 배열 두 개는 KV와 무관하지만, 층 수와 어긋난 배열은 다른 구조의 config다.
  if (modelType === 'muse_spark') {
    const perLayer = (k, ok) => {
      if (c[k] != null && !(Array.isArray(c[k]) && c[k].length === layerCount && c[k].every(ok))) {
        throw new Error(`config의 ${k}가 층 수만큼의 올바른 값 배열이 아니에요 — 계산하지 않습니다`);
      }
    };
    perLayer('layer_rope_theta', (v) => Number.isFinite(v) && v >= 0);
    perLayer('mlp_layer_types', (v) => v === 'sparse' || v === 'dense');
  }

  // DeepSeek-V4 계열: 창·압축 캐시는 compressedKv 하나로 나간다 — sliding_window를 일반 슬라이딩 경로에 넣지 않는다.
  const ds = dsFamily ? deepseekCompressedKv(c, dsFamily, layerCount) : undefined;
  const compressedKv = ds?.compressedKv;

  // 슬라이딩 윈도우가 "실제로" 쓰이는지 판별 (sliding_window 값만 있고 미사용인 모델 오탐 방지)
  const hasSlidingLayers = Array.isArray(c.layer_types) && c.layer_types.some((t) => String(t).includes('sliding'));
  const slidingActive =
    !compressedKv &&
    (c.sliding_window || 0) > 0 &&
    c.use_sliding_window !== false &&
    // layer_types가 있으면 그것이 정본이다 — 전 층 full_attention이면 sliding_window 값이 있어도 슬라이딩이 아니다
    // (transformers: sliding_window = config.sliding_window if layer_types[i] == "sliding_attention" else None).
    // 종래엔 값만 보고 켜서 slidingSplit 기본 5:1로 KV를 1/6로 깎았다(Codex 리뷰 2026-10-18: Muse 전 층 full → 248 대신 40.4 GiB).
    (Array.isArray(c.layer_types) ? hasSlidingLayers : c.sliding_window < (c.max_position_embeddings || Infinity));
  const sliding = slidingActive ? c.sliding_window : 0;
  if (sliding && !Array.isArray(c.layer_types) && !c.kv_lora_rank) {
    throw new Error('레이어별 sliding/full 구성이 없는 모델은 정확히 계산할 수 없어요');
  }
  // 슬라이딩 레이어가 있는데 쓸 수 있는 윈도우 크기를 못 읽으면, 아래 분기가 sliding 레이어를
  // 'KV 없는 linear'로 해석해 그 레이어들의 KV를 통째로 삭제한다 — 거짓 "fits"의 직행로다.
  // sliding_windows(복수형) 말고도 swa_size 등 이름이 다른 경우가 있어 개별 필드명이 아니라
  // "윈도우를 못 읽었다"는 상태 자체로 막는다.
  if (hasSlidingLayers && !(sliding > 0)) {
    throw new Error('슬라이딩 어텐션 레이어가 있는데 윈도우 크기를 읽을 수 없어 계산하지 않아요');
  }

  // layer_types로 full attention 레이어 수 파악 (하이브리드/슬라이딩 정확도)
  let fullAttnLayers, globalAttnLayers, linearAttn;
  if (Array.isArray(c.layer_types)) {
    const linearCount = c.layer_types.filter((t) => String(t).includes('linear')).length;
    if (linearCount > 0) {
      // 선형/재귀 어텐션은 고정 상태 메모리를 갖는다 — 치수를 모르면 계산할 수 없다.
      // linear_num_key_heads가 빠져도 통과하던 구멍이 있었다 — 그러면 상태 크기가 NaN이 되고,
      // NaN은 free < 0 비교를 통과해 verdict가 'yes'로 떨어진다. 다섯 치수를 모두 요구한다.
      if (!isHybridLinear || !c.linear_num_key_heads || !c.linear_num_value_heads ||
          !c.linear_key_head_dim || !c.linear_value_head_dim) {
        throw new Error('linear/recurrent attention의 고정 상태 메모리는 HuggingFace 즉석 계산에서 지원하지 않아요');
      }
      linearAttn = {
        layers: linearCount,
        numKHeads: c.linear_num_key_heads,
        numVHeads: c.linear_num_value_heads,
        headKDim: c.linear_key_head_dim,
        headVDim: c.linear_value_head_dim,
        convKernel: c.linear_conv_kernel_dim || 4,
      };
    }
    const full = c.layer_types.filter((t) => String(t).includes('full')).length;
    if (full > 0 && full < layerCount) {
      if (sliding > 0) globalAttnLayers = full; // 슬라이딩: full = 글로벌 레이어
      else fullAttnLayers = full; // 하이브리드 linear: full만 KV 보유
    }
  }

  const numExperts = c.num_local_experts || c.num_experts || c.n_routed_experts;
  const expertsPerToken = c.num_experts_per_tok;
  const isMoe = !!numExperts;

  // PLE(Per-Layer Embeddings, Gemma e2b/e4b류) 감지 — config에 두 필드가 있으면 텐서 크기 결정론 산출.
  // 검증된 gemma4_text 계열에 한해 pinned llama.cpp/GGUF 경로(입력층 텐서 → CPU/host 버퍼 배치)에서만 GPU weight
  // 차감 대상이 된다(residentParamsB·pleOffloadVerified 참조). 그 밖의 계열은 전체 상주로 fail-close.
  // 카탈로그의 손계산 값과 같은 식. 두 필드가 양의 안전 정수이고 곱도 안전 정수일 때만 산출한다 —
  // 문자열·0·음수·소수·NaN·Infinity·오버플로는 종래 `*` 강제변환으로 그럴듯한 수가 되거나 NaN이 흘렀다.
  const plePosInt = (v) => Number.isSafeInteger(v) && v > 0;
  const pleWellFormed = plePosInt(c.vocab_size_per_layer_input) && plePosInt(c.hidden_size_per_layer_input) &&
    Number.isSafeInteger(c.vocab_size_per_layer_input * c.hidden_size_per_layer_input * layerCount);
  const pleParams = pleWellFormed
    ? (c.vocab_size_per_layer_input * c.hidden_size_per_layer_input * layerCount) / 1e9
    : undefined;

  // MLA(Multi-head Latent Attention) 감지 — kv_lora_rank 있으면 압축 KV 경로(GLM-5.2/GLM-4.7-Flash 등).
  // ⚠ DeepSeek-V4류(kv_lora_rank 부재 + MQA/compressor)는 MLA 아님 → compressedKv 경로(deepseekCompressedKv).
  const mlaKvLoraRank = c.kv_lora_rank || undefined;
  const mlaRopeDim = mlaKvLoraRank ? (c.qk_rope_head_dim || 0) : undefined;

  // 슬라이딩 패턴 라벨은 layer_types 실카운트로 유도한다(gcd 약분 sliding:full). 종래엔 슬라이딩이면 무조건 '5:1'
  // (Gemma 4 관례)이라 Laguna(30:10)·Spark(27:9) 같은 3:1 배치에도 '5:1'이 붙었다. 라벨은 표시·감사용이고
  // KV 계산은 globalAttnLayers(실카운트)를 쓰므로 바이트는 불변. 카운트를 못 얻는 경우(full 0 또는 전층)는
  // slidingSplit이 실제로 5:1 비율로 나누므로 라벨도 '5:1'을 유지한다 — 계산과 라벨이 어긋나지 않게.
  let slidingPattern;
  if (sliding && !mlaKvLoraRank) {
    const slidingCount = Array.isArray(c.layer_types) ? c.layer_types.filter((t) => String(t).includes('sliding')).length : 0;
    slidingPattern = globalAttnLayers && slidingCount ? slidingPatternLabel(slidingCount, globalAttnLayers) : '5:1';
  }

  // 파라미터 수: safetensors total_size(저장 dtype 바이트)에서 역산, 없으면 이름 추정
  let totalParams = null;
  let parameterSource = null;
  if (totalSize != null && !(Number.isFinite(totalSize) && totalSize > 0)) {
    throw new Error('체크포인트 크기(total_size)가 유효한 양수가 아니에요 — 계산하지 않습니다');
  }
  const resolvedEvidence = resolveParameterCount({
    checkpointBytes: totalSize,
    safetensorsParameters: parameterEvidence?.safetensorsParameters,
    safetensorsTotal: parameterEvidence?.safetensorsTotal,
    revision: parameterEvidence?.revision,
  });
  if (resolvedEvidence) {
    if (totalSize == null) {
      const dims = checkpointSanityParams(c, layerCount);
      if (!dims) {
        throw new Error('checkpoint byte 합계 없이 safetensors 파라미터 증거를 교차검증할 수 없어 계산하지 않아요');
      }
      if (resolvedEvidence.totalParamsB > dims * 4 || resolvedEvidence.totalParamsB < dims / 4) {
        throw new Error('safetensors 파라미터 수가 config 구조 추정과 자릿수가 달라 계산하지 않아요');
      }
    }
    if (resolvedEvidence.validation === 'byte-band') {
      const dims = checkpointSanityParams(c, layerCount);
      if (dims && (resolvedEvidence.totalParamsB > dims * 2 || resolvedEvidence.totalParamsB < dims / 2)) {
        throw new Error('packed safetensors 파라미터 수가 config 구조 추정과 달라 계산하지 않아요');
      }
    }
    totalParams = resolvedEvidence.totalParamsB;
    parameterSource = resolvedEvidence.source;
  }
  // DeepSeek-V4 계열 체크포인트는 FP4 전문가 + FP8 나머지 혼합이다. quantization_config는 fp8(1B)만 선언하므로 바이트에서
  // 역산하면 FP4 전문가를 절반으로 센다(V4.1-Flash 510.3GB → 510B, 실제 논리 763.2B) — 거짓 fits 방향이라 증거를 요구한다.
  if (dsFamily && !resolvedEvidence) {
    throw new Error('DeepSeek-V4 계열은 FP4·FP8 혼합 체크포인트라 safetensors 파라미터 증거 없이 파라미터 수를 계산하지 않아요');
  }
  if (totalSize && !resolvedEvidence) {
    // 선-양자화 레포(MLX/AWQ/bnb): 저장 비트폭의 진실은 quantization(.bits) — torch_dtype은 원본 정밀도라
    // ÷2 과소계산 → 거짓 "fits" (issue #2). 합성 재현: 8bit 레포에 qbits 무시 시 params 절반 (test/parsehf.test.mjs).
    // 혼합 정밀도(일부 레이어 상위 bit)는 params 과대 방향으로만 틀림 — 보수적이라 허용.
    // 저장 정보는 멀티모달 래퍼의 *최상위*에만 있는 경우가 있다. inner(text_config)만 보면
    // 4비트를 못 읽고 2바이트로 나눠 4배 과소계산 = 거짓 "fits"가 된다.
    // 실측: Qwen/Qwen3.5-397B-A17B-GPTQ-Int4 는 quantization_config.bits=4가 raw 최상위에 있고
    // text_config는 중첩이라, inner만 읽던 종래 코드가 397B 모델을 117.8B로 계산했다.
    // 양쪽에 다 있고 서로 다르면 어느 쪽이 참인지 알 수 없으므로 거부한다.
    // 저장 정보는 최상위·중첩 어디에나 있을 수 있고 별칭(torch_dtype/dtype, quantization/
    // quantization_config)도 있다. 한 자리만 읽으면 다른 자리로 우회가 생기므로, **모든 선언을 모아**
    // 정규화한 뒤 충돌을 판정한다. (같은 키끼리만 비교하면 최상위 bits=4 + 중첩 quantization.bits=8
    // 같은 별칭 간 충돌을 놓친다.)
    const scopes = raw === inner ? [inner] : [inner, raw];
    const quantObjs = scopes.flatMap((s) => [s.quantization, s.quantization_config]).filter(Boolean);
    const bitsIn = (q) => {
      const declarations = [];
      if (q.bits != null) declarations.push(q.bits);
      if (q.load_in_4bit === true) declarations.push(4);
      if (q.load_in_8bit === true) declarations.push(8);
      const meth = String(q.quant_method || q.fmt || '').toLowerCase();
      if (meth.includes('fp8') || meth.includes('int8')) declarations.push(8);
      return declarations;
    };
    const declaredBits = [...new Set(quantObjs.flatMap(bitsIn))];
    if (declaredBits.length > 1) {
      throw new Error(`양자화 비트폭 선언이 서로 달라요(${declaredBits.join(', ')}) — 계산하지 않습니다`);
    }
    const qbits = declaredBits[0];
    if (qbits !== undefined && !(Number.isInteger(qbits) && qbits >= 1 && qbits <= 64)) {
      throw new Error(`양자화 비트폭(${qbits})이 1–64 사이 정수가 아니에요 — 계산하지 않습니다`);
    }
    const dtypes = [...new Set(
      scopes.flatMap((s) => [s.torch_dtype, s.dtype]).filter((v) => v != null).map((v) => String(v).toLowerCase())
    )];
    if (dtypes.length > 1) {
      throw new Error(`dtype 선언이 서로 달라요(${dtypes.join(', ')}) — 계산하지 않습니다`);
    }
    const dt = dtypes[0] || '';
    // 양자화를 *선언해놓고* 비트폭을 못 읽으면 2바이트로 추정하지 않는다 — 그 추정이 틀리는
    // 방향이 곧 거짓 fits다. 실측: gpt-oss는 dtype도 bits도 없이 quant_method만 "mxfp4"라
    // 종래엔 bf16으로 가정해 117B 모델을 32.6B로 계산했다(3.6배 과소, #98).
    // 반대로 양자화 선언이 아예 없으면 bf16이 HF 관례이므로 종래 기본값을 유지한다.
    if (quantObjs.length && qbits === undefined) {
      const meth = quantObjs.map((q) => q.quant_method || q.fmt).filter(Boolean).join(', ');
      throw new Error(`선언된 양자화(${meth || '방식 미상'})의 저장 비트폭을 확정할 수 없어 계산하지 않아요`);
    }
    if (!qbits && !dt) {
      throw new Error('체크포인트의 저장 dtype이 명시되지 않아 파라미터 수를 계산하지 않아요');
    }
    const dtypeBytes = qbits ? qbits / 8
      : dt.includes('float64') || dt.includes('fp64') ? 8
        : dt.includes('float32') || dt.includes('fp32') ? 4
          : dt.includes('bfloat16') || dt.includes('bf16') || dt.includes('float16') || dt.includes('fp16') ? 2
            : dt.includes('float8') || dt.includes('fp8') || dt.includes('int8') || dt.includes('uint8') ? 1
              : null;
    if (!dtypeBytes) {
      throw new Error(`체크포인트의 저장 dtype(${dt})을 해석할 수 없어 파라미터 수를 계산하지 않아요`);
    }
    totalParams = totalSize / dtypeBytes / 1e9;
    parameterSource = 'uniform-checkpoint';
    // total_size sanity check — 종래엔 아래 fallback 경로에만 교차검증이 있어서, total_size가
    // 있기만 하면 그 값을 무조건 믿었다. 그런데 index의 metadata.total_size가 아예 틀린 레포가 있다.
    // 실증: InternScience/Agents-A1-4B는 shard 2개(텍스트 32L·hidden 2560 = ~4B급 멀티모달)인데
    // total_size가 550.7GB로 적혀 있어 275.3B가 산출됐다 — 2개 샤드에 550GB는 물리적으로 불가능하다.
    //
    // 심판은 레포 *이름*이 아니라 config *치수*로 한다. 이름이 틀리고 크기가 맞는 반대 사례가
    // 실재하기 때문이다(z-lab/Qwen3.8-27B-DFlash2 = 이름 27B, 실제 1.9B 드래프트 헤드).
    // 치수는 레이어 구조와 같은 출처라, 크기 메타데이터와 독립적인 제3의 앵커가 된다.
    // 목적은 '정밀 검증'이 아니라 '자릿수 붕괴 검출'이라 배수 여유를 크게 둔다.
    // 치수 추정이 성립할 때만 심판으로 쓴다 — FFN·임베딩 항이 빠지면 어텐션만 세어 심하게
    // 과소평가되고, 그 상태로 비교하면 정상 모델을 거짓 거부한다.
    // MoE일 때 paramsFromDims는 moe_intermediate_size로만 FFN을 센다. gpt_oss처럼 전문가 FFN 폭을
    // intermediate_size에 담는 config는 FFN이 0으로 잡혀 추정이 자릿수째 무너지므로 심판으로 쓸 수 없다
    // (실증: gpt-oss-120b 추정 ~2.1B vs 실제 32.6B → 거짓 거부).
    // MoE 배치 필드가 있으면 현재 근사식은 전 레이어에 전문가가 있다고 세므로 심판에서 제외한다.
    const dims = checkpointSanityParams(c, layerCount);
    if (dims && (totalParams > dims * 4 || totalParams < dims / 4)) {
      throw new Error(
        `체크포인트 크기로 계산한 파라미터 수(${totalParams.toFixed(1)}B)가 config 구조 추정` +
          `(~${dims.toFixed(1)}B)과 자릿수가 달라요 — 레포의 total_size 메타데이터를 믿을 수 없어 계산하지 않습니다`
      );
    }
  }
  if (!totalParams) {
    // 최후 수단인 이름 추정은 config 치수와 교차검증한다. 레포 이름이 *타깃* 모델을 가리키는
    // 드래프트 헤드·어댑터가 실재한다 — z-lab/Qwen3.8-27B-DFlash2는 5레이어 draft인데 이름은 27B다.
    const named = paramsFromName(id);
    if (!named) throw new Error('safetensors 크기도 이름 단서도 없어 파라미터 수를 구할 수 없어요');
    const dims = paramsFromDims(c, layerCount);
    if (dims && (named > dims * 2 || named < dims * 0.5)) {
      throw new Error(
        `레포 이름의 파라미터 수(${named}B)가 config 구조 추정(~${dims.toFixed(1)}B)과 달라요 — 드래프트/어댑터 레포일 수 있어 계산하지 않습니다`
      );
    }
    totalParams = named;
    parameterSource = 'model-name';
  }

  // PLE GPU 차감 검증 — 계열 허용(PLE_OFFLOAD_FAMILIES)만으로는 부족하다. 세 조건을 모두 요구한다:
  //  ① 선언이 pinned 공식 검증 profile(PLE_VERIFIED_PROFILES) 하나와 완전히 일치한다 — 층 수·hidden_size·
  //     intermediate_size·PLE 두 치수 전부. 밴드 정합만으로는 E2B hidden_size_per_layer_input ≤ 379(PLE +48%)가
  //     통과하고, PLE 치수 두 개만 대조하면 E2B 40·45·46층 부풀리기(PLE 곱의 세 번째 인자)와 E2B 본체+E4B 층 수
  //     같은 profile 혼합이 ②를 통과해 거짓 fits 구멍이 남는다.
  //  ② 체크포인트 총 파라미터에서 PLE를 뺀 잔여가 config 자체가 선언한 dense body(어텐션+FFN+임베딩,
  //     checkpointSanityParams)보다 작지 않다 — 계수 1의 구조 하한. 수축된 total_size(2.4B → 잔여 0.05B)와
  //     극단적 층 수 부풀리기(70층)를 잡는다. 공식 여유: E2B 2.774B−1.641B=1.133B, E4B 5.177B−4.525B=0.653B.
  //     kv-shared 층은 K/V 사영이 없어 body가 과대될 수 있는 최대치는 E2B 15.7M·E4B 47.2M(여유의 1.4%·7.2%).
  //  ③ body 추정을 만들 수 없으면(intermediate_size·vocab_size 부재) 정합을 확인할 수 없어 검증하지 않는다.
  //     body에 들어가는 vocab_size는 pinned 다섯 필드 밖이라 별도로 양의 안전 정수를 요구한다 — 독립 리뷰 R1(2026-09-06):
  //     -1은 truthy라 임베딩 항을 음수로 만들어 하한을 낮추고, 문자열 '262144'·소수 1.5는 산술 강제변환으로 유한한
  //     양수 body가 되어 수축된 체크포인트가 verified로 열렸다. Number.isFinite(body)만으로는 못 막는다.
  //  ④ 하한에 넣는 총 파라미터는 실제 checkpoint/evidence 출처여야 한다 — 독립 리뷰 R2(2026-09-06): 이름 추정
  //     (parameterSource 'model-name')은 checkpoint bytes도 safetensors 증거도 없는데 잔여 하한을 통과해 인증됐다.
  //     이름 추정은 일반 추정으로는 그대로 남기되, 예외적 GPU 차감의 인증에는 검증된 resolvedEvidence 또는
  //     양의 안전 정수 checkpoint bytes에서 역산한 uniform-checkpoint만 허용한다.
  // 실패는 throw가 아니라 미검증(false)이다 — guardedPleParamsB가 이 플래그 하나로 GPU 차감·전제·pleOffloadGB를
  // 모두 닫아 전체 가중치 상주로 계산한다(사용자 결정: 확인되지 않은 부분 상주는 일반 fit으로 표현하지 않는다).
  const pleProfiles = PLE_OFFLOAD_FAMILIES.has(modelType) ? PLE_VERIFIED_PROFILES[modelType] : undefined;
  const pinnedPle = pleParams && pleProfiles
    ? pleProfiles.find((p) =>
      layerCount === p.layerCount && c.hidden_size === p.hiddenSize && c.intermediate_size === p.intermediateSize &&
      c.vocab_size_per_layer_input === p.vocabSizePerLayerInput && c.hidden_size_per_layer_input === p.hiddenSizePerLayerInput)
    : undefined;
  const pleBody = pinnedPle && plePosInt(c.vocab_size) ? checkpointSanityParams(c, layerCount) : null;
  const pleParamsProvenance = Boolean(resolvedEvidence) ||
    (parameterSource === 'uniform-checkpoint' && Number.isSafeInteger(totalSize) && totalSize > 0);
  const pleOffloadVerified = Boolean(
    pinnedPle && pleBody && pleParamsProvenance && totalParams && totalParams - pleParams >= pleBody
  );
  const fp4Scopes = [inner, raw, ...[inner, raw].flatMap((s) => [s.quantization, s.quantization_config])].filter(Boolean);
  const fp4Container = fp4Scopes.some((s) => String(s.quant_method || '').toLowerCase() === 'mxfp4') ? 'U8'
    : fp4Scopes.some((s) => String(s.expert_dtype || '').toLowerCase() === 'fp4') ? 'I8'
      : null;
  const nativeFp4Params = resolvedEvidence
    ? nativeFp4FromEvidence({
      container: fp4Container,
      safetensorsParameters: parameterEvidence?.safetensorsParameters,
      checkpointBytes: totalSize,
      extraBytes: ds?.engramParams ? (ds.engramParams * 1e9) / 32 : 0, // V4.1 Engram FP8 표의 E8M0 스케일(파라미터 수에 안 잡힌다)
    })
    : undefined;
  // GGUF 단계 파일 크기는 텐서 형상·형식에만 달려 있다 — 텐서 목록이 카탈로그 행의 ggufFileSource와 정확히 같은 체크포인트(공식 레포·
  // 미러·같은 형상의 파인튜닝)는 그 행의 공개 단계 크기(ggufFileBytes)를 물려받는다: dtype별 수가 반올림 없이 같고, 형상을 정하는
  // config 값이 모두 같아야 한다. 하나라도 다르면 물려받지 않는다(그때는 FP4 공식만 — 공개 파일이 없는 체크포인트와 같다).
  const evidenceParams = parameterEvidence?.safetensorsParameters;
  const sameCounts = (want) => evidenceParams && Object.keys(evidenceParams).length === Object.keys(want).length &&
    Object.entries(want).every(([dtype, n]) => evidenceParams[dtype] === n);
  const ggufTwin = nativeFp4Params
    ? MODELS.find((m) => m.ggufFileBytes && m.ggufFileSource && m.nativeFp4Params === nativeFp4Params &&
      sameCounts(m.ggufFileSource.parameters) &&
      Object.entries(m.ggufFileSource.config).every(([k, v]) => (c[k] ?? raw[k]) === v))
    : undefined;

  return {
    name: id.split('/').pop(),
    group: 'HuggingFace',
    custom: true,
    sourceId: id,
    parameterSource,
    tags: isMoe ? ['moe'] : ['dense'],
    totalParams: totalParams ? +totalParams.toFixed(1) : null,
    activeParams: null, // MoE 활성 파라미터는 config로 정확 산출 어려움 → 속도만 근사
    layerCount,
    fullAttnLayers,
    globalAttnLayers,
    linearAttn,
    kvHeads,
    kvHeadDim: headDim,
    globalHeadDim: c.global_head_dim || undefined,
    attnHeads,
    hiddenSize: c.hidden_size,
    numExperts,
    expertsPerToken,
    mlaKvLoraRank,
    mlaRopeDim,
    compressedKv,
    engramParams: ds?.engramParams,
    nativeFp4Params,
    fp4ReleaseBytes: nativeFp4Params ? totalSize : undefined, // nativeFp4FromEvidence가 이 바이트로 등식을 확인했다
    ggufFileBytes: ggufTwin ? { ...ggufTwin.ggufFileBytes } : undefined,
    mtpLayerCount,
    pleParams: pleParams ? +pleParams.toFixed(3) : undefined,
    pleOffloadVerified,
    maxContext: c.max_position_embeddings || 131072,
    // MLA가 우선 경로 → MLA 모델엔 sliding 필드 미설정(계산은 MLA 먼저 타지만 dead data 방지, correct-by-construction)
    slidingWindow: mlaKvLoraRank ? undefined : sliding || undefined,
    slidingPattern,
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
// ── 이름 해석 — 모든 표면의 단일 정본 ────────────────────────────────────────
// 표면마다 별도 matcher를 두면 같은 질의에 다른 답이 나온다(계획서 AC-1). 그리고 기존
// `.includes()` 첫-일치는 임의 선택이었다. 실측(2026-09-03 라이브 카탈로그):
//   'llama' → 후보 3개 중 Llama-3.2-3B (사용자가 8B를 의도했을 수 있다)
//   'gemma' → 후보 6개 중 가장 작은 Gemma 4 e2b
//   'qwen'  → 후보 7개 중 Qwen 3.6 27B
// 배열 순서에 따라 *작은 모델로 기우는* 편향이고, 작은 모델은 메모리를 덜 먹으니 판정이
// fits 쪽으로 틀린다 — 거짓 FITS와 같은 방향의 실패다. 그래서 모호하면 고르지 않고 후보를 돌려준다.
// ⚠ 카탈로그 배열 순서는 `?m=` 공유링크·OG 이미지가 인덱스로 참조한다(api/og.js) — 재정렬 금지.
export function normalizeNameTokens(value) {
  return String(value ?? '')
    .normalize('NFKC')
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter(Boolean);
}

const nameKey = (value) => normalizeNameTokens(value).join(' ');
// 별칭(aliases): 같은 칩·같은 메모리라 판정이 같은 다른 제품명(예: GB10 파트너 시스템, RTX PRO 6000 에디션).
// 별칭으로 찾아도 답은 정본 행(item.name) 하나다 — 별칭마다 행을 따로 두면 같은 기계가 표면마다 여러 줄로 늘어난다.
// ⚠ 별칭은 **이름 전체가 같을 때만** 해석한다(정확 일치·붙여 쓰기 정확 일치). 토큰 포함 일치에 별칭을 쓰면
//   'NVIDIA'·'ASUS'·'Dell' 같은 제조사 토큰 하나가 그 별칭을 가진 유일한 행으로 확정돼 거짓 FITS가 난다
//   (2026-10-16 Codex 리뷰 재현: --gpu NVIDIA → DGX Spark 128GB, exit 0). 부분 일치 후보 제시에는 써도 된다(해석은 안 함).
const namesOf = (item) => [item.name, ...(Array.isArray(item.aliases) ? item.aliases : [])];
// `org/repo` 별칭(공식 Hugging Face ID)은 토큰이 아니라 **ID 전체**로만 맞춘다(대소문자 무시). 토큰 키는 '/'·'-'·'_'를 모두 버려
// 'deepseek/ai-DeepSeek-V4-Flash'·'meta_models/Muse-…' 같은 **다른 레포**가 공식 별칭과 같은 키가 됐다(2.21.1 Codex 리뷰 재현:
// 확인 없는 레포가 카탈로그 행으로 둔갑해 exit 0). 일반 이름 별칭은 그대로 표기 차이를 흡수한다.
const isRepoId = (name) => String(name).includes('/');
const sameRepoId = (a, b) => String(a).trim().toLowerCase() === String(b ?? '').trim().toLowerCase();
const nameMatches = (name, query, key) => (isRepoId(name) ? sameRepoId(name, query) : nameKey(name) === key);

// -> { status: 'resolved',  match, canonicalName, matchedBy: 'exact'|'tokens' }
//  | { status: 'ambiguous', candidates: [{ name }], total }
//  | { status: 'unknown',   candidates: [{ name }], total }
function resolveByTokens(list, query, { limit = 5 } = {}) {
  const tokens = normalizeNameTokens(query);
  if (!tokens.length) return { status: 'unknown', candidates: [], total: 0 };

  const key = tokens.join(' ');
  const exact = list.find((item) => namesOf(item).some((n) => nameMatches(n, query, key)));
  if (exact) return { status: 'resolved', match: exact, canonicalName: exact.name, matchedBy: 'exact' };

  // 질의 토큰 전체를 품는 항목만 후보(정본 이름 기준 — 별칭은 위 정확 일치에서만). 유일할 때만 해석하고, 복수면 고르지 않는다.
  // 용량 토큰('48gb'·'128gb')만으로는 기기를 확정하지 않는다 — 이름에 용량이 적힌 행만 후보가 되어 같은 용량의 다른 기기를
  // 못 본다(2026-10-16 Codex 2차 리뷰: --gpu 48gb → RTX PRO 5000 48GB TIGHT, 같은 48GB인 2× RTX 4090은 NO).
  // matchRequires: 그 행을 토큰 일치로 **확정**하려면 질의에 반드시 있어야 하는 토큰(예: 'pro'+'4500' — NVIDIA는 같은 번호를
  // 'PRO' 없이 Ada·Turing 카드에 썼다. 'RTX 4500'은 24GB RTX 4500 Ada이지 32GB RTX PRO 4500 Blackwell이 아니다).
  // ⚠ 후보에서 빼는 게 아니라 확정 자격만 막는다 — 빼면 남은 한 행이 모호성 없이 확정된다
  //   (Codex 3차 리뷰: 'blackwell' → RTX PRO 6000 96GB로 확정, 'pro 48gb' → PRO 5000 48GB).
  const identifying = tokens.filter((t) => !/^\d+(?:gb|tb)$/.test(t));
  const full = identifying.length === 0 ? [] : list.filter((item) => {
    const own = new Set(normalizeNameTokens(item.name));
    return tokens.every((t) => own.has(t));
  });
  const eligible = (item) => (Array.isArray(item.matchRequires) ? item.matchRequires : []).every((t) => tokens.includes(t));
  if (full.length === 1) {
    if (!eligible(full[0])) return { status: 'unknown', candidates: [{ name: full[0].name }], total: 1 };
    return { status: 'resolved', match: full[0], canonicalName: full[0].name, matchedBy: 'tokens' };
  }
  if (full.length > 1) {
    // 후보 중 "나머지 전부가 확장하는 유일한 최소 이름"이 있으면 그것으로 해석한다.
    // 'RTX 4090'은 '2× RTX 4090' 리그 항목의 부분집합이라 gpu=4090은 단일 카드로 해석된다
    // (문서가 광고하는 질의). 반면 'A100 40GB'와 'A100 80GB'는 서로 부분집합이 아니고
    // 'Gemma 4 12b'와 'Gemma 4 31b'도 아니라서 그대로 ambiguous — 임의 선택이 아니다.
    const sets = full.map((m) => new Set(normalizeNameTokens(m.name)));
    const minimal = full.filter((_, i) => sets.every((other, j) => j === i || [...sets[i]].every((t) => other.has(t))));
    if (minimal.length === 1 && eligible(minimal[0])) {
      return { status: 'resolved', match: minimal[0], canonicalName: minimal[0].name, matchedBy: 'base-name' };
    }
    return { status: 'ambiguous', candidates: full.slice(0, limit).map((m) => ({ name: m.name })), total: full.length };
  }

  // 전체 일치가 없으면 부분 일치를 *후보로만* 제시한다 — 첫 항목을 답으로 승격하지 않는다.
  const partial = list.filter((item) => namesOf(item).some((n) => {
    const own = new Set(normalizeNameTokens(n));
    return tokens.some((t) => own.has(t));
  }));
  return { status: 'unknown', candidates: partial.slice(0, limit).map((m) => ({ name: m.name })), total: partial.length };
}

// 2차 패스: 붙여 쓴 카탈로그 이름(2026-10-08 감사 — 'Qwen3.8-27B'가 unknown이었다). 1차가 unknown이고
// `org/model`이 아닐 때만, 아래 키가 **정확히 한 항목과 같을 때만** 해석한다. 토큰 포함(부분) 일치는 쓰지 않는다 —
// 그러면 'Qwen3-35B-A3B'가 Qwen 3.6으로, 'MiniCPM5.5-1B'가 MiniCPM5-1B로 둔갑한다(Codex 리뷰 실측).
// 키: 소수점 버전은 한 토큰('3.8')으로 남겨 'Qwen3-8.27B'와 'Qwen3.8-27B'를 가르고, 영문 2자 이상 접두에
// 붙은 숫자만 가른다('qwen3.8' → 'qwen','3.8'). e2b·a3b·a100 같은 단일문자 접두는 그대로 둔다.
// 점으로 시작·끝나거나 점이 겹친 토큰('.27b' = 0.27B)은 뜻을 단정할 수 없어 키를 만들지 않는다(Codex 2차 리뷰).
function gluedNameKey(value) {
  const tokens = String(value ?? '')
    .normalize('NFKC')
    .toLowerCase()
    .split(/[^a-z0-9.]+/)
    .filter(Boolean);
  if (tokens.some((t) => /^\.|\.$|\.\./.test(t))) return null;
  return tokens
    .flatMap((t) => {
      const m = /^([a-z]{2,})(\d.*)$/.exec(t);
      return m ? [m[1], m[2]] : [t];
    })
    .join(' ');
}

export function resolveByName(list, query, opts) {
  const first = resolveByTokens(list, query, opts);
  if (first.status !== 'unknown' || String(query ?? '').includes('/')) return first;
  const key = gluedNameKey(query);
  if (key === null) return first;
  const hits = list.filter((item) => namesOf(item).some((n) => !isRepoId(n) && gluedNameKey(n) === key));
  if (hits.length !== 1) return first;
  return { status: 'resolved', match: hits[0], canonicalName: hits[0].name, matchedBy: 'exact' };
}

export const resolveLocalModel = (query, opts) => resolveByName(LOCAL_MODELS, query, opts);
export const resolveGpuByName = (query, opts) => resolveByName(GPUS, query, opts);

export const DATA_UPDATED = '2026-10';

// 이 엔진 스냅샷의 버전 — package.json version과 같이 올린다.
// 소비처(v2 영수증 /api/r 등)가 자기 package.json 버전을 엔진 버전으로 표시하던 드리프트를 막는 단일 출처.
export const ENGINE_VERSION = '2.21.4';
