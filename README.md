# FitLLM Engine

> **The memory math behind [fitllm.run](https://fitllm.run) — accurate on modern LLM architectures, where most calculators (and LLMs) are wrong.**
> Zero dependencies. One readable file: [`engine.js`](engine.js). Conformance-vector tested. MIT.

```bash
npx fitllm "GLM-4.7-Flash" --gpu 4090     # ✓ FITS — 21.9/24 GB, free 2.1 GB
npx fitllm "gpt-oss-120b" --mac 64        # ✗ WON'T FIT → what to change to make it fit
npx fitllm --detect                       # reads this machine's real hardware
```
Exit code 0/1 — usable as a **pre-download guard** in scripts: know it won't fit *before* the 40 GB download.

This is the open calculation core of FitLLM. **The math is open so you can audit it.**

Ask an LLM "does Qwen 3.6 fit my GPU?" and it pattern-matches to an architecture from its training cutoff — and usually says *no*. Catalog-based calculators lag new releases. FitLLM reads each model's **official `config.json` live**, so it's right on **day-one releases** and on the hybrid / sliding-window / MoE architectures that naive formulas get wrong.

Covers **Apple Silicon unified memory (M1–M5, Pro/Max/Ultra — up to the 512GB Mac Studio)**, **NVIDIA GPUs (RTX 20/30/40/50, workstation RTX 6000 Ada / RTX PRO 6000, datacenter A100/H100/H200/B200)**, **AMD Radeon (RX 7000/9000, PRO W7900)** and **multi-GPU presets (2×3090, 2×4090, 4×3090)** — with GGUF Q-tier weight quantization kept separate from KV-cache quantization. Every hardware number is cross-verified against **≥2 independent sources** (source URLs embedded per-value in `engine.js`).

---

## Why most LLM memory calculators are wrong

Almost every "can I run this LLM?" calculator estimates the KV cache with the textbook formula:

```
KV ≈ 2 × num_layers × num_kv_heads × head_dim × context_length × bytes
```

That assumes **every layer keeps a full-context KV cache with one uniform head shape.** True for Llama-1/2 — wrong for most 2025–2026 models:

| Model | What naive formulas miss | Naive KV | FitLLM KV | Off by |
|---|---|---|---|---|
| **Gemma 4 31B** @131K, 8-bit | 50 of 60 layers are sliding-window (keep only the last 1024 tokens); the 10 global layers use a different head shape (4 KV-heads × 512, not 16 × 256) | ~60 GB | ~5.4 GB | **11×** |
| **Qwen 3.6 27B** @131K, 8-bit | 48 of 64 layers are linear attention (Gated DeltaNet) — no growing KV cache | ~16 GB | ~4 GB | **4×** |
| **GLM-4.7-Flash** @128K, bf16 | MLA: K/V compressed into one shared latent (512+64 dims, cached once — not per-head K and V) | ~117 GB | ~6.6 GB | **17.8×** |
| Plain dense (Llama, Mistral…) | nothing — standard transformer | same | same | 1× ✅ |

An 11× error flips the verdict: a naive calculator says Gemma 4 31B *won't fit* in 64 GB at long context, when it **fits comfortably**.

### The four things they ignore
1. **Sliding-window attention** (Gemma 2/3/4, gpt-oss): most layers only keep the last *N* tokens, so their KV stops growing. Only the global layers scale with full context.
2. **Hybrid / linear attention** (Qwen 3.6, many 2026 models): linear-attention layers use a fixed-size recurrent state, not a growing KV cache.
3. **MLA — Multi-head Latent Attention** (GLM-5.2, GLM-4.7-Flash, DeepSeek family): the cache is a single low-rank latent (`kv_lora_rank` + RoPE dims) shared across all heads — per-head "2 × heads × head_dim" formulas over-count by an order of magnitude. Verified against the DeepSeek-V2 paper (arXiv:2405.04434) and the official DeepSeek-V3 inference code.
4. **Heterogeneous head dims + MoE**: global layers can use a different `head_dim` (Gemma 4: 512 vs 256). MoE keeps every expert in memory while activating only a few per token.

This engine models each layer type separately, verified against official HuggingFace `config.json` files.

---

## What it computes

```
Total = Parameters (quantization-adjusted)
      + KV cache (per layer kind: sliding / global / linear / dense)
      + Runtime overhead (quant metadata + KV block padding + activations + fixed)
      + macOS base (Apple Silicon unified memory)
```

Plus decode-speed estimate (`bandwidth ÷ active-params`) and an `parseHfConfig()` that turns any HuggingFace config into the model shape above.

## Usage

```js
import { simulate, LOCAL_MODELS, estimateSpeed, parseHfConfig } from './engine.js';

const model = LOCAL_MODELS.find((m) => m.name === 'Gemma 4 31b');
const sim = simulate(model, /*ram*/ 64, /*ctx*/ 131072, /*bits*/ 8);
// → { used, free, verdict: 'yes'|'tight'|'no', param, kv, rt, os, maxContext, ... }

estimateSpeed(model, 'M5 Max', 8, /*gpuCores*/ 40); // ≈ tok/s

// any HuggingFace model:
const m = parseHfConfig('Qwen/Qwen3-32B', configJson, totalSizeBytes);
```

## Verification

- Architecture values checked against official HuggingFace `config.json`.
- Gemma 4 31B full-context KV reproduces **20.78 GiB**, matching the published [architecture analysis](https://kaitchup.substack.com/p/gemma-4-31b-and-26b-a4b-architecture). Reproduce it by hand:

```
global: 10 layers × 2(K,V) × 4 heads × 512 dim × 2 B × 262,144 = 21,474,836,480 B
local:  50 layers × 2(K,V) × 16 heads × 256 dim × 2 B × 1,024  =    838,860,800 B
total = 22,313,697,280 B ÷ 1024³ = 20.78 GiB
```

- Calibration: Qwen 3.6 35B-A3B @128K, 8-bit ≈ **54 GB** (matches real local runs).
- MLA per-token cost: GLM-4.7-Flash = (512 + 64) × 2 B × 47 layers = **54,144 B/token** — pinned by conformance vectors.

All figures are estimates — real usage varies with the runtime (MLX/Ollama/llama.cpp), OS state, and quantization scheme.

## Conformance vectors

[`vectors/fit-vectors-v1.json`](vectors/fit-vectors-v1.json) pins **14 language-neutral test vectors** (exact KV bytes, per-token costs, fit verdicts) derived from official `config.json` values. **Any implementation in any language conforms if every vector passes** — run ours with `node vectors/run.mjs`. Port the engine, keep the vectors.

## Principles

**No ads. No login. No affiliate links. Output is never for sale.** Fit is a *winnable*, verifiable claim; raw tok/s is not — so this engine refuses speed predictions rather than dress a guess as precision.

## Help calibrate

Ran a model and measured real peak memory? **[Report a measurement](../../issues/new?labels=measurement)** — it improves the estimates for everyone.

## Built by

[yongha](https://yonghakim.com) — [GitHub](https://github.com/click6067-ship-it). Powers [fitllm.run](https://fitllm.run).

## License

MIT © click6067-ship-it
