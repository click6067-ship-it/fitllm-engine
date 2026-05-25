# FitLLM Engine

> **The memory & speed math behind [FitLLM](https://fitllm-v2.vercel.app) — accurate on modern LLM architectures, where most calculators are wrong.**
> Zero dependencies. One file: [`engine.js`](engine.js). MIT.

This is the open calculation core of FitLLM. The UI/design lives in a separate private repo; **the math is open so you can audit it.**

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
| Plain dense (Llama, Mistral…) | nothing — standard transformer | same | same | 1× ✅ |

An 11× error flips the verdict: a naive calculator says Gemma 4 31B *won't fit* in 64 GB at long context, when it **fits comfortably**.

### The three things they ignore
1. **Sliding-window attention** (Gemma 2/3/4): most layers only keep the last *N* tokens, so their KV stops growing. Only the ~1/6 global layers scale with full context.
2. **Hybrid / linear attention** (Qwen 3.6, many 2026 models): linear-attention layers use a fixed-size recurrent state, not a growing KV cache.
3. **Heterogeneous head dims + MoE**: global layers can use a different `head_dim` (Gemma 4: 512 vs 256). MoE keeps every expert in memory while activating only a few per token.

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
- Gemma 4 31B full-context KV reproduces **20.78 GiB**, matching the published [architecture analysis](https://kaitchup.substack.com/p/gemma-4-31b-and-26b-a4b-architecture).
- Calibration: Qwen 3.6 35B-A3B @128K, 8-bit ≈ **54 GB** (matches real local runs).
- M5 memory bandwidth from Apple's official M5 Pro (307 GB/s) / M5 Max (460–614 GB/s) specs.

All figures are estimates — real usage varies with the runtime (MLX/Ollama), OS state, and quantization scheme.

## Help calibrate

Ran a model and measured real peak memory? **[Report a measurement](../../issues/new?labels=measurement)** — it improves the estimates for everyone.

## License

MIT © click6067-ship-it
