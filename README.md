# FitLLM Engine

FitLLM is an open-source, zero-dependency engine that checks whether a local LLM fits on a GPU or Apple Silicon Mac using architecture-aware memory math.

[![npm](https://img.shields.io/npm/v/fitllm-engine?color=cb3837&label=npm)](https://www.npmjs.com/package/fitllm-engine)
[![conformance](https://img.shields.io/badge/conformance_vectors-28%2F28-brightgreen)](vectors/fit-vectors-v1.json)
[![license](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![zero deps](https://img.shields.io/badge/dependencies-0-success)](package.json)

![npx fitllm — one-line fit verdict with the full memory breakdown](assets/demo.gif)

> **Live: https://fitllm.run · Bilingual  · Free · No ads · No login**
> 
> **Open engine:** [fitllm-engine](https://github.com/click6067-ship-it/fitllm-engine) (MIT · npm `fitllm-engine` · `npx fitllm`)
> 
> Zero dependencies. One readable file: [`engine.js`](engine.js). Conformance-vector tested. MIT.

## Quick start

```bash
npx fitllm "Gemma 4 12b" --gpu "RTX 4090"     # one line, exit 0 fits / 1 won't — run before you download
npx fitllm --top --gpu 4090                    # what can this hardware run?
npx fitllm --top --detect --json               # detect every supported local GPU; agent-ready JSON
npx fitllm Qwen/Qwen3-32B --detect --json      # public Hugging Face ID or URL; unsupported configs fail closed
npx fitllm "Gemma 4 12b" --detect --json --why # include architecture, evidence, assumptions, and exact memory inputs
npm install fitllm-engine                      # use the same engine as a library (see Usage)
```

## Remote MCP server

Connect any Streamable HTTP MCP client to `https://fitllm.run/api/mcp`:

```json
{
  "mcpServers": {
    "fitllm": {
      "url": "https://fitllm.run/api/mcp"
    }
  }
}
```

## Tools

- `check_llm_fit` — check one model against a GPU, multi-GPU rig, or Mac and return the verdict, memory breakdown, and a fix when it does not fit.
- `what_fits_on_hardware` — rank the supported local models that fit the given GPU, multi-GPU rig, or Mac.
- `list_supported` — list the built-in model and hardware names accepted by the fit checker.

The server is read-only, stateless, and requires no authentication.

```bash
npx fitllm "GLM-4.7-Flash" --gpu 4090     # ✓ FITS — 21.9/24 GB, free 2.1 GB
npx fitllm "gpt-oss-120b" --mac 64        # ✗ WON'T FIT → what to change to make it fit
npx fitllm "Qwen 3.6 35B" --gpu "5090 + 3090"   # multi-GPU rig — VRAM pools (56GB), even mixed cards
npx fitllm --top --detect --json          # detect hardware + what fits, for people or agents
npx fitllm "Gemma 4 12b" --detect         # one verdict on this machine
```

`--detect` reads all `nvidia-smi` adapters, uses Apple Silicon unified memory on arm64 macOS, and can resolve an exact catalog GPU name through Windows/WSL PowerShell. It never uses `Win32_VideoController.AdapterRAM`, serials, PNP IDs, or full environment dumps. Intel-only, ambiguous, and unsupported adapters stop with exit 2 instead of borrowing a nearby GPU's memory.

**Why a CLI?** The "will it run?" question is born in the terminal — one line before `ollama pull`. No install, no tab-switching, and it reads your *actual* hardware with `--detect` instead of asking you to know your VRAM. Exit code 0/1 makes it a **pre-download guard**:

```bash
# in your model-pull script — stop BEFORE the 40 GB download:
npx fitllm "gpt-oss-120b" --detect || { echo "won't fit — aborting pull"; exit 1; }
```

### Put the guard in the download path

The CLI exit contract composes directly with model runtimes. The download or launch runs only after a `FITS` or `TIGHT` verdict; invalid inputs stop with exit 2.

```bash
npx fitllm "Gemma 4 12b" --detect && ollama pull gemma4:12b
npx fitllm "Llama-3.1-8B-Instruct" --detect && llama-cli -m ./Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
```

For CI, use the repository's composite Action. It needs no secret and returns the full `--json --why` result as `steps.preflight.outputs.result`:

```yaml
permissions:
  contents: read
steps:
  - name: Check model memory before download
    id: preflight
    uses: click6067-ship-it/fitllm-engine@v2.12.0
    with:
      model: Gemma 4 12b
      gpu: RTX 4090
      ctx: '8192'
```

Set exactly one of `gpu` or `mac`. The Action preserves the CLI contract: exit 0 means fits/tight, 1 means it will not fit, and 2 means the request is invalid. Inputs cross the GitHub expression boundary through environment variables and are passed to Node as a Bash argument array.

Agents can collect a typed measurement without uploading anything:

```bash
npx fitllm measure "Qwen 3.6 27B" --detect --measured 15.3 --kind system_total_peak --unit GiB --runtime "llama.cpp b6400"
```

The command validates the conditions and prints a candidate JSON object plus a prefilled GitHub issue URL. Submission remains a human action. Public Hugging Face IDs (`org/model`) are fetched with a bounded config/index reader and accepted only when `parseHfConfig()` supports the architecture.

This is the open calculation core of FitLLM. **The math is open so you can audit it.**

Ask an LLM "does Qwen 3.6 fit my GPU?" and it pattern-matches to an architecture from its training cutoff — and usually says *no*. Catalog-based calculators lag new releases. The CLI, API, and MCP use a curated catalog pinned to official configs. The web calculator can additionally inspect a pasted Hugging Face ID's **official `config.json` live**, so supported architectures work on **day-one releases** — including the hybrid / sliding-window / MoE structures that naive formulas get wrong.

Covers **Apple Silicon unified memory (M1–M6, Pro/Max/Ultra — up to the 512GB Mac Studio)**, **NVIDIA GPUs (RTX 20/30/40/50, workstation RTX 6000 Ada / RTX PRO 6000, datacenter A100/H100/H200/B200)**, **AMD Radeon (RX 7000/9000, PRO W7900)** and **multi-GPU presets (2×3090, 2×4090, 4×3090)** — with GGUF Q-tier weight quantization kept separate from KV-cache quantization. Hardware entries carry their source URLs per-value in `engine.js`; new entries require **≥2 independent sources** ([CONTRIBUTING](CONTRIBUTING.md)).

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
| **Qwen 3.8 27B** @256K, F16 KV | same shape, newest generation: KV lives on 16 of 64 layers only | 64.0 GiB | **16.0 GiB** | **4×** |
| **GLM-4.7-Flash** @128K, bf16 | MLA: K/V compressed into one shared latent (512+64 dims, cached once — not per-head K and V) | ~117 GB | ~6.6 GB | **17.8×** |
| Plain dense (Llama, Mistral…) | nothing — standard transformer | same | same | 1× ✅ |

An 11× error flips the verdict: a naive calculator says Gemma 4 31B *won't fit* in 64 GB at long context, when it **fits comfortably**.

### The five things they ignore
1. **Sliding-window attention** (Gemma 2/3/4, gpt-oss): most layers only keep the last *N* tokens, so their KV stops growing. Only the global layers scale with full context.
2. **Hybrid / linear attention** (Qwen 3.6 / 3.8, many 2026 models): linear-attention layers use a fixed-size recurrent state, not a growing KV cache. That state is modeled too, as its own component (`linearState`) — it is a constant per sequence, so it never inflates the context curve.
3. **MLA — Multi-head Latent Attention** (GLM-5.2, GLM-4.7-Flash, DeepSeek family): the cache is a single low-rank latent (`kv_lora_rank` + RoPE dims) shared across all heads — per-head "2 × heads × head_dim" formulas over-count by an order of magnitude. Verified against the DeepSeek-V2 paper (arXiv:2405.04434) and the official DeepSeek-V3 inference code.
4. **Heterogeneous head dims + MoE**: global layers can use a different `head_dim` (Gemma 4: 512 vs 256). MoE keeps every expert in memory while activating only a few per token.
5. **PLE — Per-Layer Embeddings** (Gemma 4 e2b/e4b): under the pinned llama.cpp/GGUF path the `per_layer_token_embd` tensor is read lazily or stays host-resident instead of being loaded as an ordinary GPU-resident weight, so only the non-PLE weights are counted against VRAM for the verified Gemma 4 e2b/e4b entries. Counting all 5.1B params against a GPU over-predicts e2b's resident weights by ~1.9× and flips small-card verdicts. This deduction is conditional: it holds only under that pinned path, a runtime that loads PLE tensors onto the accelerator (vLLM, for example) invalidates the estimate, and unverified families keep their full weights resident. On Apple Silicon system RAM *is* accelerator memory, so total params stay correct there. The exact premise and its pinned sources are listed under [Structural premises](#structural-premises); a direct measurement on a Gemma 4 GGUF is welcome in issue #7.

This engine models each layer type separately, verified against official HuggingFace `config.json` files.

---

## What it computes

```
Total = Parameters (quantization-adjusted)
      + KV cache (per layer kind: sliding / global / linear / dense)
      + Runtime overhead (quant metadata + KV block padding + activations + fixed)
      + macOS base (Apple Silicon unified memory)
```

Plus a `parseHfConfig()` that turns configs from verified, modeled Hugging Face architecture families into the model shape above; unsupported structures fail closed instead of returning a guess. (No token/s prediction — deliberately: speed depends on runtime/backend in ways a static model can't claim honestly. Fit is a verifiable claim; speed is not.)

## Structural premises

Some verdicts are valid only under a structural premise about the artifact or the runtime path. The engine derives the active premises from the same normalized model fields the calculation uses and attaches them to affected results as `structuralAssumptions`, an array of `{ id, statement }`. Unaffected results, such as plain GQA models, carry no such key, so legacy JSON shapes are unchanged. The CLI prints each active premise as a `premise [id]:` line in text mode and includes the array in `--json` (also per row in `--top --json`); `--why` and the composite Action forward the same array. There is no runtime selector, no confidence score, and no speed claim: a premise tells you what the memory math assumes, not how fast the model runs.

- `mla-compressed-latent-cache` — KV memory assumes a compressed-latent MLA artifact or mode; legacy non-MLA GGUF or an explicitly uncompressed mode invalidates this estimate. Sources: [DeepSeek-V3 inference `model.py`](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py) · [llama.cpp `convert_hf_to_gguf.py`](https://github.com/ggml-org/llama.cpp/blob/master/convert_hf_to_gguf.py) · [llama.cpp `llama-kv-cache.cpp`](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-kv-cache.cpp).
- `ple-llamacpp-non-gpu-residency` — GPU weight memory excludes the verified Gemma 4 PLE tensors only under the pinned llama.cpp/GGUF lazy-or-host-resident path; an accelerator-loading runtime invalidates this estimate. Sources: [Gemma 4 E2B `config.json`](https://huggingface.co/google/gemma-4-E2B-it/blob/main/config.json) (text body `model_type` is exactly `gemma4_text`) · pinned llama.cpp `8b4b3558f1459c13e4aa38d5c94d306a00dc6acd`: [Gemma 4 construction](https://github.com/ggml-org/llama.cpp/blob/8b4b3558f1459c13e4aa38d5c94d306a00dc6acd/src/models/gemma4.cpp) · [loader header](https://github.com/ggml-org/llama.cpp/blob/8b4b3558f1459c13e4aa38d5c94d306a00dc6acd/src/llama-model-loader.h) · [loader implementation](https://github.com/ggml-org/llama.cpp/blob/8b4b3558f1459c13e4aa38d5c94d306a00dc6acd/src/llama-model-loader.cpp). The deduction applies only to the catalog entries `Gemma 4 e2b` and `Gemma 4 e4b` (`pleOffloadVerified: true`) and to parsed configs whose text body is exactly `gemma4_text`; a look-alike field on any other family keeps the full weights resident on the GPU, so an unverified config cannot flip a verdict toward "fits".
- `mtp-ordinary-generation` — KV memory assumes ordinary non-speculative generation; an MTP draft context is not included. Sources: [llama.cpp `llama-hparams.cpp`](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-hparams.cpp) · [llama.cpp `llama-model.cpp`](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-model.cpp) · [llama.cpp `convert_hf_to_gguf.py`](https://github.com/ggml-org/llama.cpp/blob/master/convert_hf_to_gguf.py) · [vLLM `qwen3_next_mtp.py`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3_next_mtp.py).

```bash
npx fitllm "GLM-4.7-Flash" --gpu "RTX 4090"          # text adds:  premise [mla-compressed-latent-cache]: KV memory assumes …
npx fitllm "GLM-4.7-Flash" --gpu "RTX 4090" --json   # adds "structuralAssumptions": [{ "id": "…", "statement": "…" }]
npx fitllm "Llama-3.1-8B-Instruct" --gpu "RTX 4090" --json   # plain GQA: no structuralAssumptions key at all
```

## Usage

```js
// from npm:  npm install fitllm-engine
import { simulate, structuralAssumptions, LOCAL_MODELS, GPUS, gpuDevice, parseHfConfig } from 'fitllm-engine';
// …or vendored single-file:
// import { simulate, structuralAssumptions, LOCAL_MODELS, GPUS, gpuDevice, parseHfConfig } from './engine.js';

const model = LOCAL_MODELS.find((m) => m.name === 'Gemma 4 31b');
const sim = simulate(model, /*ram*/ 64, /*ctx*/ 131072, /*bits*/ 8);
// → { used, free, verdict: 'yes'|'tight'|'no', param, kv, rt, os, maxContext, ... }

// affected results carry the active structural premises; plain GQA results have no such key:
const gpu = gpuDevice(GPUS.find((g) => g.name === 'RTX 4090'));
const mla = simulate(LOCAL_MODELS.find((m) => m.name === 'GLM-4.7-Flash'), gpu, 8192, { weightBpw: 4.8944, kvBits: 16 });
mla.structuralAssumptions; // → [{ id: 'mla-compressed-latent-cache', statement: '…' }]
structuralAssumptions(model, gpu); // → [] for an unaffected model (pure; same inputs as the calculation)

// a config from a modeled Hugging Face architecture family:
const m = parseHfConfig('Qwen/Qwen3-32B', configJson, totalSizeBytes);
```

## Verification

From a repository checkout, recompute the typed measured-vs-predicted ledger with `npm run benchmark:accuracy` (or add `-- --json`). Run the pinned estimator differential with `npm run benchmark:differential`. To recapture its uncut source output from a separately downloaded official llmfit release artifact:

```bash
npm run benchmark:capture:llmfit -- --binary /path/to/llmfit --artifact /path/to/llmfit-v1.1.12-x86_64-unknown-linux-gnu.tar.gz
```

The differential is labeled `architecture_differential_not_runtime_accuracy`: it demonstrates how estimators treat GQA, sliding-window attention, and MLA, but it is not runtime measurement evidence. The methodology, competitor input format, checksums, and precommitted kill conditions are in [`benchmarks/README.md`](benchmarks/README.md). The current evidence does **not** clear the comparative accuracy claim gate.

- Architecture values checked against official HuggingFace `config.json`.
- Gemma 4 31B full-context KV reproduces **20.78 GiB**, matching the published [architecture analysis](https://kaitchup.substack.com/p/gemma-4-31b-and-26b-a4b-architecture). Reproduce it by hand:

```
global: 10 layers × 2(K,V) × 4 heads × 512 dim × 2 B × 262,144 = 21,474,836,480 B
local:  50 layers × 2(K,V) × 16 heads × 256 dim × 2 B × 1,024  =    838,860,800 B
total = 22,313,697,280 B ÷ 1024³ = 20.78 GiB
```

- MLA per-token cost: GLM-4.7-Flash = (512 + 64) × 2 B × 47 layers = **54,144 B/token** — pinned by conformance vectors.

All figures are estimates — real usage varies with the runtime (MLX/Ollama/llama.cpp), OS state, and quantization scheme.

## Conformance vectors

[`vectors/fit-vectors-v1.json`](vectors/fit-vectors-v1.json) pins **28 language-neutral test vectors** (exact KV bytes, per-token costs, fit verdicts) derived by hand from official `config.json` values — e.g. *"Gemma 4 31B at 262,144 ctx, bf16 = exactly 22,313,697,280 bytes"*. **Any implementation in any language conforms if every vector passes** — run ours with `node vectors/run.mjs`.

**Why this matters:** the formulas are easy to copy; a verified answer key is not. If you port this engine to Python, Rust or Go, you don't become an untrusted fork — pass the vectors and you're a **conformant implementation of the same standard**. Port the engine, keep the vectors.

## The Fit Census — every model × every device, one truth table

[`census/`](census/README.md) holds **8,424 verdicts** (24 models incl. draft tier × 93 GPUs/Macs × quant tiers) computed by this engine — as CSV/JSON you can import, chart or cite, plus a starter matrix ("biggest model that fits comfortably per device"). Regenerate it yourself: `npm run census`. Real-world measurements land next to predictions via [`fixtures/`](fixtures/README.md) PRs — **predicted vs. measured, in public.**

## Embed a fit badge

Show whether a model runs on given hardware — live from the engine, one line in any README or model card:

```markdown
![fits](https://img.shields.io/endpoint?url=https%3A%2F%2Ffitllm.run%2Fapi%2Fbadge%3Fmodel%3DGLM-4.7-Flash%26gpu%3D4090)
```

![fits](https://img.shields.io/endpoint?url=https%3A%2F%2Ffitllm.run%2Fapi%2Fbadge%3Fmodel%3DGLM-4.7-Flash%26gpu%3D4090)

Params: `model` (name, fuzzy), `gpu` (name, fuzzy) **or** `ram` (GB, Apple unified memory), optional `quant` (GGUF tier / 4|8|16), `ctx`, `kv`. Verdict color: green fits · yellow tight · red won't fit.

**Why embed it?** The #1 question under every model card and local-AI tutorial is *"will it run on my machine?"* The badge answers it **live from the engine** — recomputed when the data updates, not a stale claim frozen into your README. If you publish models or write guides: one line replaces a whole FAQ paragraph and cuts the "it OOM'd on my 8GB card" issues before they're filed.

## Ask your AI assistant (MCP)

The engine runs as a public **MCP server** at `https://fitllm.run/api/mcp` — connect it once and your assistant answers *"can I run X on my Y?"* with this engine's math instead of guessing from stale training data (LLMs routinely get KV-cache math wrong — see the 17.8× table above).

- **Claude** (web / desktop / mobile): Settings → Connectors → **Add custom connector** → paste `https://fitllm.run/api/mcp`
- **Claude Code**: `claude mcp add --transport http fitllm https://fitllm.run/api/mcp`
- **Cursor / Windsurf**: add to `mcp.json` → `{ "mcpServers": { "fitllm": { "url": "https://fitllm.run/api/mcp" } } }`
- **ChatGPT**: Settings → Apps → Advanced → Developer mode → add MCP server (Plus/Pro)

Tools: `check_llm_fit` (verdict + full memory breakdown + fix suggestion — supports multi-GPU rigs like `"RTX 5090 + RTX 3090"`), `what_fits_on_hardware` (ranked list for your machine), `list_supported`. Resources: `fitllm://models`, `fitllm://hardware`, `fitllm://census`, `fitllm://engine`. **Intentionally open**: read-only, stateless, no auth, no secrets — every call is a pure function of public data.

Listed on: [official MCP registry](https://registry.modelcontextprotocol.io) (`run.fitllm/fitllm`) · [Glama](https://glama.ai/mcp/connectors/run.fitllm/fitllm) · [mcp.so](https://mcp.so/servers/fitllm) · [Smithery](https://smithery.ai/server/click6067/fitllm)

## For agents & scripts — plain HTTP API

No MCP client? One GET, no auth, no key — JSON by default, plain text for curl:

```bash
curl 'https://fitllm.run/api/check?model=gemma%204%2031b&gpu=4090'
# multi-GPU rigs: gpu=5090%2B3090 · Mac: ram=64 · usage: curl https://fitllm.run/api/check
```

Open data: the full **Fit Census** (8,424 verdicts, **CC0**) at [fitllm.run/data](https://fitllm.run/data) and on [Hugging Face Datasets](https://huggingface.co/datasets/click6067/fitllm-fit-census). Try the engine in-browser: [HF Space demo](https://huggingface.co/spaces/click6067/fitllm).

## Principles

**No ads. No login. No affiliate links. Output is never for sale.** Fit is a *winnable*, verifiable claim; raw tok/s is not — so this engine refuses speed predictions rather than dress a guess as precision.

## Help calibrate

Ran a model and measured real peak memory? **[Report a measurement](../../issues/new?template=measurement.yml)** — it improves the estimates for everyone.

## Built by

[yongha](https://yonghakim.com) — [GitHub](https://github.com/click6067-ship-it). Powers [fitllm.run](https://fitllm.run).

## License

MIT © click6067-ship-it
