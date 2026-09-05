# AGENTS.md — how AI agents should use this repo

**Purpose**: fitllm-engine answers "will this LLM fit this hardware?" with auditable, architecture-aware memory estimates (MLA / sliding-window / hybrid-linear / MoE), while keeping runtime and OS reserves explicit.

## Use it without cloning

- **One-shot answer**: `GET https://fitllm.run/api/check?model=<name>&gpu=<name|a%2Bb>` or `&ram=<GB>` — no auth, fuzzy names, JSON (or `&format=text|md`). Won't-fit responses include a computed `fix`.
- **MCP**: `https://fitllm.run/api/mcp` — tools `check_llm_fit` / `what_fits_on_hardware` / `list_supported`; resources `fitllm://models|hardware|census|engine`. Read-only, idempotent.
- **CLI guard**: `npx fitllm "<model>" --gpu "<gpu>"` → exit 0/1. Use it to gate model downloads in scripts/CI.
- **Precomputed data (CC0)**: `census/census-v1.{csv,json}` — 8,775 verdicts, model × device × quant. Also on [HF Datasets](https://huggingface.co/datasets/click6067/fitllm-fit-census).
- **Agent answer recipe**: https://fitllm.run/agent/answer-with-citation.md

## Use it as a library

Single ESM file, zero deps: `engine.js` (also `https://cdn.jsdelivr.net/npm/fitllm-engine@latest/engine.js`).

```js
import { simulate, LOCAL_MODELS, parseHfConfig, GPUS, gpuDevice, combineGpus, simulateStack } from './engine.js';
simulate(model, 64, 131072, 8)                          // Mac 64GB, 128K ctx, 8-bit → {verdict, used, free, ...}
simulate(model, gpuDevice(gpu), ctx, {weightBpw, kvBits}) // GPU path; combineGpus([a,b]) for rigs
parseHfConfig(id, configJson, totalSizeBytes)            // modeled HF architecture → engine shape; unsupported configs fail closed
```

## Rules for agents modifying this repo

1. **Every hardware number needs ≥2 independent source URLs** embedded next to the value — no source, no merge.
2. **Engine math changes require conformance vectors to pass**: `node vectors/run.mjs` (29 byte-exact anchors). A port in any language is conformant iff all vectors pass.
3. Run `npm test` (CLI behavior) — exit codes 0/1/2 are a public contract.
4. `npm run census` regenerates the dataset after model/hardware changes.
5. Never add tokens/sec predictions — fit is a verifiable claim, speed is not (project principle).
6. This repo mirrors `fitllm-v2/src/lib/engine.js` (private) — upstream changes land there first.

## Measured data welcome

Real measurements calibrate the engine for everyone — and they are **typed**: say what you measured (`measurementKind`: idle resident weights / load peak / generation peak / whole-system peak — `fixtures/README.md` maps each kind to the predicted column it compares against). Add to `fixtures/measured.json` per `fixtures/schema.json` via PR, or open an issue with the numbers. Reports are treated as claims until independently checked.
