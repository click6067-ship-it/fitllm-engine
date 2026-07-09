# Contributing

The fastest way to help: **add a model or a GPU as a data-only PR.** No math changes needed.

## Adding a model
1. Fetch the **official** `config.json`: `https://huggingface.co/<org>/<model>/raw/main/config.json` (gated repos: cite an identical ungated mirror AND the official repo).
2. Fill the fields exactly as in existing `MODELS` entries: `layerCount, kvHeads, kvHeadDim, attnHeads, hiddenSize, maxContext`, MoE (`numExperts, expertsPerToken`), sliding (`slidingWindow, globalAttnLayers`), hybrid (`fullAttnLayers`), MLA (`mlaKvLoraRank, mlaRopeDim`).
3. `totalParams` from `model.safetensors.index.json → metadata.total_size ÷ dtype bytes` — **watch the dtype** (bf16 = 2 bytes, fp8 = 1; MXFP4-native models: use the model card's stated count).
4. Leave `benchmarks: null` unless you cite official numbers.

## Adding hardware
Every number needs **≥2 independent sources** (different domains — vendor spec + reputable database/review), cited as URLs in the entry's `sources` field. Single-sourced numbers are rejected — one confidently-wrong figure costs more than a missing one.

## Before you open the PR
```bash
npm test        # node --test + conformance vectors — must pass
```
If you changed any math (rare): add a vector to `vectors/fit-vectors-v1.json` whose expected value you derived **by hand from the config**, not from the engine's own output.

## Report a real measurement
Ran a model and measured actual peak memory? [Open a measurement issue](../../issues/new?labels=measurement) — estimate-vs-measured reports calibrate the overhead constants for everyone.
