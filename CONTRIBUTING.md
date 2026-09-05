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
Ran a model and measured actual peak memory? Two ways:
- **PR (preferred)**: add one object to [`fixtures/measured.json`](fixtures/measured.json) per [`fixtures/schema.json`](fixtures/schema.json) — it appears in the public [Fit Census](census/README.md) as a predicted-vs-measured row. Runtime + version required (reproducibility).
- Or [open a measurement issue](../../issues/new?template=measurement.yml).

### Partial-residency / SSD expert offload reports (elastic MoE)
"It runs under partial residency" observations are recorded in the ledger but **never change verdicts by themselves** — the promotion gates live in [issue #6](../../issues/6):
- **Gate A (UI measured-exception note)**: ≥2 independent reporters · `measurementKind: generation_peak` with completed prefill+decode (context + generated token counts) · raw artifacts (runtime version/commit, launch flags, completion status, wall-clock prefill/decode timing, logs/screenshots, report URL) · canonical model/quant/runtime/platform ids. `idle_resident` observations don't qualify — the schema forbids comparing them to totals.
- **Gate B (engine modeling)**: a deterministic residency derivation from runtime source/docs — only then does the engine gain a runtime dimension.

The one disclosed structural exception, the Gemma 4 e2b/e4b PLE deduction (`ple-llamacpp-non-gpu-residency`, shown with the verdict), is not storage paging: the pinned llama.cpp/GGUF path assigns that input-layer tensor to CPU/host buffers instead of accelerator memory, the host memory it needs is not budgeted by the discrete-GPU verdict, and a runtime that loads PLE onto the accelerator invalidates the estimate. It does not open a CPU/RAM/SSD offload path; a Gemma 4 GGUF measurement that names the runtime and revision belongs in the measurement template above.

Estimate-vs-measured reports calibrate the overhead constants for everyone — they're the most valuable contribution this repo takes.
