# Local LLM Fit Census v1 — 2026-07-11

**6,048 verdicts**: 18 models × 88 devices (36 GPUs + 52 Mac configs) × per-platform quant tiers.
Every number computed by [fitllm-engine](https://github.com/click6067-ship-it/fitllm-engine) from official `config.json` values — architecture-aware (MLA, sliding-window, hybrid attention, MoE). **Reproduce it yourself: `npm run census`.**

Assumptions: context = min(8K, model max) · KV cache F16 · platform reserve/headroom per engine. Interactive per-combo pages: [fitllm.run/can-i-run](https://fitllm.run/can-i-run).

## Starter matrix — biggest model that fits comfortably (~4-bit, 8K ctx)

| Device | Memory | Biggest comfortable fit |
|---|---|---|
| M1 8GB | 8GB | ❌ none comfortably at ~4-bit |
| M2 16GB | 16GB | **Gemma 4 e2b** (5.1B) — free 3.32GB, up to ~11K ctx |
| M4 32GB | 32GB | **GLM-4.7-Flash** (30B) — free 7.13GB, up to ~17K ctx |
| M5 Max 64GB | 64GB | **Qwen 3.6 35B-A3B** (35B) — free 37.32GB, up to ~262K ctx |
| M4 Max 128GB | 128GB | **gpt-oss-120b** (117B) — free 58.41GB, up to ~131K ctx |
| M3 Ultra 512GB | 512GB | **GLM-5.2** (753B) — free 110.25GB, up to ~70K ctx |
| RTX 3060 12GB | 12GB | **Gemma 4 12b** (11.95B) — free 1.63GB, up to ~30K ctx |
| RTX 4060 Ti 16GB | 16GB | **Gemma 4 12b** (11.95B) — free 5.63GB, up to ~110K ctx |
| RTX 4090 | 24GB | **GLM-4.7-Flash** (30B) — free 2.13GB, up to ~19K ctx |
| RTX 5090 | 32GB | **Qwen 3.6 35B-A3B** (35B) — free 7.24GB, up to ~117K ctx |
| RX 7900 XTX | 24GB | **GLM-4.7-Flash** (30B) — free 2.13GB, up to ~19K ctx |
| RTX PRO 6000 Blackwell | 96GB | **gpt-oss-120b** (117B) — free 18.76GB, up to ~131K ctx |
| A100 80GB | 80GB | **Qwen 3.6 35B-A3B** (35B) — free 55.24GB, up to ~262K ctx |
| H200 141GB | 141GB | **gpt-oss-120b** (117B) — free 63.76GB, up to ~131K ctx |

## Smallest device that runs each model (~4-bit, 8K ctx)

| Model | Params | Smallest GPU | Smallest Mac |
|---|---|---|---|
| GLM-4.7-Flash | 30B | RTX 4090 | M5 32GB |
| GLM-5.2 | 753B | — | M3 Ultra 512GB |
| gpt-oss-20b | 21B | RTX 5080 | M5 24GB |
| gpt-oss-120b | 117B | A100 80GB | M3 Max 96GB |
| Qwen 3.6 27B | 27.2B | RTX 4090 | M5 32GB |
| Qwen 3.6 35B-A3B | 35B | RTX 5090 | M5 32GB |
| Qwen-AgentWorld-35B-A3B | 34.7B | RTX 5090 | M5 32GB |
| Gemma 4 e2b | 5.1B | RTX 4060 Ti 8GB | M5 16GB |
| Gemma 4 e4b | 8B | RTX 4060 Ti 8GB | M5 16GB |
| Gemma 4 12b | 11.95B | RTX 2080 Ti | M5 16GB |
| Gemma 4 26b A4B | 25.5B | RX 7900 XT | M5 32GB |
| Gemma 4 31b | 30.7B | RTX 4090 | M5 32GB |
| Llama-3.2-3B-Instruct | 3.2B | RTX 4060 Ti 8GB | M5 16GB |
| Llama-3.1-8B-Instruct | 8B | RTX 3080 10GB | M5 16GB |
| Qwen3-0.6B | 0.596B | RTX 4060 Ti 8GB | M5 16GB |
| Qwen3-1.7B | 1.721B | RTX 4060 Ti 8GB | M5 16GB |
| Llama-3.2-1B-Instruct | 1.236B | RTX 4060 Ti 8GB | M5 16GB |
| Gemma-3-1B-it | 1B | RTX 4060 Ti 8GB | M5 16GB |

## Full data

- [`census-v1.csv`](census-v1.csv) / [`census-v1.json`](census-v1.json) — every model × device × quant verdict with the full predicted breakdown (`predicted_total_to_run_gb` = weights + KV + runtime + reserve — what the verdict uses; `predicted_resident_weights_gb` = quantized weights **plus ~12% runtime weight overhead** (non-quantized parts, buffers) — the number resident-weights measurements should be compared against; `predicted_param_gb` = quantized weights alone) and max context. Machine-readable; import it, chart it, cite it.
- **Measurements are typed** (from [`fixtures/measured.json`](../fixtures/README.md), community PRs): `measurement_kind` says what was measured. `idle_resident` readings (e.g. oMLX `actual_size`) are a resident-weights **floor** — compare them to `predicted_resident_weights_gb`, not to the total; only `system_total_peak` is comparable to `predicted_total_to_run_gb`. An idle_resident value below the predicted total is expected, not an over-prediction. Ledger holds 9 entries; 3 join this census (exact model+device+quant match required — the rest cover models/devices outside the catalog or carry unconfirmed attribution).

All figures are estimates; real usage varies with runtime, driver and OS state. Verdicts: ✅ fits comfortably · ⚠️ tight · ❌ won't fit.
