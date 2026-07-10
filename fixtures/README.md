# Measured fixtures — predicted vs. reality

The engine predicts memory from architecture math. **This folder holds what actually happened on real hardware** — community-submitted measurements, cross-referenced by the [Fit Census](../census/README.md) next to every prediction.

Why it matters: theoretical math can be right on paper and off in practice (runtime overhead, fragmentation, driver reservations). A public predicted-vs-measured ledger makes the engine's accuracy **falsifiable** — and calibrates the overhead constants for everyone.

**Measurements are typed.** Different tools measure different things, and comparing the wrong pair looks like a prediction error when it isn't:

| `measurementKind` | What it is | Compare against |
|---|---|---|
| `idle_resident` | Model weights resident, no active generation — a **floor** (oMLX `actual_size`, `ollama ps`) | `predicted_resident_weights_gb` |
| `load_peak` | Peak while loading the model | between weights and total |
| `generation_peak` | Peak during active generation (process-level) | weights + KV + runtime (no OS reserve) |
| `system_total_peak` | Whole-device peak incl. OS/driver | `predicted_total_to_run_gb` (= fitllm's verdict number) |

An `idle_resident` reading **below** fitllm's total is expected — the total deliberately includes KV, runtime overhead and OS/GPU reserve, because it answers "can I run this", not "how big are the weights".

## Submit a measurement (PR)

1. Run a model. Note the memory reading **and what kind it is** (see table above; e.g. `nvidia-smi`, Activity Monitor, `ollama ps`, oMLX `/admin/api/models`).
2. Add one object to [`measured.json`](measured.json) following [`schema.json`](schema.json) — exact engine model/device names, quant, context, KV precision, `measurementKind`, unit if you know it (GiB vs GB), runtime **with version**, and a link to your fuller report (issue or the PR itself).
3. `npm test` must still pass. That's it.

Ground rules: reproducible conditions only (runtime + version required) · measurements are evidence to investigate, never auto-truth · one measurement per object — don't average across runs yourself.
