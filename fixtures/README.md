# Measured fixtures — predicted vs. reality

The engine predicts memory from architecture math. **This folder holds what actually happened on real hardware** — community-submitted measurements, cross-referenced by the [Fit Census](../census/README.md) as a `measured` column next to every prediction.

Why it matters: theoretical math can be right on paper and off in practice (runtime overhead, fragmentation, driver reservations). A public predicted-vs-measured ledger makes the engine's accuracy **falsifiable** — and calibrates the overhead constants for everyone.

## Submit a measurement (PR)

1. Run a model. Note the **peak** memory (e.g. `nvidia-smi`, Activity Monitor, `ollama ps`).
2. Add one object to [`measured.json`](measured.json) following [`schema.json`](schema.json) — exact engine model/device names, quant, context, KV precision, runtime **with version**, and a link to your fuller report (issue or the PR itself).
3. `npm test` must still pass. That's it.

Ground rules: reproducible conditions only (runtime + version required) · measurements are evidence to investigate, never auto-truth · one measurement per object — don't average across runs yourself.
