# ADR-0001: One truth, two projections — resident weights vs total-to-run

- **Status**: Accepted (implemented 2026-07-10)
- **Context date**: 2026-07-10 · issues [#1](https://github.com/click6067-ship-it/fitllm-engine/issues/1), [#6](https://github.com/click6067-ship-it/fitllm-engine/issues/6), PR [#5](https://github.com/click6067-ship-it/fitllm-engine/pull/5)
- **Review**: Claude + Codex, blind-then-adversarial (both independently converged on the same architecture)

## Decision

FitLLM's number that verdicts use is **`predicted_total_to_run_gb` = weights + KV cache + runtime overhead + OS/GPU reserve**. Runtime tools that report resident model weights (e.g. oMLX `actual_size`, idle) read lower than this total **by design** — that is a different metric, not a prediction error.

1. **One semantic layer.** `simulate()` is the single source of truth: verdict + breakdown (`param`, `kv`, `rtDyn`, `reserve`). No surface may compute or restate its own variant of these numbers.
2. **Two projections.** Human surfaces (fitllm.run, can-i-run, census README) project verdict-first and may **omit** fields. Agent surfaces (census CSV/JSON, llms.txt, HTTP API, CLI `--json`, HF dataset) must be **lossless**: full breakdown with self-describing column names.
3. **Core vs evidence.** Core fields (verdict, total, breakdown) are identical everywhere — surfaces may omit, never alter. Evidence fields (measurements, `measurement_kind`, units, provenance, calibration metadata) may appear on agent surfaces only, in their own namespace, and never overwrite core numbers.
4. **Measurements are typed.** Every ledger entry carries `measurementKind`. `idle_resident` compares against `predicted_resident_weights_gb`; only `system_total_peak` compares against the total. Cross-kind accuracy ratios are a category error.
5. **Enforced, not declared.** `test/invariants.test.mjs` gates: engine equation (`used == param + kv + rtDyn + reserve`), CLI breakdown sum, fixtures schema (kind required), census alias/typing consistency. The 2GB Apple double-display bug (fixed the same day, fitllm-v2 PR #41) is the standing proof that declarations without tests drift.

## Rejected alternatives

- **Different numbers per audience** (lenient for humans, precise for agents): agents read the human page too; any divergence is a public contradiction, and Google requires structured data to match visible content. Rejected outright.
- **Switching the verdict to resident weights** (matches measurements better): produces "fits!"-then-OOM, the worst failure mode for a can-i-run product. The verdict stays on total-to-run.
- **UI split now** (showing weights vs total on the live page): deferred until the data-surface split has community feedback. The human projection is allowed to stay simple.

## Calibration policy

Engine constants change only on accumulated, typed, multi-runtime evidence — never on a single batch. Current evidence (9 oMLX/M-series points, 2026-07): measured resident lands within ±10% of `predicted_resident_weights_gb` (= param × 1.12); the 12% param-overhead constant stands. Elastic/SSD expert offload (resident < full weights, issue #6) is out of scope for this ADR.

## Open items (tracked, not promised)

- Verdict confidence display (categorical ✅/⚠️/❌ hides estimate error bars) — next in line after ledger growth.
- Reserve constants are representative estimates pending `system_total_peak` measurements.
- QAPage JSON-LD suitability on can-i-run pages (Google guidance review).
