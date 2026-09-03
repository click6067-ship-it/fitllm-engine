# Accuracy benchmark contract

FitLLM's public accuracy claim is blocked until independent evidence clears this contract. The benchmark compares like with like; it does not treat a resident-weight reading as whole-system peak memory.

## Reproduce the current report

```bash
npm run census:check
npm run benchmark:accuracy
npm run benchmark:accuracy -- --json
```

The current fixture ledger is useful input, but its entries remain claims unless independently verified. `community_unverified` rows can appear in the report; only `verified` or `maintainer_verified` rows with the immutable evidence fields below count toward the public claim gate.

## Independent oracle

For each case, preserve a unique `caseId`, the model revision, artifact hash, runtime version/commit, full launch flags, hardware, driver/OS, raw memory artifact SHA-256 and HTTPS source, reporter, independent verifier, prefill/decode completion, context, KV precision, and measurement kind. The report's string inequality check (`verifiedBy` differs from `reporter`) is only a machine guard; pull-request review must confirm they are actually different people. Duplicate case IDs do not count. Prefer independent readings from:

- Hugging Face Transformers initialized on the `meta` device for architecture and parameter accounting.
- llama.cpp's own `KV self size` and runtime memory logs for GGUF execution.
- The official DeepSeek inference implementation for MLA cases.

Include ordinary dense/GQA controls as well as sliding-window, hybrid-linear, MLA, MoE, and heterogeneous-head models. Publish every exclusion and report tail errors, not only a favorable median.

## Pinned competitor input

Competitor rows are future, reviewed inputs to `buildAccuracyReport(census, measurements, competitorRows)`:

```json
{
  "caseId": "immutable-case-id",
  "competitor": "name@version-or-commit",
  "absolutePercentageErrorPct": 12.4,
  "falseFit": true,
  "rawOutputSha256": "...",
  "source": "https://..."
}
```

Run every product against the same precommitted cases and conditions. The claim gate pairs rows by `caseId`, requires an immutable competitor identifier (`name@version-or-commit`), HTTPS raw-output source, and SHA-256, and compares only a single pinned competitor's matched cohort. Do not fill a competitor output by reusing FitLLM's formulas or census.

## Claim gate and kill conditions

A comparative accuracy claim stays blocked until there are at least 30 independently verified compatible measurements, 3 runtimes, 6 architecture families, and pinned competitor output. FitLLM must have both at least 10% relative MAPE improvement and a strictly lower false-fit rate.

Kill condition: abandon the public accuracy-leader position if the first three independent pilot cases show no directional advantage, if the full set improves MAPE by less than 10%, or if false-fit rate is not lower. Keep publishing the ledger either way; the auditable oracle remains useful without a superlative.
