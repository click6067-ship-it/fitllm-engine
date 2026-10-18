# DeepSeek-V4 family + Muse Spark Open (2.21.0)

**State (2026-10-18):** engine + tests done (386/386 · vectors 46/46 · census:check byte-identical) · Codex gpt-6-astra review R1 FAIL (4 blockers + 1 major) → fixed → R2 PASS → PR → tag v2.21.0 → vendor into fitllm-v2 → verify npm + fitllm.run.
**Blockers:** none. **Follow-ups (not in this release):** all-`sliding_attention` layer_types still falls back to a 5:1 split (over-count, pre-existing on master); GLM-5.2/5.3 catalog rows omit the DSA indexer-key cache (needs its own check).

## Intent
`npx fitllm-engine@2.20.0 deepseek-ai/DeepSeek-V4.1-Flash-1017 --gpu "DGX Spark" --count 4` refused the model ("비표준 압축·희소·혼합 어텐션"), and the catalog had no DeepSeek or Muse rows. Model the DeepSeek-V4 attention memory from DeepSeek's own inference code so V4-Flash, V4-Pro, V4.1-Flash and V4.1-Flash-1017 compute, and add catalog rows — without guessing: anything the reference code does not establish stays a conservative upper bound or fails closed.

## Examples (must-haves ✱)
- ✱ V4.1-Flash-1017 KV = 1,600 cached elements/token (4 source layers 2·8·14·20; ratio 2,2,2,1; 512 main + 128 indexer) + 8 B/token Engram history; in DeepSeek's FP4 formats the same elements are the tech report's 890 B/token.
- ✱ V4-Flash 3,440 elements/token (CSA 21 × (512+128)/4 + HCA 20 × 512/128); V4-Pro 4,924.
- ✱ The reported command answers: HF ID × 4× DGX Spark, Q4_K_M, 8K → WON'T FIT, 629.72 / 512 GiB (exit 1), same numbers as the catalog row.
- ✱ Muse Spark Open: mirror config is the gated original's git blob (e6df3a8); standard GQA, 16 full + 46 sliding(2048) → existing sliding path; the gated HF ID resolves to the catalog row.
- Off-layout configs (unverified ratios, KV sharing that the reference code would not run, layer_types, two KV heads, no safetensors evidence) fail closed.

## Non-goals
- No Engram host/SSD offload credit (all weights resident; on GB10 host memory is the same pool anyway).
- No FP8/FP4 serving-format scales on top of the chosen KV precision.
- Vision-Exp (#101) and transformers-format deepseek_v4 configs (Nex-N2.5-Max) stay rejected.

## Review fixes (Codex R1)
- 512 MiB indexer prefill workspace per unit (vLLM VLLM_SPARSE_INDEXER_MAX_LOGITS_MB); whole-prompt prefill of the reference generate.py is declared out of scope.
- layer_types is authoritative for sliding in every family (all full → no sliding).
- Cluster split also needs experts (incl. DSpark) and vocab to divide.
- V4.1 Engram tables never below FP8 + scale (8.25 bit).
- CSA2 candidate mask ratio must match later indexers.

## Decisions
- Catalog totalParams = HF logical safetensors sum (whole checkpoint incl. Engram/MTP/DSpark/vision): 763.2 / 290.9 / 1598.8 / 822.0.
- DeepSeek HF IDs keep the live HF path (no alias); only Muse's gated original ID is an alias.
- KV cache is replicated per tensor-parallel rank on GB10 clusters (reference code: wkv/Compressor/Indexer.wk are unsplit Linear).
