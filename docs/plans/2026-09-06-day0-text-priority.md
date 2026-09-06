# Day-0 Text-Scope Priority Implementation Plan
Contract-Version: 1

**Goal:** Prevent newly discovered multimodal checkpoints from consuming all three day-0 mutation slots before text-generation candidates are evaluated. **Architecture:** Keep discovery, evidence pinning, capability classification, and both reviewed pipeline tags unchanged; reorder only the valid-revision evaluation queue into deterministic pipeline tiers, preserving the existing new-issue-before-update rule inside each tier. **Stack:** Node.js ESM, `node:test`, local read-only Hugging Face/GitHub API dry-run.

## Contract

- Source: `/tmp/fitllm-supervisor/day0-current-queue-audit-2026-09-06.md:1` (SHA-256 `af410a714e7f925b4b93733461f1f70acdb7b94d31c7c15de0922ffa2ad06e70`); `commit:865ac37066ce2e586326e8d41377643e351ab3d2`; user-message:`ㄱㄱ`.
- JTBD: When official model activity exceeds the three-mutation safety cap, maintainers see text-generation candidates relevant to FitLLM's numerical product before unsupported image/multimodal tracking candidates, without losing fail-closed evidence for either class.
- Preserve: Both reviewed pipeline tags, official namespace allowlist, discovery sources and ordering within a tier, evidence byte/revision/license gates, numeric-null capability classification, existing issue trust and human-text preservation, three-mutation cap, twelve-evidence cap, dry-run no-mutation contract, issue body format, engine math, package version, CLI, census, and all published surfaces.
- Approved removals: NONE
- Removal approval: N/A
- User-visible surfaces: github_issue_automation
- AC-1 [surface:github_issue_automation]: For valid revisions, `text-generation` candidates are evaluated before unknown-pipeline candidates, which are evaluated before `image-text-to-text` candidates.
- AC-2 [surface:github_issue_automation]: Within each pipeline tier, candidates without a trusted managed issue remain ahead of candidates with one, and the existing deterministic discovery order remains the final tie-breaker.
- AC-3 [surface:github_issue_automation]: Invalid or revision-conflicted candidates remain after every valid-revision tier and continue to produce fail-closed blocked records without starving a later valid candidate.
- AC-4 [internal]: Discovery still accepts both `text-generation` and `image-text-to-text`; capability classification, evidence manifests, mutation/evidence caps, issue trust, apply-time safety, and issue serialization are byte-behavior unchanged except for queue order.
- AC-5 [internal]: A regression fixture with at least three newer `image-text-to-text` candidates and text-generation candidates proves the first three planned mutations are text-generation and that multimodal candidates remain available after capacity is freed.
- AC-6 [internal]: Target tests and the full suite pass with zero skips; package version, engine, vectors, census, workflow permissions, and publish configuration are unchanged.
- AC-7 [surface:github_issue_automation]: A fresh local `npm run day0:dry-run` on the fixed commit performs no GitHub mutations and no remote workflow dispatch, reports zero source failures, and no longer spends all three current mutation slots on `image-text-to-text` candidates.

## Global Constraints

- Base commit is exactly `865ac37066ce2e586326e8d41377643e351ab3d2`; stop if the worktree starts elsewhere.
- TDD: add the mixed-pipeline starvation test and observe the intended failure before changing production code.
- Do not delete or exclude `image-text-to-text` candidates; this change is priority only.
- Do not trigger GitHub Actions, create/update/close issues, publish npm, tag, release, or deploy while implementing or reviewing.
- Do not change `engine.js`, `package.json`, `.github/day0-sources.json`, `.github/workflows/day0-watch.yml`, census files, or vectors.
- Commit author and committer must be `click6067-ship-it <click6067@gmail.com>`.
- One fixed-SHA review by a different backbone is required; another review round is allowed only for concrete findings.

## Target Tests

- AC-1 -> 실측 게이트: `test/day0-watch.test.mjs` mixed-pipeline mutation-cap order assertion
- AC-2 -> 실측 게이트: `test/day0-watch.test.mjs` per-tier new-before-update and deterministic-order assertion
- AC-3 -> 실측 게이트: existing invalid-revision starvation regression in `test/day0-watch.test.mjs`
- AC-4 -> 실측 게이트: existing discovery, classification, mutation-plan, and apply-time suites plus excluded-file audit
- AC-5 -> 실측 게이트: two new mixed-pipeline queue-order regression assertions
- AC-6 -> 실측 게이트: full `npm test`, `git diff --check`, and exact changed-file allowlist
- AC-7 -> 실측 게이트: local `npm run day0:dry-run` summary/manifest inspection with zero external writes

---

### Task 1: Seal mixed-pipeline starvation as a failing test

**Files:** Modify `test/day0-watch.test.mjs`. **Interfaces:** Consume the public `runDay0Watch()` API with sealed fake HF feeds and supplied GitHub issues; assert planned mutation order and retained candidates. **Target tests:** AC-1, AC-2, AC-5.

- [ ] Step 1: Add a local fixture helper that constructs pinned, parser-supported llama-like checkpoints while varying `pipeline_tag`, creation time, and whether a trusted managed issue already exists.
- [ ] Step 2: Build a feed where three newer `image-text-to-text` candidates currently fill the mutation cap while three older `text-generation` candidates are available; include both issue-less and trusted-issue cases inside a tier.
- [ ] Step 3: Assert the first three mutation model IDs are the text-generation candidates in new-before-update then existing deterministic order, and assert an image candidate is still present when `maxIssueMutationsPerRun` is raised in the test policy.
- [ ] Step 4: Run `node --test --test-name-pattern='pipeline|text-generation candidates' test/day0-watch.test.mjs`; record the pre-change failure showing image candidates currently win.

### Task 2: Apply the minimum valid-candidate queue ordering

**Files:** Modify `.github/scripts/day0-core.mjs`; Test `test/day0-watch.test.mjs`. **Interfaces:** Consume already-discovered candidates plus the existing trusted-issue partition; produce a valid-revision queue ordered by pipeline tier, issue presence, then `candidateSort`, followed by invalid revisions.

- [ ] Step 1: Add a private helper equivalent to:

```js
function pipelinePriority(tag) {
  if (tag === 'text-generation') return 0;
  if (tag === null || tag === undefined) return 1;
  return 2;
}
```

- [ ] Step 2: Build an `issuePriorityById` map (`0` without trusted issue, `1` with trusted issue), and sort a copy of `candidatesWithValidRevision` by pipeline priority, then issue priority, then the existing `candidateSort(a, b)`.
- [ ] Step 3: Form `evaluationCandidates` from that sorted valid queue followed by `candidatesWithInvalidRevision`; do not alter discovery output or any later evidence/mutation logic.
- [ ] Step 4: Run the new target tests, the existing starvation tests, and `node --test test/day0-watch.test.mjs`.

### Task 3: Verify current live queue locally and hand off a fixed SHA

**Files:** Modify only the two planned source/test files plus this plan; create receipts under `/tmp/fitllm-supervisor/`. **Interfaces:** Consume the fixed candidate commit; produce test logs, a local live dry-run summary, changed-file audit, and handoff report. **Target tests:** AC-3, AC-4, AC-6, AC-7.

- [ ] Step 1: Run `npm test`, `git diff --check`, and verify the changed-file set is exactly `.github/scripts/day0-core.mjs`, `test/day0-watch.test.mjs`, and this plan.
- [ ] Step 2: Run `npm run day0:dry-run` locally without mutation environment variables; verify mode `dry-run`, zero source failures, and inspect all three planned mutation manifests' pipeline tags.
- [ ] Step 3: Verify `engine.js`, `package.json`, `.github/day0-sources.json`, `.github/workflows/day0-watch.yml`, census, and vectors are byte-identical to base.
- [ ] Step 4: Commit with the required identity as `fix(day0): prioritize text candidates before mutation cap` and write `/tmp/fitllm-supervisor/day0-text-priority-fable-report-2026-09-06.md` with commit/tree/test/dry-run receipts and remaining limits.
- [ ] Step 5: Send `worker_done`; do not push, open a PR, run a remote workflow, mutate issues, publish, or deploy.
