# Distribution Preflight Implementation Plan
Contract-Version: 1

**Goal:** FitLLM을 다운로드 전 CI·터미널 검문소로 만들고, 판정 근거와 경쟁 출력이 에이전트가 검증할 수 있는 형태로 남게 한다. **Architecture:** 기존 `simulate()`를 단일 계산 정본으로 유지하고 설명·GitHub Action·벤치마크는 그 결과를 소비하는 얇은 표면으로만 추가한다. 경쟁 비교는 고정 버전의 원출력과 SHA-256을 저장하되 실측 정확도 claim gate에는 독립 검증 자료만 들어간다. **Stack:** Node.js ESM, zero-dependency CLI, composite GitHub Action, node:test.

## Contract

- Source: user-message:고레버리지작업-남은단계-전부-진행, `docs/strategy/2026-09-03-competitor-benchmark.md:98`, commit:794f806c89568333e59327433465351f3ff6ece5
- JTBD: 모델을 받거나 CI 작업을 시작하기 전에 사용자는 동일한 FitLLM 판정으로 실패를 차단하고, 사람·에이전트는 그 판정의 입력과 한계를 감사할 수 있다.
- Preserve: 기존 CLI 인자·기본 출력·exit 0/1/2, fail-closed HF/하드웨어 처리, `simulate()` 계산, receipt 계약, 비교 정확도 claim gate.
- Approved removals: NONE
- Removal approval: N/A
- User-visible surfaces: npm_cli, github_action, github_readme, benchmark_report
- AC-1 [surface:npm_cli]: `--why --json`은 모델·하드웨어·quant·context·attention 구조·각 메모리 항·출처·가정을 구조화해 내고 기존 결과 값과 정확히 일치한다.
- AC-2 [surface:npm_cli]: `--why` 없는 기존 JSON/텍스트 계약과 exit code는 바뀌지 않는다.
- AC-3 [surface:github_action]: action은 model과 정확히 하나의 gpu/mac 입력을 받아 저장소 안의 CLI를 실행하며 FITS/TIGHT는 성공, WON'T FIT은 실패, 잘못된 입력은 오류로 구분한다.
- AC-4 [internal]: Action 입력은 GitHub expression을 셸 명령 문자열로 삽입하지 않고 환경변수→Bash 배열로 전달한다.
- AC-5 [surface:benchmark_report]: llmfit v1.1.12 공식 Linux artifact의 공식 SHA-256을 확인한 뒤 3개 고정 명령의 절단 없는 JSON 원출력과 각 출력 SHA-256을 재생성한다.
- AC-6 [surface:benchmark_report]: differential report는 Llama 표준 GQA 대조군과 Gemma sliding-window·GLM MLA 반례를 분리하며 이를 실측 정확도 증거로 세지 않는다.
- AC-7 [internal]: `benchmark:accuracy`는 committed competitor rows를 실제로 읽지만 독립 검증 measurement와 caseId가 없으면 claim gate가 계속 BLOCKED다.
- AC-8 [surface:github_readme]: README는 GitHub Action, Ollama/llama.cpp preflight, `--why`, 경쟁 원출력 재현 명령을 복사 가능한 형태로 제공한다.

## Global Constraints

- 새 계산식을 Action·설명·벤치마크에 복제하지 않는다.
- 속도 순위와 일반적 “가장 정확함” 주장은 금지한다.
- 경쟁 출력은 공식 릴리스 artifact, 버전, checksum, 전체 stdout을 고정한다.
- Action은 checkout 외 네트워크·시크릿을 요구하지 않는다.
- 외부 저장소에 PR/게시물을 자동 제출하지 않는다.

## Target Tests

- AC-1 -> `test/explain.test.mjs::structured basis mirrors simulation`
- AC-2 -> `test/cli.test.mjs::without --why preserves legacy JSON shape`
- AC-3 -> `test/action.test.mjs::composite action propagates CLI exit contracts`
- AC-4 -> `test/action.test.mjs::action inputs use env and argv arrays`
- AC-5 -> `test/differential-report.test.mjs::manifest pins release and three raw outputs`
- AC-6 -> `test/differential-report.test.mjs::report labels controls counterexamples and non-claim status`
- AC-7 -> `test/accuracy-report.test.mjs::committed competitors cannot bypass independent evidence`
- AC-8 -> `test/readme-claims.test.mjs::distribution preflight examples stay present`

---

### Task 1: 판정 근거 모델

**Files:** Create `bin/explain.mjs`, `test/explain.test.mjs`. **Interfaces:** Consumes `{ model, device, simulation, quantLabel, kvBits, ctx, modelSource, detection }`; Produces `buildExplanation(input)` plain JSON. **Target tests:** AC-1.

- [x] Step 1: Gemma sliding-window, GLM MLA, Qwen hybrid-linear, Llama GQA의 정확 필드와 simulation breakdown 동일성을 단언하는 실패 테스트를 작성한다.
- [x] Step 2: `node --test test/explain.test.mjs`가 missing module로 실패함을 확인한다.
- [x] Step 3: model의 실재 필드만 복사하고 `attention.kind`를 `hybrid-linear|mla|sliding-window|gqa`, `memoryGiB`를 `simulation`에서 만드는 순수 함수를 구현한다.
- [x] Step 4: exact deepEqual 테스트와 전체 `npm test`를 통과시킨다.

### Task 2: `--why` CLI 연결

**Files:** Modify `bin/fitllm.mjs` help/JSON/text result paths; Modify `test/cli.test.mjs`. **Interfaces:** Consumes `buildExplanation()`; Produces optional `basis` JSON or deterministic text lines. **Target tests:** AC-1, AC-2.

- [x] Step 1: `--why --json` basis와 무플래그 legacy key set을 exact 비교하는 실패 테스트를 작성한다.
- [x] Step 2: 기존 CLI가 basis를 내지 않아 실패하는 것을 확인한다.
- [x] Step 3: `has('--why')`일 때만 explanation을 만들고 JSON에는 `basis`, 텍스트에는 model/attention/hardware/assumptions 4행을 추가한다.
- [x] Step 4: CLI·invariant·vector 전체 회귀를 통과시킨다.

### Task 3: GitHub Action preflight

**Files:** Create `action.yml`, `scripts/action-entry.sh`, `test/action.test.mjs`; Modify `README.md`. **Interfaces:** Consumes action inputs `model`, `gpu|mac`, `quant`, `ctx`, `kv`, `count`; Produces `result` JSON output and CLI-compatible job status. **Target tests:** AC-3, AC-4, AC-8.

- [x] Step 1: action metadata, one-of device validation, env-only expressions, Bash array quoting, exit propagation을 검사하는 실패 테스트를 작성한다.
- [x] Step 2: action 파일 부재 실패를 확인한다.
- [x] Step 3: composite action과 entry script를 구현한다. entry script는 `args=("$INPUT_MODEL")`, 선택 입력마다 `args+=(--flag "$value")`, 마지막에 `node "$GITHUB_ACTION_PATH/bin/fitllm.mjs" "${args[@]}" --json`만 호출한다.
- [x] Step 4: 임시 `GITHUB_OUTPUT`에서 fits=0, no-fit=1, invalid=2와 multiline result를 실제 셸 실행으로 검증한다.

### Task 4: 고정 경쟁 differential ledger

**Files:** Create `scripts/capture-llmfit.mjs`, `scripts/differential-report.mjs`, `benchmarks/llmfit-v1.1.12/manifest.json`, three raw JSON files, `test/differential-report.test.mjs`; Modify `package.json`, `scripts/accuracy-report.mjs`, `benchmarks/README.md`, `test/accuracy-report.test.mjs`. **Interfaces:** Capture consumes verified llmfit binary path; Produces exact raw stdout+manifest. Report consumes manifest+engine; Produces controls/counterexamples with `evidenceClass=architecture_differential_not_runtime_accuracy`. **Target tests:** AC-5, AC-6, AC-7.

- [x] Step 1: artifact checksum, binary version, case count, raw SHA, exact fp16 KV fields, non-claim label을 요구하는 실패 테스트를 작성한다.
- [x] Step 2: 빈 benchmark tree가 실패함을 확인한다.
- [x] Step 3: official artifact SHA `6a97338862c87e497c844ccd29a16512a147335631c179744b4f6cc87a36ead1`과 `llmfit 1.1.12`를 확인한 뒤 Llama-3.1-8B, Gemma-4-31B, GLM-4.7-Flash `plan --context 8192 --quant Q4_K_M --json` 전체 stdout을 저장하는 capture를 구현한다.
- [x] Step 4: raw JSON의 fp16 KV를 같은 모델·ctx의 FitLLM `simulate()`와 비교하고 표준 GQA 일치/구조 반례 차이를 분리한 Markdown/JSON 보고서를 구현한다.
- [x] Step 5: committed `benchmarks/competitors.json`을 accuracy report에 전달하되 빈/부적격 자료가 gate를 절대 통과하지 못함을 테스트한다.

### Task 5: 공개 계약·릴리스 검증

**Files:** Modify `README.md`, `package.json`, `engine.js`; Generate census artifacts only if version is serialized there. **Interfaces:** Consumes completed features; Produces `2.9.0` package candidate and reproducible docs. **Target tests:** all above.

- [x] Step 1: README examples를 테스트로 고정하고 실패를 확인한다.
- [x] Step 2: `--why`, Action exact tag, Ollama/llama.cpp `fitllm && downloader` preflight, differential reproduction을 문서화한다.
- [x] Step 3: package/engine version을 함께 `2.9.0`으로 올리고 census 결정성을 재생성한다.
- [x] Step 4: `npm test`, vectors, `census:check`, Action smoke, differential report, clean `npm pack --dry-run`을 실행한다.
- [ ] Step 5: 지정 author로 의도 파일만 커밋하고 PR·CI·protected npm release를 수행한다.
