# Detect, Measure, and Accuracy Ledger Implementation Plan
Contract-Version: 1

**Goal:** `npx fitllm`이 지원 가능한 로컬 하드웨어를 안전하게 자동 감지하고, Hugging Face ID를 fail-closed로 계산하며, 사용자가 실제 측정을 표준 형식으로 보고하고 누구나 정확도를 재계산하게 한다. **Architecture:** OS 명령 출력 파싱을 순수 함수로 분리하고 NVIDIA 실측 VRAM, Windows/WSL의 exact catalog identity, Apple Silicon RAM만 신뢰한다. CLI는 원격 HF config를 제한적으로 가져오고, 측정 보고서는 제출하지 않고 검증된 prefilled issue payload만 만든다. 정확도 원장은 measurement kind별 동종 지표만 비교한다. **Stack:** Node.js >=18, zero runtime dependencies, node:test, existing single-file engine.

## Contract

- Source: user-message:내-컴퓨터-자동-감지, user-message:전략-전부-승인, user-message:cc와-피어-협업-1위
- JTBD: 사용자는 하드웨어 이름을 몰라도 다운로드 전에 로컬 CLI가 지원 장비를 찾게 하고, 새 모델 ID와 실측 데이터를 같은 감사 가능한 루프로 연결한다.
- Preserve: CLI exit 0/1/2 계약, zero dependencies, 기존 `--gpu`/`--mac`/`--top`, canonical receipt 제한, fail-closed architecture gate, no tokens/sec predictions, engine math and conformance vectors.
- Approved removals: NONE
- Removal approval: N/A
- User-visible surfaces: npm_cli, README, measurement_issue, accuracy_report, CI
- AC-1 [surface:npm_cli]: `nvidia-smi`의 모든 유효 GPU 행을 읽고 각 행의 실제 VRAM을 사용해 멀티GPU를 합산한다. 잘못된 행이나 8장 초과는 exit 2다.
- AC-2 [surface:npm_cli]: NVIDIA catalog 승격은 exact identity+VRAM 일치만 허용하고, 미등록/Laptop 변형은 실제 감지 이름·VRAM으로 계산하되 canonical receipt를 만들지 않는다.
- AC-3 [surface:npm_cli]: Windows와 WSL에서 NVIDIA가 없으면 PowerShell의 adapter 이름을 읽되 exact 단일 catalog GPU만 사용한다. Intel·generic·모호한 VRAM 변형은 지원 장비로 승격하지 않고 감지된 이름을 포함한 설명형 exit 2를 낸다.
- AC-4 [surface:npm_cli]: macOS는 `arm64`일 때만 unified memory를 사용하며 Intel Mac은 exit 2다. OS별 GPU reserve는 보수적으로 명시한다.
- AC-5 [surface:npm_cli]: `--json`은 하위호환 판정 필드와 함께 `detection.source`, `confidence`, adapter 목록을 노출하며 민감한 장치 식별자·명령 원문은 싣지 않는다.
- AC-6 [surface:npm_cli]: catalog에 없는 `org/model` 또는 Hugging Face URL은 8초 제한·1MB config 상한으로 공식 config/index를 읽고 `parseHfConfig` gate를 통과한 경우만 계산한다. 실패·게이트·미지원 구조는 exit 2이며 임의 근사나 영수증을 만들지 않는다.
- AC-7 [surface:measurement_issue]: `npx fitllm measure <model> ...`는 measured value, kind, unit, runtime/version, exact conditions를 검증해 JSON 후보와 prefilled GitHub issue URL을 출력하지만 자동 제출하지 않는다.
- AC-8 [surface:accuracy_report]: 정확도 리포트는 `idle_resident`↔predicted resident, `system_total_peak`↔predicted total만 비교하고 단위 변환·제외 이유·evidence level·표본 수를 공개한다.
- AC-9 [surface:accuracy_report]: 공개 “accuracy #1” 주장은 최소 30개 독립 재현 측정, 3개 runtime, 6개 architecture family와 pinned competitor 결과에서 MAPE와 false-fit rate 모두 우위일 때만 허용한다. 3개 파일럿에서 방향성이 없거나 본 게이트에서 10% 상대 MAPE 개선이 없으면 해당 포지셔닝을 폐기한다.
- AC-10 [surface:README]: one-command detect, HF ID, measurement report, benchmark 재현, CI pre-download guard를 복사 가능한 명령으로 문서화하고 현재 증거 수준을 과장하지 않는다.
- AC-11 [surface:CI]: 기존 전체 테스트·conformance·census·alias dry-run이 통과하고 새 정확도 리포트가 결정적으로 재현된다.

## Global Constraints

- 장치 serial, PNPDeviceID, 전체 환경변수, 사용자 경로를 stdout/보고서에 기록하지 않는다.
- `Win32_VideoController.AdapterRAM`의 32-bit 한계를 VRAM 근거로 사용하지 않는다.
- 실제 측정과 예측은 종류와 단위가 같을 때만 오차를 계산한다.
- 외부 바이너리를 다운로드·실행하거나 GitHub issue를 자동 생성하지 않는다.
- `engine.js` 수식 변경은 이번 계획의 기본 범위가 아니며 필요 시 별도 증거와 vector 변경 계약이 필요하다.

## Target Tests

- AC-1 -> ../../test/detect-hardware.test.mjs::test_multi_nvidia
- AC-2 -> ../../test/detect-identity.test.mjs::test_exact_identity_and_vram
- AC-3 -> ../../test/detect-hardware.test.mjs::test_windows_wsl_fallback
- AC-4 -> ../../test/detect-hardware.test.mjs::test_apple_silicon_only
- AC-5 -> ../../test/cli-detect.test.mjs::test_json_detection_metadata
- AC-6 -> ../../test/hf-model.test.mjs::test_bounded_fail_closed_fetch
- AC-7 -> ../../test/measurement-report.test.mjs::test_validated_prefill_without_submit
- AC-8 -> ../../test/accuracy-report.test.mjs::test_typed_metric_pairing
- AC-9 -> ../../test/accuracy-report.test.mjs::test_public_claim_gate
- AC-10 -> ../../test/readme-claims.test.mjs::test_approved_cli_workflows
- AC-11 -> 수동: full test, census, alias, accuracy commands

---

### Task 1: 하드웨어 감지 계층 분리·확장

**Files:** Create `bin/detect-hardware.mjs`, `test/detect-hardware.test.mjs`, `test/cli-detect.test.mjs`; Modify `bin/detect-resolver.mjs`, `bin/fitllm.mjs`, `test/detect-identity.test.mjs`. **Interfaces:** Consumes bounded OS command output; Produces typed detection result consumed by existing device constructors. **Target tests:** AC-1..AC-5.

- [x] Step 1: multi-NVIDIA, malformed output, Laptop mismatch, Windows AMD exact name, ambiguous 4060 Ti, WSL Intel, Apple Silicon, Intel Mac 반례를 먼저 실패시킨다.
- [x] Step 2: injectable executor를 쓰는 detector와 exact name resolver를 구현한다.
- [x] Step 3: CLI가 detection metadata를 JSON에 추가하고 기존 비감지 JSON/exit 계약이 유지되게 한다.
- [x] Step 4: 이 컴퓨터(WSL + Intel Iris Xe)에서 unsupported를 정확히 보고하는지 실측한다.

### Task 2: Hugging Face ID fail-closed CLI

**Files:** Create `bin/hf-model.mjs`, `test/hf-model.test.mjs`; Modify `bin/fitllm.mjs`, `test/cli.test.mjs`. **Interfaces:** Consumes `org/model` or HF URL; Produces parsed engine model or typed exit-2 error. **Target tests:** AC-6.

- [x] Step 1: strict ID, timeout, oversized config, gated/404, malformed JSON, unsupported architecture, valid config fixtures를 작성한다.
- [x] Step 2: Node built-in fetch와 AbortController만 사용해 config와 total size를 가져온다.
- [x] Step 3: custom HF model은 결과에 source ID를 남기고 receipt를 `n/a`로 처리한다.
- [x] Step 4: 네트워크 없는 단위 테스트와 허용된 실제 public 모델 1건 smoke test를 분리한다.

### Task 3: 한 명령 측정 보고 루프

**Files:** Create `bin/measurement-report.mjs`, `test/measurement-report.test.mjs`; Modify `bin/fitllm.mjs`, `.github/ISSUE_TEMPLATE/measurement.yml`, `fixtures/README.md`. **Interfaces:** Consumes model/device/quant/context/measurement metadata; Produces validated issue body, URL, and candidate payload without submission. **Target tests:** AC-7.

- [x] Step 1: missing runtime version, invalid kind/unit/value, unsupported detected hardware, raw secret-like fields를 거부하는 테스트를 작성한다.
- [x] Step 2: `measure` subcommand가 기존 판정을 재사용하고 예측·실측 종류를 명확히 병기하게 한다.
- [x] Step 3: issue template 필드를 fixture schema와 맞추고 자동 제출이 없음을 문서화한다.

### Task 4: 공개 정확도 원장과 경쟁 kill gate

**Files:** Create `benchmarks/README.md`, `scripts/accuracy-report.mjs`, `test/accuracy-report.test.mjs`; Modify `package.json`, `README.md`. **Interfaces:** Consumes census + typed measured fixtures (+ future pinned competitor results); Produces deterministic Markdown/JSON coverage and error metrics. **Target tests:** AC-8..AC-10.

- [x] Step 1: synthetic measured fixtures로 올바른 pair, 단위 변환, unmatched/unknown 제외를 검증한다.
- [x] Step 2: 현재 원장의 실제 표본 수·근거 등급을 산출하고 부족하면 부족하다고 출력한다.
- [x] Step 3: competitor 입력 포맷, pinned version, architecture/runtime coverage, false-fit 우선 gate와 kill 조건을 문서화한다.
- [x] Step 4: README에 공개 재현 명령과 현재 claim 상태를 추가한다.

### Task 5: 검증·CC 반례 리뷰·통합 준비

**Files:** Modify only defects found in planned files. **Interfaces:** Consumes completed diff; Produces full test/vector evidence and CC review resolution. **Target tests:** all above.

- [x] Step 1: 타깃 테스트, `npm test`, `npm run census:check`, `npm run alias:dry-run`, `npm run benchmark:accuracy`를 실행한다.
- [x] Step 2: CLI 실제 `--top --detect --json`과 public HF 모델 smoke를 실행하고 stdout에 민감정보가 없는지 확인한다.
- [x] Step 3: CC Opus에 contract와 diff를 주어 감지 오인·측정 비교 오류·과장 claim을 독립 리뷰시킨다.
- [ ] Step 4: 지정 author로 저장소별 의도 파일만 커밋하고 원격 HEAD/PR/CI는 실제 수행한 범위만 보고한다.
