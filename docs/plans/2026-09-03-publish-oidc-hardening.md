# Publish OIDC Hardening Implementation Plan
Contract-Version: 1

**Goal:** npm OIDC 권한을 검증·빌드 코드와 분리하고, 두 패키지의 고정된 tarball 무결성 및 provenance를 발행 전후에 강제한다. **Architecture:** 권한 없는 `verify` job이 exact tag/SHA를 검증하고 canonical·alias tarball 및 SHA512 manifest를 artifact로 만든다. protected `npm-publish` environment를 쓰는 최소 `publish` job만 OIDC를 받아 `--ignore-scripts`로 고정 artifact를 발행하고, 다시 권한 없는 `postcondition` job이 registry integrity와 provenance를 검증한다. **Stack:** GitHub Actions, Node.js 22.14.0, npm 11.5.1, `node:test`.

## Contract

- Source: user-message:/root/fix_pr110_oidc
- JTBD: `fitllm-engine`과 `fitllm`을 장기 토큰 없이 발행하되 tag-controlled 코드가 OIDC를 탈취하거나 부분 발행이 다른 payload로 굳어지는 경로를 차단한다.
- Preserve: exact `vX.Y.Z` tag 발행, package/engine version 일치, 전체 테스트·census gate, 기존 동일 버전의 안전한 멱등 재실행, readiness=false 유지.
- Approved removals: NONE
- Removal approval: N/A
- User-visible surfaces: GitHub_Actions, npm_registry
- AC-1 [surface:GitHub_Actions]: dispatch 입력은 env로만 셸에 들어가고 anchored `^v[0-9]+\.[0-9]+\.[0-9]+$`를 통과해야 한다.
- AC-2 [internal]: checkout/test/import/lifecycle 실행 job에는 `id-token: write`가 없으며 publish job만 protected `npm-publish` environment에서 OIDC를 받는다.
- AC-3 [internal]: push release는 `github.sha`와 tag commit이 일치하고 Node/npm은 exact version으로 고정된다.
- AC-4 [surface:npm_registry]: canonical/alias tarball과 expected SHA512가 한 artifact로 고정되고, 기존 및 신규 registry version의 `dist.integrity`가 expected 값과 정확히 일치해야 한다.
- AC-5 [surface:npm_registry]: canonical/alias payload는 package name 외 byte parity를 만족하고, 발행 뒤 두 버전 모두 provenance metadata 및 signature audit를 통과해야 한다.
- AC-6 [internal]: `NPM_TRUSTED_PUBLISHING_READY` 기본값/저장소 설정은 변경하지 않고 tag 생성·publish·merge를 수행하지 않는다.

## Global Constraints

- 작업 범위는 `/home/click/ghq/github.com/click6067-ship-it/fitllm-engine/.worktrees/publish-oidc`뿐이다.
- npm trusted publisher 두 항목의 environment name은 GitHub protected environment `npm-publish`와 정확히 같아야 한다.
- publish job은 checkout하지 않고 package-controlled 스크립트를 import/실행하지 않으며 모든 npm lifecycle을 무시한다.
- GitHub actions와 Node/npm 도구 버전은 변경 불가능한 exact pin을 사용한다.

## Target Tests

- AC-1 -> ../../test/publish-workflow.test.mjs::test_dispatch_tag_env_exact
- AC-2 -> ../../test/publish-workflow.test.mjs::test_oidc_minimal_job
- AC-3 -> ../../test/publish-workflow.test.mjs::test_push_sha_and_exact_tools
- AC-4 -> ../../test/publish-workflow.test.mjs::test_immutable_integrity_artifact
- AC-5 -> ../../test/publish-workflow.test.mjs::test_payload_parity_and_provenance
- AC-6 -> ../../test/publish-workflow.test.mjs::test_readiness_external_gate

---

### Task 1: 적대적 workflow 계약 테스트

**Files:** Create `test/publish-workflow.test.mjs`; Modify `package.json` test glob은 기존 `test/*.test.mjs`가 자동 포함하므로 변경 없음. **Interfaces:** Consumes `.github/workflows/publish.yml` text / Produces release security invariants. **Target tests:** 위 Target Tests 전체.

- [ ] Step 1: workflow를 job block 단위로 읽고 direct `${{ inputs.tag }}` shell interpolation, broad OIDC, 비고정 versions, checkout in publish, integrity/provenance 누락을 exact assertion으로 거부하는 테스트를 작성한다.
- [ ] Step 2: `node --test test/publish-workflow.test.mjs`가 현재 workflow의 broad `id-token: write`와 glob tag 검증 때문에 FAIL하는지 확인한다.
- [ ] Step 3: 테스트는 구현 문자열 전체 복제 대신 보안 불변식의 필수/금지 패턴만 소유하도록 유지한다.
- [ ] Step 4: 구현 후 `node --test test/publish-workflow.test.mjs` PASS를 확인한다.
- [ ] Step 5: 구현과 함께 한 커밋으로 묶는다.

### Task 2: 무권한 artifact 준비와 최소 OIDC publish 분리

**Files:** Modify `.github/workflows/publish.yml` 전체. **Interfaces:** Produces `release-artifacts` containing canonical tgz, alias tgz, `release-manifest.tsv`; Consumes that artifact only in publish/postcondition jobs. **Target tests:** AC-1~AC-5 tests.

- [ ] Step 1: Task 1 실패가 구조적 결함을 재현하는지 확인한다.
- [ ] Step 2: top-level `permissions: {}`와 job별 최소 권한, env-only exact regex tag resolver, push `github.sha` binding, exact setup-node/npm pins를 구현한다.
- [ ] Step 3: verify job에서 `npm ci --ignore-scripts`, 테스트·census, 두 deterministic pack, name 외 payload parity, SHA512 manifest 생성 및 artifact upload를 구현한다.
- [ ] Step 4: `npm-publish` environment의 publish job에만 `id-token: write`를 부여하고 checkout 없이 artifact SHA512 확인 → 기존 registry integrity preflight → missing tarball `npm publish --ignore-scripts` → 즉시 integrity 확인을 구현한다.
- [ ] Step 5: 권한 없는 postcondition job에서 두 package의 exact integrity, SLSA provenance metadata, `npm audit signatures`를 검증한다.

### Task 3: 로컬·원격 검증 및 PR 갱신

**Files:** Modify PR 110 body only after commit/push. **Interfaces:** Consumes committed diff / Produces auditable validation evidence. **Target tests:** `node --test test/publish-workflow.test.mjs`; `npm test`; `npm run census:check`; `npm run alias:dry-run`; YAML parse/static checks.

- [ ] Step 1: targeted test와 전체 npm test를 실행한다.
- [ ] Step 2: census determinism과 alias dry-run을 실행하고, 현재 `fitllm-engine@2.8.2` 및 `fitllm@2.8.2`가 미발행인지 read-only 조회한다.
- [ ] Step 3: workflow YAML parse와 금지 패턴 정적 검사를 실행하고 diff/status를 검토한다.
- [ ] Step 4: 지정 author로 의도한 파일만 커밋하고 branch를 push한 뒤 remote HEAD와 PR head SHA를 확인한다.
- [ ] Step 5: PR body에 job 경계·무결성/provenance 계약·남은 단 하나의 npm UI trusted-publisher/environment 연결 blocker를 반영한다.
