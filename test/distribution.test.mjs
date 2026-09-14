// 배포 파이프라인 계약 — 바이너리 이름·설치 스크립트·패키징 매니페스트가 서로 어긋나지 않는가.
//
// 여기서 지키는 실패는 구체적이다: 빌드가 내는 자산 이름과 install.sh 가 조립하는 이름이
// 한 글자라도 다르면, 스크립트는 404 를 받고 사용자는 "설치가 안 되네" 로 끝난다. 그 대응을
// 사람 눈으로 맞춰 두면 반드시 언젠가 어긋나므로 기계가 붙든다.
import { test } from 'node:test';
import { strict as assert } from 'node:assert';
import { readFileSync } from 'node:fs';
import { TARGETS, assetName, isHostTarget } from '../scripts/build-binaries.mjs';
import { renderFormula, renderScoop } from '../scripts/gen-packaging.mjs';

const read = (rel) => readFileSync(new URL(`../${rel}`, import.meta.url), 'utf8');
const INSTALL_SH = read('packaging/install.sh');
const VERSION = JSON.parse(read('package.json')).version;

test('설치 스크립트가 조립하는 자산 이름 = 빌드가 내는 자산 이름', () => {
  // install.sh: asset="fitllm-v${version}-${os}-${arch}${libc}"  (libc 는 "" 또는 "-musl")
  assert.match(INSTALL_SH, /asset="fitllm-v\$\{version\}-\$\{os\}-\$\{arch\}\$\{libc\}"/);
  for (const target of TARGETS) {
    if (target.os === 'windows') continue; // install.sh 는 POSIX 전용 — 아래에서 따로 검증
    const libc = target.libc === 'musl' ? '-musl' : '';
    assert.equal(assetName(VERSION, target), `fitllm-v${VERSION}-${target.os}-${target.arch}${libc}`);
  }
});

test('install.sh 의 uname 매핑이 빌드 타깃 집합을 정확히 덮는다', () => {
  const posix = TARGETS.filter((t) => t.os !== 'windows');
  // 스크립트가 인정하는 os/arch 토큰
  for (const os of new Set(posix.map((t) => t.os))) {
    assert.ok(INSTALL_SH.includes(`os="${os}"`), `install.sh 가 ${os} 를 매핑하지 않는다`);
  }
  for (const arch of new Set(posix.map((t) => t.arch))) {
    assert.ok(INSTALL_SH.includes(`arch="${arch}"`), `install.sh 가 ${arch} 를 매핑하지 않는다`);
  }
  // musl 타깃을 빌드하면서 스크립트가 musl 을 감지하지 않으면 alpine 사용자는 깨진 바이너리를 받는다.
  assert.ok(posix.some((t) => t.libc === 'musl'));
  assert.ok(INSTALL_SH.includes('libc="-musl"'));
  assert.match(INSTALL_SH, /alpine-release|ldd --version/);
});

test('install.sh 는 검증 전에 PATH 로 실행파일을 옮기지 않는다 (fail-closed 순서)', () => {
  const verifyAt = INSTALL_SH.indexOf('checksum mismatch');
  const moveAt = INSTALL_SH.indexOf('mv "$tmp/fitllm"');
  assert.ok(verifyAt > 0 && moveAt > 0);
  assert.ok(verifyAt < moveAt, '체크섬 대조가 설치보다 뒤에 있다 — 변조된 바이너리가 PATH 에 들어간다');
  // 체크섬 파일이 없으면 "그냥 설치" 로 새지 않는다.
  assert.match(INSTALL_SH, /checksum file not found[^\n]*refusing to install/);
  assert.match(INSTALL_SH, /^set -eu$/m);
});

test('패키징 매니페스트는 릴리스 자산만 가리키고, 빠진 타깃이면 생성을 거부한다', () => {
  const manifest = {
    schema_version: 1,
    version: VERSION,
    assets: TARGETS.map((t) => ({
      name: assetName(VERSION, t), os: t.os, arch: t.arch, libc: t.libc,
      bytes: 1, sha256: 'a'.repeat(64),
    })),
  };
  const rb = renderFormula(manifest);
  const scoop = JSON.parse(renderScoop(manifest));
  for (const t of TARGETS.filter((x) => ['darwin', 'linux'].includes(x.os) && x.libc !== 'musl')) {
    assert.ok(rb.includes(assetName(VERSION, t)), `formula 에 ${t.os}/${t.arch} 자산이 없다`);
  }
  assert.equal(scoop.version, VERSION);
  assert.ok(scoop.architecture['64bit'].url.endsWith('windows-x64.exe'));

  // 타깃이 빠지면 조용히 건너뛰지 않고 던진다.
  const missing = { ...manifest, assets: manifest.assets.filter((a) => a.os !== 'darwin') };
  assert.throws(() => renderFormula(missing), /missing darwin/);
});

test('배포 표면에 속도 주장이 새지 않는다', () => {
  // 2026-06-05 봉인 결정(속도 예측 거부)은 새 표면에도 그대로 적용된다.
  for (const [label, text] of [['install.sh', INSTALL_SH], ['formula', read('packaging/fitllm.rb')]]) {
    assert.doesNotMatch(text, /tok\/s|tokens per second|faster than/i, `${label} 에 속도 주장이 있다`);
  }
});

test('isHostTarget 는 실행 가능한 타깃만 parity 검증 대상으로 고른다', () => {
  const linuxGnuX64 = TARGETS.find((t) => t.os === 'linux' && t.arch === 'x64' && t.libc === 'gnu');
  assert.equal(isHostTarget(linuxGnuX64, 'linux', 'x64'), true);
  assert.equal(isHostTarget(linuxGnuX64, 'darwin', 'x64'), false);
  // musl 바이너리는 glibc 호스트에서 실행을 보장할 수 없으므로 parity 대상이 아니다.
  const musl = TARGETS.find((t) => t.libc === 'musl' && t.arch === 'x64');
  assert.equal(isHostTarget(musl, 'linux', 'x64'), false);
});
