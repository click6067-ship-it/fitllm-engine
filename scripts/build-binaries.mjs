#!/usr/bin/env node
// 단일 실행 바이너리 빌드 — Node 설치 없이 `fitllm` 이 돌게 한다.
//
// 왜 이게 가능한가: 이 패키지는 런타임 의존성이 0개이고 `node:child_process` 하나 말고는
// 전부 상대 import 다. 그래서 엔진 코드를 **한 줄도 바꾸지 않고** 런타임째로 감싸기만 하면 된다.
// 엔진을 Rust/Go 로 포팅하지 않는 이유도 같다 — engine.js 가 자산이고, 포팅은 진실의 사본을
// 두 벌 만들어 둘이 갈라지게 하는 일이다.
//
// 크기 정직성: Bun 런타임이 통째로 박히므로 타깃당 60–94 MB 다(darwin-arm64 가 가장 작다).
// 같은 부류의 Rust 도구가 5–6 MB 인 것에 비하면 확실히 크다. 이 사실은 README 에도 그대로 적는다.
//
// usage: node scripts/build-binaries.mjs [--out <dir>] [--targets <a,b>] [--skip-parity]
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { mkdirSync, readFileSync, readdirSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const ENTRY = join(ROOT, 'bin', 'fitllm.mjs');

// 봉인된 타깃 목록. `os`/`arch`/`libc` 는 install.sh 가 `uname` 결과로 고르는 키라
// 여기서 바꾸면 설치 스크립트도 같이 바꿔야 한다(양쪽 테스트가 이 대응을 고정한다).
export const TARGETS = Object.freeze([
  { bun: 'bun-darwin-arm64', os: 'darwin', arch: 'arm64', libc: null, ext: '' },
  { bun: 'bun-darwin-x64', os: 'darwin', arch: 'x64', libc: null, ext: '' },
  { bun: 'bun-linux-x64', os: 'linux', arch: 'x64', libc: 'gnu', ext: '' },
  { bun: 'bun-linux-arm64', os: 'linux', arch: 'arm64', libc: 'gnu', ext: '' },
  { bun: 'bun-linux-x64-musl', os: 'linux', arch: 'x64', libc: 'musl', ext: '' },
  { bun: 'bun-linux-arm64-musl', os: 'linux', arch: 'arm64', libc: 'musl', ext: '' },
  { bun: 'bun-windows-x64', os: 'windows', arch: 'x64', libc: null, ext: '.exe' },
]);

export function assetName(version, target) {
  const libc = target.libc === 'musl' ? '-musl' : '';
  return `fitllm-v${version}-${target.os}-${target.arch}${libc}${target.ext}`;
}

// 호스트에서 바로 실행할 수 있는 타깃인지 — parity 검증을 돌릴 수 있는 대상을 고른다.
export function isHostTarget(target, platform = process.platform, arch = process.arch) {
  const osMatch = (platform === 'darwin' && target.os === 'darwin')
    || (platform === 'linux' && target.os === 'linux' && target.libc === 'gnu')
    || (platform === 'win32' && target.os === 'windows');
  const archMatch = (arch === 'arm64' && target.arch === 'arm64') || (arch === 'x64' && target.arch === 'x64');
  return osMatch && archMatch;
}

const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');

function parseArgs(argv) {
  const out = { outDir: join(ROOT, 'dist-bin'), targets: null, parity: true };
  for (let i = 0; i < argv.length; i += 1) {
    if (argv[i] === '--out') { out.outDir = argv[i + 1]; i += 1; }
    else if (argv[i] === '--targets') { out.targets = argv[i + 1].split(','); i += 1; }
    else if (argv[i] === '--skip-parity') { out.parity = false; }
    else throw new Error(`unknown argument: ${argv[i]}`);
  }
  return out;
}

// 바이너리가 npm 경로와 **문자 단위로 같은 답**을 내는지. 여기서 갈리면 배포하면 안 된다 —
// 사용자는 설치 경로에 따라 다른 숫자를 보게 되고, 그게 이 제품에서 가장 나쁜 실패다.
const PARITY_CASES = Object.freeze([
  ['MiniCPM5-2B', '--gpu', 'RTX 4090'],
  ['Gemma 4 31b', '--gpu', 'RTX 5090'],
  ['Nex-N2.5-Pro', '--mac', '512'],
]);

function runCapture(cmd, args) {
  try {
    return execFileSync(cmd, args, { encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] });
  } catch (error) {
    // 비-0 종료도 정상 경로다(fit 안 되면 exit 1) — stdout 을 그대로 비교 대상으로 쓴다.
    if (error.stdout != null) return error.stdout;
    throw error;
  }
}

export function buildAll({ outDir, targets, parity } = parseArgs(process.argv.slice(2))) {
  const version = JSON.parse(readFileSync(join(ROOT, 'package.json'), 'utf8')).version;
  const selected = targets ? TARGETS.filter((t) => targets.includes(t.bun)) : TARGETS;
  if (selected.length === 0) throw new Error('no targets selected');
  rmSync(outDir, { recursive: true, force: true });
  mkdirSync(outDir, { recursive: true });

  const assets = [];
  for (const target of selected) {
    const name = assetName(version, target);
    const outfile = join(outDir, name);
    execFileSync('bun', ['build', ENTRY, '--compile', `--target=${target.bun}`, '--outfile', outfile], {
      cwd: ROOT, stdio: ['ignore', 'pipe', 'inherit'],
    });
    const bytes = readFileSync(outfile);
    const digest = sha256(bytes);
    writeFileSync(`${outfile}.sha256`, `${digest}  ${name}\n`);
    assets.push({
      name, target: target.bun, os: target.os, arch: target.arch, libc: target.libc,
      bytes: bytes.length, sha256: digest,
    });
    process.stderr.write(`built ${name} — ${(bytes.length / 1048576).toFixed(1)} MB\n`);

    if (parity && isHostTarget(target)) {
      for (const args of PARITY_CASES) {
        const fromBinary = runCapture(outfile, args);
        const fromNode = runCapture(process.execPath, [ENTRY, ...args]);
        if (fromBinary !== fromNode) {
          throw new Error(`parity mismatch on ${name} for ${args.join(' ')}\n--- binary ---\n${fromBinary}\n--- node ---\n${fromNode}`);
        }
      }
      process.stderr.write(`  parity ok (${PARITY_CASES.length} cases, byte-identical to the node path)\n`);
    }
  }

  const manifest = {
    schema_version: 1,
    name: 'fitllm',
    version,
    built_from: 'bun build --compile',
    // 크기를 숨기지 않는다 — 설치 문서가 이 값을 그대로 인용한다.
    note: 'Each binary embeds the Bun runtime, so it is much larger than a Rust/Go equivalent. No Node installation is required to run it.',
    assets: assets.sort((a, b) => a.name.localeCompare(b.name)),
  };
  writeFileSync(join(outDir, 'manifest.json'), `${JSON.stringify(manifest, null, 2)}\n`);
  return manifest;
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const manifest = buildAll(parseArgs(process.argv.slice(2)));
  process.stdout.write(`${JSON.stringify({ version: manifest.version, assets: manifest.assets.length })}\n`);
}
