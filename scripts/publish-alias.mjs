#!/usr/bin/env node
// 별칭 패키지 발행 — 같은 tarball을 `fitllm` 이름으로도 올린다.
//
// 왜 필요한가: 우리가 3개월간 모든 표면(README·사이트·llmrepos·HF·이슈)에 광고한 CTA는
// `npx fitllm` 인데, npm에 실재하는 패키지는 `fitllm-engine` 뿐이라 E404가 났다.
// `fitllm`은 패키지 이름이 아니라 이 패키지 안의 bin 이름이었다 — 즉 전역 설치한 사람만 됐다.
// 이미 외부에 나간 문구는 회수할 수 없으므로, 문구를 고치는 대신 이름을 실재하게 만든다.
//
// 작업트리는 절대 수정하지 않는다(임시 디렉터리에 스테이징 후 그쪽에서 publish).
// 중간에 죽어도 레포에 name:"fitllm" 이 남지 않게 하려는 의도적 설계.
//
// usage:
//   node scripts/publish-alias.mjs --dry-run   # 검증만
//   node scripts/publish-alias.mjs             # 실제 발행

import { readFileSync, writeFileSync, mkdtempSync, cpSync, existsSync } from 'node:fs'
import { execFileSync } from 'node:child_process'
import { tmpdir } from 'node:os'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

const ALIAS_NAME = 'fitllm'
const root = join(dirname(fileURLToPath(import.meta.url)), '..')
const dryRun = process.argv.includes('--dry-run')

const pkg = JSON.parse(readFileSync(join(root, 'package.json'), 'utf8'))

if (pkg.name === ALIAS_NAME) {
  console.error(`정본 패키지 이름이 이미 "${ALIAS_NAME}" 이다 — 이 스크립트는 불필요하다.`)
  process.exit(2)
}

// 스테이징: files[] + npm이 항상 포함하는 README/LICENSE
const stage = mkdtempSync(join(tmpdir(), 'fitllm-alias-'))
for (const entry of [...pkg.files, 'README.md', 'LICENSE']) {
  const src = join(root, entry)
  if (!existsSync(src)) {
    console.error(`빠진 파일: ${entry}`)
    process.exit(2)
  }
  cpSync(src, join(stage, entry), { recursive: true })
}

// 별칭 manifest — 이름만 다르고 내용은 동일. scripts는 배포본에 필요없어 제거.
const { scripts, ...rest } = pkg
writeFileSync(
  join(stage, 'package.json'),
  JSON.stringify({ ...rest, name: ALIAS_NAME }, null, 2) + '\n'
)

// bin이 실제로 도는지 스테이징본으로 먼저 확인 — 깨진 걸 또 올리지 않기 위해
const binPath = join(stage, pkg.bin[ALIAS_NAME])
execFileSync(process.execPath, [binPath, '--help'], { stdio: 'ignore' })

console.log(`staged: ${stage}`)
console.log(`publish: ${pkg.name}@${pkg.version}  →  ${ALIAS_NAME}@${pkg.version}`)

execFileSync('npm', ['publish', '--access', 'public', ...(dryRun ? ['--dry-run'] : [])], {
  cwd: stage,
  stdio: 'inherit',
})

console.log(dryRun ? '\ndry-run 통과 (실제 발행 안 함)' : `\n발행 완료: ${ALIAS_NAME}@${pkg.version}`)
