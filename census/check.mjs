#!/usr/bin/env node
// census 결정성 검증 — 커밋된 census와 같은 CENSUS_DATE로 재생성한 4개 산출물이 byte-identical한지.
// 실패 = 엔진 데이터/생성기와 커밋 산출물의 드리프트(조용한 스테일 방지). `npm run census:check`.
import { execFileSync } from 'node:child_process';
import { readFileSync, mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const dir = new URL('./', import.meta.url).pathname;
const committed = JSON.parse(readFileSync(join(dir, 'census-v1.json'), 'utf8'));
if (!/^\d{4}-\d{2}-\d{2}$/.test(committed.generated)) {
  console.error(`✗ committed census generated 필드 형식 오류: ${committed.generated}`);
  process.exit(1);
}
const out = mkdtempSync(join(tmpdir(), 'census-check-')) + '/';
let ok = true;
try {
  execFileSync(process.execPath, [join(dir, 'generate.mjs')], {
    env: { ...process.env, CENSUS_DATE: committed.generated, CENSUS_OUT: out },
    stdio: ['ignore', 'ignore', 'inherit'],
  });
  for (const f of ['census-v1.json', 'census-v1.csv', 'manifest.json', 'README.md']) {
    const a = readFileSync(join(dir, f));
    const b = readFileSync(join(out, f));
    if (a.equals(b)) console.log(`✓ ${f} byte-identical`);
    else { console.error(`✗ ${f} differs from a deterministic regeneration`); ok = false; }
  }
} finally {
  // 임시 산출물은 성공/실패(재생성 오류·비교 불일치 포함) 어느 경로에서도 남기지 않는다 — /tmp 누수 차단
  rmSync(out, { recursive: true, force: true });
}
process.exit(ok ? 0 : 1);
