// --detect 신원 해석 회귀 (Sol P1: Laptop/Ti 부분문자열 오매칭 → 거짓 yes + 무효 영수증).
// 1) 순수 리졸버 단위검증  2) fake nvidia-smi(PATH 주입) 서브프로세스 종단검증.
import test from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdtempSync, writeFileSync, chmodSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { resolveDetectedGpu } from '../bin/detect-resolver.mjs';
import { GPUS } from '../engine.js';

test('test_exact_identity_and_vram', () => {
  assert.equal(resolveDetectedGpu('NVIDIA GeForce RTX 4090', 24, GPUS)?.name, 'RTX 4090');
  assert.equal(resolveDetectedGpu('NVIDIA GeForce RTX 4090', 16, GPUS), null);
  assert.equal(resolveDetectedGpu('NVIDIA GeForce RTX 4090 Laptop GPU', 16, GPUS), null);
});

test('순수 리졸버: Laptop 변형은 exact 불일치로 null — 데스크톱 4090 승격 금지 (Sol 반례 1)', () => {
  assert.equal(resolveDetectedGpu('NVIDIA GeForce RTX 4090 Laptop GPU', 16, GPUS), null);
  // 이름이 우연히 벗겨져도 VRAM 3중 방어가 잡는다
  assert.equal(resolveDetectedGpu('NVIDIA GeForce RTX 4090', 16, GPUS), null);
});

test('순수 리졸버: "RTX 3090 Ti"는 절대 RTX 3090이 되지 않는다 (Sol 반례 2 — exact identity)', () => {
  const g = resolveDetectedGpu('NVIDIA GeForce RTX 3090 Ti', 24, GPUS);
  assert.notEqual(g?.name, 'RTX 3090'); // 핵심 우려: 비-Ti 승격 금지
  // 카탈로그에 RTX 3090 Ti 정확 엔트리(24GB)가 실재하므로 exact identity는 그 엔트리로 확정된다.
  // (지시의 'null' 기대는 카탈로그 실재와 모순 — 카탈로그에 없는 변형이 null임은 아래 케이스가 증명)
  assert.equal(g?.name, 'RTX 3090 Ti');
});

test('순수 리졸버: 카탈로그에 없는 변형 토큰은 null — 벗기지 않으므로 승격 불가', () => {
  assert.equal(resolveDetectedGpu('NVIDIA GeForce RTX 5090 Ti', 32, GPUS), null);      // 무존재 Ti 변형
  assert.equal(resolveDetectedGpu('NVIDIA GeForce RTX 4090 Mobile', 24, GPUS), null);  // Mobile 변형
  assert.equal(resolveDetectedGpu('Unknown Accelerator X1', 48, GPUS), null);
});

test('순수 리졸버: 정확 신원 + VRAM 일치만 확정 (vendor/GPU 토큰 제거 허용)', () => {
  assert.equal(resolveDetectedGpu('NVIDIA GeForce RTX 4090', 24, GPUS)?.name, 'RTX 4090');
  assert.equal(resolveDetectedGpu('NVIDIA RTX 4080 SUPER GPU', 16, GPUS)?.name, 'RTX 4080 SUPER'); // 변형도 정확 엔트리가 있으면 그 엔트리로만
});

function runDetect(smiLine, ...extra) {
  const shim = mkdtempSync(join(tmpdir(), 'fake-smi-'));
  try {
    const exe = join(shim, 'nvidia-smi');
    writeFileSync(exe, `#!/bin/sh\necho "${smiLine}"\n`);
    chmodSync(exe, 0o755);
    return spawnSync(process.execPath, [new URL('../bin/fitllm.mjs', import.meta.url).pathname, 'Qwen 3.8 27B', '--detect', ...extra], {
      encoding: 'utf8',
      env: { ...process.env, PATH: `${shim}:${process.env.PATH}` },
    });
  } finally {
    setTimeout(() => rmSync(shim, { recursive: true, force: true }), 0);
  }
}

test('종단(fake nvidia-smi): Laptop 4090 16GB — 감지된 실제 16GB로 계산, 영수증 n/a', () => {
  const r = runDetect('NVIDIA GeForce RTX 4090 Laptop GPU, 16384');
  assert.match(r.stdout, /Laptop GPU \(detected\)/);       // 카탈로그 오인 없이 실제 이름 사용
  assert.match(r.stdout, /\/ 16 GB/);                       // 판정 분모가 감지 VRAM 16GB
  assert.doesNotMatch(r.stdout, /receipt: https/);          // 무효 영수증 발급 금지
  assert.match(r.stdout, /receipt: n\/a/);
});

test('종단(fake nvidia-smi): 데스크톱 4090 24GB — 카탈로그 확정, canonical 영수증 발급', () => {
  const r = runDetect('NVIDIA GeForce RTX 4090, 24564');
  assert.match(r.stdout, /\/ 24 GB/);
  assert.match(r.stdout, /receipt: https:\/\/fitllm\.run\/r\/qwen-3-8-27b-q4_k_m-on-rtx-4090/);
});
