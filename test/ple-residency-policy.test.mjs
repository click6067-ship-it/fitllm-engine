// PLE resident-scope policy — 2.15.0 public port (private docs/plans/2026-09-05-ple-residency-policy-correction.md).
//
// 2.14.1의 전제 문장 'lazy-or-host-resident'는 저장장치 페이징 계열(lazy/on-disk 읽기)을 차감 근거에 섞었다.
// 최종 정책: 차감의 유일한 근거는 pinned llama.cpp/GGUF 경로가 per_layer_token_embd 입력층 텐서를 accelerator가 아닌
// CPU/host 버퍼에 배치한다는 사실이다. 그 host 메모리는 GPU 판정 예산에 넣지 않고, PLE를 accelerator에 적재하는
// 런타임에서는 추정이 무효이며, Apple 통합 메모리는 전체 상주다. SSD/NVMe·익스퍼트 페이징·스왑은 어디서도 용량이 아니다.
// 전제 id·수식·숫자는 그대로다 — 이 파일은 문장·문서·CLI/Action 표면만 고정한다.
import test from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync, spawnSync } from 'node:child_process';
import { mkdtempSync, readFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { GPUS, LOCAL_MODELS, gpuDevice, simulate, simulateStack, structuralAssumptions } from '../engine.js';

const ROOT = new URL('..', import.meta.url).pathname;
const BIN = new URL('../bin/fitllm.mjs', import.meta.url).pathname;
const ENTRY = new URL('../scripts/action-entry.sh', import.meta.url).pathname;
const read = (rel) => readFileSync(new URL(`../${rel}`, import.meta.url), 'utf8');
const run = (args) => {
  try { return { out: execFileSync(process.execPath, [BIN, ...args], { encoding: 'utf8' }), code: 0 }; }
  catch (e) { return { out: (e.stdout || '') + (e.stderr || ''), code: e.status }; }
};
function runAction(inputs) {
  const outputFile = join(mkdtempSync(join(tmpdir(), 'fitllm-action-')), 'output');
  const result = spawnSync('bash', [ENTRY], {
    cwd: ROOT,
    encoding: 'utf8',
    env: {
      ...process.env,
      GITHUB_ACTION_PATH: ROOT,
      GITHUB_OUTPUT: outputFile,
      INPUT_MODEL: '', INPUT_GPU: '', INPUT_MAC: '', INPUT_QUANT: '', INPUT_CTX: '8192', INPUT_KV: '16', INPUT_COUNT: '1',
      ...inputs,
    },
  });
  return { ...result, output: readFileSync(outputFile, 'utf8') };
}

const PLE_PREMISE = Object.freeze({
  id: 'ple-llamacpp-non-gpu-residency',
  statement: 'GPU weight memory excludes the verified Gemma 4 PLE tensors only because the pinned llama.cpp/GGUF path assigns the per_layer_token_embd input-layer tensor to CPU/host buffers instead of accelerator memory; that host memory is not budgeted here, and a runtime that loads PLE onto the accelerator invalidates this estimate.',
});
const STORAGE_PATH = /lazy|on[- ]disk|SSD|NVMe|stream|swap|mmap|pag(e|ing)/i;
const PINNED_LLAMACPP = '8b4b3558f1459c13e4aa38d5c94d306a00dc6acd';
const PINNED_SOURCES = ['src/llama-model.cpp', 'src/llama-arch.cpp', 'src/models/gemma4.cpp', 'src/llama-model-loader.h', 'src/llama-model-loader.cpp']
  .map((file) => `https://github.com/ggml-org/llama.cpp/blob/${PINNED_LLAMACPP}/${file}`);
const gpu = (name, env) => gpuDevice(GPUS.find((g) => g.name === name), env);
const e2b = () => LOCAL_MODELS.find((m) => m.name === 'Gemma 4 e2b');
const e4b = () => LOCAL_MODELS.find((m) => m.name === 'Gemma 4 e4b');
const close = (actual, expected, digits, label = '') =>
  assert.ok(Math.abs(actual - expected) < 10 ** -digits / 2, `${label} ${actual} ≉ ${expected}`);

test('PLE premise statement names pinned host placement and a runtime conditional, never a storage path', () => {
  const rtx4090 = gpu('RTX 4090');
  assert.deepEqual(structuralAssumptions(e2b(), rtx4090), [PLE_PREMISE]);
  assert.deepEqual(structuralAssumptions(e4b(), rtx4090), [PLE_PREMISE]);
  assert.doesNotMatch(PLE_PREMISE.statement, STORAGE_PATH);
  assert.ok(PLE_PREMISE.statement.includes('input-layer tensor to CPU/host buffers'));
  assert.ok(PLE_PREMISE.statement.includes('host memory is not budgeted'));
  assert.ok(PLE_PREMISE.statement.includes('invalidates this estimate'));
  // Apple 통합 메모리: 전제도 차감도 없다 — 전체 가중치 상주(fail-closed 보존)
  assert.deepEqual(structuralAssumptions(e2b(), 64), []);
  const apple = simulate(e2b(), 64, 8192, 8);
  const appleNoPle = simulate({ ...e2b(), pleOffloadVerified: false }, 64, 8192, 8);
  close(apple.param, appleNoPle.param, 6, 'apple param');
  assert.equal('structuralAssumptions' in apple, false);
  assert.equal(apple.pleOffloadGB, 0);
});

test('PLE non-GPU residency is pinned-path gated with the corrected statement and unchanged numbers', () => {
  const device = gpu('RTX 3060 8GB', 'linux-headless');
  const verified = simulate(e2b(), device, 8192, { weightBpw: 16, kvBits: 16 });
  const unverified = simulate({ ...e2b(), name: 'Invented PLE', pleOffloadVerified: false }, device, 8192, { weightBpw: 16, kvBits: 16 });
  close(verified.param, 5.1241, 4, 'verified param');
  close(verified.pleOffloadGB, 4.3754, 4, 'verified pleOffloadGB');
  assert.deepEqual(verified.structuralAssumptions, [PLE_PREMISE]);
  assert.doesNotMatch(JSON.stringify(verified), /lazy-or-host|SSD|NVMe/i);
  close(unverified.param, 9.4995, 4, 'unverified param');
  assert.equal(unverified.pleOffloadGB, 0);
  assert.equal('structuralAssumptions' in unverified, false);
  // 스택: 같은 전제는 한 번만, 부품 객체와 별칭 공유 없음
  const stack = simulateStack([
    { model: e2b(), ctx: 8192, weightBpw: 16, kvBits: 16 },
    { model: e4b(), ctx: 8192, weightBpw: 16, kvBits: 16 },
  ], gpu('RTX 4090'));
  assert.deepEqual(stack.structuralAssumptions, [PLE_PREMISE]);
  assert.notEqual(stack.structuralAssumptions[0], stack.parts[0].structuralAssumptions[0]);
});

test('engine source anchors the PLE deduction on pinned input-layer host placement, not lazy or on-disk reads', () => {
  const source = read('engine.js');
  assert.equal(source.includes('시스템 RAM 상주'), false);
  assert.equal(source.includes('keeps per-layer token embeddings in host RAM'), false);
  assert.equal(source.includes('lazy-or-host-resident'), false);
  assert.ok(source.includes('assigns the per_layer_token_embd input-layer tensor to CPU/host buffers'));
  // 배치 사실을 뒷받침하는 pinned 출처 다섯 개(placement 두 개 + 기존 loader/construction 세 개)가 소스에 있어야 한다
  for (const url of PINNED_SOURCES) assert.ok(source.includes(url), `engine.js missing ${url}`);
  // 2.15.0 parser guard: 완전 profile 대조와 body 정합이 소스에 실재한다
  assert.ok(source.includes('PLE_VERIFIED_PROFILES'));
  assert.ok(source.includes("gemma4_text: Object.freeze(["));
});

test('CLI text, JSON, --why and --top forward the corrected premise on GPU and none on Apple', () => {
  // CLI 기본 GPU 환경 프리셋은 엔진 테스트의 linux-headless가 아니라 reserve가 더 큰 기본값이라 RTX 3060 8GB FP16은 CLI에서 no다.
  // 전제 검사는 fit 여부와 무관하므로 CLI 기본값에서 실제로 fits인 RTX 4090 FP16(used ≈ 8.1 GiB)으로 exit 0 경로를 고정한다.
  const text = run(['Gemma 4 e2b', '--gpu', 'RTX 4090', '--quant', 'FP16']);
  assert.equal(text.code, 0, text.out);
  assert.ok(text.out.includes(`premise [${PLE_PREMISE.id}]: ${PLE_PREMISE.statement}`), text.out);
  assert.doesNotMatch(text.out, /lazy-or-host|SSD|NVMe/i);
  const json = JSON.parse(run(['Gemma 4 e2b', '--gpu', 'RTX 4090', '--quant', 'FP16', '--json']).out);
  assert.deepEqual(json.structuralAssumptions, [PLE_PREMISE]);
  // 8GB 카드에서는 CLI 기본 환경으로 no(exit 1)지만 전제는 같은 문장으로 그대로 실린다
  const small = run(['Gemma 4 e2b', '--gpu', 'RTX 3060 8GB', '--quant', 'FP16', '--json']);
  assert.equal(small.code, 1, small.out);
  assert.deepEqual(JSON.parse(small.out).structuralAssumptions, [PLE_PREMISE]);
  const why = JSON.parse(run(['Gemma 4 e2b', '--gpu', 'RTX 4090', '--quant', 'FP16', '--json', '--why']).out);
  assert.deepEqual(why.structuralAssumptions, [PLE_PREMISE]);
  assert.deepEqual(why.basis.structuralAssumptions, [PLE_PREMISE]);
  assert.doesNotMatch(JSON.stringify(why), /lazy-or-host/i);
  const top = JSON.parse(run(['--top', '--gpu', 'RTX 4090', '--quant', 'FP16', '--json']).out);
  const row = top.fits.find((r) => r.model === 'Gemma 4 e2b');
  assert.ok(row, 'Gemma 4 e2b FP16 fits an RTX 4090');
  assert.deepEqual(row.structuralAssumptions, [PLE_PREMISE]);
  const mac = JSON.parse(run(['Gemma 4 e2b', '--mac', '16', '--json']).out);
  assert.equal('structuralAssumptions' in mac, false);
  assert.doesNotMatch(run(['Gemma 4 e2b', '--mac', '16']).out, /premise \[/);
});

test('composite Action forwards the corrected premise and keeps the exit contract', () => {
  const affected = runAction({ INPUT_MODEL: 'Gemma 4 e2b', INPUT_GPU: 'RTX 4090', INPUT_QUANT: 'FP16' });
  assert.equal(affected.status, 0, affected.output);
  assert.match(affected.output, /exit-code=0/);
  assert.ok(affected.output.includes(PLE_PREMISE.id));
  assert.ok(affected.output.includes(PLE_PREMISE.statement));
  assert.doesNotMatch(affected.output, /lazy-or-host|SSD|NVMe/i);
  const smallCard = runAction({ INPUT_MODEL: 'Gemma 4 e2b', INPUT_GPU: 'RTX 3060 8GB', INPUT_QUANT: 'FP16' });
  assert.equal(smallCard.status, 1);
  assert.match(smallCard.output, /exit-code=1/);
  assert.ok(smallCard.output.includes(PLE_PREMISE.statement));
  const apple = runAction({ INPUT_MODEL: 'Gemma 4 e2b', INPUT_MAC: '16' });
  assert.equal(apple.status, 0);
  assert.doesNotMatch(apple.output, /structuralAssumptions/);
});

test('README states the final resident-scope policy with the exact premise and five pinned placement sources', () => {
  const readme = read('README.md');
  assert.ok(readme.includes(`\`${PLE_PREMISE.id}\` — ${PLE_PREMISE.statement}`));
  for (const url of PINNED_SOURCES) assert.ok(readme.includes(url), `README missing ${url}`);
  assert.ok(readme.includes('https://huggingface.co/google/gemma-4-E2B-it/blob/main/config.json'));
  assert.doesNotMatch(readme, /lazy-or-host|lazily|host-resident|on-disk/i);
  assert.equal(readme.includes('keeps per-layer token embeddings in host RAM'), false);
  // 정책 단락: 저장장치 페이징 계열은 판정 밖, PLE는 배치 사실이라는 구분 — 수치·속도 없음
  const policy = readme.split('\n').filter((l) => l.startsWith('**Resident scope.**'));
  assert.equal(policy.length, 1, 'exactly one resident-scope policy paragraph');
  const [line] = policy;
  assert.ok(line.includes('SSD/NVMe streaming, expert paging, swap and general CPU/RAM offload stay outside ordinary FitLLM fit verdicts'));
  assert.ok(line.includes('a placement fact, not storage paging'));
  assert.ok(line.includes('Gemma 4 e2b/e4b'));
  assert.ok(line.includes('assigns that input-layer tensor to CPU/host buffers instead of accelerator memory'));
  assert.ok(line.includes(`\`${PLE_PREMISE.id}\``));
  assert.ok(line.includes('host/system memory is not budgeted by the discrete-GPU verdict'));
  assert.ok(line.includes('a runtime that loads PLE onto the accelerator invalidates the PLE estimate'));
  assert.ok(line.includes('Apple unified memory'));
  assert.doesNotMatch(line, /tok\/s|token\/s|faster|slower|speed/i);
  assert.doesNotMatch(line, /\d+(?:\.\d+)?\s?(?:GB|GiB|MB|MiB)\b/i);
  // PLE 설명 항목: host 메모리 미예산·Apple 전체 상주·조건부 — 저장장치 문구 없음
  const pleItem = readme.split('\n').find((l) => l.startsWith('5. **PLE — Per-Layer Embeddings**'));
  assert.ok(pleItem, 'PLE item present');
  assert.ok(pleItem.includes('assigns the `per_layer_token_embd` input-layer tensor to CPU/host buffers'));
  assert.ok(pleItem.includes('not budgeted'));
  assert.ok(pleItem.includes('unverified families keep their full weights resident'));
  assert.doesNotMatch(pleItem, /lazy|SSD|NVMe/i);
  // 저장장치 = 용량 문구는 README 어디에도 없다
  assert.doesNotMatch(readme, /(?:SSD|NVMe|swap|offload)[^\n.]*(?:as capacity|counts? as|is supported|supported as)/i);
});

test('CONTRIBUTING separates the disclosed PLE placement fact from storage-paging residency reports', () => {
  const contrib = read('CONTRIBUTING.md');
  assert.ok(contrib.includes('Gate A'));
  assert.ok(contrib.includes('Gate B'));
  assert.ok(contrib.includes(`\`${PLE_PREMISE.id}\``));
  assert.ok(contrib.includes('not storage paging'));
  assert.ok(contrib.includes('never change verdicts by themselves'));
  assert.doesNotMatch(contrib, /lazy-or-host/i);
});

test('census README discloses the PLE deduction for Gemma 4 e2b/e4b GPU rows with engine-derived magnitudes', () => {
  const readme = read('census/README.md');
  const rows = JSON.parse(read('census/census-v1.json')).data;
  const rtx4090 = gpu('RTX 4090');
  const excluded = (m, bpw) => simulate(m, rtx4090, 8192, { weightBpw: bpw, kvBits: 16 }).pleOffloadGB.toFixed(2);
  const line = readme.split('\n').find((l) => l.includes(`\`${PLE_PREMISE.id}\``));
  assert.ok(line, 'census README discloses the PLE premise');
  assert.equal(line.startsWith('- **Structural premise disclosed'), true);
  assert.ok(line.includes(`${e2b().pleParams}B of e2b's ${e2b().totalParams}B`));
  assert.ok(line.includes(`${e4b().pleParams}B of e4b's ${e4b().totalParams}B`));
  assert.ok(line.includes(`at Q4_K_M the excluded weights are ~${excluded(e2b(), 4.8944)} GB for e2b and ~${excluded(e4b(), 4.8944)} GB for e4b; at FP16 ~${excluded(e2b(), 16)} GB and ~${excluded(e4b(), 16)} GB`));
  for (const col of ['`predicted_param_gb`', '`predicted_resident_weights_gb`', '`predicted_total_to_run_gb`', '`free_gb`', '`max_context`']) assert.ok(line.includes(col), col);
  assert.ok(line.includes("assigns that input-layer tensor to CPU/host buffers rather than the discrete GPU's memory pool"));
  assert.ok(line.includes('host/system memory that this census neither budgets nor verifies'));
  assert.ok(line.includes('a runtime that loads PLE onto the accelerator invalidates those rows'));
  assert.ok(line.includes('the Mac rows (Apple unified memory) count the full weights'));
  assert.ok(line.includes('No row treats SSD/NVMe streaming, expert paging or swap as capacity'));
  assert.ok(line.includes('https://github.com/click6067-ship-it/fitllm-engine#structural-premises'));
  assert.doesNotMatch(line, /lazy/i);
  // 공개 행 자체는 그대로다: e2b GPU Q4_K_M 행의 predicted_param_gb는 (params − PLE) × bpw 산술과 일치한다
  const row = rows.find((r) => r.model === 'Gemma 4 e2b' && r.platform === 'gpu' && r.device === 'RTX 3060 12GB' && r.quant === 'Q4_K_M');
  assert.ok(row, 'e2b RTX 3060 12GB Q4_K_M row');
  assert.ok(Math.abs(row.predicted_param_gb - ((e2b().totalParams - e2b().pleParams) * 1e9 * (4.8944 / 8)) / 1024 ** 3) < 0.0051);
  const macRow = rows.find((r) => r.model === 'Gemma 4 e2b' && r.platform === 'mac' && r.quant === '4bit');
  assert.ok(macRow, 'e2b Mac 4bit row');
  assert.ok(Math.abs(macRow.predicted_param_gb - simulate(e2b(), macRow.memory_gb, 8192, { weightBpw: 4, kvBits: 16 }).param) < 0.0051);
  assert.ok(rows.filter((r) => (r.model === 'Gemma 4 e2b' || r.model === 'Gemma 4 e4b') && r.platform === 'gpu').length > 0);
});
