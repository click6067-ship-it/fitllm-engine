// `fitllm scan` — 설치된 모델을 읽어 판정한다.
//
// ⚠️ 이 테스트가 증명하지 **못하는** 것: 진짜 Ollama/LM Studio 와 붙는가.
// 이 개발 머신에는 둘 다 설치돼 있지 않다. 종단 검증은 CI 가 실제 Ollama 를 띄워서 한다
// (.github/workflows/verify-install-paths.yml 의 ollama-scan job). 픽스처 통과를
// "런타임 검증됨"으로 읽지 말 것.
import { test } from 'node:test';
import { strict as assert } from 'node:assert';
import { listOllama, listLmStudio } from '../bin/scan-providers.mjs';
import {
  buildScan, renderScan, catalogQueryFor, resolveInstalled, detectDevice,
} from '../bin/scan.mjs';
import { GPUS, gpuDevice } from '../engine.js';

// 공식 GET /api/tags 스키마 그대로(ollama/docs/api.md).
const OLLAMA_TAGS = {
  models: [
    {
      name: 'qwen3:0.6b',
      model: 'qwen3:0.6b',
      modified_at: '2026-09-20T08:06:48.639712648-07:00',
      size: 522653484,
      digest: '0a8c266910232fd3291e71e5ba1e058cc5af9d411192cf88b6d30e92b6e73163',
      details: {
        parent_model: '', format: 'gguf', family: 'qwen3', families: ['qwen3'],
        parameter_size: '0.6B', quantization_level: 'Q4_K_M',
      },
    },
    {
      name: 'some-unknown-model:latest',
      model: 'some-unknown-model:latest',
      size: 7200000000,
      digest: 'deadbeef',
      details: { format: 'gguf', family: 'mystery', parameter_size: '12B', quantization_level: 'Q5_K_M' },
    },
  ],
};

const okResponse = (body) => ({ ok: true, json: async () => body });

test('scanReadsOfficialOllamaSchema: 공식 스키마를 그대로 읽고 기형 항목만 버린다', async () => {
  const withJunk = { models: [...OLLAMA_TAGS.models, null, 42, { details: {} }, { name: '' }] };
  const result = await listOllama({ fetchImpl: async () => okResponse(withJunk), env: {} });
  assert.equal(result.ok, true);
  // 기형 4개는 버리고 정상 2개만 — 한 줄이 이상하다고 나머지를 못 보게 하지 않는다.
  assert.equal(result.models.length, 2);
  assert.deepEqual(result.models[0], {
    id: 'qwen3:0.6b', sizeBytes: 522653484, quant: 'Q4_K_M',
    paramSizeLabel: '0.6B', family: 'qwen3', format: 'gguf',
  });
});

test('scanOnlyTalksToLoopback: 외부 호스트를 부르지 않는다', async () => {
  const calls = [];
  const spy = async (url, init) => { calls.push({ url, method: init?.method }); return okResponse({ models: [] }); };

  await listOllama({ fetchImpl: spy, env: {} });
  assert.equal(calls.length, 1);
  assert.match(calls[0].url, /^http:\/\/127\.0\.0\.1:11434\/api\/tags$/);
  assert.equal(calls[0].method, 'GET', '조회 말고 다른 메서드를 쓰면 런타임을 건드릴 수 있다');

  // OLLAMA_HOST 는 존중하되 루프백이 아니면 **호출 자체를 안 한다**.
  calls.length = 0;
  const remote = await listOllama({ fetchImpl: spy, env: { OLLAMA_HOST: 'http://evil.example.com:11434' } });
  assert.equal(calls.length, 0, '루프백이 아닌 호스트로 나갔다');
  assert.equal(remote.ok, false);
  assert.match(remote.note, /loopback/);

  calls.length = 0;
  await listOllama({ fetchImpl: spy, env: { OLLAMA_HOST: 'localhost:11434' } });
  assert.equal(calls.length, 1, '정상 루프백 표기는 허용돼야 한다');
});

test('scanTreatsAbsentRuntimeAsNormal: 꺼져 있으면 에러가 아니라 ok:false 다', async () => {
  const refused = await listOllama({ fetchImpl: async () => { throw new Error('ECONNREFUSED'); }, env: {} });
  assert.deepEqual(refused, { provider: 'ollama', ok: false, models: [], note: 'not running' });

  const missing = listLmStudio({ execImpl: () => { throw new Error('ENOENT'); } });
  assert.equal(missing.ok, false);
  assert.match(missing.note, /not found/);

  const garbled = listLmStudio({ execImpl: () => 'not json at all' });
  assert.equal(garbled.ok, false);
  assert.equal(garbled.models.length, 0);
});

test('scanNeverGuessesAnUnmappedModel: 매핑 실패는 숫자가 아니라 사실로 답한다', async () => {
  const scan = await buildScan({
    fetchImpl: async () => okResponse(OLLAMA_TAGS),
    execImpl: () => { throw new Error('ENOENT'); },
    device: gpuDevice(GPUS.find((g) => g.name === 'RTX 4090')),
  });
  const items = scan.providers.find((p) => p.provider === 'ollama').models;

  const known = items.find((i) => i.installedId === 'qwen3:0.6b');
  assert.equal(known.catalogModel, 'Qwen3-0.6B');
  assert.equal(known.resolution, 'resolved');
  assert.ok(known.fit && known.fit.verdict, '카탈로그에 있는 모델은 판정이 나와야 한다');

  const unknown = items.find((i) => i.installedId === 'some-unknown-model:latest');
  assert.equal(unknown.catalogModel, null);
  assert.equal(unknown.fit, null, '모르는 모델에 판정을 붙이면 안 된다');
  // 런타임이 알려준 측정된 사실은 그대로 남는다.
  assert.equal(unknown.reported.sizeBytes, 7200000000);
  assert.equal(unknown.reported.quant, 'Q5_K_M');

  const text = renderScan(scan);
  assert.match(text, /not in the catalog, no verdict/);
  assert.match(text, /on disk: 6\.7 GB · Q5_K_M/);
  // 미매핑 항목 줄에 GB 판정 수치가 섞이면 안 된다.
  assert.doesNotMatch(text, /some-unknown-model[^\n]*(FITS|TIGHT|WON'T FIT)/);
});

test('scanDoesNotClaimEmptyWhenNothingAnswered: 아무 provider 도 없으면 단정하지 않는다', async () => {
  const scan = await buildScan({
    fetchImpl: async () => { throw new Error('ECONNREFUSED'); },
    execImpl: () => { throw new Error('ENOENT'); },
    device: null,
  });
  assert.equal(scan.availableProviders.length, 0);
  const text = renderScan(scan);
  assert.match(text, /No supported runtime answered/);
  assert.doesNotMatch(text, /no models installed|설치된 모델 없음/i);
  // 다음에 뭘 하면 되는지까지 있어야 안내가 완성된다.
  assert.match(text, /fitllm "Gemma 4 31b" --detect/);
});

test('scanWithoutHardwareListsButDoesNotJudge: 감지 실패 시 판정을 지어내지 않는다', async () => {
  const scan = await buildScan({
    fetchImpl: async () => okResponse(OLLAMA_TAGS),
    execImpl: () => { throw new Error('ENOENT'); },
    device: null,
  });
  assert.equal(scan.device, null);
  for (const item of scan.providers.find((p) => p.provider === 'ollama').models) {
    assert.equal(item.fit, null);
  }
  const text = renderScan(scan);
  assert.match(text, /not detected/);
  assert.match(text, /no hardware detected/);
});

test('scanJsonCarriesTheSameFacts: --json 이 사람용 출력과 같은 사실을 담는다', async () => {
  const scan = await buildScan({
    fetchImpl: async () => okResponse(OLLAMA_TAGS),
    execImpl: () => { throw new Error('ENOENT'); },
    device: gpuDevice(GPUS.find((g) => g.name === 'RTX 4090')),
  });
  const serialized = JSON.stringify(scan);
  for (const key of ['provider', 'installedId', 'catalogModel', 'resolution', 'fit', 'reported']) {
    assert.ok(serialized.includes(`"${key}"`), `--json 스키마에 ${key} 가 없다`);
  }
  assert.equal(scan.installedCount, 2);
  assert.deepEqual(scan.availableProviders, ['ollama']);
});

test('scanIdMapping: 태그와 네임스페이스를 떼고 해석한다', () => {
  // 태그는 보존한다 — Ollama 에서 :0.6b 는 모델 정체성이고, 떼면 ambiguous 가 된다.
  assert.equal(catalogQueryFor('qwen3:0.6b'), 'qwen3:0.6b');
  assert.equal(catalogQueryFor('lmstudio-community/Gemma-4-31b-GGUF'), 'Gemma-4-31b-GGUF');
  assert.equal(catalogQueryFor(''), '');
  assert.equal(resolveInstalled('qwen3:0.6b').model?.name, 'Qwen3-0.6B');
  // 해석기가 모르면 모르는 것 — 비슷한 이름을 갖다 붙이지 않는다.
  assert.equal(resolveInstalled('totally-made-up:7b').status, 'unresolved');
  assert.equal(resolveInstalled('totally-made-up:7b').model, null);
});

test('scanAddsNoSpeedClaim: 속도 주장을 싣지 않는다 (2026-06-05 봉인)', async () => {
  const scan = await buildScan({
    fetchImpl: async () => okResponse(OLLAMA_TAGS),
    execImpl: () => { throw new Error('ENOENT'); },
    device: gpuDevice(GPUS.find((g) => g.name === 'RTX 4090')),
  });
  for (const text of [renderScan(scan), JSON.stringify(scan)]) {
    assert.doesNotMatch(text, /tok\/s|tokens per second|faster/i);
  }
});

test('scanDetectFailureIsNotFatal: 하드웨어 감지가 던져도 scan 은 죽지 않는다', () => {
  assert.equal(detectDevice({ detect: () => { throw new Error('nvidia-smi exploded'); } }), null);
});
