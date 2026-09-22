// `fitllm measure --from-ollama` — 런타임이 보고한 상주량을 회수해 후보 측정으로 만든다.
//
// 이 파일이 지키는 핵심은 **무엇을 제출하지 않는가**다. 실측 제보는 주장이고, 해석할 수
// 없는 숫자가 census 에 들어가면 그 오류는 우리 책임이 된다. 그래서 통과 경로보다
// 거부 경로의 테스트가 더 많다.
import { test } from 'node:test';
import { strict as assert } from 'node:assert';
import { residentFromPsEntry, listRunningOllama } from '../bin/scan-providers.mjs';
import {
  buildRuntimeMeasurements, renderRuntimeMeasurements, PARTIAL_VERIFICATION_NOTE,
} from '../bin/measure-from-runtime.mjs';
import { GPUS, gpuDevice } from '../engine.js';

const GiB = 1073741824;
const device = () => gpuDevice(GPUS.find((g) => g.name === 'RTX 4090'));

// 공식 GET /api/ps 스키마 그대로(ollama/docs/api.md).
const psEntry = (over = {}) => ({
  name: 'qwen3:0.6b',
  model: 'qwen3:0.6b',
  size: 2 * GiB,
  digest: 'abc',
  details: { format: 'gguf', family: 'qwen3', parameter_size: '0.6B', quantization_level: 'Q4_K_M' },
  expires_at: '2026-09-23T14:38:31.83753-07:00',
  size_vram: 2 * GiB,
  ...over,
});

const stubFetch = (ps, version = '0.12.3') => async (url) => {
  if (url.endsWith('/api/ps')) return { ok: true, json: async () => ps };
  if (url.endsWith('/api/version')) return version === null
    ? { ok: false, json: async () => ({}) }
    : { ok: true, json: async () => ({ version }) };
  throw new Error(`unexpected url: ${url}`);
};

test('measureGateRejectsWhatItCannotInterpret: size_vram 이 size 와 다르거나 0이면 통과시키지 않는다', () => {
  // 전부 VRAM 에 있을 때만 통과.
  assert.equal(residentFromPsEntry(psEntry()).residentBytes, 2 * GiB);

  // 일부만 VRAM — Ollama 문서가 이 상태를 정의하지 않고, 부분 상주는 이 엔진이 모델링하지
  // 않는 영역이다. 추론으로 밀어붙이지 않는다.
  const partial = residentFromPsEntry(psEntry({ size_vram: 1 * GiB }));
  assert.equal(partial.residentBytes, null);
  assert.match(partial.reason, /part of the model/);

  // CPU 전용.
  const cpu = residentFromPsEntry(psEntry({ size_vram: 0 }));
  assert.equal(cpu.residentBytes, null);
  assert.match(cpu.reason, /not resident in VRAM/);

  // 기형.
  for (const bad of [null, 42, {}, psEntry({ size: 'nope' }), psEntry({ size_vram: undefined })]) {
    const out = residentFromPsEntry(bad);
    assert.ok(out === null || out.residentBytes === null, `${JSON.stringify(bad)} 가 통과했다`);
  }
});

test('measureProducesCandidateFromLoadedModel: 로드된 모델에서 후보와 이슈 URL 이 나온다', async () => {
  const result = await buildRuntimeMeasurements({
    fetchImpl: stubFetch({ models: [psEntry()] }), env: {}, device: device(),
  });
  assert.equal(result.ok, true);
  assert.equal(result.candidates.length, 1);
  const [c] = result.candidates;
  assert.equal(c.report.candidate.model, 'Qwen3-0.6B');
  assert.equal(c.report.candidate.measurementKind, 'idle_resident', '측정 종류를 새로 만들지 않는다');
  assert.equal(c.report.candidate.measuredPeakGB, 2);
  assert.equal(c.report.candidate.unit, 'GiB');
  assert.match(c.report.candidate.runtime, /Ollama 0\.12\.3/);
  assert.equal(c.report.submitted, false, '이 명령은 아무것도 제출하지 않는다');
  assert.ok(c.report.issueUrl.startsWith('https://github.com/'));

  const text = renderRuntimeMeasurements(result);
  // 부분 검증이라는 사실이 출력에 있어야 한다 — 없으면 전체 검증으로 오해된다.
  assert.ok(text.includes(PARTIAL_VERIFICATION_NOTE));
  assert.match(PARTIAL_VERIFICATION_NOTE, /does not measure peak memory during generation/);
});

test('measureSkipsUnmappedModels: 카탈로그에 없으면 대조할 예측이 없다', async () => {
  const result = await buildRuntimeMeasurements({
    fetchImpl: stubFetch({ models: [psEntry({ name: 'mystery:9b', model: 'mystery:9b' })] }),
    env: {}, device: device(),
  });
  assert.equal(result.candidates.length, 0);
  assert.equal(result.skipped.length, 1);
  assert.match(result.skipped[0].reason, /not in the catalog/);
});

test('measureRefusesWithoutRuntimeVersion: 버전을 못 읽으면 후보를 만들지 않는다', async () => {
  const result = await buildRuntimeMeasurements({
    fetchImpl: stubFetch({ models: [psEntry()] }, null), env: {}, device: device(),
  });
  assert.equal(result.ok, false);
  assert.equal(result.candidates.length, 0);
  assert.match(result.note, /version/);
  // 지어낸 런타임 문자열로 우회하지 않는다.
  assert.doesNotMatch(JSON.stringify(result), /unknown|latest/i);
});

test('measureRefusesWithoutHardware: 기기를 모르면 보고서가 기기를 특정하지 못한다', async () => {
  const result = await buildRuntimeMeasurements({
    fetchImpl: stubFetch({ models: [psEntry()] }), env: {}, device: null,
  });
  assert.equal(result.candidates.length, 0);
  assert.match(result.skipped[0].reason, /hardware not detected/);
});

test('measureIsReadOnlyAndLoopbackOnly: 조회만 하고 루프백만 부른다', async () => {
  const calls = [];
  const spy = async (url, init) => {
    calls.push({ url, method: init?.method });
    return url.endsWith('/api/version')
      ? { ok: true, json: async () => ({ version: '0.12.3' }) }
      : { ok: true, json: async () => ({ models: [] }) };
  };
  await buildRuntimeMeasurements({ fetchImpl: spy, env: {}, device: device() });
  assert.ok(calls.length >= 1);
  for (const call of calls) {
    assert.match(call.url, /^http:\/\/127\.0\.0\.1:11434\//);
    assert.equal(call.method, 'GET', '조회 외 메서드는 런타임 상태를 바꿀 수 있다');
  }
  // 로드·언로드 같은 조작 경로를 부르지 않는다.
  for (const forbidden of ['/api/generate', '/api/pull', '/api/create', '/api/delete', '/api/chat']) {
    assert.ok(!calls.some((c) => c.url.includes(forbidden)), `${forbidden} 를 호출했다`);
  }

  calls.length = 0;
  const remote = await buildRuntimeMeasurements({
    fetchImpl: spy, env: { OLLAMA_HOST: 'http://elsewhere.example.com:11434' }, device: device(),
  });
  assert.equal(calls.length, 0, '루프백이 아닌 호스트로 나갔다');
  assert.equal(remote.ok, false);
});

test('measureTellsYouWhatToDoWhenNothingIsLoaded: 로드된 게 없으면 다음 행동을 알려준다', async () => {
  const stopped = await buildRuntimeMeasurements({
    fetchImpl: async () => { throw new Error('ECONNREFUSED'); }, env: {}, device: device(),
  });
  assert.equal(stopped.ok, false);
  const text = renderRuntimeMeasurements(stopped);
  assert.match(text, /ollama run <model>/);

  const empty = await buildRuntimeMeasurements({
    fetchImpl: stubFetch({ models: [] }), env: {}, device: device(),
  });
  assert.equal(empty.ok, true);
  assert.match(renderRuntimeMeasurements(empty), /No submittable measurement/);
});

test('measureAddsNoSpeedClaim: 속도 주장을 싣지 않는다 (2026-06-05 봉인)', async () => {
  const result = await buildRuntimeMeasurements({
    fetchImpl: stubFetch({ models: [psEntry()] }), env: {}, device: device(),
  });
  for (const text of [renderRuntimeMeasurements(result), JSON.stringify(result)]) {
    assert.doesNotMatch(text, /tok\/s|tokens per second|faster/i);
  }
});

test('measurePassesThroughListRunningShape: provider 반환 모양이 계약대로다', async () => {
  const out = await listRunningOllama({ fetchImpl: stubFetch({ models: [psEntry()] }), env: {} });
  assert.deepEqual(Object.keys(out).sort(), ['note', 'ok', 'provider', 'running', 'version']);
  assert.equal(out.version, '0.12.3');
  assert.equal(out.running[0].id, 'qwen3:0.6b');
});

test('measurePairsLikeWithLike: idle_resident 에는 상주 예측을 짝지어야 한다', async () => {
  const { simulate, LOCAL_MODELS } = await import('../engine.js');
  const result = await buildRuntimeMeasurements({
    fetchImpl: stubFetch({ models: [psEntry()] }), env: {}, device: device(),
  });
  const c = result.candidates[0].report.candidate;
  const sim = simulate(LOCAL_MODELS.find((m) => m.name === 'Qwen3-0.6B'), device(), 8192,
    { weightBpw: 4.8944, kvBits: 16 });

  // 측정한 양과 예측한 양이 같아야 비교가 성립한다.
  assert.equal(c.predictedGB, sim.param + sim.kv + sim.linearState);
  assert.equal(c.predictedMetric, 'resident_weights_plus_kv_gb',
    '예측이 무엇인지 라벨이 없으면 검토자가 다른 양과 비교하게 된다');

  // generation_peak 용 값(used-reserve)을 짝지으면 안 된다 — 런타임 오버헤드가 섞여 있다.
  assert.notEqual(c.predictedGB, sim.used - sim.reserve);
  assert.ok(Math.abs((sim.used - sim.reserve) - c.predictedGB) > 0.3,
    '두 양이 사실상 같다면 이 테스트가 지키는 구분이 사라진 것이다');
});

// Grok 교차검수(2026-09-23) 지적: Number() 강제변환에 기대면 타입이 다른 값이 통과한다.
// Ollama JSON 으로 도달하기는 어렵지만, 이 게이트가 막기로 한 것이 정확히 "해석할 수 없는 값"이다.
test('measureGateRequiresRealByteCounts: 강제변환으로 통과하는 값을 막는다', () => {
  const coercible = [
    [true, true], ['100', '100'], [[100], [100]], ['0x100', 256], ['1e3', 1000],
    [' 100 ', 100], [1.5, 1.5], [Number.MAX_SAFE_INTEGER + 2, Number.MAX_SAFE_INTEGER + 2],
  ];
  for (const [size, vram] of coercible) {
    const out = residentFromPsEntry({ model: 'm', size, size_vram: vram });
    assert.ok(out === null || out.residentBytes === null,
      `size=${String(size)} vram=${String(vram)} 가 통과했다`);
  }
  // 진짜 정수 바이트는 그대로 통과한다.
  assert.equal(residentFromPsEntry({ model: 'm', size: 100, size_vram: 100 }).residentBytes, 100);
});
