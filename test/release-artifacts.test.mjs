// 2.15.0 release artifacts — 버전 표면, 패키지 계약 불변, 30 conformance vectors 바이트 보존, census 파생 수치·해시.
// 원칙: 수치는 이전 릴리스에서 복사하지 않고 엔진·산출물에서 유도해 문서 리터럴과 대조한다(드리프트 방지).
import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { existsSync, readFileSync } from 'node:fs';
import { ENGINE_VERSION, GPUS, GPU_QUANTS, LOCAL_MODELS, MACBOOK_RAM_GROUPS } from '../engine.js';

const RELEASE_VERSION = '2.15.0';
const RELEASE_DATE = '2026-09-06';
const url = (rel) => new URL(`../${rel}`, import.meta.url);
const read = (rel) => readFileSync(url(rel), 'utf8');
const readJson = (rel) => JSON.parse(read(rel));
const sha256 = (rel) => createHash('sha256').update(readFileSync(url(rel))).digest('hex');

// 2.14.1(619585f57396523d8d173179180340cbd881c324)의 vectors 바이트 — 2.15.0은 벡터를 추가·수정·재수출하지 않는다.
const VECTORS_JSON_SHA256 = 'bcc806a79eef9192e08152ded754c5228f4d0dfd54b53c9eb87904e6a8fcd8b8';
const VECTORS_RUNNER_SHA256 = '08692a4704ed9fcabdb9cd796568401538c291aaf20f7f7504026181c6e3ef7b';

test('release version surfaces agree on 2.15.0 and the package contract is unchanged', () => {
  const pkg = readJson('package.json');
  const lock = readJson('package-lock.json');
  assert.equal(pkg.version, RELEASE_VERSION);
  assert.equal(lock.version, RELEASE_VERSION);
  assert.equal(lock.packages[''].version, RELEASE_VERSION);
  assert.equal(ENGINE_VERSION, RELEASE_VERSION);
  assert.ok(read('engine.js').includes(`export const ENGINE_VERSION = '${RELEASE_VERSION}';`));
  // 계약 불변: 이름·Node floor·exports·bin·files·라이선스 (alias `fitllm`은 publish-alias.mjs가 이름만 바꾼다)
  assert.equal(pkg.name, 'fitllm-engine');
  assert.equal(pkg.type, 'module');
  assert.equal(pkg.main, 'engine.js');
  assert.deepEqual(pkg.exports, { '.': './engine.js' });
  assert.deepEqual(pkg.bin, { fitllm: 'bin/fitllm.mjs' });
  assert.deepEqual(pkg.files, ['engine.js', 'bin', 'vectors']);
  assert.deepEqual(pkg.engines, { node: '>=18' });
  assert.equal(pkg.license, 'MIT');
  assert.deepEqual(lock.packages[''].engines, { node: '>=18' });
  assert.deepEqual(lock.packages[''].bin, { fitllm: 'bin/fitllm.mjs' });
  assert.equal(Object.keys(lock.packages).length, 1); // zero deps
  for (const entry of [...pkg.files, 'README.md', 'LICENSE']) assert.ok(existsSync(url(entry)), `${entry} missing from tree`);
  // README/AGENTS: Action 예시는 새 태그를 가리키고 낡은 버전 문자열은 남지 않는다
  const readme = read('README.md');
  assert.ok(readme.includes(`uses: click6067-ship-it/fitllm-engine@v${RELEASE_VERSION}`));
  assert.equal(readme.includes('2.14.1'), false);
  assert.equal(read('AGENTS.md').includes('2.14.1'), false);
});

test('the 30 conformance vectors are byte-identical to 2.14.1 with 12/3/4/11 kinds and no new export paths', () => {
  assert.equal(sha256('vectors/fit-vectors-v1.json'), VECTORS_JSON_SHA256);
  assert.equal(sha256('vectors/run.mjs'), VECTORS_RUNNER_SHA256);
  const { version, vectors } = readJson('vectors/fit-vectors-v1.json');
  assert.equal(version, '1.5.0');
  assert.equal(vectors.length, 30);
  const kinds = {};
  for (const v of vectors) kinds[v.kind] = (kinds[v.kind] || 0) + 1;
  assert.deepEqual(kinds, { kv_total_bytes: 12, kv_per_token_bytes: 3, linear_state_bytes: 4, verdict: 11 });
  assert.equal(vectors.some((v) => v.model === 'GLM-5.3'), false); // GLM-5.3은 벡터를 추가하지 않는다(GLM-5.2와 수학 동일)
  const pkg = readJson('package.json');
  assert.equal('./vectors' in pkg.exports, false);
  assert.equal('./vectors/*' in pkg.exports, false);
});

test('census counts, date, schema, URLs, licence boundary and checksums are derived and mutually consistent', () => {
  const census = readJson('census/census-v1.json');
  const manifest = readJson('census/manifest.json');
  const measured = readJson('fixtures/measured.json');
  const censusReadme = read('census/README.md');
  const macConfigs = Object.values(MACBOOK_RAM_GROUPS).reduce((n, rams) => n + rams.length, 0);
  const devices = GPUS.length + macConfigs;
  const rowsPerModel = GPUS.length * GPU_QUANTS.length + macConfigs * 3;
  const rows = LOCAL_MODELS.length * rowsPerModel;
  assert.equal(LOCAL_MODELS.length, 27);
  assert.equal(devices, 93);
  assert.equal(census.version, 1);
  assert.equal(census.schema_version, 2);
  assert.equal(census.generated, RELEASE_DATE);
  assert.equal(census.verdicts, rows);
  assert.equal(census.data.length, rows);
  assert.ok(census.definitions?.measurement_kind, 'definitions ship the measurement_kind semantics');
  assert.equal(manifest.version, 1);
  assert.equal(manifest.schema_version, 2);
  assert.equal(manifest.generated, RELEASE_DATE);
  assert.equal(manifest.rows, rows);
  assert.equal(manifest.devices, devices);
  assert.equal(manifest.measured_rows, measured.length);
  assert.equal(manifest.license, 'CC0-1.0');
  assert.equal(manifest.license_url, 'https://creativecommons.org/publicdomain/zero/1.0/');
  assert.deepEqual(manifest.canonical, {
    site: 'https://fitllm.run/data/',
    json: 'https://fitllm.run/data/census-v1.json',
    csv: 'https://fitllm.run/data/census-v1.csv',
    source: 'https://github.com/click6067-ship-it/fitllm-engine/tree/master/census',
  });
  assert.deepEqual(manifest.engine, {
    name: 'fitllm-engine', license: 'MIT',
    npm: 'https://www.npmjs.com/package/fitllm-engine', repo: 'https://github.com/click6067-ship-it/fitllm-engine',
  });
  assert.deepEqual(manifest.sha256, { 'census-v1.json': sha256('census/census-v1.json'), 'census-v1.csv': sha256('census/census-v1.csv') });
  // CSV 행수 = JSON 행수 (헤더 1행)
  assert.equal(read('census/census-v1.csv').trimEnd().split('\n').length, rows + 1);
  // 사람용 README는 같은 날짜·유도 수치를 싣는다
  assert.ok(censusReadme.startsWith(`# Local LLM Fit Census v1 — ${RELEASE_DATE}\n`));
  assert.ok(censusReadme.includes(`**${rows.toLocaleString('en-US')} verdicts**: ${LOCAL_MODELS.length} models × ${devices} devices (${GPUS.length} GPUs + ${macConfigs} Mac configs)`));
  assert.ok(censusReadme.includes('| GLM-5.3 | 753B |'));
  // 타입 있는 실측 컬럼은 모든 행에 존재한다(값은 null 허용)
  for (const key of ['measured_peak_gb', 'measurement_kind', 'measured_ctx', 'measurement_match', 'measured_unit', 'measured_evidence_level', 'measured_source']) {
    assert.ok(census.data.every((r) => key in r), `every row carries ${key}`);
  }
  // GLM-5.3 행은 GLM-5.2 행과 model 이름만 다르다(계산 동일)
  const strip = (r) => JSON.stringify({ ...r, model: undefined });
  const glm52 = census.data.filter((r) => r.model === 'GLM-5.2');
  const glm53 = census.data.filter((r) => r.model === 'GLM-5.3');
  assert.equal(glm52.length, rowsPerModel);
  assert.equal(glm53.length, rowsPerModel);
  assert.deepEqual(glm53.map(strip), glm52.map(strip));
  // 낡은 2.14.1 수치는 어느 문서에도 남지 않는다
  for (const rel of ['README.md', 'AGENTS.md', 'census/README.md']) assert.equal(read(rel).includes('9,126'), false, `${rel} still says 9,126`);
});

test('README and AGENTS model/verdict counts are derived from the catalog and census, not copied from a prior release', () => {
  const manifest = readJson('census/manifest.json');
  const readme = read('README.md');
  const agents = read('AGENTS.md');
  const verdicts = `${manifest.rows.toLocaleString('en-US')} verdicts`;
  assert.ok(readme.includes(`**${verdicts}** (${LOCAL_MODELS.length} models incl. draft tier × ${manifest.devices} GPUs/Macs × quant tiers)`), 'README census sentence');
  assert.ok(readme.includes(`Open data: the full **Fit Census** (${verdicts}, **CC0**)`), 'README open-data sentence');
  assert.ok(agents.includes(`${verdicts}, model × device × quant`), 'AGENTS census line');
  assert.ok(agents.includes('30 byte-exact anchors'));
});

test('README and CONTRIBUTING claims stay bounded: GLM-5.3 is listed as supported, never as universally runnable or more accurate', () => {
  const readme = read('README.md');
  const contrib = read('CONTRIBUTING.md');
  assert.ok(readme.includes('GLM-5.3'), 'README names GLM-5.3 support');
  assert.ok(readme.includes('GLM-5.2, GLM-5.3, GLM-4.7-Flash'), 'MLA family list includes GLM-5.3 next to GLM-5.2');
  for (const text of [readme, contrib]) {
    assert.doesNotMatch(text, /anyone can run GLM|run GLM-5\.3 on any|GLM-5\.3[^\n]*(?:tok\/s|tokens\/s|faster)/i);
    assert.doesNotMatch(text, /accuracy (?:improved|up) by|\d+\s?% (?:more |higher )?accura|most accurate|#1 accuracy/i);
    assert.doesNotMatch(text, /lazy-or-host-resident/i);
  }
  assert.doesNotMatch(readme, /(?:in|within) one second|1-second|instant verdict/i);
});
