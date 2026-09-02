#!/usr/bin/env node
// Local LLM Fit Census — 전 조합 진실표 생성기. 누구나 재생성: `npm run census`
// 산출: census-v1.json(기계용) + census-v1.csv + README.md(스타터 매트릭스·모델별 최소장비)
// measured 열은 ../fixtures/measured.json(커뮤니티 실측 PR)에서 교차 참조 — 예측 vs 실측을 공개 원장으로.
import { writeFileSync, readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { LOCAL_MODELS, GPUS, MACBOOK_RAM_GROUPS, GPU_QUANTS, gpuDevice, simulate, calcMaxContext, calcRuntimeOverhead, DATA_UPDATED } from '../engine.js';

// CENSUS_OUT: 결정성 검증(census/check.mjs)이 임시 디렉터리로 재생성할 때 사용. 기본 = 이 디렉터리(커밋 산출물).
const OUT = process.env.CENSUS_OUT ? process.env.CENSUS_OUT.replace(/\/?$/, '/') : new URL('./', import.meta.url).pathname;
const measured = JSON.parse(readFileSync(new URL('../fixtures/measured.json', import.meta.url), 'utf8'));
const measuredMap = new Map(measured.map((m) => [[m.model, m.device, m.quant].join('|'), m]));

const devices = [
  ...GPUS.map((g) => ({ platform: 'gpu', name: g.name, memoryGB: g.vramGB, arg: gpuDevice(g) })),
  ...Object.entries(MACBOOK_RAM_GROUPS).flatMap(([chip, rams]) => rams.map((ram) => ({ platform: 'mac', name: `${chip} ${ram}GB`, memoryGB: ram, arg: ram }))),
];
const quantsFor = (p) => (p === 'gpu' ? GPU_QUANTS.map((q) => ({ id: q.tier, bpw: q.bpw })) : [4, 8, 16].map((b) => ({ id: `${b}bit`, bpw: b })));

const rows = [];
for (const m of LOCAL_MODELS) {
  for (const d of devices) {
    for (const q of quantsFor(d.platform)) {
      const ctx = Math.min(8192, m.maxContext);
      const s = simulate(m, d.arg, ctx, { weightBpw: q.bpw, kvBits: 16 });
      const maxCtx = calcMaxContext(m, d.arg, { weightBpw: q.bpw, kvBits: 16 });
      const meas = measuredMap.get([m.name, d.name, q.id].join('|'));
      const ov = calcRuntimeOverhead(m, ctx, { weightBpw: q.bpw, kvBits: 16 }, s.device);
      rows.push({
        model: m.name, params_b: m.totalParams, device: d.name, platform: d.platform, memory_gb: d.memoryGB,
        quant: q.id, ctx, kv: 'F16', verdict: s.verdict,
        // 예측 분해 — 에이전트가 실측과 "같은 종류끼리" 비교할 수 있게 컬럼명이 의미를 싣는다(one truth, two projections).
        used_gb: +s.used.toFixed(2),                                        // v1 하위호환 별칭 = predicted_total_to_run_gb
        predicted_total_to_run_gb: +s.used.toFixed(2),                      // 판정 기준: weights+KV+runtime+reserve
        predicted_param_gb: +s.param.toFixed(2),                            // 순수 가중치(양자화 bpw 기준)
        predicted_resident_weights_gb: +(s.param + ov.paramOverheadGB).toFixed(2), // 상주 가중치 예측(비양자 임베딩 등 +12%) — idle_resident 실측과 비교하는 컬럼
        kv_cache_gb: +s.kv.toFixed(2), linear_state_gb: +s.linearState.toFixed(3), runtime_dynamic_gb: +s.rtDyn.toFixed(2), reserve_gb: +s.reserve.toFixed(2),
        free_gb: +s.free.toFixed(2), max_context: maxCtx,
        // 실측 — 타입 없이 예측 옆에 붙이지 않는다(resident 바닥값을 total과 직접 비교하면 오독).
        measured_peak_gb: meas ? meas.measuredPeakGB : null,                // v1 하위호환 — 의미는 measurement_kind가 정의
        measurement_kind: meas ? (meas.measurementKind || 'unknown') : null,
        measured_ctx: meas ? meas.ctx : null,
        measurement_match: meas ? (meas.ctx === ctx ? 'same_ctx' : 'different_ctx') : null,
        measured_unit: meas ? (meas.unit || null) : null,
        measured_evidence_level: meas ? (meas.evidenceLevel || 'community_unverified') : null, // 제보=주장 원칙 — 검증 등급을 소비자에게 그대로 노출
        measured_source: meas ? meas.source : null,
      });
    }
  }
}

// CENSUS_DATE 명시 시 그 날짜로 고정(결정적 재생성·byte-identical 검증용). 미지정 = 오늘(신규 릴리스 생성).
if (process.env.CENSUS_DATE && !/^\d{4}-\d{2}-\d{2}$/.test(process.env.CENSUS_DATE)) {
  console.error(`CENSUS_DATE must be YYYY-MM-DD (got "${process.env.CENSUS_DATE}")`);
  process.exit(2);
}
const generated = process.env.CENSUS_DATE || new Date().toISOString().slice(0, 10);
const header = {
  version: 1, schema_version: 2, generated, engine_data: DATA_UPDATED, verdicts: rows.length,
  assumptions: 'ctx=min(8192,model max), KV cache F16, engine reserve/headroom per platform',
  definitions: {
    predicted_total_to_run_gb: 'Total memory required to RUN: weights + KV cache + linear-attention state + runtime overhead + OS/GPU reserve. This is what the verdict uses. (used_gb is its v1 alias.)',
    linear_state_gb: 'Fixed recurrent state of hybrid linear-attention (Gated DeltaNet) layers. Unlike KV cache this does NOT grow with context — it is a constant per sequence. Exactly 0 for standard full/sliding attention models.',
    predicted_resident_weights_gb: 'Predicted resident model weights incl. non-quantized parts — compare against idle_resident measurements.',
    measurement_kind: 'What the measured number is: idle_resident (resident weights floor, e.g. oMLX actual_size) / load_peak / generation_peak / system_total_peak. Only system_total_peak is directly comparable to predicted_total_to_run_gb; idle_resident readings SHOULD be lower than the total — that is not a prediction error.',
    measurement_match: 'same_ctx = measured at this row’s ctx; different_ctx = measured under a different context length (KV portion not directly comparable).',
    units: 'Predicted *_gb columns are GiB-based (1024³), matching nvidia-smi/Activity Monitor style labels; measured_unit records the measurement’s own unit if reported.',
  },
  regenerate: 'npm run census', measured_from: 'fixtures/measured.json',
};
writeFileSync(OUT + 'census-v1.json', JSON.stringify({ ...header, data: rows }, null, 1));

const cols = Object.keys(rows[0]);
const cell = (v) => (v == null ? '' : typeof v === 'number' ? String(v) : /[",]/.test(v) ? JSON.stringify(v) : v);
writeFileSync(OUT + 'census-v1.csv', [cols.join(','), ...rows.map((r) => cols.map((c) => cell(r[c])).join(','))].join('\n') + '\n');

// ── manifest: UI 파싱 없이 데이터셋을 인용·검증할 수 있는 기계용 명세 (라이선스·행수·체크섬·정본 URL) ──
const sha256 = (f) => createHash('sha256').update(readFileSync(OUT + f)).digest('hex');
writeFileSync(OUT + 'manifest.json', JSON.stringify({
  name: 'FitLLM Fit Census',
  version: header.version, schema_version: header.schema_version, generated: header.generated,
  license: 'CC0-1.0', license_url: 'https://creativecommons.org/publicdomain/zero/1.0/',
  rows: rows.length, devices: devices.length, measured_rows: measured.length,
  canonical: {
    site: 'https://fitllm.run/data/',
    json: 'https://fitllm.run/data/census-v1.json',
    csv: 'https://fitllm.run/data/census-v1.csv',
    source: 'https://github.com/click6067-ship-it/fitllm-engine/tree/master/census',
  },
  sha256: { 'census-v1.json': sha256('census-v1.json'), 'census-v1.csv': sha256('census-v1.csv') },
  engine: { name: 'fitllm-engine', license: 'MIT', npm: 'https://www.npmjs.com/package/fitllm-engine', repo: 'https://github.com/click6067-ship-it/fitllm-engine' },
}, null, 1));

// ── README: 사람용 요약 ──
const ICON = { yes: '✅', tight: '⚠️', no: '❌' };
const q4of = (p) => (p === 'gpu' ? 'Q4_K_M' : '4bit');
const find = (model, device, quant) => rows.find((r) => r.model === model && r.device === device && r.quant === quant);
const REP_DEVICES = ['M1 8GB', 'M2 16GB', 'M4 32GB', 'M5 Max 64GB', 'M4 Max 128GB', 'M3 Ultra 512GB', 'RTX 3060 12GB', 'RTX 4060 Ti 16GB', 'RTX 4090', 'RTX 5090', 'RX 7900 XTX', 'RTX PRO 6000 Blackwell', 'A100 80GB', 'H200 141GB'];

const starter = REP_DEVICES.map((name) => {
  const d = devices.find((x) => x.name === name);
  if (!d) return null;
  const q4 = q4of(d.platform);
  const fits = rows.filter((r) => r.device === name && r.quant === q4 && r.verdict === 'yes').sort((a, b) => b.params_b - a.params_b);
  const top = fits[0];
  return `| ${name} | ${d.memoryGB}GB | ${top ? `**${top.model}** (${top.params_b}B) — free ${top.free_gb}GB, up to ~${Math.round(top.max_context / 1000)}K ctx` : '❌ none comfortably at ~4-bit'} |`;
}).filter(Boolean).join('\n');

const minDevice = LOCAL_MODELS.map((m) => {
  const min = (p) => devices.filter((d) => d.platform === p).sort((a, b) => a.memoryGB - b.memoryGB)
    .find((d) => { const r = find(m.name, d.name, q4of(p)); return r && r.verdict !== 'no'; });
  const g = min('gpu'), mac = min('mac');
  return `| ${m.name} | ${m.totalParams}B | ${g ? g.name : '—'} | ${mac ? mac.name : '—'} |`;
}).join('\n');

const md = `# Local LLM Fit Census v1 — ${generated}

**${rows.length.toLocaleString('en-US')} verdicts**: ${LOCAL_MODELS.length} models × ${devices.length} devices (${GPUS.length} GPUs + ${devices.length - GPUS.length} Mac configs) × per-platform quant tiers.
Every number computed by [fitllm-engine](https://github.com/click6067-ship-it/fitllm-engine) from official \`config.json\` values — architecture-aware (MLA, sliding-window, hybrid attention, MoE). **Reproduce it yourself: \`npm run census\`.**

Assumptions: context = min(8K, model max) · KV cache F16 · platform reserve/headroom per engine. Interactive per-combo pages: [fitllm.run/can-i-run](https://fitllm.run/can-i-run).

## Starter matrix — biggest model that fits comfortably (~4-bit, 8K ctx)

| Device | Memory | Biggest comfortable fit |
|---|---|---|
${starter}

## Smallest device that runs each model (~4-bit, 8K ctx)

| Model | Params | Smallest GPU | Smallest Mac |
|---|---|---|---|
${minDevice}

## Full data

- [\`census-v1.csv\`](census-v1.csv) / [\`census-v1.json\`](census-v1.json) — every model × device × quant verdict with the full predicted breakdown (\`predicted_total_to_run_gb\` = weights + KV + runtime + reserve — what the verdict uses; \`predicted_resident_weights_gb\` = quantized weights **plus ~12% runtime weight overhead** (non-quantized parts, buffers) — the number resident-weights measurements should be compared against; \`predicted_param_gb\` = quantized weights alone) and max context. Machine-readable; import it, chart it, cite it.
- **Measurements are typed** (from [\`fixtures/measured.json\`](../fixtures/README.md), community PRs): \`measurement_kind\` says what was measured. \`idle_resident\` readings (e.g. oMLX \`actual_size\`) are a resident-weights **floor** — compare them to \`predicted_resident_weights_gb\`, not to the total; only \`system_total_peak\` is comparable to \`predicted_total_to_run_gb\`. An idle_resident value below the predicted total is expected, not an over-prediction. Ledger holds ${measured.length} entr${measured.length === 1 ? 'y' : 'ies'}; ${(() => { const j = rows.filter((r) => r.measured_peak_gb != null).length; return `${j} join this census (exact model+device+quant match required — the rest cover models/devices outside the catalog or carry unconfirmed attribution)`; })()}.

All figures are estimates; real usage varies with runtime, driver and OS state. Verdicts: ✅ fits comfortably · ⚠️ tight · ❌ won't fit.
`;
writeFileSync(OUT + 'README.md', md);
console.log(`census: ${rows.length} verdicts · ${devices.length} devices · measured=${measured.length} → census-v1.{json,csv} + manifest.json + README.md`);
