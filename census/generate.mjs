#!/usr/bin/env node
// Local LLM Fit Census — 전 조합 진실표 생성기. 누구나 재생성: `npm run census`
// 산출: census-v1.json(기계용) + census-v1.csv + README.md(스타터 매트릭스·모델별 최소장비)
// measured 열은 ../fixtures/measured.json(커뮤니티 실측 PR)에서 교차 참조 — 예측 vs 실측을 공개 원장으로.
import { writeFileSync, readFileSync } from 'node:fs';
import { LOCAL_MODELS, GPUS, MACBOOK_RAM_GROUPS, GPU_QUANTS, gpuDevice, simulate, calcMaxContext, DATA_UPDATED } from '../engine.js';

const OUT = new URL('./', import.meta.url).pathname;
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
      rows.push({
        model: m.name, params_b: m.totalParams, device: d.name, platform: d.platform, memory_gb: d.memoryGB,
        quant: q.id, ctx, kv: 'F16', verdict: s.verdict, used_gb: +s.used.toFixed(2), free_gb: +s.free.toFixed(2),
        max_context: maxCtx, measured_peak_gb: meas ? meas.measuredPeakGB : null, measured_source: meas ? meas.source : null,
      });
    }
  }
}

const generated = new Date().toISOString().slice(0, 10);
const header = { version: 1, generated, engine_data: DATA_UPDATED, verdicts: rows.length, assumptions: 'ctx=min(8192,model max), KV cache F16, engine reserve/headroom per platform', regenerate: 'npm run census', measured_from: 'fixtures/measured.json' };
writeFileSync(OUT + 'census-v1.json', JSON.stringify({ ...header, data: rows }, null, 1));

const cols = Object.keys(rows[0]);
const cell = (v) => (v == null ? '' : typeof v === 'number' ? String(v) : /[",]/.test(v) ? JSON.stringify(v) : v);
writeFileSync(OUT + 'census-v1.csv', [cols.join(','), ...rows.map((r) => cols.map((c) => cell(r[c])).join(','))].join('\n') + '\n');

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

**${rows.length.toLocaleString()} verdicts**: ${LOCAL_MODELS.length} models × ${devices.length} devices (${GPUS.length} GPUs + ${devices.length - GPUS.length} Mac configs) × per-platform quant tiers.
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

- [\`census-v1.csv\`](census-v1.csv) / [\`census-v1.json\`](census-v1.json) — every model × device × quant verdict with used/free GB and max context. Machine-readable; import it, chart it, cite it.
- **\`measured\` column**: real-world measurements from [\`fixtures/measured.json\`](../fixtures/README.md) — community-submitted via PR. Predicted-vs-measured, in public. ${measured.length === 0 ? '_No measurements yet — [be the first](../fixtures/README.md)._' : `${measured.length} measurement(s) so far.`}

All figures are estimates; real usage varies with runtime, driver and OS state. Verdicts: ✅ fits comfortably · ⚠️ tight · ❌ won't fit.
`;
writeFileSync(OUT + 'README.md', md);
console.log(`census: ${rows.length} verdicts · ${devices.length} devices · measured=${measured.length} → census-v1.{json,csv} + README.md`);
