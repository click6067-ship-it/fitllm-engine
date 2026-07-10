#!/usr/bin/env node
// 모델카드용 Fit Matrix 블록 생성기 — HF 모델카드/README 아웃리치를 1커맨드로 (마케팅 2차).
// 사용: node scripts/fit-matrix.mjs "<model name>"  → 마크다운 블록 stdout
// 원칙: 트렌딩(이미 트래픽 있는) 모델 한정 아웃리치 — 배지는 신뢰 증폭기지 콜드스타트 해법이 아님(ICSE 2018 실증)
import { LOCAL_MODELS, GPUS, gpuDevice, simulate, calcMaxContext, formatTokens } from '../engine.js';
const EN = (ko, en) => en;
const q = (process.argv[2] || '').toLowerCase();
const m = LOCAL_MODELS.find((x) => x.name.toLowerCase() === q) || LOCAL_MODELS.find((x) => x.name.toLowerCase().includes(q));
if (!m) { console.error('unknown model — node scripts/fit-matrix.mjs "<name>"'); process.exit(2); }
const PICK = ['RTX 3060 12GB', 'RTX 4090', 'RTX 5090', 'A100 80GB'];
const rows = PICK.map((n) => {
  const g = GPUS.find((x) => x.name === n); if (!g) return null;
  const d = gpuDevice(g);
  const s = simulate(m, d, Math.min(8192, m.maxContext), { weightBpw: 4.85, kvBits: 16 });
  const mc = calcMaxContext(m, d, { weightBpw: 4.85, kvBits: 16 });
  const slug = `${m.name}-q4_k_m-on-${g.name}`.toLowerCase().replace(/[^a-z0-9._+-]+/g, '-').replace(/-+/g, '-');
  const V = { yes: '✅ fits', tight: '⚠️ tight', no: '❌ won’t fit' }[s.verdict];
  return `| ${g.name} (${g.vramGB}GB) | ${V} @ Q4_K_M | ${s.verdict === 'no' ? '—' : '~' + formatTokens(mc, EN)} | [receipt](https://fitllm.run/r/${slug}) |`;
}).filter(Boolean);
console.log(`## Will it run locally? (${m.name})

| GPU | verdict | max context | details |
|---|---|---|---|
${rows.join('\n')}

Live math (updates with the engine): [fitllm.run](https://fitllm.run/?m=${LOCAL_MODELS.indexOf(m)}) · \`npx fitllm "${m.name}" --detect\` · [how it's computed](https://github.com/click6067-ship-it/fitllm-engine) (MIT, config.json-derived)`);
