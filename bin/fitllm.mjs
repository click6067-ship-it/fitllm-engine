#!/usr/bin/env node
// fitllm — "will it run?" in one line. Zero deps. MIT. https://fitllm.run
import { execFileSync } from 'node:child_process';
import { canonicalReceiptSlug } from './receipt-slug.mjs';
import {
  LOCAL_MODELS, GPUS, GPU_QUANTS, MACBOOK_RAM_GROUPS,
  gpuDevice, combineGpus, simulate, calcMaxContext, suggestFix, suggestFixGpu, formatTokens, fmtGB,
} from '../engine.js';

const argv = process.argv.slice(2);
const flag = (name) => { const i = argv.indexOf(name); return i >= 0 ? argv[i + 1] : undefined; };
const has = (name) => argv.includes(name);
const FLAGS_WITH_VALUE = ['--gpu', '--mac', '--quant', '--ctx', '--kv', '--count'];
const positional = argv.filter((a, i) => !a.startsWith('--') && !FLAGS_WITH_VALUE.includes(argv[i - 1]));

const HELP = `fitllm — will this LLM fit your hardware? (open engine: github.com/click6067-ship-it/fitllm-engine)

usage:
  npx fitllm <model> --gpu "RTX 4090"        GPU fit (Q4_K_M, 8K ctx defaults)
  npx fitllm <model> --gpu "5090 + 3090"     multi-GPU rig — VRAM pools across cards
  npx fitllm <model> --gpu 3090 --count 2    N identical cards (2× RTX 3090)
  npx fitllm <model> --mac 64                Apple Silicon fit (8-bit default)
  npx fitllm <model> --detect                read this machine's real hardware (best-effort)
  npx fitllm --top [N] --gpu 4090            what CAN I run? — best quant per model that fits
  npx fitllm --list                          list built-in models & hardware
options:
  --quant Q4_K_M|Q5_K_M|Q6_K|Q8_0|FP16 | 4|8|16    weight quant (GPU tiers | Mac bits)
  --ctx N          context tokens (default 8192)
  --kv 16|8|4      KV-cache quant (default 16 = F16)
  --count N        copies of --gpu (1-8, default 1)
  --json           machine-readable output
exit codes: 0 fits · 1 won't fit · 2 error (guard-friendly: run before you download)`;

if (has('--help') || argv.length === 0) { console.log(HELP); process.exit(argv.length ? 0 : 2); }

if (has('--list')) {
  console.log('MODELS:');
  for (const m of LOCAL_MODELS) console.log(`  ${m.name}  (${m.totalParams}B${m.activeParams && m.activeParams < m.totalParams ? `, ${m.activeParams}B active` : ''})`);
  console.log('GPUS:');
  for (const g of GPUS) console.log(`  ${g.name}  (${g.vramGB}GB)`);
  console.log('MACS:');
  for (const [chip, rams] of Object.entries(MACBOOK_RAM_GROUPS)) console.log(`  ${chip}: ${rams.join('/')}GB`);
  process.exit(0);
}

const TOP = has('--top'); // --top 모드는 모델 인자 불필요 (전 카탈로그 스캔)
let model = null;
if (!TOP) {
  const q = (positional[0] || '').toLowerCase();
  model = LOCAL_MODELS.find((m) => m.name.toLowerCase() === q) || LOCAL_MODELS.find((m) => m.name.toLowerCase().includes(q));
  if (!model) { console.error(`unknown model: "${positional[0] || ''}" — try: npx fitllm --list`); process.exit(2); }
}

// ── device: --gpu | --mac | --detect ──
let device, isGpu, hwLabel;
const gpuName = flag('--gpu');
const macRam = flag('--mac');
if (gpuName) {
  // "5090 + 3090" 조합(이종 허용) + --count N 복제 → VRAM 풀링 (combineGpus)
  const parts = gpuName.split(/[+,&]/).map((s) => s.trim()).filter(Boolean);
  const found = [];
  for (const p of parts) {
    const pq = p.toLowerCase();
    const g = GPUS.find((x) => x.name.toLowerCase() === pq) || GPUS.find((x) => x.name.toLowerCase().includes(pq));
    if (!g) { console.error(`unknown GPU: "${p}" — try: npx fitllm --list`); process.exit(2); }
    found.push(g);
  }
  const count = Math.min(Math.max(parseInt(flag('--count'), 10) || 1, 1), 8);
  const list = [];
  for (let i = 0; i < count; i++) list.push(...found);
  device = combineGpus(list, 'windows-display'); isGpu = true;
  hwLabel = `${device.gpu.name} (${device.memoryGB}GB${list.length > 1 ? ' pooled' : ''})`;
} else if (macRam) {
  const ram = parseInt(macRam, 10);
  if (!Number.isFinite(ram) || ram < 8) { console.error('--mac needs RAM in GB (e.g. --mac 64)'); process.exit(2); }
  device = ram; isGpu = false; hwLabel = `Mac ${ram}GB unified memory`;
} else if (has('--detect')) {
  try {
    const out = execFileSync('nvidia-smi', ['--query-gpu=name,memory.total', '--format=csv,noheader,nounits'], { encoding: 'utf8' }).trim().split('\n')[0];
    const [name, mem] = out.split(',').map((s) => s.trim());
    const vram = Math.round(parseInt(mem, 10) / 1024);
    const g = GPUS.find((x) => name.toLowerCase().includes(x.name.toLowerCase())) || { name: `${name} (detected)`, vramGB: vram, bandwidthGBs: 0, series: 'detected' };
    device = gpuDevice(g, 'windows-display'); isGpu = true; hwLabel = `${g.name} (${g.vramGB}GB, detected)`;
  } catch {
    if (process.platform === 'darwin') {
      try {
        const bytes = parseInt(execFileSync('sysctl', ['-n', 'hw.memsize'], { encoding: 'utf8' }).trim(), 10);
        const ram = Math.round(bytes / 1024 ** 3);
        device = ram; isGpu = false; hwLabel = `this Mac (${ram}GB unified memory, detected)`;
      } catch { console.error('detect failed — pass --gpu "<name>" or --mac <GB>'); process.exit(2); }
    } else { console.error('detect failed (no nvidia-smi) — pass --gpu "<name>" or --mac <GB>'); process.exit(2); }
  }
} else { console.error('need --gpu "<name>", --mac <GB>, or --detect'); process.exit(2); }

// ── quant / ctx / kv ──
const kvRaw = parseInt(flag('--kv'), 10);
const kvBits = [16, 8, 4].includes(kvRaw) ? kvRaw : 16;
const rawQ = flag('--quant');

// ── --top: "이 하드웨어에서 뭘 돌릴 수 있나" — 모델별 최고 quant 티어로 fit 스캔 ──
if (TOP) {
  const nRaw = flag('--top');
  const n = /^\d+$/.test(nRaw || '') ? Math.max(1, parseInt(nRaw, 10)) : Infinity; // --top 뒤가 플래그면 전체
  const ctxReq = parseInt(flag('--ctx'), 10) || 8192;
  // 티어는 최고 정밀도부터 시도 (fit되는 가장 좋은 quant를 그 모델의 대표로). --quant 지정 시 그 티어만.
  let tiers;
  if (isGpu) {
    if (rawQ) {
      const t = GPU_QUANTS.find((x) => x.tier.toLowerCase() === String(rawQ).toLowerCase());
      if (!t) { console.error(`unknown GGUF tier "${rawQ}" — one of: ${GPU_QUANTS.map((x) => x.tier).join(', ')}`); process.exit(2); }
      tiers = [{ label: t.tier, bpw: t.bpw }];
    } else tiers = [...GPU_QUANTS].sort((a, b) => b.bpw - a.bpw).map((t) => ({ label: t.tier, bpw: t.bpw }));
  } else {
    const b = parseInt(rawQ, 10);
    tiers = ([4, 8, 16].includes(b) ? [b] : [16, 8, 4]).map((bits) => ({ label: `${bits}-bit`, bpw: bits }));
  }
  const fits = [];
  for (const m of LOCAL_MODELS) {
    if (!m.totalParams) continue;
    for (const t of tiers) {
      const c = Math.min(ctxReq, m.maxContext);
      const s = simulate(m, device, c, { weightBpw: t.bpw, kvBits });
      if (s.verdict !== 'no') {
        fits.push({ m, t, s, ctx: c, maxCtx: calcMaxContext(m, device, { weightBpw: t.bpw, kvBits }) });
        break; // 최고 티어에서 멈춤 — 이 모델의 대표
      }
    }
  }
  fits.sort((a, b) => b.m.totalParams - a.m.totalParams); // 파라미터 수 내림차순 — 크기 순일 뿐 품질 순위 아님
  const top = Number.isFinite(n) ? fits.slice(0, n) : fits;
  if (has('--json')) {
    console.log(JSON.stringify({
      hardware: hwLabel, ctx: ctxReq, kvBits, rankedBy: 'total parameter count (size only — not a quality ranking)',
      fits: top.map(({ m, t, s, maxCtx }) => ({
        model: m.name, params_b: m.totalParams, quant: t.label, verdict: s.verdict,
        usedGB: +s.used.toFixed(2), freeGB: +s.free.toFixed(2), maxContext: maxCtx,
      })),
      engine: 'fitllm-engine',
    }, null, 2));
  } else {
    const EN = (ko, en) => en;
    if (!top.length) {
      console.log(`✗ nothing in the built-in catalog fits ${hwLabel} at ${formatTokens(ctxReq, EN)} — try --kv 8 or a smaller --ctx`);
    } else {
      console.log(`What fits ${hwLabel} @ ${formatTokens(ctxReq, EN)}${kvBits !== 16 ? `, KV Q${kvBits}` : ''} — best quant per model, largest first (size ≠ quality):`);
      const W = Math.max(...top.map(({ m }) => m.name.length));
      top.forEach(({ m, t, s, maxCtx }, i) => {
        const mark = s.verdict === 'tight' ? '△' : '✓';
        console.log(`  ${String(i + 1).padStart(2)}. ${m.name.padEnd(W)}  ${String(m.totalParams).padStart(6)}B  ${t.label.padEnd(6)}  ${mark} ${fmtGB(s.used)}/${s.memoryGB} GB${maxCtx >= 1024 ? `  max ctx ~${formatTokens(maxCtx, EN)}` : ''}`);
      });
      console.log(`  every number from official config.json — audit: github.com/click6067-ship-it/fitllm-engine`);
    }
  }
  process.exit(top.length ? 0 : 1);
}
let weightBpw, quantLabel;
if (isGpu) {
  const tier = GPU_QUANTS.find((x) => x.tier.toLowerCase() === String(rawQ || 'Q4_K_M').toLowerCase());
  if (!tier) { console.error(`unknown GGUF tier "${rawQ}" — one of: ${GPU_QUANTS.map((x) => x.tier).join(', ')}`); process.exit(2); }
  weightBpw = tier.bpw; quantLabel = tier.tier;
} else {
  const bits = parseInt(rawQ, 10);
  weightBpw = [4, 8, 16].includes(bits) ? bits : 8;
  quantLabel = `${weightBpw}-bit`;
}
const ctx = Math.min(parseInt(flag('--ctx'), 10) || 8192, model.maxContext);

// ── verdict ──
const EN = (ko, en) => en; // CLI output is English
const s = simulate(model, device, ctx, { weightBpw, kvBits });
const maxCtx = calcMaxContext(model, device, { weightBpw, kvBits });
if (has('--json')) {
  console.log(JSON.stringify({
    model: model.name, hardware: hwLabel, quant: quantLabel, kvBits, ctx,
    verdict: s.verdict, usedGB: +s.used.toFixed(2), memoryGB: s.memoryGB, freeGB: +s.free.toFixed(2),
    breakdown: { paramGB: +s.param.toFixed(2), kvGB: +s.kv.toFixed(2), linearStateGB: +s.linearState.toFixed(2), overheadGB: +s.rtDyn.toFixed(2), reserveGB: +s.reserve.toFixed(2) }, // reserve 병기 분해는 rtDyn(s.rt는 Apple 고정 2GB 중복). linearState=선형어텐션 고정상태(표준 어텐션은 0)
    maxContext: maxCtx, engine: 'fitllm-engine',
  }, null, 2));
} else {
  const WORD = { yes: 'FITS', tight: 'TIGHT', no: "WON'T FIT" };
  const mark = s.verdict === 'no' ? '✗' : s.verdict === 'tight' ? '△' : '✓';
  console.log(`${mark} ${WORD[s.verdict]} — ${model.name} on ${hwLabel} @ ${quantLabel}, ${formatTokens(ctx, EN)}${kvBits !== 16 ? `, KV Q${kvBits}` : ''}`);
  const lsPart = s.linearState > 0 ? ` + linear-state ${fmtGB(s.linearState)}` : ''; // 하이브리드 선형 어텐션만 표시(표준 어텐션은 0이라 생략)
  console.log(`  weights ${fmtGB(s.param)} + KV ${fmtGB(s.kv)}${lsPart} + overhead ${fmtGB(s.rtDyn)} + reserve ${fmtGB(s.reserve)} = ${fmtGB(s.used)} / ${s.memoryGB} GB  (${s.verdict === 'no' ? 'short by ' + fmtGB(-s.free) : 'free ' + fmtGB(s.free)} GB)`);
  if (maxCtx >= 1024) console.log(`  max context at this quant: ~${formatTokens(maxCtx, EN)}`);
  if (s.verdict === 'no') {
    const fix = isGpu ? suggestFixGpu(model, s.device, ctx, { weightBpw, kvBits }, EN) : suggestFix(model, device, ctx, weightBpw, EN);
    console.log(`  → ${fix.text}`);
    console.log(`  (this just saved you a ~${fmtGB(s.param)} GB download that would have OOM'd)`); // 절약 영수증
  }
  // canonical 슬러그(v2 /r 파서와 동일 규칙) — 비기본 ctx/kv도 영수증이 같은 조건으로 재계산하게 토큰 포함
  const slug = canonicalReceiptSlug({
    modelName: model.name, quantLabel, isGpu,
    hwLabel: hwLabel.split(' (')[0], ramGB: isGpu ? 0 : device,
    ctx, kvBits, maxContext: model.maxContext,
  });
  console.log(`  receipt: https://fitllm.run/r/${slug}`);
  console.log('  every number from official config.json — audit: github.com/click6067-ship-it/fitllm-engine');
}
process.exit(s.verdict === 'no' ? 1 : 0);
