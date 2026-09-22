// `fitllm scan` — 이미 받아둔 모델을 읽어 "이거 내 기기에서 되나"를 한 번에 답한다.
//
// 여기서 새로 계산하는 건 없다. provider 가 설치 목록을 주면 해석은 engine 의
// resolveLocalModel, 하드웨어는 detectHardware, 판정은 simulate 가 그대로 한다 —
// CLI 와 웹이 같은 질문에 다른 답을 내면 안 되기 때문이다.
//
// 가장 중요한 규칙: **매핑 실패는 정상 경로다.** 이름이 비슷하다고 카탈로그 행을
// 갖다 붙이면, 사용자는 자기가 가진 것과 다른 모델의 숫자를 받는다. 그건 숫자가
// 없는 것보다 나쁘다. 해석 못 하면 런타임이 알려준 **측정된 사실만** 인용한다.
import { listAllInstalled } from './scan-providers.mjs';
import { detectHardware } from './detect-hardware.mjs';
import {
  LOCAL_MODELS, GPUS, gpuDevice, simulate, resolveLocalModel, fmtGB,
} from '../engine.js';

const DEFAULT_CTX = 8192;
const DEFAULT_QUANT = { weightBpw: 4.8944, kvBits: 16 }; // Q4_K_M — GGUF 배포본의 사실상 기본

function bytesLabel(bytes) {
  if (!Number.isFinite(bytes) || bytes < 0) return null;
  return `${(bytes / 1073741824).toFixed(1)} GB`;
}

// Ollama 는 "qwen3:0.6b", LM Studio 는 "org/repo" 꼴이다. **네임스페이스만** 떼고
// 나머지는 엔진 해석기에 그대로 넘긴다.
// 태그를 떼지 않는 이유: Ollama 에서 `:0.6b` 는 장식이 아니라 모델 정체성이다. 떼면
// "qwen3" 만 남아 0.6B 와 1.7B 사이에서 ambiguous 가 되고, 사용자가 실제로 가진 게
// 무엇인지 아는 상태에서 모른다고 답하게 된다(실측: resolveLocalModel('qwen3:0.6b')
// 는 이미 Qwen3-0.6B 로 해석한다).
export function catalogQueryFor(installedId) {
  return (String(installedId || '').split('/').pop() || '').trim();
}

export function resolveInstalled(installedId, resolver = resolveLocalModel) {
  const query = catalogQueryFor(installedId);
  if (!query) return { status: 'unresolved', model: null };
  const result = resolver(query);
  if (result?.status === 'resolved' && result.canonicalName) {
    const model = LOCAL_MODELS.find((m) => m.name === result.canonicalName) || null;
    return model ? { status: 'resolved', model } : { status: 'unresolved', model: null };
  }
  return { status: result?.status === 'ambiguous' ? 'ambiguous' : 'unresolved', model: null };
}

function describeDevice(device) {
  if (!device) return null;
  if (device.type === 'gpu') {
    const count = device.gpuCount && device.gpuCount > 1 ? ` ×${device.gpuCount}` : '';
    return `${device.gpu?.name || 'GPU'}${count} (${device.memoryGB}GB)`;
  }
  return `Mac ${device.memoryGB}GB unified`;
}

// 감지에 실패하면 판정을 지어내지 않는다. 목록은 그대로 보여주되 판정 자리는 비운다.
export function detectDevice({ detect = detectHardware } = {}) {
  let detection;
  try { detection = detect({ catalog: { GPUS } }); } catch { return null; }
  const gpu = detection?.gpu || detection?.resolved || null;
  if (gpu && Number.isFinite(gpu.vramGB)) return gpuDevice(gpu);
  if (Number.isFinite(detection?.unifiedMemoryGB)) {
    return { type: 'apple', memoryGB: detection.unifiedMemoryGB, ram: detection.unifiedMemoryGB };
  }
  return null;
}

export async function buildScan(deps = {}) {
  const providers = await listAllInstalled(deps);
  const device = deps.device !== undefined ? deps.device : detectDevice(deps);
  const ctx = Number.isFinite(deps.ctx) ? deps.ctx : DEFAULT_CTX;

  const groups = providers.map((result) => ({
    provider: result.provider,
    ok: result.ok,
    note: result.note,
    models: result.models.map((installed) => {
      const resolution = resolveInstalled(installed.id, deps.resolver);
      let verdict = null;
      if (resolution.status === 'resolved' && device) {
        try {
          const sim = simulate(resolution.model, device, ctx, DEFAULT_QUANT);
          verdict = { verdict: sim.verdict, usedGB: sim.used, freeGB: sim.free, maxContext: sim.maxContext };
        } catch { verdict = null; }
      }
      return {
        installedId: installed.id,
        // 런타임이 알려준 측정된 사실 — 해석 실패해도 이건 참이다.
        reported: {
          sizeBytes: installed.sizeBytes, quant: installed.quant,
          paramSizeLabel: installed.paramSizeLabel, family: installed.family,
        },
        catalogModel: resolution.status === 'resolved' ? resolution.model.name : null,
        resolution: resolution.status,
        fit: verdict,
      };
    }),
  }));

  return {
    device: device ? { label: describeDevice(device), type: device.type, memoryGB: device.memoryGB } : null,
    ctx,
    quant: 'Q4_K_M',
    providers: groups,
    installedCount: groups.reduce((n, g) => n + g.models.length, 0),
    availableProviders: groups.filter((g) => g.ok).map((g) => g.provider),
  };
}

const VERDICT_MARK = { yes: '✓ FITS', tight: '△ TIGHT', no: "✗ WON'T FIT" };

export function renderScan(scan) {
  const lines = [];
  if (!scan.availableProviders.length) {
    // 빈 목록을 "설치된 모델 없음"으로 단정하지 않는다 — 우리가 못 본 것뿐일 수 있다.
    lines.push('No supported runtime answered on this machine.');
    for (const g of scan.providers) lines.push(`  ${g.provider}: ${g.note || 'unavailable'}`);
    lines.push('');
    lines.push('fitllm scan reads Ollama (loopback API) and LM Studio (lms CLI). Start one, or');
    lines.push('check a model by name instead:  fitllm "Gemma 4 31b" --detect');
    return lines.join('\n');
  }

  lines.push(scan.device
    ? `Hardware: ${scan.device.label} · quant ${scan.quant} · ctx ${scan.ctx.toLocaleString('en-US')}`
    : 'Hardware: not detected — listing installed models without a verdict');
  lines.push('');

  for (const group of scan.providers) {
    if (!group.ok) continue;
    lines.push(`${group.provider} (${group.models.length})`);
    if (!group.models.length) lines.push('  no models installed');
    for (const item of group.models) {
      const size = bytesLabel(item.reported.sizeBytes);
      const facts = [size, item.reported.quant].filter(Boolean).join(' · ');
      if (item.fit) {
        const mark = VERDICT_MARK[item.fit.verdict] || item.fit.verdict;
        lines.push(`  ${mark}  ${item.installedId}  →  ${item.catalogModel}`);
        lines.push(`        ${fmtGB(item.fit.usedGB)} used, ${fmtGB(item.fit.freeGB)} free${facts ? `  (on disk: ${facts})` : ''}`);
      } else if (item.resolution === 'resolved') {
        lines.push(`  —      ${item.installedId}  →  ${item.catalogModel}  (no hardware detected)`);
      } else {
        // 추측 금지: 판정 자리에 숫자가 아니라 사실을 둔다.
        lines.push(`  ?      ${item.installedId}  —  not in the catalog, no verdict${facts ? `  (on disk: ${facts})` : ''}`);
      }
    }
    lines.push('');
  }
  const unresolved = scan.providers.flatMap((g) => g.models).filter((m) => m.resolution !== 'resolved').length;
  if (unresolved) {
    lines.push(`${unresolved} installed model(s) are not in the catalog. Their size is what the runtime reported;`);
    lines.push('no memory estimate is made for an architecture this engine has not verified.');
  }
  return lines.join('\n').trimEnd();
}
