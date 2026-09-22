// `fitllm measure --from-ollama` — 런타임이 **이미 알고 있는** 실측값을 회수한다.
//
// 왜 이게 필요한가: census 는 10,530 판정을 담고 있는데 실측 대조는 9건이다. 예측이
// 맞는지 공개적으로 확인할 방법이 사실상 없다는 뜻이고, "정확도가 제품"인 프로젝트에서
// 그게 가장 약한 고리다. 종전 경로는 사용자가 메모리를 직접 재서 --measured 로 넣어야
// 했고, 그걸 할 줄 아는 사람은 거의 없다.
//
// 새 측정 기계는 만들지 않는다. Ollama 가 /api/ps 로 상주량을 이미 보고하므로 그걸 읽어
// 기존 buildMeasurementReport 에 넣을 뿐이다. 측정 종류도 새로 만들지 않는다 —
// idle_resident 가 이미 있고 size_vram 이 정확히 그것이다.
//
// 제출은 사람이 한다. 이 명령은 이슈를 열지 않고 네트워크로 아무것도 보내지 않는다.
// 제보는 주장이고, 검증 없이 census 에 들어가면 그 오류는 우리 책임이 된다.
import { listRunningOllama } from './scan-providers.mjs';
import { resolveInstalled, detectDevice } from './scan.mjs';
import { buildMeasurementReport } from './measurement-report.mjs';
import { simulate, fmtGB, gpuDevice, combineGpus, resolveGpuByName, appleDevice } from '../engine.js';

const BYTES_PER_GIB = 1073741824;
const DEFAULT_CTX = 8192;
const DEFAULT_QUANT = { weightBpw: 4.8944, kvBits: 16 };

// size_vram 은 **상주량**이지 생성 중 최대가 아니다. 우리 판정은 system_total_peak 계열이라
// 이 실측은 그 일부만 검증한다. 후보에 그 사실을 같이 싣지 않으면, 읽는 사람이 전체 검증으로
// 오해한다 — 숫자를 주는 것보다 숫자의 범위를 주는 게 중요하다.
export const PARTIAL_VERIFICATION_NOTE =
  'idle_resident: weights plus KV resident in VRAM with the model loaded and idle. '
  + 'This does not measure peak memory during generation, so it verifies part of the prediction, not all of it.';

// 하드웨어 자동감지는 NVIDIA 가 아니거나 nvidia-smi 가 없으면 실패한다. 그때 실측을
// 기여할 방법이 없으면 곤란하다 — 정작 판정이 애매해 실측이 가장 필요한 사람이 그런 기기를
// 쓰는 경우가 많다. 기존 measure 경로가 이미 --gpu/--mac 을 받으므로 같은 표기를 허용한다.
// 사용자가 스스로 밝힌 기기는 추측이 아니라 진술이고, 보고서에 그대로 기록된다.
export function deviceFromFlags({ gpu, mac, count } = {}) {
  if (mac != null && mac !== '') {
    const ram = Number(mac);
    if (!Number.isFinite(ram) || ram <= 0) throw new Error(`--mac must be a positive number of GB, got: ${mac}`);
    return appleDevice(ram);
  }
  if (gpu != null && gpu !== '') {
    const resolved = resolveGpuByName(String(gpu));
    if (!resolved || resolved.status !== 'resolved') {
      throw new Error(`unknown GPU: ${gpu}. Use a catalog name, for example "RTX 4090".`);
    }
    const n = Number(count);
    const card = resolved.gpu || resolved.match || resolved.value;
    return Number.isInteger(n) && n > 1 ? combineGpus(Array(n).fill(card)) : gpuDevice(card);
  }
  return null;
}

export async function buildRuntimeMeasurements(deps = {}) {
  const runtime = await listRunningOllama(deps);
  if (!runtime.ok) {
    return { ok: false, note: runtime.note || 'not running', candidates: [], skipped: [] };
  }
  if (!runtime.version) {
    // buildMeasurementReport 는 숫자를 포함한 runtime 문자열을 요구한다. 못 읽으면
    // "Ollama (unknown)" 같은 걸 지어내지 않고 후보를 만들지 않는다.
    return { ok: false, note: 'could not read the Ollama version; nothing submitted', candidates: [], skipped: [] };
  }
  const device = deps.device !== undefined ? deps.device : detectDevice(deps);
  const ctx = Number.isFinite(deps.ctx) ? deps.ctx : DEFAULT_CTX;
  const candidates = [];
  const skipped = [];

  for (const entry of runtime.running) {
    if (!entry.residentBytes) {
      skipped.push({ id: entry.id, reason: entry.reason || 'no resident size reported' });
      continue;
    }
    const resolution = resolveInstalled(entry.id, deps.resolver);
    if (resolution.status !== 'resolved') {
      // 예측값이 없으면 대조할 것이 없다. 이름이 비슷하다고 갖다 붙이지 않는다.
      skipped.push({ id: entry.id, reason: 'not in the catalog, so there is no prediction to compare against' });
      continue;
    }
    if (!device) {
      skipped.push({ id: entry.id, reason: 'hardware not detected, so the report would not identify the machine' });
      continue;
    }
    const measuredGiB = entry.residentBytes / BYTES_PER_GIB;
    let sim;
    try { sim = simulate(resolution.model, device, ctx, DEFAULT_QUANT); } catch {
      skipped.push({ id: entry.id, reason: 'engine could not produce a prediction for this configuration' });
      continue;
    }
    try {
      const report = buildMeasurementReport({
        model: resolution.model.name,
        hardware: device.type === 'gpu' ? device.gpu.name : `Mac ${device.memoryGB}GB unified`,
        quant: 'Q4_K_M',
        ctx,
        kvBits: DEFAULT_QUANT.kvBits,
        measured: Number(measuredGiB.toFixed(3)),
        kind: 'idle_resident',
        unit: 'GiB',
        runtime: `Ollama ${runtime.version}`,
        // ⚠️ 예측값은 **측정 종류와 같은 양**이어야 한다. idle_resident 는 상주 가중치+KV 이므로
        //    param+kv+linearState 를 쓴다. 기존 measure 경로가 generation_peak 에 쓰는
        //    used-reserve 는 런타임 오버헤드를 포함해 여기서는 34% 가량 크다(Qwen3-0.6B/4090
        //    기준 1.632 vs 1.215). 그 값을 짝지으면 검토자가 "엔진이 과대예측한다"고 읽고
        //    엔진을 실측에 맞추려 들 수 있다 — 이 레포가 '54GB 앵커'로 한 번 겪은 실패다.
        predicted: sim.param + sim.kv + sim.linearState,
        predictedMetric: 'resident_weights_plus_kv_gb',
      });
      candidates.push({ installedId: entry.id, report, predictedTotalGB: sim.used });
    } catch (error) {
      skipped.push({ id: entry.id, reason: `report rejected: ${error.message}` });
    }
  }
  return { ok: true, note: null, candidates, skipped, version: runtime.version };
}

export function renderRuntimeMeasurements(result) {
  const lines = [];
  if (!result.ok) {
    lines.push(`No measurement taken: ${result.note}.`);
    lines.push('');
    lines.push('This reads what Ollama already reports for a loaded model. Load one and try again:');
    lines.push('  ollama run <model>   # in another shell, then re-run this command');
    return lines.join('\n');
  }
  if (!result.candidates.length) {
    lines.push('No submittable measurement from the models currently loaded.');
    for (const s of result.skipped) lines.push(`  ${s.id}: ${s.reason}`);
    return lines.join('\n');
  }
  lines.push(`${result.candidates.length} measurement candidate(s) from Ollama ${result.version}:`);
  lines.push('');
  lines.push(PARTIAL_VERIFICATION_NOTE);
  lines.push('');
  for (const candidate of result.candidates) {
    const r = candidate.report.candidate || candidate.report;
    lines.push(`${candidate.installedId}  →  ${r.model}`);
    // 비교 가능한 양을 **먼저·가깝게** 둔다. 총량을 measured 바로 옆에 놓으면 2 vs 3.6 이
    // 나란히 읽혀 "엔진이 80% 과대예측"으로 오해된다 — JSON 에서 고친 문제가 사람용 줄에
    // 그대로 남아 있었다(2026-09-23 실제 실행에서 발견).
    lines.push(`  measured ${r.measuredPeakGB} GiB resident · engine predicts ${fmtGB(r.predictedGB)} resident  ← compare these`);
    lines.push(`  (the full prediction to run it is ${fmtGB(candidate.predictedTotalGB)}, which includes runtime overhead and reserve — not what this measures)`);
    lines.push('');
    lines.push(JSON.stringify(r, null, 2));
    if (candidate.report.issueUrl) {
      lines.push('');
      lines.push(`Submit (opens a prefilled issue, nothing is sent from here):\n  ${candidate.report.issueUrl}`);
    }
    lines.push('');
  }
  for (const s of result.skipped) lines.push(`skipped ${s.id}: ${s.reason}`);
  return lines.join('\n').trimEnd();
}
