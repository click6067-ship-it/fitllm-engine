import { execFileSync } from 'node:child_process';
import { resolveDetectedGpu, resolveDetectedGpuByName } from './detect-resolver.mjs';

const MAX_ADAPTERS = 8;
const POWERSHELL_ADAPTERS = [
  '-NoLogo',
  '-NoProfile',
  '-NonInteractive',
  '-Command',
  'Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name | ConvertTo-Json -Compress',
];

function detectedGpu(name, vramGB) {
  return { name: `${name} (detected)`, vramGB, bandwidthGBs: 0, series: 'detected' };
}

function boundedLines(output) {
  const lines = String(output).trim().split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
  if (!lines.length) throw new Error('hardware detection returned no adapters');
  if (lines.length > MAX_ADAPTERS) throw new Error(`hardware detection returned more than ${MAX_ADAPTERS} adapters`);
  return lines;
}

export function parseNvidiaSmi(output, catalog) {
  const rows = boundedLines(output).map((line) => {
    const splitAt = line.lastIndexOf(',');
    if (splitAt < 1) throw new Error(`invalid nvidia-smi row: ${line}`);
    const name = line.slice(0, splitAt).trim();
    const memoryMiB = Number(line.slice(splitAt + 1).trim());
    if (!name || !Number.isFinite(memoryMiB) || memoryMiB <= 0) throw new Error(`invalid nvidia-smi row: ${line}`);
    const vramGB = Math.round(memoryMiB / 1024);
    if (vramGB <= 0) throw new Error(`invalid nvidia-smi memory: ${line}`);
    const catalogGpu = resolveDetectedGpu(name, vramGB, catalog);
    return {
      gpu: catalogGpu || detectedGpu(name, vramGB),
      adapter: { name: catalogGpu?.name || name, vramGB, catalogMatch: Boolean(catalogGpu) },
    };
  });
  return {
    kind: 'gpu',
    source: 'nvidia-smi',
    confidence: 'measured-vram',
    gpus: rows.map((row) => row.gpu),
    adapters: rows.map((row) => row.adapter),
  };
}

export function resolvePowerShellAdapters(output, catalog) {
  let decoded;
  try {
    decoded = JSON.parse(String(output).trim());
  } catch {
    throw new Error('PowerShell returned invalid display-adapter data');
  }
  const names = (Array.isArray(decoded) ? decoded : [decoded])
    .map((name) => String(name || '').trim())
    .filter(Boolean);
  if (!names.length) throw new Error('PowerShell returned no display adapters');
  if (names.length > MAX_ADAPTERS) throw new Error(`PowerShell returned more than ${MAX_ADAPTERS} display adapters`);
  const matches = names.map((name) => ({ name, gpu: resolveDetectedGpuByName(name, catalog) })).filter((row) => row.gpu);
  if (!matches.length) throw new Error(`${names.join(', ')} — detected display adapter is not supported; pass --gpu "<name>"`);
  return {
    kind: 'gpu',
    source: 'windows-cim',
    confidence: 'catalog-identity',
    gpus: matches.map((row) => row.gpu),
    adapters: matches.map((row) => ({ name: row.gpu.name, catalogMatch: true })),
  };
}

function isWsl(env) {
  return Boolean(env.WSL_INTEROP || env.WSL_DISTRO_NAME);
}

function run(execFile, command, args) {
  return execFile(command, args, { encoding: 'utf8', timeout: 8_000, maxBuffer: 64 * 1024 });
}

export function detectHardware({
  execFile = execFileSync,
  platform = process.platform,
  arch = process.arch,
  env = process.env,
  catalog,
} = {}) {
  let nvidiaOutput;
  let nvidiaError;
  try {
    nvidiaOutput = run(execFile, 'nvidia-smi', [
      '--query-gpu=name,memory.total',
      '--format=csv,noheader,nounits',
    ]);
  } catch (error) {
    nvidiaError = error;
  }
  if (nvidiaOutput != null) {
    const result = parseNvidiaSmi(nvidiaOutput, catalog);
    // Detection proves adapter identity and installed VRAM, not that the process owns a
    // headless card. Keep the conservative display reserve until the user explicitly
    // selects a runtime environment; this also keeps detected receipts reproducible.
    return { ...result, environment: 'windows-display' };
  }
  if (platform === 'darwin') {
    if (arch !== 'arm64') throw new Error('Intel Mac detection is not supported; pass an explicit supported device');
    try {
      const bytes = Number(run(execFile, 'sysctl', ['-n', 'hw.memsize']).trim());
      const brand = run(execFile, 'sysctl', ['-n', 'machdep.cpu.brand_string']).trim();
      const ramGB = Math.round(bytes / 1024 ** 3);
      if (!Number.isFinite(bytes) || bytes <= 0 || ramGB < 8 || !/^Apple\s+/i.test(brand)) {
        throw new Error('invalid Apple Silicon hardware data');
      }
      return {
        kind: 'apple',
        source: 'macos-sysctl',
        confidence: 'measured-memory',
        ramGB,
        chip: brand.replace(/^Apple\s+/i, ''),
        adapters: [{ name: brand }],
      };
    } catch (error) {
      throw new Error(`Apple Silicon detection failed: ${error.message}`);
    }
  }

  if (platform === 'win32' || isWsl(env)) {
    try {
      const result = resolvePowerShellAdapters(run(execFile, 'powershell.exe', POWERSHELL_ADAPTERS), catalog);
      return { ...result, environment: 'windows-display' };
    } catch (error) {
      throw new Error(`hardware detection failed: ${error.message}`);
    }
  }

  const reason = nvidiaError instanceof Error ? nvidiaError.message : String(nvidiaError);
  throw new Error(`hardware detection failed (nvidia-smi unavailable: ${reason}); pass --gpu "<name>" or --mac <GB>`);
}
