import test from 'node:test';
import assert from 'node:assert/strict';
import { detectHardware, parseNvidiaSmi, resolvePowerShellAdapters } from '../bin/detect-hardware.mjs';
import { GPUS } from '../engine.js';

test('test_multi_nvidia', () => {
  const result = parseNvidiaSmi('NVIDIA GeForce RTX 4090, 24564\nNVIDIA GeForce RTX 3090, 24576\n', GPUS);
  assert.equal(result.adapters.length, 2);
  assert.deepEqual(result.gpus.map((gpu) => gpu.name), ['RTX 4090', 'RTX 3090']);
  assert.equal(result.gpus.reduce((sum, gpu) => sum + gpu.vramGB, 0), 48);
  assert.equal(result.confidence, 'measured-vram');
  assert.throws(() => parseNvidiaSmi('NVIDIA GeForce RTX 4090, not-a-number\n', GPUS), /invalid nvidia-smi/i);
  assert.throws(() => parseNvidiaSmi(`${'NVIDIA GeForce RTX 4090, 24564\n'.repeat(9)}`, GPUS), /more than 8/i);

  const detected = detectHardware({
    execFile: () => 'NVIDIA GeForce RTX 5080, 16384\n',
    platform: 'linux',
    arch: 'x64',
    env: {},
    catalog: GPUS,
  });
  assert.equal(detected.environment, 'windows-display', 'auto-detect must preserve the conservative receipt-compatible reserve');
});

test('test_windows_wsl_fallback', () => {
  const amd = resolvePowerShellAdapters(JSON.stringify(['AMD Radeon RX 7900 XTX', 'Intel(R) Iris(R) Xe Graphics']), GPUS);
  assert.deepEqual(amd.gpus.map((gpu) => gpu.name), ['RX 7900 XTX']);
  assert.equal(amd.confidence, 'catalog-identity');

  const pro = resolvePowerShellAdapters(JSON.stringify('AMD Radeon PRO W7900'), GPUS);
  assert.deepEqual(pro.gpus.map((gpu) => gpu.name), ['Radeon PRO W7900']);

  assert.throws(
    () => resolvePowerShellAdapters(JSON.stringify('NVIDIA GeForce RTX 4090'), [
      { name: 'RTX 4090', vramGB: 24 },
      { name: 'RTX 4090 48GB', vramGB: 48 },
    ]),
    /not supported/i,
  );

  assert.throws(
    () => resolvePowerShellAdapters(JSON.stringify('Intel(R) Iris(R) Xe Graphics'), GPUS),
    /Intel\(R\) Iris\(R\) Xe Graphics.*not supported/i,
  );

  const execFile = (command) => {
    if (command === 'nvidia-smi') throw Object.assign(new Error('missing'), { code: 'ENOENT' });
    if (command === 'powershell.exe') return JSON.stringify('Intel(R) Iris(R) Xe Graphics');
    throw new Error(`unexpected ${command}`);
  };
  assert.throws(() => detectHardware({ execFile, platform: 'linux', arch: 'x64', env: { WSL_INTEROP: '/run/WSL/1' }, catalog: GPUS }), /Iris.*not supported/i);
});

test('test_apple_silicon_only', () => {
  const execFile = (command, args) => {
    if (command === 'nvidia-smi') throw new Error('missing');
    if (command === 'sysctl' && args.at(-1) === 'hw.memsize') return String(64 * 1024 ** 3);
    if (command === 'sysctl' && args.at(-1) === 'machdep.cpu.brand_string') return 'Apple M4 Max';
    throw new Error(`unexpected ${command}`);
  };
  const detected = detectHardware({ execFile, platform: 'darwin', arch: 'arm64', env: {}, catalog: GPUS });
  assert.equal(detected.kind, 'apple');
  assert.equal(detected.ramGB, 64);
  assert.equal(detected.chip, 'M4 Max');
  assert.throws(() => detectHardware({ execFile, platform: 'darwin', arch: 'x64', env: {}, catalog: GPUS }), /Intel Mac.*not supported/i);
});
