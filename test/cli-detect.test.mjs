import test from 'node:test';
import assert from 'node:assert/strict';
import { chmodSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { spawnSync } from 'node:child_process';

const BIN = new URL('../bin/fitllm.mjs', import.meta.url).pathname;

test('test_json_detection_metadata', () => {
  const dir = mkdtempSync(join(tmpdir(), 'fitllm-detect-'));
  try {
    const shim = join(dir, 'nvidia-smi');
    writeFileSync(shim, '#!/bin/sh\nprintf "NVIDIA GeForce RTX 4090, 24564\\nNVIDIA GeForce RTX 3090, 24576\\n"\n');
    chmodSync(shim, 0o755);
    const result = spawnSync(process.execPath, [BIN, 'Qwen 3.8 27B', '--detect', '--json'], {
      encoding: 'utf8',
      env: { ...process.env, PATH: `${dir}:${process.env.PATH}` },
    });
    assert.equal(result.status, 0, result.stderr);
    const output = JSON.parse(result.stdout);
    assert.equal(output.memoryGB, 48);
    assert.equal(output.detection.source, 'nvidia-smi');
    assert.equal(output.detection.confidence, 'measured-vram');
    assert.deepEqual(output.detection.adapters.map((adapter) => adapter.name), ['RTX 4090', 'RTX 3090']);
    assert.doesNotMatch(result.stdout, /PNPDeviceID|WSL_INTEROP/);

    const human = spawnSync(process.execPath, [BIN, 'Qwen 3.8 27B', '--detect'], {
      encoding: 'utf8',
      env: { ...process.env, PATH: `${dir}:${process.env.PATH}` },
    });
    assert.equal(human.status, 0, human.stderr);
    assert.match(human.stdout, /RTX 4090 24GB \+ RTX 3090 24GB/);
    assert.match(human.stdout, /layer split|single layer/i);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});
