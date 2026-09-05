import assert from 'node:assert/strict';
import test from 'node:test';

import { buildExplanation } from '../bin/explain.mjs';
import { GPUS, LOCAL_MODELS, gpuDevice, simulate } from '../engine.js';

const device = gpuDevice(GPUS.find((gpu) => gpu.name === 'RTX 4090'), 'windows-display');

function explain(name) {
  const model = LOCAL_MODELS.find((candidate) => candidate.name === name);
  const simulation = simulate(model, device, 8192, { weightBpw: 4.8944, kvBits: 16 });
  return {
    simulation,
    explanation: buildExplanation({
      model,
      device,
      simulation,
      quantLabel: 'Q4_K_M',
      kvBits: 16,
      ctx: 8192,
      modelSource: null,
      detection: null,
    }),
  };
}

test('structured basis mirrors simulation without recomputing memory', () => {
  const { simulation, explanation } = explain('Gemma 4 31b');
  assert.equal(explanation.schemaVersion, 1);
  assert.deepEqual(explanation.configuration, {
    quant: 'Q4_K_M',
    weightBpw: 4.8944,
    kvBits: 16,
    contextTokens: 8192,
  });
  assert.deepEqual(explanation.memoryGiB, {
    weights: simulation.param,
    kvCache: simulation.kv,
    linearState: simulation.linearState,
    runtimeDynamic: simulation.rtDyn,
    reserve: simulation.reserve,
    total: simulation.used,
    available: simulation.memoryGB,
    free: simulation.free,
  });
  assert.deepEqual(explanation.model, {
    name: 'Gemma 4 31b',
    source: { type: 'catalog' },
    totalParamsB: 30.7,
    activeParamsB: 30.7,
    layers: 60,
    maxContext: 262144,
    tags: ['dense'],
  });
  assert.equal(explanation.hardware.evidence.status, 'VERIFIED');
  assert.equal(explanation.hardware.memoryGB, 24);
  assert.equal(explanation.verdict, simulation.verdict);
});

test('attention basis distinguishes GQA, sliding-window, MLA, and hybrid-linear', () => {
  assert.deepEqual(explain('Llama-3.1-8B-Instruct').explanation.attention, {
    kind: 'gqa',
    totalLayers: 32,
    kvLayers: 32,
    kvHeads: 8,
    kvHeadDim: 128,
  });
  assert.deepEqual(explain('Gemma 4 31b').explanation.attention, {
    kind: 'sliding-window',
    totalLayers: 60,
    kvLayers: 60,
    kvHeads: 16,
    kvHeadDim: 256,
    slidingWindow: 1024,
    slidingPattern: '5:1',
    globalKvHeads: 4,
    globalHeadDim: 512,
  });
  assert.deepEqual(explain('GLM-4.7-Flash').explanation.attention, {
    kind: 'mla',
    totalLayers: 47,
    kvLayers: 47,
    kvHeads: 20,
    kvHeadDim: 256,
    mlaKvLoraRank: 512,
    mlaRopeDim: 64,
  });
  assert.deepEqual(explain('Qwen 3.8 27B').explanation.attention, {
    kind: 'hybrid-linear',
    totalLayers: 64,
    kvLayers: 16,
    kvHeads: 4,
    kvHeadDim: 256,
    linearAttention: {
      layers: 48,
      numKHeads: 16,
      numVHeads: 48,
      headKDim: 128,
      headVDim: 128,
      convKernel: 4,
    },
  });
});

test('remote model provenance and detected hardware stay explicit', () => {
  const { simulation } = explain('Llama-3.1-8B-Instruct');
  const model = simulation.model;
  const basis = buildExplanation({
    model,
    device,
    simulation,
    quantLabel: 'Q4_K_M',
    kvBits: 16,
    ctx: 8192,
    modelSource: { type: 'huggingface', id: 'meta-llama/Llama-3.1-8B-Instruct', revision: 'a'.repeat(40) },
    detection: { kind: 'gpu', environment: 'linux-headless', adapters: [{ name: 'RTX 4090', vramGB: 24 }] },
  });
  assert.deepEqual(basis.model.source, {
    type: 'huggingface',
    id: 'meta-llama/Llama-3.1-8B-Instruct',
    revision: 'a'.repeat(40),
  });
  assert.equal(basis.hardware.detected, true);
  assert.match(basis.limitations.join('\n'), /not a speed prediction/i);
  assert.match(basis.limitations.join('\n'), /runtime.*reserve.*estimates/i);
});

test('structured basis forwards active premises without adding an empty key', () => {
  const affected = explain('GLM-4.7-Flash');
  assert.deepEqual(affected.explanation.structuralAssumptions, affected.simulation.structuralAssumptions);
  const plain = explain('Llama-3.1-8B-Instruct');
  assert.equal('structuralAssumptions' in plain.explanation, false);
});
