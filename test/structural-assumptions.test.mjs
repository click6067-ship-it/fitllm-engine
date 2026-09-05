import test from 'node:test';
import assert from 'node:assert/strict';
import {
  GPUS, LOCAL_MODELS, gpuDevice, simulate, structuralAssumptions,
} from '../engine.js';

const gpu = gpuDevice(GPUS.find((candidate) => candidate.name === 'RTX 4090'));

test('structuralAssumptions exposes stable ids and plain GQA keeps the legacy shape', () => {
  const mla = LOCAL_MODELS.find((model) => model.name === 'GLM-4.7-Flash');
  const gqa = LOCAL_MODELS.find((model) => model.name === 'Llama-3.1-8B-Instruct');
  assert.deepEqual(structuralAssumptions(mla, gpu), [{
    id: 'mla-compressed-latent-cache',
    statement: 'KV memory assumes a compressed-latent MLA artifact or mode; legacy non-MLA GGUF or an explicitly uncompressed mode invalidates this estimate.',
  }]);
  assert.deepEqual(structuralAssumptions({ ...gqa, mtpLayerCount: 1 }, 64), [{
    id: 'mtp-ordinary-generation',
    statement: 'KV memory assumes ordinary non-speculative generation; an MTP draft context is not included.',
  }]);
  const plain = simulate(gqa, gpu, 8192, { weightBpw: 4.8944, kvBits: 16 });
  assert.equal('structuralAssumptions' in plain, false);
});

test('verified PLE excludes GPU weights only under the pinned path while an invented family keeps full GPU weights', () => {
  const verified = LOCAL_MODELS.find((model) => model.name === 'Gemma 4 e2b');
  const invented = { ...verified, name: 'Invented PLE', pleOffloadVerified: false };
  assert.deepEqual(
    LOCAL_MODELS.filter((model) => model.pleOffloadVerified === true).map((model) => model.name),
    ['Gemma 4 e2b', 'Gemma 4 e4b'],
  );
  const verifiedResult = simulate(verified, gpu, 8192, { weightBpw: 16, kvBits: 16 });
  const inventedResult = simulate(invented, gpu, 8192, { weightBpw: 16, kvBits: 16 });
  assert.deepEqual(verifiedResult.structuralAssumptions, [{
    id: 'ple-llamacpp-non-gpu-residency',
    // 2.15.0 residency-policy correction: 근거는 pinned llama.cpp의 입력층 host 배치 사실 — lazy/on-disk 경로는 근거가 아니다
    statement: 'GPU weight memory excludes the verified Gemma 4 PLE tensors only because the pinned llama.cpp/GGUF path assigns the per_layer_token_embd input-layer tensor to CPU/host buffers instead of accelerator memory; that host memory is not budgeted here, and a runtime that loads PLE onto the accelerator invalidates this estimate.',
  }]);
  assert.ok(Math.abs(verifiedResult.param - 5.1241) < 5e-4);
  assert.ok(Math.abs(inventedResult.param - 9.4995) < 5e-4);
  assert.equal(inventedResult.pleOffloadGB, 0);
  assert.equal('structuralAssumptions' in inventedResult, false);
});
