function attentionBasis(model) {
  const basis = {
    totalLayers: model.layerCount,
    kvLayers: model.fullAttnLayers ?? model.layerCount,
    kvHeads: model.kvHeads,
    kvHeadDim: model.kvHeadDim,
  };

  if (model.linearAttn) {
    return {
      kind: 'hybrid-linear',
      ...basis,
      linearAttention: {
        layers: model.linearAttn.layers,
        numKHeads: model.linearAttn.numKHeads,
        numVHeads: model.linearAttn.numVHeads,
        headKDim: model.linearAttn.headKDim,
        headVDim: model.linearAttn.headVDim,
        convKernel: model.linearAttn.convKernel,
      },
    };
  }

  if (model.mlaKvLoraRank) {
    return {
      kind: 'mla',
      ...basis,
      mlaKvLoraRank: model.mlaKvLoraRank,
      mlaRopeDim: model.mlaRopeDim,
    };
  }

  if (model.slidingWindow) {
    return {
      kind: 'sliding-window',
      ...basis,
      slidingWindow: model.slidingWindow,
      ...(model.slidingPattern ? { slidingPattern: model.slidingPattern } : {}),
      ...(model.globalKvHeads ? { globalKvHeads: model.globalKvHeads } : {}),
      ...(model.globalHeadDim ? { globalHeadDim: model.globalHeadDim } : {}),
    };
  }

  return { kind: 'gqa', ...basis };
}

function hardwareBasis(device, detection) {
  if (device.type === 'gpu') {
    return {
      type: 'gpu',
      name: device.gpu.name,
      memoryGB: device.memoryGB,
      environment: device.env,
      gpuCount: device.gpuCount,
      detected: Boolean(detection),
      evidence: {
        status: device.gpu.status ?? (detection ? 'DETECTED' : 'UNVERIFIED'),
        verifiedAt: device.gpu.verifiedAt ?? null,
        sources: device.gpu.sources?.vramGB ?? [],
      },
    };
  }

  return {
    type: 'apple',
    name: detection?.chip ? `${detection.chip} Mac` : 'Apple Silicon Mac',
    memoryGB: device.memoryGB,
    environment: 'apple-unified',
    detected: Boolean(detection),
    evidence: {
      status: detection ? 'DETECTED' : 'USER_PROVIDED',
      verifiedAt: null,
      sources: detection?.source ? [detection.source] : [],
    },
  };
}

export function buildExplanation({
  model,
  device,
  simulation,
  quantLabel,
  kvBits,
  ctx,
  modelSource,
  detection,
}) {
  return {
    schemaVersion: 1,
    verdict: simulation.verdict,
    model: {
      name: model.name,
      source: modelSource ?? { type: 'catalog' },
      totalParamsB: model.totalParams,
      activeParamsB: model.activeParams ?? model.totalParams,
      layers: model.layerCount,
      maxContext: model.maxContext,
      tags: [...(model.tags ?? [])],
    },
    hardware: hardwareBasis(device, detection),
    configuration: {
      quant: quantLabel,
      weightBpw: simulation.weightBpw,
      kvBits,
      contextTokens: ctx,
    },
    attention: attentionBasis(model),
    memoryGiB: {
      weights: simulation.param,
      kvCache: simulation.kv,
      linearState: simulation.linearState,
      runtimeDynamic: simulation.rtDyn,
      reserve: simulation.reserve,
      total: simulation.used,
      available: simulation.memoryGB,
      free: simulation.free,
    },
    assumptions: [
      'Model architecture inputs come from the catalog or the pinned Hugging Face revision shown above.',
      'The configured quantization applies to model weights; KV-cache precision is reported separately.',
      'A fit verdict estimates memory capacity for one inference workload at the requested context.',
    ],
    limitations: [
      'This is a memory-capacity estimate, not a speed prediction.',
      'Runtime overhead and reserve are estimates; real usage varies by runtime, OS, drivers, and workload.',
    ],
    ...(simulation.structuralAssumptions
      ? { structuralAssumptions: simulation.structuralAssumptions }
      : {}),
  };
}
