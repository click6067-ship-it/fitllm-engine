#!/usr/bin/env node
// Conformance runner — an implementation conforms if every vector passes.
import { readFileSync } from 'node:fs';
import { strict as assert } from 'node:assert';
import { LOCAL_MODELS, GPUS, gpuDevice, simulate, calcKVCache, calcLinearState } from '../engine.js';

const { vectors, version } = JSON.parse(readFileSync(new URL('./fit-vectors-v1.json', import.meta.url), 'utf8'));
const byModel = (n) => LOCAL_MODELS.find((m) => m.name === n);
const toDevice = (d) => (d.type === 'mac' ? d.ram : gpuDevice(GPUS.find((g) => g.name === d.gpu), d.env));
let pass = 0, fail = 0;
for (const v of vectors) {
  try {
    const model = byModel(v.model);
    assert.ok(model, `unknown model ${v.model}`);
    if (v.kind === 'kv_total_bytes') assert.equal(calcKVCache(model, v.ctx, v.kvBits).totalBytes, v.expect);
    else if (v.kind === 'kv_per_token_bytes') assert.equal(calcKVCache(model, v.ctx ?? 1, v.kvBits).kvPerToken, v.expect);
    else if (v.kind === 'linear_state_bytes') assert.equal(calcLinearState(model).totalBytes, v.expect);
    else if (v.kind === 'verdict') assert.equal(simulate(model, toDevice(v.device), v.ctx, { weightBpw: v.weightBpw, kvBits: v.kvBits }).verdict, v.expect);
    else throw new Error(`unknown kind ${v.kind}`);
    pass++; console.log(`PASS ${v.id}`);
  } catch (e) { fail++; console.error(`FAIL ${v.id}: ${e.message}`); }
}
console.log(`\nfit-vectors v${version}: ${pass}/${pass + fail} pass`);
process.exit(fail ? 1 : 0);
