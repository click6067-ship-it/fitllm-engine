import test from 'node:test';
import assert from 'node:assert/strict';
import { fetchHfModel, parseHfId } from '../bin/hf-model.mjs';

const REVISION = 'c1899de289a04d12100db370d81485cdf75e47ca';
const API = 'https://huggingface.co/api/models/Qwen/Qwen3-32B?blobs=true';

// index의 metadata.total_size는 레포에 따라 파라미터 수를 담는다(GLM-4.7-Flash 실측).
// 여기서는 그 값(1234)과 실제 shard byte 합계(6234)를 일부러 다르게 두어, 수집기가 어느
// 쪽을 checkpoint bytes로 채택하는지 단언한다.
const SHARD_A = 3117;
const SHARD_B = 3117;
const METADATA = {
  sha: REVISION,
  safetensors: { parameters: { BF16: 1_000 }, total: 1_000 },
  siblings: [
    { rfilename: 'model-00001-of-00002.safetensors', size: SHARD_A },
    { rfilename: 'model-00002-of-00002.safetensors', size: SHARD_B },
  ],
};
const INDEX = {
  metadata: { total_size: 1234 },
  weight_map: { a: 'model-00001-of-00002.safetensors', b: 'model-00002-of-00002.safetensors' },
};

test('test_bounded_fail_closed_fetch', async () => {
  assert.equal(parseHfId('https://huggingface.co/Qwen/Qwen3-32B/tree/main'), 'Qwen/Qwen3-32B');
  assert.equal(parseHfId('not-an-id'), null);

  const requested = [];
  const validFetch = async (url, options = {}) => {
    requested.push(url);
    if (url === API) return new Response(JSON.stringify(METADATA), { status: 200 });
    if (url.endsWith('/config.json')) return new Response(JSON.stringify({ model_type: 'llama', hidden_size: 1024, dtype: 'bfloat16' }), { status: 200 });
    if (url.endsWith('/model.safetensors.index.json')) return new Response(JSON.stringify(INDEX), { status: 200 });
    if (options.method === 'HEAD') return new Response('', { status: 404 });
    return new Response('', { status: 404 });
  };
  const loaded = await fetchHfModel('Qwen/Qwen3-32B', { fetchImpl: validFetch });
  assert.equal(loaded.id, 'Qwen/Qwen3-32B');
  // 실제 shard byte 합계를 채택하고 index의 total_size(1234)는 바이트로 쓰지 않는다.
  assert.equal(loaded.totalSize, SHARD_A + SHARD_B);
  assert.notEqual(loaded.totalSize, INDEX.metadata.total_size);
  assert.equal(loaded.revision, REVISION);
  assert.deepEqual(loaded.parameterEvidence, {
    revision: REVISION,
    safetensorsParameters: { BF16: 1_000 },
    safetensorsTotal: 1_000,
  });
  // metadata를 먼저 읽고, config·index는 봉인된 revision에서만 읽는다.
  assert.equal(requested[0], API);
  assert.match(requested[1], new RegExp(`/resolve/${REVISION}/config\\.json$`));
  assert.match(requested[2], new RegExp(`/resolve/${REVISION}/model\\.safetensors\\.index\\.json$`));

  // weight_map이 siblings에 없는 파일을 가리키면 합계를 만들지 않는다(부분합 금지).
  const missingShard = await fetchHfModel('Qwen/Qwen3-32B', {
    fetchImpl: async (url, options = {}) => {
      if (url === API) {
        return new Response(JSON.stringify({
          ...METADATA,
          siblings: [{ rfilename: 'model-00001-of-00002.safetensors', size: SHARD_A }],
        }), { status: 200 });
      }
      if (url.endsWith('/config.json')) return new Response(JSON.stringify({ model_type: 'llama', hidden_size: 1024, dtype: 'bfloat16' }), { status: 200 });
      if (url.endsWith('/model.safetensors.index.json')) return new Response(JSON.stringify(INDEX), { status: 200 });
      if (options.method === 'HEAD') return new Response('', { status: 404 });
      return new Response('', { status: 404 });
    },
  });
  assert.equal(missingShard.totalSize, null);
  assert.equal(missingShard.parameterEvidence.safetensorsParameters.BF16, 1_000);

  // 파라미터 증거도 checkpoint 크기도 없으면 추정하지 않는다.
  await assert.rejects(
    fetchHfModel('Qwen/Qwen3-32B', {
      fetchImpl: async (url, options = {}) => {
        if (url === API) return new Response(JSON.stringify({ sha: REVISION, siblings: [] }), { status: 200 });
        if (url.endsWith('/config.json')) return new Response(JSON.stringify({ model_type: 'llama', hidden_size: 1024, dtype: 'bfloat16' }), { status: 200 });
        if (options.method === 'HEAD') return new Response('', { status: 404 });
        return new Response('', { status: 404 });
      },
    }),
    /refusing to guess/i,
  );

  await assert.rejects(
    fetchHfModel('Qwen/Qwen3-32B', { fetchImpl: async () => new Response('{}', { status: 200 }) }),
    /revision/i,
  );

  // config.json은 자체 1MB 상한을 그대로 지킨다.
  await assert.rejects(
    fetchHfModel('Qwen/Qwen3-32B', {
      fetchImpl: async (url) => {
        if (url === API) return new Response(JSON.stringify(METADATA), { status: 200 });
        return new Response('x'.repeat(1_000_001), { status: 200 });
      },
    }),
    /too large/i,
  );

  // metadata도 상한이 있다.
  await assert.rejects(
    fetchHfModel('Qwen/Qwen3-32B', { fetchImpl: async () => new Response('x'.repeat(5_000_001), { status: 200 }) }),
    /too large/i,
  );

  await assert.rejects(
    fetchHfModel('Qwen/Qwen3-32B', { fetchImpl: async () => new Response('', { status: 403 }) }),
    /gated|private|not found/i,
  );

  await assert.rejects(
    fetchHfModel('Qwen/Qwen3-32B', {
      timeoutMs: 100,
      fetchImpl: async (url) => {
        if (url === API) {
          return new Promise((resolve) => setTimeout(
            () => resolve(new Response(JSON.stringify(METADATA), { status: 200 })),
            70,
          ));
        }
        return new Promise((resolve) => setTimeout(
          () => resolve(new Response(JSON.stringify({ model_type: 'llama', hidden_size: 1024, dtype: 'bfloat16' }), { status: 200 })),
          70,
        ));
      },
    }),
    /timed out/i,
  );
});
