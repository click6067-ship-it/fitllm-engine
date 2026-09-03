import test from 'node:test';
import assert from 'node:assert/strict';
import { fetchHfModel, parseHfId } from '../bin/hf-model.mjs';

test('test_bounded_fail_closed_fetch', async () => {
  assert.equal(parseHfId('https://huggingface.co/Qwen/Qwen3-32B/tree/main'), 'Qwen/Qwen3-32B');
  assert.equal(parseHfId('not-an-id'), null);

  const requested = [];
  const revision = 'c1899de289a04d12100db370d81485cdf75e47ca';
  const validFetch = async (url, options = {}) => {
    requested.push(url);
    if (url.endsWith('/config.json')) return new Response(JSON.stringify({ model_type: 'llama', hidden_size: 1024 }), { status: 200, headers: { 'x-repo-commit': revision } });
    if (url.endsWith('/model.safetensors.index.json')) return new Response(JSON.stringify({ metadata: { total_size: 1234 } }), { status: 200 });
    if (options.method === 'HEAD') return new Response('', { status: 404 });
    return new Response('', { status: 404 });
  };
  const loaded = await fetchHfModel('Qwen/Qwen3-32B', { fetchImpl: validFetch });
  assert.equal(loaded.id, 'Qwen/Qwen3-32B');
  assert.equal(loaded.totalSize, 1234);
  assert.equal(loaded.revision, revision);
  assert.match(requested[1], new RegExp(`/resolve/${revision}/model\\.safetensors\\.index\\.json$`));

  await assert.rejects(
    fetchHfModel('Qwen/Qwen3-32B', { fetchImpl: async () => new Response('{}', { status: 200 }) }),
    /revision/i,
  );

  await assert.rejects(
    fetchHfModel('Qwen/Qwen3-32B', { fetchImpl: async () => new Response('x'.repeat(1_000_001), { status: 200, headers: { 'x-repo-commit': revision } }) }),
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
        if (url.endsWith('/config.json')) {
          return new Promise((resolve) => setTimeout(
            () => resolve(new Response('{}', { status: 200, headers: { 'x-repo-commit': revision } })),
            70,
          ));
        }
        return new Promise((resolve) => setTimeout(
          () => resolve(new Response(JSON.stringify({ metadata: { total_size: 1234 } }), { status: 200 })),
          70,
        ));
      },
    }),
    /timed out/i,
  );
});
