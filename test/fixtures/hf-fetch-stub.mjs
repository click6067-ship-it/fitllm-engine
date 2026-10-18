// CLI 를 네트워크 없이 Hugging Face 경로로 돌리기 위한 fetch 스텁(`node --import`).
// FITLLM_TEST_HF_SCENARIO: 'architecture' = config 는 받지만 엔진이 아키텍처를 거부한다,
//                          'gated'        = config.json 이 401(gated 레포).
const SHA = 'a'.repeat(40);
const json = (value, status = 200) => new Response(JSON.stringify(value), {
  status, headers: { 'content-type': 'application/json' },
});

globalThis.fetch = async (input) => {
  const url = new URL(String(input));
  const scenario = process.env.FITLLM_TEST_HF_SCENARIO;
  if (url.hostname !== 'huggingface.co') throw new Error(`unexpected host in test: ${url.hostname}`);
  if (url.pathname.startsWith('/api/models/')) {
    return json({ sha: SHA, siblings: [{ rfilename: 'config.json', size: 512 }, { rfilename: 'model.safetensors', size: 2_000_000_000 }] });
  }
  if (url.pathname.endsWith('/config.json')) {
    if (scenario === 'gated') return json({ error: 'gated' }, 401);
    // num_loops 는 엔진이 계산하지 않는 반복 레이어 구조다 — 아키텍처 거부 경로를 탄다.
    return json({
      model_type: 'llama', num_hidden_layers: 2, num_attention_heads: 2, num_key_value_heads: 2,
      hidden_size: 128, head_dim: 64, intermediate_size: 256, vocab_size: 1024, torch_dtype: 'bfloat16',
      num_loops: 2,
    });
  }
  if (url.pathname.endsWith('/model.safetensors.index.json')) return json({ error: 'not found' }, 404);
  throw new Error(`unexpected path in test: ${url.pathname}`);
};
