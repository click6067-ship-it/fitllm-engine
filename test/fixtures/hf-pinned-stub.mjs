// CLI 를 네트워크 없이 실제 Hugging Face 경로(fetchHfModel → parseHfConfig)로 돌리는 fetch 스텁(`node --import`).
// hf-api-evidence-2026-10-18.json 에 고정한 레포만 응답한다 — API 메타데이터(리비전·샤드 크기·safetensors 파라미터),
// 같은 리비전의 config.json(test/fixtures/<org>__<repo>.config.json), 샤드 이름만 담은 index.
import { readFileSync } from 'node:fs';

const evidence = JSON.parse(readFileSync(new URL('./hf-api-evidence-2026-10-18.json', import.meta.url), 'utf8'));
const json = (value, status = 200) => new Response(JSON.stringify(value), {
  status, headers: { 'content-type': 'application/json' },
});

globalThis.fetch = async (input) => {
  const url = new URL(String(input));
  if (url.hostname !== 'huggingface.co') throw new Error(`unexpected host in test: ${url.hostname}`);
  const path = decodeURIComponent(url.pathname);
  const api = path.match(/^\/api\/models\/([^/]+\/[^/]+)$/);
  if (api) {
    const repo = evidence[api[1]];
    if (!repo?.shards) return json({ error: 'not found' }, 404);
    return json({
      sha: repo.sha,
      safetensors: repo.safetensors,
      siblings: [{ rfilename: 'config.json', size: 3311 }, ...repo.shards],
    });
  }
  const file = path.match(/^\/([^/]+\/[^/]+)\/resolve\/([a-f0-9]{40})\/(.+)$/);
  const repo = file && evidence[file[1]];
  if (!repo || repo.sha !== file[2]) throw new Error(`unexpected path in test: ${path}`);
  if (file[3] === 'config.json') {
    return new Response(readFileSync(new URL(`./${file[1].replace('/', '__')}.config.json`, import.meta.url)), { status: 200 });
  }
  if (file[3] === 'model.safetensors.index.json') {
    return json({ weight_map: Object.fromEntries(repo.shards.map((s, i) => [`tensor.${i}`, s.rfilename])) });
  }
  throw new Error(`unexpected path in test: ${path}`);
};
