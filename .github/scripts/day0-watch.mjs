#!/usr/bin/env node
// day0-watch — HF 트렌딩을 감시해 카탈로그에 없는 신규 모델을 day0-candidate 이슈로 올린다.
// Day-0 플레이북(발견→라이브 ≤24h)의 "발견" 단계 기계화. 판단·추가는 사람+플레이북이 한다(자동 추가 금지).
// 중복 방지 = 이슈 타이틀 정본(라벨 day0-candidate, state=all — 한 번 올라온 id는 다시 안 올림).
// DRY_RUN=1 이면 이슈 생성 없이 출력만.
const REPO = process.env.GITHUB_REPOSITORY || 'click6067-ship-it/fitllm-engine';
const TOKEN = process.env.GH_TOKEN || process.env.GITHUB_TOKEN;
const DRY = process.env.DRY_RUN === '1';
const MAX_NEW = 3; // 러닝당 신규 이슈 상한 (스팸 가드)

const gh = async (path, init = {}) => {
  const r = await fetch(`https://api.github.com${path}`, {
    ...init,
    headers: { Authorization: `Bearer ${TOKEN}`, Accept: 'application/vnd.github+json', 'X-GitHub-Api-Version': '2022-11-28', ...(init.headers || {}) },
  });
  if (!r.ok && r.status !== 422) throw new Error(`${path} → ${r.status} ${await r.text()}`);
  return r.status === 422 ? null : r.json();
};

// 1) HF 트렌딩 (텍스트 생성만) — 견고 fetch: HF가 일시적으로 HTML(레이트리밋·인터스티셜)을 주면
//    JSON.parse 크래시 대신 재시도 후 graceful skip. 발견 크론은 best-effort라 한 런 건너뛰어도 무해
//    (다음 6h 런이 따라잡음). 상류 일시장애로 실패 메일을 쏘지 않는다.
const HF_URL = 'https://huggingface.co/api/models?sort=trendingScore&direction=-1&limit=20&filter=text-generation';
const sleep = (ms) => new Promise((res) => setTimeout(res, ms));
async function fetchTrending() {
  for (let attempt = 1; attempt <= 3; attempt++) {
    try {
      const r = await fetch(HF_URL, { headers: { Accept: 'application/json', 'User-Agent': 'fitllm-day0-watch' } });
      const ct = r.headers.get('content-type') || '';
      const text = await r.text();
      if (!r.ok) throw new Error(`HF ${r.status}: ${text.slice(0, 120)}`);
      if (!ct.includes('json')) throw new Error(`non-JSON (${ct || 'no content-type'}): ${text.slice(0, 120)}`);
      const data = JSON.parse(text);
      if (!Array.isArray(data)) throw new Error('payload is not an array');
      return data;
    } catch (e) {
      console.warn(`HF trending fetch attempt ${attempt}/3 failed: ${e.message}`);
      if (attempt < 3) await sleep(attempt * 2000);
    }
  }
  return null;
}

const trending = await fetchTrending();
if (!trending) {
  console.log('day0-watch: HF trending unavailable (transient) — skipping this run.');
  process.exit(0);
}

// 2) 카탈로그 소프트 매치 (이름 부분일치 — 정본 dedup은 이슈)
const { LOCAL_MODELS } = await import('../../engine.js');
const norm = (s) => String(s).toLowerCase().replace(/[^a-z0-9]+/g, '');
const catalogNorms = LOCAL_MODELS.map((m) => norm(m.name));
const inCatalog = (id) => { const n = norm(id.split('/').pop()); return catalogNorms.some((c) => n.includes(c) || c.includes(n)); };

// 3) 기존 day0 이슈 (state=all — 닫힌 것도 재알림 금지)
const seen = new Set();
for (let page = 1; page <= 3; page++) {
  const issues = await gh(`/repos/${REPO}/issues?labels=day0-candidate&state=all&per_page=100&page=${page}`);
  if (!issues?.length) break;
  for (const i of issues) { const m = i.title.match(/^day0: (.+)$/); if (m) seen.add(m[1]); }
}

// 4) 라벨 보장 (있으면 422 → null 무시)
if (!DRY) await gh(`/repos/${REPO}/labels`, { method: 'POST', body: JSON.stringify({ name: 'day0-candidate', color: '7c3aed', description: 'HF trending model not in catalog — Day-0 playbook trigger' }) });

// 5) 신규 후보 → 이슈
let created = 0;
for (const m of trending) {
  if (created >= MAX_NEW) break;
  const id = m.id || m.modelId;
  if (!id || seen.has(id) || inCatalog(id)) continue;
  if (/[-_.](gguf|awq|gptq|mlx|bnb|exl2|int[48]|nvfp4|mxfp4|fp4|fp8)\b/i.test(id)) continue; // 양자화 미러 repo = 노이즈 (원본이 따로 트렌딩됨). fp4/fp8 추가 — issue #13(NVFP4 미러) 유출 방지
  const body = [
    `HF trending detected (trendingScore ${m.trendingScore ?? '?'}, downloads ${m.downloads ?? '?'}, likes ${m.likes ?? '?'}).`,
    '',
    `- Model: https://huggingface.co/${id}`,
    `- Config: https://huggingface.co/${id}/raw/main/config.json`,
    `- Weights index: https://huggingface.co/${id}/raw/main/model.safetensors.index.json`,
    '',
    'Day-0 playbook (판단은 사람이): ① config 확보 ② 아키텍처 판별 — 엔진이 모델링 못 하는 구조면 **추가하지 않는다**(틀린 숫자>없는 숫자) ③ 이중 검증 ④ 엔진 append + 손계산 테스트 ⑤ SEO/배포 ⑥ 벡터 1개. 트리거 기준: "can I run X"가 검색될 모델인가.',
    '',
    '_Auto-filed by day0-watch (6h cron). Close with a reason if not a candidate — closed issues are never re-filed._',
  ].join('\n');
  if (DRY) { console.log(`DRY: would file "day0: ${id}"`); created++; continue; }
  await gh(`/repos/${REPO}/issues`, { method: 'POST', body: JSON.stringify({ title: `day0: ${id}`, body, labels: ['day0-candidate'] }) });
  console.log(`filed: day0: ${id}`);
  created++;
}
console.log(`day0-watch: trending=${trending.length}, seen=${seen.size}, filed=${created}${DRY ? ' (dry)' : ''}`);
