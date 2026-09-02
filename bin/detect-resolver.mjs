// --detect GPU 신원 해석(순수 함수) — vendor/generic 토큰만 벗긴 **exact token identity**.
// 종전 결함(Sol P1): 부분문자열 매치라 "RTX 4090 Laptop GPU"(16GB)→데스크톱 4090, Ti 변형이
// 비-Ti로 오인될 수 있었다. 규칙:
//   ① 앞쪽 vendor 토큰(NVIDIA/GeForce)과 말단 generic 토큰(GPU)만 제거
//   ② 남은 문자열이 카탈로그 이름과 **정확히 일치**해야 확정 — Ti/SUPER/Laptop/Mobile 등 변형
//      토큰은 벗기지 않으므로, 카탈로그에 그 변형의 정확 엔트리가 없으면 절대 다른 엔트리로
//      승격되지 않는다(예: "RTX 3090 Ti"는 RTX 3090이 될 수 없음 — Ti 엔트리에만 매치 가능)
//   ③ 감지 VRAM(반올림 GB)이 카탈로그와 정확히 일치해야 확정(동명 이용량 변형 방어)
// 불일치/변형 잔존 → null: 호출부는 감지된 실제 이름·VRAM으로 계산하고 영수증은 n/a.
export function resolveDetectedGpu(detectedName, detectedVramGB, catalog) {
  const tokens = String(detectedName).trim().split(/\s+/);
  while (tokens.length && /^(nvidia|geforce)$/i.test(tokens[0])) tokens.shift();
  while (tokens.length && /^gpu$/i.test(tokens[tokens.length - 1])) tokens.pop();
  const core = tokens.join(' ').toLowerCase();
  if (!core) return null;
  const g = catalog.find((x) => x.name.toLowerCase() === core);
  if (!g) return null;
  if (!Number.isFinite(detectedVramGB) || detectedVramGB !== g.vramGB) return null;
  return g;
}
