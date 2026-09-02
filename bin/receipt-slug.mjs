// 영수증 canonical 슬러그 — fitllm-v2 `api/_receipt-slug.js`와 동일 규칙(패리티 테스트로 고정).
// 형식: <model-dashed>-<quant-token>[-ctx<n>][-kv<b>]-on-<hw-dashed | mac-Ngb>
// 기본(생략) = ctx min(8192, maxContext) · kv 16. Mac "N-bit"은 "Nbit" 토큰.
const seg = (s) => String(s).toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '');

export function quantToken(quantLabel) {
  const m = String(quantLabel).match(/^(\d+)-bit$/i);
  return m ? `${m[1]}bit` : String(quantLabel).toLowerCase();
}

export function defaultCtxFor(maxContext) {
  return Number.isFinite(maxContext) ? Math.min(8192, maxContext) : 8192;
}

// v2 /r 파서의 표현 한계(계약 미러 — 정본: fitllm-v2 api/r.js). CLI는 이 밖의 입력도 "계산"은 하되,
// v2가 200을 못 주는 조합에는 영수증 URL을 발급하지 않는다(깨진 링크 안내 금지).
export const V2_RECEIPT_LIMITS = { macMinGB: 8, macMaxGB: 2048, maxCards: 8 };

export function receiptRepresentable({ isGpu, ramGB, totalCards = 1, catalogGpu = true }) {
  if (!isGpu) return Number.isFinite(ramGB) && ramGB >= V2_RECEIPT_LIMITS.macMinGB && ramGB <= V2_RECEIPT_LIMITS.macMaxGB;
  return catalogGpu && totalCards <= V2_RECEIPT_LIMITS.maxCards;
}

export function canonicalReceiptSlug({ modelName, quantLabel, isGpu, hwLabel, ramGB, ctx, kvBits, maxContext }) {
  const hw = isGpu ? seg(String(hwLabel).replace(/ \+ /g, ' plus ')) : `mac-${ramGB}gb`;
  const extras = [];
  if (Number.isFinite(ctx) && ctx !== defaultCtxFor(maxContext)) extras.push(`-ctx${ctx}`);
  if ([4, 8].includes(kvBits)) extras.push(`-kv${kvBits}`);
  return `${seg(modelName)}-${quantToken(quantLabel)}${extras.join('')}-on-${hw}`;
}
