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

export function canonicalReceiptSlug({ modelName, quantLabel, isGpu, hwLabel, ramGB, ctx, kvBits, maxContext }) {
  const hw = isGpu ? seg(String(hwLabel).replace(/ \+ /g, ' plus ')) : `mac-${ramGB}gb`;
  const extras = [];
  if (Number.isFinite(ctx) && ctx !== defaultCtxFor(maxContext)) extras.push(`-ctx${ctx}`);
  if ([4, 8].includes(kvBits)) extras.push(`-kv${kvBits}`);
  return `${seg(modelName)}-${quantToken(quantLabel)}${extras.join('')}-on-${hw}`;
}
