# MXFP4·FP4로 출시된 모델의 가중치 크기 (2026-10-23)

**상태:** 로컬 가지 `fix/mxfp4-native-size`에만 있다. push·태그·npm·HF·사이트 반영은 하지 않았다(공개 저장소라 사용자 결정).
**검증:** `npm test` 410/410 · 벡터 49/49 · `census:check` 4/4 byte-identical · Codex gpt-6-astra high 검수 3회차 **SHIP**(새 맥락 3회: 1차 REVISE — PLE 이중 차감·공개 파일보다 작은 계산, 2차 REVISE — 파생 모델이 원본 크기를 물려받음·작은 U8 몫이 헤더 여유에 숨음, 모두 고치고 회귀 시험으로 고정).

## 결론
- gpt-oss-120b Q8_0 가중치 **115.6 → 59.0 GiB**(공개 파일 59.0). DGX Spark 1대 8K: **안 들어감(129.5/128) → 들어감(72.9/128)**.
- 같은 결함이 DeepSeek-V4-Flash · V4-Pro · V4.1-Flash-1017(FP4 전문가)에도 있었다. 다섯 행만 바뀌고 나머지 29개 모델 값은 비트 단위로 같다.
- FP8로 출시된 모델(GLM-5.3, Nex-N2.5-Pro)은 결함이 아니다. 그대로 뒀다.

## 원인
gpt-oss는 전문가 가중치(120b: 114.7B, 전체의 98%)를 MXFP4로 내놓는다. MXFP4는 4bit 값 + 32개마다 스케일 1바이트로 4.25 bit다.
공개 GGUF는 Q4_K_M부터 F16까지 어느 단계든 이 텐서를 그대로 두고 나머지 2%만 바꾼다. 그래서 단계와 상관없이 58.5–60.9 GiB다.
엔진은 모델 전체(116.8B)에 단계 평균 bit를 곱했다. 그래서 8bit는 115.6 GiB, 16bit는 217.6 GiB로 실제 파일의 2–3.6배가 됐다.

## 고친 것 (부류 단위)
1. **출시 FP4 텐서는 출시 크기 위로 세지 않는다.** 행에 `nativeFp4Params`(HF가 보고하는 FP4 텐서 논리 수)를 둔다. 4.25 bit보다 높은 단계에서는 이 몫을 4.25 bit로, 나머지를 그 단계 bit로 센다. 해당 행은 gpt-oss-20b·120b, DeepSeek-V4-Flash·V4-Pro·V4.1-Flash-1017 다섯 개다.
2. **실제 파일보다 작게 세지 않는다.** 1번 공식만 쓰면 gpt-oss 파일보다 0.1–3.7% 작았다. hidden 2880이 K-quant 블록 256으로 나눠지지 않아 llama.cpp가 일부 텐서를 Q8_0으로 두기 때문이다. 그래서 하한을 둔다.
   - 그 단계의 공개 GGUF가 알려져 있으면 가장 큰 파일 아래로 세지 않는다(`ggufFileBytes`, gpt-oss: unsloth·bartowski·ggml-org).
   - 모르면 출시 체크포인트 크기 아래로 세지 않는다(`fp4ReleaseBytes`, F16 단계는 +0.25%). DeepSeek-V4는 같은 이름 단계의 공개 파일이 없어(unsloth UD-* 혼합만 있음) 이쪽이다.
3. **HF 즉석 파싱도 같은 답.** 설정이 `mxfp4`(gpt-oss) 또는 `expert_dtype: fp4`(DeepSeek-V4)를 선언해야 FP4 수를 받는다. 샤드 바이트도 "FP4 수 × 4.25/8 + 나머지 dtype 바이트"와 헤더 여유 안에서 맞아야 한다. 같은 컨테이너를 1바이트 정수로 읽어도 맞는 애매한 증거는 받지 않는다. 텐서 목록(dtype별 수, 형상 config)이 카탈로그 행과 정확히 같을 때만 그 행의 공개 파일 크기를 물려받는다. 하나라도 다르면 자기 출시 바이트를 하한으로 쓴다. 확인이 안 되면 할인 없이 크게 센다.
4. **PLE와 겹치지 않게.** PLE 차감이 인증된 모델에는 FP4 할인을 적용하지 않는다. Codex 1차 검수의 합성 입력에서 가중치가 −1.0 GiB까지 내려갔다. 2차 검수가 찾은 두 구멍(어휘를 늘린 파생 모델이 원본 크기를 물려받음, 작은 U8 몫이 헤더 여유에 숨음)도 막고 회귀 시험으로 고정했다.
5. **전제 문구** `native-fp4-expert-weights`를 다섯 행 결과에 붙인다. README 전제 목록에 출처와 함께 넣었다.
6. **FP8 출시본은 바꾸지 않았다.** GGUF에는 FP8 타입이 없어서 Q8_0·BF16 GGUF가 FP8을 다시 쓴다. 공개 파일도 엔진 단계 공식과 0.2% 안에서 맞는다(GLM-5.3 Q8_0 746.3 GiB · BF16 1,404.4 GiB, 출시본 703.7). 출시 크기를 상한으로 두면 Q8_0 파일보다 5.7% 작게 세서, 거짓으로 "들어간다"고 답하게 된다.

## 전후 표 — 가중치 GiB (2.21.4 → 지금 · 공개 파일)
| 모델 | Q4_K_M | Q8_0 | FP16 | Mac 8bit | 공개 파일(가장 큰 것) |
|---|---|---|---|---|---|
| gpt-oss-120b | 66.55 → **58.53** | 115.59 → **59.03** | 217.56 → **60.88** | 108.78 → 60.77 | Q4_K_M 58.53 · Q8_0 59.03 · F16 60.88 · 원본 60.77 |
| gpt-oss-20b | 11.91 → **10.87** | 20.68 → **11.28** | 38.93 → **12.85** | 19.46 → 12.82 | Q4_K_M 10.87 · Q8_0 11.28 · F16 12.85 · 원본 12.82 |
| DeepSeek-V4-Flash | 165.75 → 148.66 | 287.88 → 148.66 | 541.86 → 154.09 | 270.92 → 148.66 | UD-Q4_K_XL 144.44 · UD-Q8_K_XL 150.75 · 원본 148.66 |
| DeepSeek-V4-Pro | 910.97 → 805.33 | 1582.21 → 805.33 | 2978.09 → 826.65 | 1489.00 → 805.33 | 원본 805.33 |
| DeepSeek-V4.1-Flash-1017 | 511.66 → 475.25 | 755.28 → 479.56 | 1421.62 → 659.44 | 716.51 → 475.25 | UD-Q8_K_XL 473.46(+DSpark·비전 별도 8.3) · 원본 475.25 |

gpt-oss Mac 8·16bit도 출시본(120b 60.77 · 20b 12.82)이 하한이다. Mac 4bit(4 < 4.25)는 다섯 행 모두 그대로다.

### 판정이 바뀐 대표 사례 (8K, KV F16)
| 경우 | 2.21.4 | 지금 |
|---|---|---|
| gpt-oss-120b · DGX Spark 1대 · Q4_K_M | 들어감 80.43/128 | 들어감 72.41 |
| gpt-oss-120b · DGX Spark 1대 · Q8_0 | **안 들어감 129.47** | **들어감 72.92** |
| gpt-oss-120b · DGX Spark 1대 · Q8_0 · 128K | 안 들어감 138.01 | 들어감 81.46 |
| gpt-oss-120b · RTX PRO 6000 96GB · Q8_0 · 32K | 안 들어감 133.74 | 들어감 70.40 |
| gpt-oss-20b · RTX 4060 Ti/5060 Ti 16GB · Q4_K_M | 빠듯 15.80/16 | 들어감 14.64 |
| gpt-oss-20b · 16GB · Q8_0 | 안 들어감 25.63 | 들어감 15.10 |
| gpt-oss-20b · 16GB · FP16 | 안 들어감 46.07 | 안 들어감 16.85 |
| DeepSeek-V4-Flash · DGX Spark 2대 · Q8_0 | 안 들어감 350.85/256 | 들어감 194.91 |

**바뀐 범위(전수):** 34개 모델 × (GPU 47종 × GGUF 5단계 + Mac 메모리 크기 13종 × 4·8·16bit) × 문맥(8K·32K·128K·모델 최대) = 34,250개 조합을 고치기 전후로 비교했다. 바뀐 4,932개는 전부 위 다섯 모델이다. 판정은 모두 "들어가는 쪽"으로만 바뀌었다. census 13,804행에서도 바뀐 행은 다섯 모델 1,745행뿐이다.

## 남은 차이·위험
- DeepSeek-V4의 FP16 단계는 FP8 몫이 16bit로 늘어나 출시본보다 크다(V4.1 659.4 vs 475.3 GiB; Engram 196.6B가 FP8). GGUF F16 파일 기준으로 맞춘 것이다. vLLM처럼 FP8 출시본을 그대로 돌리면 더 적게 든다. 이 차이는 런타임을 고르는 기능이 있어야 표현할 수 있다(GLM-5.3 등 FP8 출시본도 같다).
- 공개 파일이 없는 FP4 체크포인트(DeepSeek-V4, 형상이 다른 파인튜닝)는 출시본을 하한으로 세서 보수적이다. 대신 실제 4bit 파일보다 크게 나온다(V4-Flash Q4_K_M 148.66 vs UD-Q4_K_XL 144.44, +2.9%). unsloth UD-Q8_K_XL(150.75)처럼 일부 텐서를 출시본보다 높은 정밀도로 다시 편 혼합 빌드는 Q8_0 단계보다 크다.
- Mac은 모든 모델과 같이 공칭 bit + 가중치 12% 여유로 센다. 공개 MLX 파일은 대부분 이 안에 든다(gpt-oss-120b 4bit 58.05 vs 60.92 · 8bit 59.03 vs 68.06 · 16bit 60.77 vs 68.06). V4.1-Flash MLX 4bit(1017과 같은 형상의 전 체크포인트)만 514.9 vs 507.0 GiB로 1.5% 넘는다. 이번에 바꾼 부분이 아니라 따로 기록한다.
- 벡터 계약이 바뀐다(1.10.0 → 1.11.0, 46 → 49개). 기대값 2개를 바꿨다: gpt-oss-120b Spark 1대 Q6_K 32K tight → yes, V4-Flash Spark 4대 Q8_0 224K tight → yes. 대신 3개를 넣었다. 하나는 Spark 1대 Q8_0 8K가 yes인지 본다. 하나는 GB10을 일반 그래픽카드로 셈하는지 Laguna S 8K로 가려낸다. 하나는 KV 복제 경계를 V4-Flash Spark 2대 144K로 옮긴다. 다른 언어로 엔진을 옮긴 구현은 이 규칙을 따라야 통과한다.

## 출처 (HF 파일 목록 `?blobs=true`, 2026-10-23 조회 · 리비전 고정)
- 원본: openai/gpt-oss-120b@b5c939d 65,248,893,184 B · openai/gpt-oss-20b@6cee5e8 13,761,316,904 B · deepseek-ai/DeepSeek-V4-Flash@60d8d70 159,617,149,040 · DeepSeek-V4-Pro@b5968e9 864,721,029,744 · DeepSeek-V4.1-Flash-1017@1bae0e1 510,296,708,312 · zai-org/GLM-5.3@aca966e 755,632,050,320 · nex-agi/Nex-N2.5-Pro@0f389ab 407,130,522,144
- gpt-oss GGUF: unsloth/gpt-oss-120b-GGUF@ff1a82d · bartowski/openai_gpt-oss-120b-GGUF@8d64014 · ggml-org/gpt-oss-120b-GGUF@7200b74 · unsloth/gpt-oss-20b-GGUF@d449b42 · bartowski/openai_gpt-oss-20b-GGUF@e39ba3a · ggml-org/gpt-oss-20b-GGUF@15dcfc7 (단계별 바이트는 `engine.js` 행 주석과 `ggufFileBytes`)
- MLX: mlx-community/gpt-oss-120b-MXFP4-Q4@bce781b 62,330,057,589 · -MXFP4-Q8@81e5ac3 63,386,948,506 · -mxfp4-bf16@a0ed0ff 65,248,907,820 · lmstudio-community/gpt-oss-120b-MLX-8bit@1c03082 124,168,924,424(전문가를 8bit로 다시 편 빌드)
- DeepSeek GGUF: unsloth/DeepSeek-V4-Flash-GGUF@e3aa0d6 · unsloth/DeepSeek-V4.1-Flash-1017-GGUF@60ad365 · unsloth/DeepSeek-V4-Pro-0813-GGUF@6d05361
- FP8 비교: unsloth/GLM-5.3-GGUF@346b359 Q8_0 801,357,677,216 · BF16 1,507,988,027,936 · bartowski/Nex-N2.5-Pro-GGUF@f7c3c07 Q8_0 421,576,669,344
- llama.cpp@970b51c: `ggml/include/ggml.h`(GGML_TYPE_MXFP4 있음, FP8 타입 없음) · `src/llama-quant.cpp`(이미 양자화된 텐서는 `--allow-requantize` 없이 다시 양자화하지 않음)

## 릴리스 체크리스트 — 준비만, 실행하지 않음 (공개·사용자 결정)
1. 공개 저장소에 올리기: `git push -u origin fix/mxfp4-native-size` → PR → CI 녹색 → 머지.
2. 패치 버전 2.21.4 → 2.21.5. 바꿀 곳: `package.json`, `package-lock.json`(두 곳), `engine.js` `ENGINE_VERSION`, README Action 예시 `@v2.21.5`. 버전을 고정한 시험 4개도 고친다: `test/spark-x2-5`, `granite-4-2-30b`, `readme-claims`, `release-artifacts`의 `RELEASE_VERSION`. 끝나면 `npm test`.
3. 릴리스 날짜(UTC)로 `npm run census`를 다시 돌리고 `release-artifacts`의 `RELEASE_DATE`를 맞춘다. 지금 census 날짜는 2026-10-23으로 고정해 뒀다.
4. 변경 기록 한 줄(GitHub Release 본문): `2.21.5 — Shipped FP4 expert weights (gpt-oss MXFP4, DeepSeek-V4 FP4) count at their 4.25-bit storage, and gpt-oss tiers never below the largest public GGUF: gpt-oss-120b Q8_0 115.6 → 59.0 GiB, one DGX Spark now fits. Conformance vectors 1.11.0 (49).`
5. 머지 커밋에 경량 태그 `v2.21.5`를 push한다. `publish.yml`이 `npm-publish` 승인을 기다리면, 사용자 지시가 있을 때만 승인한다(`gh api … pending_deployments`). 그러면 npm `fitllm-engine`·`fitllm` 2.21.5가 나가고, `binaries.yml`이 실행 파일과 packaging 자동 커밋을 만든다.
6. 확인: 빈 캐시에서 `npx -y fitllm@2.21.5 gpt-oss-120b --gpu "DGX Spark" --quant Q8_0` → FITS 72.9/128.
7. fitllm-v2 들여오기(2.21.4 선례 f084bc3):
   - `git archive v2.21.5`로 내보내고 `scripts/releases/fitllm-engine-2.21.5.json`을 봉인한다(catalog_sha256 = sha256(JSON.stringify(MODELS))).
   - `node scripts/sync-engine.mjs --source-dir <내보낸 폴더> --release-contract <json> --output-dir <새 폴더>`를 돌리고, 번들 파일을 복사한다(`release-contract.json` 제외).
   - `docs/launch/hf-space-fitllm/engine.js`에 엔진을 미러한다.
   - census를 `docs/launch/hf-dataset-fitllm-census/`에 복사하고 `census-jsonl` → `census-staging-manifest` → `census-readme`를 돌린다.
   - content-dates: 전후 dist의 `user-surface-hashes.mjs`를 비교해 바뀐 클래스 날짜를 올리고 `computeClassDigest`로 다시 계산한다.
   - vitest → PR → 머지.
8. 사이트 배포(수동): 머지된 master에서 분리한 작업 트리에서 `vercel pull --yes --environment=production` → `TZ=UTC vercel build --prod` → `vercel deploy --prebuilt --prod`. 확인할 것: `/api/check?model=gpt-oss-120b&gpu=DGX%20Spark&quant=Q8_0` → yes, 계산기 화면, `/data` 표.
9. HF Space·Dataset 올리기(사용자 몫, 토큰 필요): fitllm-v2의 `docs/launch/hf-space-fitllm/engine.js`, `docs/launch/hf-dataset-fitllm-census/`.
