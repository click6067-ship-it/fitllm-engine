#!/usr/bin/env bash
set -uo pipefail

write_output() {
  local result="$1"
  local code="$2"
  local delimiter="FITLLM_RESULT_${RANDOM}_${RANDOM}_$$"
  while grep -Fxq "$delimiter" <<< "$result"; do
    delimiter="FITLLM_RESULT_${RANDOM}_${RANDOM}_$$"
  done
  {
    printf 'result<<%s\n' "$delimiter"
    printf '%s\n' "$result"
    printf '%s\n' "$delimiter"
    printf 'exit-code=%s\n' "$code"
  } >> "$GITHUB_OUTPUT"
}

if [[ -z "${INPUT_MODEL:-}" ]]; then
  message='model is required'
  printf '%s\n' "$message" >&2
  write_output "$message" 2
  exit 2
fi

if [[ -n "${INPUT_GPU:-}" && -n "${INPUT_MAC:-}" ]] || [[ -z "${INPUT_GPU:-}" && -z "${INPUT_MAC:-}" ]]; then
  message='set exactly one of gpu or mac'
  printf '%s\n' "$message" >&2
  write_output "$message" 2
  exit 2
fi

args=("$INPUT_MODEL")
if [[ -n "${INPUT_GPU:-}" ]]; then
  args+=(--gpu "$INPUT_GPU")
  [[ -n "${INPUT_COUNT:-}" ]] && args+=(--count "$INPUT_COUNT")
else
  args+=(--mac "$INPUT_MAC")
fi
[[ -n "${INPUT_QUANT:-}" ]] && args+=(--quant "$INPUT_QUANT")
[[ -n "${INPUT_CTX:-}" ]] && args+=(--ctx "$INPUT_CTX")
[[ -n "${INPUT_KV:-}" ]] && args+=(--kv "$INPUT_KV")

result="$(node "$GITHUB_ACTION_PATH/bin/fitllm.mjs" "${args[@]}" --json --why 2>&1)"
code=$?
write_output "$result" "$code"

if [[ "$code" -eq 2 ]]; then
  printf '%s\n' "$result" >&2
else
  printf '%s\n' "$result"
fi
exit "$code"
