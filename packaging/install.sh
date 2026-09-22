#!/bin/sh
# FitLLM installer — https://fitllm.run/install.sh
#
#   curl -fsSL https://fitllm.run/install.sh | sh
#
# Installs a single self-contained `fitllm` binary. No Node, no package manager.
#
# Design rules, because this script runs as `curl | sh` on other people's machines:
#   1. Fail closed. Unsupported platform, failed download, or a checksum mismatch
#      all exit non-zero and install nothing. A wrong binary on PATH is worse than
#      no binary on PATH.
#   2. Never write an executable into PATH before its sha256 has been verified
#      against the checksum file published with the release.
#   3. No sudo. Default target is a user-writable directory.
#
# Env overrides: FITLLM_VERSION (default: latest release), FITLLM_INSTALL_DIR (default: ~/.local/bin)
set -eu

REPO="click6067-ship-it/fitllm-engine"
INSTALL_DIR="${FITLLM_INSTALL_DIR:-$HOME/.local/bin}"

die() { printf 'fitllm: %s\n' "$*" >&2; exit 1; }
info() { printf '%s\n' "$*" >&2; }

need() { command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"; }
need uname
need mkdir
need mv
need chmod

if command -v curl >/dev/null 2>&1; then
  fetch() { curl -fsSL "$1" -o "$2"; }
  fetch_stdout() { curl -fsSL "$1"; }
elif command -v wget >/dev/null 2>&1; then
  fetch() { wget -qO "$2" "$1"; }
  fetch_stdout() { wget -qO- "$1"; }
else
  die "need curl or wget"
fi

# /releases/latest 는 /releases/tag/vX.Y.Z 로 리다이렉트한다. 최종 URL 에서 태그만 떼어낸다.
redirect_tag() {
  if command -v curl >/dev/null 2>&1; then
    curl -fsSLI -o /dev/null -w '%{url_effective}' "$1" 2>/dev/null \
      | sed -n 's|.*/releases/tag/v\{0,1\}\([^/]*\)$|\1|p'
  else
    wget -qS --spider --max-redirect=10 "$1" 2>&1 \
      | sed -n 's|.*Location:.*/releases/tag/v\{0,1\}\([^ /]*\).*|\1|p' | tail -n 1
  fi
}

# --- platform ---------------------------------------------------------------
os_raw="$(uname -s)"
arch_raw="$(uname -m)"

case "$os_raw" in
  Darwin) os="darwin" ;;
  Linux) os="linux" ;;
  *) die "unsupported operating system: $os_raw (supported: Darwin, Linux; on Windows use the .exe from the releases page or 'scoop install fitllm')" ;;
esac

case "$arch_raw" in
  arm64|aarch64) arch="arm64" ;;
  x86_64|amd64) arch="x64" ;;
  *) die "unsupported architecture: $arch_raw (supported: arm64, x86_64)" ;;
esac

# musl vs glibc. Getting this wrong produces a binary that dies with a loader
# error at first run, so detect rather than assume.
libc=""
if [ "$os" = "linux" ]; then
  if [ -f /etc/alpine-release ]; then
    libc="-musl"
  elif command -v ldd >/dev/null 2>&1 && ldd --version 2>&1 | grep -qi musl; then
    libc="-musl"
  fi
fi

# --- version ----------------------------------------------------------------
version="${FITLLM_VERSION:-}"
if [ -z "$version" ]; then
  # 1순위: /releases/latest 의 리다이렉트 목적지에서 태그를 읽는다.
  # api.github.com 은 비인증 호출에 IP 당 시간당 60회 제한이 있어, 공유 IP·회사망·CI 에서
  # 403 이 뜬다(2026-09-22 실측: GitHub Actions 러너에서 재현). 리다이렉트 경로는 그 제한을
  # 받지 않으므로 이쪽을 먼저 쓰고, API 는 폴백으로만 남긴다.
  version="$(redirect_tag "https://github.com/$REPO/releases/latest")"
fi
if [ -z "$version" ]; then
  version="$(fetch_stdout "https://api.github.com/repos/$REPO/releases/latest" 2>/dev/null \
    | sed -n 's/.*"tag_name"[[:space:]]*:[[:space:]]*"v\{0,1\}\([^"]*\)".*/\1/p' \
    | head -n 1)"
fi
[ -n "$version" ] || die "could not determine the latest release version; set FITLLM_VERSION=x.y.z"

asset="fitllm-v${version}-${os}-${arch}${libc}"
base="https://github.com/$REPO/releases/download/v${version}/${asset}"

# --- download + verify ------------------------------------------------------
tmp="$(mktemp -d 2>/dev/null || mktemp -d -t fitllm)"
cleanup() { rm -rf "$tmp"; }
trap cleanup EXIT INT TERM

info "fitllm ${version} — ${os}/${arch}${libc}"
fetch "$base" "$tmp/fitllm" || die "download failed: $base"
fetch "$base.sha256" "$tmp/fitllm.sha256" || die "checksum file not found: $base.sha256 (refusing to install an unverified binary)"

expected="$(cut -d' ' -f1 < "$tmp/fitllm.sha256")"
[ -n "$expected" ] || die "checksum file is empty or malformed"

if command -v sha256sum >/dev/null 2>&1; then
  actual="$(sha256sum "$tmp/fitllm" | cut -d' ' -f1)"
elif command -v shasum >/dev/null 2>&1; then
  actual="$(shasum -a 256 "$tmp/fitllm" | cut -d' ' -f1)"
else
  die "need sha256sum or shasum to verify the download"
fi

[ "$actual" = "$expected" ] || die "checksum mismatch — expected $expected, got $actual. Nothing was installed."

# --- install ----------------------------------------------------------------
mkdir -p "$INSTALL_DIR" || die "cannot create $INSTALL_DIR"
chmod 755 "$tmp/fitllm"
mv "$tmp/fitllm" "$INSTALL_DIR/fitllm" || die "cannot write to $INSTALL_DIR (set FITLLM_INSTALL_DIR to a writable path)"

info "installed: $INSTALL_DIR/fitllm"
case ":${PATH}:" in
  *":$INSTALL_DIR:"*) info "run: fitllm \"Gemma 4 31b\" --gpu \"RTX 4090\"" ;;
  *) info "note: $INSTALL_DIR is not on your PATH. Add it, or run it directly:"
     info "      $INSTALL_DIR/fitllm \"Gemma 4 31b\" --gpu \"RTX 4090\"" ;;
esac
