#!/usr/bin/env bash
# ocr-region.sh
# Send text to the running TTS API queue, mirroring what
# `uv run ocr-tts api send-text` ("speak") and `api send-region` ("region")
# do, but driving curl directly.
#
# speak and region both POST to /queue with {"text","voice","speed"}; they
# differ only in where text comes from. For "region" pass the text that was
# OCR'd from the screen selection.
#
# Usage:
#   scripts/ocr-region.sh speak "<text>" [--voice V] [--speed S] [--host H] [--port P]
#   scripts/ocr-region.sh region "<text>" [--voice V] [--speed S] [--host H] [--port P]
#
# Options:
#   --voice V   Piper voice name or alias (default: male)
#   --speed S   Speed multiplier (default: 1.0)
#   --host  H   Server host (default: 127.0.0.1)
#   --port  P   Server port (default: 8000)

set -euo pipefail

DEFAULT_VOICE="male"
DEFAULT_SPEED="1.0"
DEFAULT_HOST="127.0.0.1"
DEFAULT_PORT="8000"

# Friendly voice aliases -> resolved Piper voice names (mirrors text2speech).
declare -A VOICE_ALIASES=(
  [male]="en_US-hfc_male-medium"
  [female]="en_US-hfc_female-medium"
)

usage() {
  sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
  exit "${1:-0}"
}

require_curl() {
  if ! command -v curl >/dev/null 2>&1; then
    echo "error: 'curl' is required but was not found" >&2
    exit 1
  fi
}

resolve_voice() {
  local v="${1:-$DEFAULT_VOICE}"
  if [[ -n "${VOICE_ALIASES[$v]:-}" ]]; then
    printf '%s' "${VOICE_ALIASES[$v]}"
  else
    printf '%s' "$v"
  fi
}

# Escape a string for embedding inside a JSON double-quoted string.
escape_json() {
  local s="$1"
  s=${s//\\/\\\\}
  s=${s//\"/\\\"}
  s=${s//$'\n'/\\n}
  printf '%s' "$s"
}

main() {
  (($# < 1)) && usage 1

  local cmd="$1"
  shift

  case "$cmd" in
    -h|--help|help) usage 0 ;;
    speak|region) ;;
    *) echo "unknown command: $cmd" >&2; usage 1 ;;
  esac

  local voice="$DEFAULT_VOICE" speed="$DEFAULT_SPEED"
  local host="$DEFAULT_HOST" port="$DEFAULT_PORT"
  local text="" text_set=0

  while (($# > 0)); do
    case "$1" in
      --voice) voice="${2:?--voice requires a value}"; shift 2 ;;
      --speed) speed="${2:?--speed requires a value}"; shift 2 ;;
      --host)  host="${2:?--host requires a value}";   shift 2 ;;
      --port)  port="${2:?--port requires a value}";   shift 2 ;;
      -h|--help) usage 0 ;;
      --*) echo "unknown option: $1" >&2; usage 1 ;;
      *)
        if (( text_set )); then
          echo "unexpected extra argument: $1" >&2; usage 1
        fi
        text="$1"; text_set=1; shift ;;
    esac
  done

  if (( text_set == 0 )); then
    echo "error: $cmd requires a <text> argument" >&2
    usage 1
  fi

  require_curl

  local resolved escaped payload url
  resolved="$(resolve_voice "$voice")"
  escaped="$(escape_json "$text")"
  payload=$(printf '{"text": "%s", "voice": "%s", "speed": %s}' \
    "$escaped" "$resolved" "$speed")
  url="http://$host:$port/queue"

  printf 'POST %s\npayload: %s\n' "$url" "$payload"
  curl --silent --show-error --fail-with-body \
    --request POST "$url" \
    --header "Content-Type: application/json" \
    --data "$payload"
}

main "$@"
