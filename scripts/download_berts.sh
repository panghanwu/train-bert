#!/usr/bin/env bash
# Download every model listed in berts.yml into the HF cache: $HF_HOME/hub, or ~/.cache/huggingface/hub when unset.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
YAML="${SCRIPT_DIR}/../berts.yml"
ENV_FILE="${ENV_FILE:-${SCRIPT_DIR}/../.env}"

# Pick up HF_TOKEN / HF_HOME from the env file without exporting unrelated secrets
if [ -f "$ENV_FILE" ]; then
  while IFS='=' read -r key value; do
    case "$key" in
      HF_TOKEN|HF_HOME)
        value="${value%$'\r'}"; value="${value#[\"\']}"; value="${value%[\"\']}"
        [ -n "${!key:-}" ] || export "$key=$value"
        ;;
    esac
  done < "$ENV_FILE"
fi

# Without HF_HOME the hf CLI falls back to its own default; only resolve it here for display
HF_CACHE="${HF_HOME:-${XDG_CACHE_HOME:-${HOME}/.cache}/huggingface}"

DRY_RUN=0
LIST_ONLY=0
ALL_FILES=0
# Skip TensorFlow / Flax / ONNX duplicates by default; PyTorch weights are enough
EXCLUDE=('*.msgpack' '*.h5' '*.ot' 'onnx/*' '*.onnx' '*.onnx_data')
# Repos with safetensors also drop their duplicated pytorch_model*.bin
BIN_EXCLUDE=('pytorch_model*.bin' 'pytorch_model*.bin.index.json')
# Files fetched for a tokenizer-only repo (tokenizer_id in the YAML)
TOKENIZER_INCLUDE=('config.json' 'tokenizer*' 'vocab*' 'merges.txt' 'special_tokens_map.json' '*.model')

usage() {
  cat <<EOF
Usage: ${0##*/} [options] [name ...]

Download the models declared in berts.yml. With no name given, download all.
An entry's tokenizer_id, when present, is fetched as tokenizer files only.

Options:
  -f, --file PATH   YAML source (default: ${YAML})
  -l, --list        List the models in the YAML and exit
  -n, --dry-run     Resolve and print what would be downloaded, without fetching
  -a, --all         Download every file, including TF/Flax/ONNX and duplicated .bin weights
  -h, --help        Show this help

HF_TOKEN / HF_HOME are read from \$ENV_FILE (default: the repo .env) unless already set.
Destination: \$HF_HOME/hub, or ~/.cache/huggingface/hub when unset  ->  ${HF_CACHE}/hub
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    -f|--file)    YAML="$2"; shift 2 ;;
    -l|--list)    LIST_ONLY=1; shift ;;
    -n|--dry-run) DRY_RUN=1; shift ;;
    -a|--all)     ALL_FILES=1; shift ;;
    -h|--help)    usage; exit 0 ;;
    -*)           echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    *)            break ;;
  esac
done

[ -f "$YAML" ] || { echo "YAML not found: ${YAML}" >&2; exit 1; }
command -v hf >/dev/null 2>&1 || {
  echo 'hf CLI not found. Install it with: pip install -U "huggingface_hub[cli]"' >&2
  exit 1
}

# Flatten the YAML list into "<name>\t<hf_id>\t<tokenizer_id>" lines.
# Each "- " line starts a new entry; keys may sit on the dash line or below it.
parse_yaml() {
  awk '
    function flush() { if (n != "" && h != "") print n "\t" h "\t" (t == "" ? "-" : t); n = h = t = "" }
    function val(s) { sub(/^[^:]*:[[:space:]]*/, "", s); gsub(/\r|["'"'"']|[[:space:]]+$/, "", s); return s }
    /^[[:space:]]*-[[:space:]]/ { flush(); sub(/-/, " ") }
    /^[[:space:]]*name:/         { n = val($0) }
    /^[[:space:]]*hf_id:/        { h = val($0) }
    /^[[:space:]]*tokenizer_id:/ { t = val($0) }
    END { flush() }
  ' "$1"
}

ENTRIES="$(parse_yaml "$YAML")"
[ -n "$ENTRIES" ] || { echo "No 'name'/'hf_id' pairs found in ${YAML}" >&2; exit 1; }

if [ "$LIST_ONLY" -eq 1 ]; then
  printf '%s\n' "$ENTRIES" | awk -F'\t' '{ printf "  %-32s %s%s\n", $1, $2, ($3 == "-" ? "" : "  (tokenizer: " $3 ")") }'
  exit 0
fi

# Keep only the requested names, preserving the order given on the command line
if [ $# -gt 0 ]; then
  SELECTED=''
  for want in "$@"; do
    line="$(printf '%s\n' "$ENTRIES" | awk -F'\t' -v w="$want" '$1 == w { print; exit }')"
    [ -n "$line" ] || { echo "Unknown model name: ${want} (try --list)" >&2; exit 1; }
    SELECTED="${SELECTED}${line}"$'\n'
  done
  ENTRIES="${SELECTED%$'\n'}"
fi

has_safetensors() {
  hf models info "$1" --expand siblings < /dev/null 2>/dev/null | grep -q '\.safetensors"'
}

total="$(printf '%s\n' "$ENTRIES" | wc -l)"
echo "HF_HOME     : ${HF_HOME:-(unset, hf default)}"
echo "Destination : ${HF_CACHE}/hub"
echo "HF_TOKEN    : $([ -n "${HF_TOKEN:-}" ] && echo set || echo unset)"
echo "Models      : ${total}"
echo

i=0
failed=()
while IFS=$'\t' read -r name hf_id tok_id; do
  [ -n "$hf_id" ] || continue
  i=$((i + 1))
  printf '[%d/%d] %s  (%s)\n' "$i" "$total" "$name" "$hf_id"

  args=()
  if [ "$ALL_FILES" -eq 0 ]; then
    for pat in "${EXCLUDE[@]}"; do args+=(--exclude "$pat"); done
    if has_safetensors "$hf_id"; then
      for pat in "${BIN_EXCLUDE[@]}"; do args+=(--exclude "$pat"); done
    fi
  fi
  cmd=(hf download "$hf_id" "${args[@]}")
  [ "$DRY_RUN" -eq 1 ] && cmd+=(--dry-run)

  if ! "${cmd[@]}" < /dev/null; then
    echo "  !! failed: ${hf_id}" >&2
    failed+=("${name} (${hf_id})")
  fi

  if [ "$tok_id" != '-' ] && [ "$tok_id" != "$hf_id" ]; then
    echo "  + tokenizer: ${tok_id}"
    args=()
    for pat in "${TOKENIZER_INCLUDE[@]}"; do args+=(--include "$pat"); done
    cmd=(hf download "$tok_id" "${args[@]}")
    [ "$DRY_RUN" -eq 1 ] && cmd+=(--dry-run)
    if ! "${cmd[@]}" < /dev/null; then
      echo "  !! failed: ${tok_id}" >&2
      failed+=("${name} tokenizer (${tok_id})")
    fi
  fi
  echo
done <<< "$ENTRIES"

if [ "${#failed[@]}" -gt 0 ]; then
  echo "Finished with ${#failed[@]} failure(s):" >&2
  printf '  - %s\n' "${failed[@]}" >&2
  exit 1
fi

echo "All ${total} model(s) are in ${HF_CACHE}/hub"
