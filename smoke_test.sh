#!/usr/bin/env bash
# Smoke test: a few MLM steps plus the linear probe on synthetic data, for one or more ids in berts.yml.
#
#   ./smoke_test.sh          # id 0 (tiny-random-bert)
#   ./smoke_test.sh 0 2 7    # several models, each run independently
#
# Writes to runs/<bert>/smoke-test-YYMMDDHHMMSS/ and never touches .env: the data paths come from .env.smoke.
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

ENV_SMOKE="${REPO_DIR}/.env.smoke"
OUT_ROOT='runs'
RUN_NAME="smoke-test-$(date +%y%m%d%H%M%S)"
IDS=("$@")
[ "${#IDS[@]}" -gt 0 ] || IDS=(0)

# MLM settings: small blocks so the eval split still yields several, and frequent eval/logging
MLM_ARGS=(-e 2 --max-seq-length 64 --batch-size 8 --eval-steps 5 --logging-steps 2)

case "${1:-}" in
  -h|--help) sed -n '2,7p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
esac
for id in "${IDS[@]}"; do
  [[ "$id" =~ ^[0-9]+$ ]] || { echo "Not a berts.yml id: ${id}" >&2; exit 2; }
done

# 1. data: regenerate when .env.smoke or a file it points to is missing
data_ok() {
  [ -f "$ENV_SMOKE" ] || return 1
  local f
  for f in $(grep -E '^(MLM|PROBE)_(TRAIN|EVAL)_FILE=' "$ENV_SMOKE" | cut -d= -f2-); do
    [ -f "$f" ] || return 1
  done
}
if ! data_ok; then
  echo '== Generating smoke data'
  python scripts/make_smoke_test.py --env-file "$ENV_SMOKE" || exit 1
fi
set -a
# shellcheck disable=SC1090
. "$ENV_SMOKE"
set +a
export HF_HUB_OFFLINE=1  # every model must already be in the cache; downloads happen below, explicitly

echo "== Output: ${OUT_ROOT}/<bert>/${RUN_NAME}"
echo "== Data  : ${MLM_TRAIN_FILE}"
echo

# 2. per model: download if missing, train, probe standalone, check outputs
passed=()
failed=()
for id in "${IDS[@]}"; do
  read -r name hf_id tok_id < <(python -c "
from common import load_bert
b = load_bert($id)
print(b.name, b.hf_id, b.tokenizer_id or '-')
") || { failed+=("id ${id}: not in berts.yml"); continue; }
  echo "========== [${id}] ${name} (${hf_id})"

  missing=0
  for repo in "$hf_id" "$tok_id"; do
    [ "$repo" = '-' ] && continue
    python -c "from huggingface_hub import snapshot_download as s; s('$repo', local_files_only=True)" 2>/dev/null || missing=1
  done
  if [ "$missing" -eq 1 ]; then
    echo "-- Downloading ${name}"
    HF_HUB_OFFLINE=0 scripts/download_berts.sh "$name" || { failed+=("${name}: download"); continue; }
  fi

  run_dir="${OUT_ROOT}/${name}/${RUN_NAME}"
  mkdir -p "$run_dir"  # train_mlm.py only refuses an existing mlm/ inside it

  echo "-- train_mlm.py"
  if ! python train_mlm.py -b "$id" -o "$OUT_ROOT" -r "$RUN_NAME" "${MLM_ARGS[@]}" 2>&1 | tee "${run_dir}/train_mlm.out"; then
    failed+=("${name}: train_mlm.py")
    continue
  fi

  echo "-- probe.py (standalone, on the best checkpoint)"
  if ! python probe.py "${run_dir}/mlm/best" -o "$run_dir" -p probe-standalone 2>&1 | tee "${run_dir}/probe.out"; then
    failed+=("${name}: probe.py")
    continue
  fi

  errors=()
  for f in mlm/args.json mlm/train.log mlm/history.jsonl mlm/history.png mlm/best/best.json mlm/best/config.json \
           probe/metrics.csv probe/summary.md probe/head.parquet; do
    [ -e "${run_dir}/${f}" ] || errors+=("missing ${run_dir}/${f}")
  done
  for f in metrics.csv summary.md head.parquet probe.log; do
    [ -e "${run_dir}/probe-standalone/${f}" ] || errors+=("missing ${run_dir}/probe-standalone/${f}")
  done
  for col in is_pos grade score flag; do
    grep -q "^| ${col} |" "${run_dir}/probe/summary.md" 2>/dev/null || errors+=("label ${col} not probed")
  done
  grep -q '`note`' "${run_dir}/probe/summary.md" 2>/dev/null || errors+=('string column note not listed as skipped')

  if [ "${#errors[@]}" -gt 0 ]; then
    printf '  !! %s\n' "${errors[@]}" >&2
    failed+=("${name}: ${#errors[@]} check(s) failed")
  else
    passed+=("$name")
  fi
  echo
done

# 3. summary
echo "========== Smoke test summary (${OUT_ROOT}/<bert>/${RUN_NAME})"
for n in "${passed[@]}"; do echo "  PASS  ${n}"; done
for n in "${failed[@]}"; do echo "  FAIL  ${n}"; done
[ "${#failed[@]}" -eq 0 ]
