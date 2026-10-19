#!/usr/bin/env bash
# Stage A: probe the hub weights of every model, as the baseline.
# Stage B: MLM continued pretraining at each learning rate, then probe the best checkpoint of each run.
# Finished steps are skipped and unfinished ones restart, so after a crash or a stop just run it again.
#
#   scripts/run_experiments.sh        # stages A then B
#   scripts/run_experiments.sh B      # stage B only
#
# Every result ends in runs/<bert>/<run>/probe/, with <run> = hub or lr<learning rate>.
set -uo pipefail
cd "$(dirname "$0")/.."

# ===== edit here =====
OUT_ROOT='runs'
IDS=(1 2 3 4 5 6 7 8 9)
declare -A LRS=(  # three learning rates per berts.yml id
  [1]='3e-5 1e-4 3e-4'
  [2]='3e-5 1e-4 3e-4'
  [3]='3e-5 1e-4 3e-4'
  [4]='3e-5 1e-4 3e-4'
  [5]='3e-5 1e-4 3e-4'
  [6]='1e-5 3e-5 1e-4'
  [7]='3e-6 1e-5 3e-5'
  [8]='3e-6 1e-5 3e-5'
  [9]='3e-6 1e-5 3e-5'
)
declare -A LOG_STEPS=(  # train records every N steps, about every 30 s from the measured step time
  [1]=500 [2]=500 [3]=200 [4]=200 [5]=200 [6]=100 [7]=200 [8]=100 [9]=50
)
declare -A PROBE_LEN=([1]=512 [2]=512)  # probe truncation; ids not listed use DEFAULT_PROBE_LEN
DEFAULT_PROBE_LEN=2048
LAUNCH=(accelerate launch --multi_gpu --num_processes 8)
MLM_ARGS=(
  -e 5 --batch-size 32 --grad-accum-steps 1 --max-seq-length 512 --mixed-precision bf16
  --weight-decay 0.01 --warmup-ratio 0.06 --eval-steps 500 --num-proc 32
)
PROBE_GPU=0
PROBE_ARGS=(--batch-size 256)
# =====================

STAGES="${*:-A B}"
export HF_DATASETS_DISABLE_PROGRESS_BARS=1 HF_HUB_DISABLE_PROGRESS_BARS=1 TQDM_DISABLE=1  # keep the log readable
set -a; . ./.env; set +a  # always the real data, even if the shell still holds .env.smoke
mkdir -p "$OUT_ROOT"
LOG="${OUT_ROOT}/experiments-$(date +%y%m%d%H%M%S).log"
exec > >(tee "$LOG") 2>&1
failed=()

set_aside() {  # keep an unfinished output folder for inspection, out of the way of the rerun
  if [ -e "$1" ]; then
    mv "$1" "$1.failed-$(date +%y%m%d%H%M%S)"
    echo "  moved unfinished $1 aside"
  fi
}

bert_info() {  # prints: name hf_id tokenizer_source
  python -c "from common import load_bert; b = load_bert($1); print(b.name, b.hf_id, b.tokenizer_source)"
}

probe() {  # probe <model> <tokenizer> <run_dir> <id>
  local model=$1 tok=$2 run_dir=$3 len=${PROBE_LEN[$4]:-$DEFAULT_PROBE_LEN}
  if [ -f "${run_dir}/probe/metrics.csv" ]; then
    echo '  probe done, skipped'
    return 0
  fi
  set_aside "${run_dir}/probe"
  CUDA_VISIBLE_DEVICES=$PROBE_GPU python probe.py "$model" --tokenizer "$tok" -o "$run_dir" -p probe \
    --max-seq-length "$len" "${PROBE_ARGS[@]}"
}

stage_a() {
  for id in "${IDS[@]}"; do
    read -r name hf_id tok < <(bert_info "$id") || { failed+=("A id ${id}: not in berts.yml"); continue; }
    echo "========== [A] ${name}: hub weights ($(date '+%F %T'))"
    probe "$hf_id" "$tok" "${OUT_ROOT}/${name}/hub" "$id" || failed+=("A ${name}: probe")
  done
}

stage_b() {
  for id in "${IDS[@]}"; do
    read -r name hf_id tok < <(bert_info "$id") || { failed+=("B id ${id}: not in berts.yml"); continue; }
    for lr in ${LRS[$id]}; do
      run="lr${lr}"
      run_dir="${OUT_ROOT}/${name}/${run}"
      echo "========== [B] ${name}: ${run} ($(date '+%F %T'))"
      if grep -qis 'best eval/loss' "${run_dir}/mlm/train.log"; then  # logged only after the last step
        echo '  training done, skipped'
      else
        set_aside "${run_dir}/mlm"
        "${LAUNCH[@]}" train_mlm.py -b "$id" -o "$OUT_ROOT" -r "$run" --learning-rate "$lr" \
          --logging-steps "${LOG_STEPS[$id]:-50}" "${MLM_ARGS[@]}" \
          || { failed+=("B ${name} ${run}: train"); continue; }
      fi
      probe "${run_dir}/mlm/best" "${run_dir}/mlm/best" "$run_dir" "$id" || failed+=("B ${name} ${run}: probe")
    done
  done
}

for stage in $STAGES; do
  case "$stage" in
    A) stage_a ;;
    B) stage_b ;;
    *) echo "Unknown stage: ${stage}" >&2 ;;
  esac
done

echo "========== Done ($(date '+%F %T')), log in ${LOG}"
if [ "${#failed[@]}" -gt 0 ]; then
  printf '  FAIL  %s\n' "${failed[@]}"
  exit 1
fi
echo '  all steps finished'
