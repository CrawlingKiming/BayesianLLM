#!/usr/bin/env bash
HF_ROOT="/hpc/group/laberlabs/dc430/BayesianLLM/.cache/huggingface"
export HF_HOME="$HF_ROOT"
export TRANSFORMERS_CACHE="$HF_ROOT/transformers"
export HF_DATASETS_CACHE="$HF_ROOT/datasets"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
set -euo pipefail

# Root of the repo (BayesianLLM)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ENV_NAME=${ENV_NAME:-gpoe}
DATA_DIR="data/UF-P-2"
TRAIN_JSONL="$DATA_DIR/train.jsonl"
MODEL_NAME=${MODEL_NAME:-meta-llama/Llama-3.2-1B}
REF_MODEL_NAME=${REF_MODEL_NAME:-$MODEL_NAME}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/gpoe_llama3_ufp2}
ADAPTER1=${ADAPTER1:-expert_helpfulness}
ADAPTER2=${ADAPTER2:-expert_honesty}

echo "[1/3] Running UF-P-2 data builder...(Currently Skip)"
#conda run -n "$ENV_NAME" python data/make_uf_p2.py --out_dir "$DATA_DIR"
#-meta-llama/Llama-3.2-1B
if [[ ! -f "$TRAIN_JSONL" ]]; then
  echo "Dataset file $TRAIN_JSONL not found. Aborting." >&2
  exit 1
fi

echo "[2/3] Starting GPOE training with M=2 experts..."
conda run -n "$ENV_NAME" python examples/train_gpoe.py \
  --model_name "$MODEL_NAME" \
  --ref_model_name "$REF_MODEL_NAME" \
  --train_path "$TRAIN_JSONL" \
  --output_dir "$OUTPUT_DIR" \
  --num_experts 2 \
  --adapter_names "$ADAPTER1" "$ADAPTER2" \
  --per_device_train_batch_size 1 \
  --grad_accum_steps 4 \
  --mc_samples 4 \
  --beta 0.5 \
  --lambda_offdiag 0.5 \
  --learning_rate 5e-5 \
  --max_steps 200 \
  --save_steps 100 \
  --logging_steps 10 \
  --max_length 1024

echo "[3/3] Training complete. Outputs saved to $OUTPUT_DIR"
