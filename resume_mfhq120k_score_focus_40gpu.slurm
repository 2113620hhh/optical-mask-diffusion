#!/usr/bin/env bash
# Five 8-GPU nodes: one torchrun launcher per node, eight ranks per launcher.
# The account is inherited from the submitting user; the partition is set to this cluster's DCU partition.
# Submit from this directory with:
#   sbatch resume_mfhq120k_score_focus_40gpu.slurm
# Override the Python environment when needed, for example:
#   sbatch --export=ALL,PYTHON_BIN=/path/to/python resume_finetune_multifidelity_high_quality120k_from_epoch1_40gpu.slurm
#SBATCH --job-name=mfhq120k-score-focus
#SBATCH --partition=hx1hdnormal01
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=1
#SBATCH --gres=dcu:Hygon:8
#SBATCH --cpus-per-task=96
#SBATCH --exclusive
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
WORKDIR="$SCRIPT_DIR"
readonly EXPECTED_NODES=5
readonly GPUS_PER_NODE=8
readonly EXPECTED_WORLD_SIZE=$((EXPECTED_NODES * GPUS_PER_NODE))

DATASET_DIR="${DATASET_DIR:-$WORKDIR/real_circuit_pipe_hard_expert_120k_multifidelity_high_quality_v3}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-$WORKDIR/runs/mfhq120k_field_print_linear_asf10_40gpu_v1/checkpoints/latest.pt}"
OUT_DIR="${OUT_DIR:-$WORKDIR/runs/mfhq120k_field_print_linear_asf10_40gpu_v2_lr7e-7}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-$WORKDIR/训练扩散专家掩码模型_修正角谱版.py}"
PYTHON_BIN="${PYTHON_BIN:-$WORKDIR/../miniconda3/envs/3torch251_py310_dtk25042/bin/python}"
CONDA_ROOT="${CONDA_ROOT:-$WORKDIR/../miniconda3}"
CONDA_ENV="${CONDA_ENV:-3torch251_py310_dtk25042}"

# Keep the validated scratch-run batch size. With 40 ranks this is a global
# batch of 200; --samples-per-epoch remains global at 96,000 samples.
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-3}"
SAMPLES_PER_EPOCH="${SAMPLES_PER_EPOCH:-96000}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-139}"
LR="${LR:-7e-7}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DATASET_CACHE_SIZE="${DATASET_CACHE_SIZE:-8}"
LOG_EVERY="${LOG_EVERY:-20}"
SAVE_STEP_EVERY="${SAVE_STEP_EVERY:-0}"
VAL_EVERY="${VAL_EVERY:-2}"
VAL_MAX_BATCHES="${VAL_MAX_BATCHES:-512}"
STAGE_DATASET_LOCAL="${STAGE_DATASET_LOCAL:-1}"

cd "$WORKDIR"

module purge
module load \
  compiler/dtk/25.04.2 \
  mpi/ucx/1.18.0/dtk-25.04/mlnx \
  mpi/openmpi/5.0.3/gcc-8.5.0/mlnx \
  app/rccl/dtk-25.04/26.01.1 \
  app/rccl/topos/default
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$CONDA_ROOT/envs/$CONDA_ENV/lib:${LD_LIBRARY_PATH:-}"
ulimit -u 2000000 || true

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Submit this script with sbatch; do not run it directly."
  exit 1
fi
if [[ "${SLURM_NNODES:-0}" -ne "$EXPECTED_NODES" ]]; then
  echo "Expected $EXPECTED_NODES nodes ($EXPECTED_WORLD_SIZE GPUs), received ${SLURM_NNODES:-unset} nodes."
  exit 1
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing Python environment: $PYTHON_BIN"
  exit 1
fi
if [[ ! -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]]; then
  echo "Missing Conda activation script: $CONDA_ROOT/etc/profile.d/conda.sh"
  exit 1
fi
if [[ ! -f "$TRAIN_SCRIPT" ]]; then
  echo "Missing training entry point: $TRAIN_SCRIPT"
  exit 1
fi
if [[ ! -f "$DATASET_DIR/metadata.json" ]]; then
  echo "Missing complete high-quality dataset: $DATASET_DIR"
  exit 1
fi
if [[ ! -f "$BASE_CHECKPOINT" ]]; then
  echo "Missing resume checkpoint: $BASE_CHECKPOINT"
  exit 1
fi
if [[ -e "$OUT_DIR" ]]; then
  echo "Refusing to overwrite existing score-focus output: $OUT_DIR" >&2
  exit 1
fi

"$PYTHON_BIN" - "$DATASET_DIR/metadata.json" "$BASE_CHECKPOINT" <<'PY'
import json
import sys
import torch

metadata_path, checkpoint_path = sys.argv[1:]
with open(metadata_path, "r", encoding="utf-8") as handle:
    metadata = json.load(handle)

if not metadata.get("quality", {}).get("complete", False):
    raise SystemExit("Dataset metadata is not marked complete")
if metadata.get("mask_resolution") != 512 or metadata.get("target_resolution") != 192:
    raise SystemExit("Dataset resolution is not 512x512 mask / 192x192 target")
if metadata.get("forward_model") != "linear_asf_reference_with_angular_spectrum_proxy":
    raise SystemExit(f"Unexpected dataset forward model: {metadata.get('forward_model')!r}")

checkpoint = torch.load(checkpoint_path, map_location="cpu")
epoch = int(checkpoint.get("epoch", -1))
step = int(checkpoint.get("global_step", -1))
model_config = checkpoint.get("model_config", {})
if int(model_config.get("input_channels", 0)) != 8:
    raise SystemExit(f"Unexpected model input_channels: {model_config.get('input_channels')!r}")
checkpoint_metadata = checkpoint.get("dataset_metadata", {})
for key, expected in (("mask_resolution", 512), ("target_resolution", 192)):
    if checkpoint_metadata.get(key, expected) != expected:
        raise SystemExit(f"Checkpoint {key}={checkpoint_metadata.get(key)!r} does not match {expected}")
if "optimizer" not in checkpoint:
    raise SystemExit("Checkpoint does not contain optimizer state")
print(f"Dataset and resume checkpoint verified: epoch={epoch}, global_step={step}")
PY

available_kb="$(df -Pk "$WORKDIR" | awk 'NR==2 {print $4}')"
if (( available_kb < 10 * 1024 * 1024 )); then
  echo "At least 10 GiB free disk is required for atomic checkpoint saves."
  exit 1
fi

# Every rank needs the optimizer-inclusive checkpoint.  Loading it directly from
# the shared filesystem makes 40 processes contend for the same 7.9 GiB file.
# Stage one verified copy on each node's local NVMe before launching torchrun.
CHECKPOINT_BASENAME="$(basename "$BASE_CHECKPOINT")"
CHECKPOINT_BYTES="$(stat -c '%s' "$BASE_CHECKPOINT")"
DATASET_BYTES="$(du -sb "$DATASET_DIR" | awk '{print $1}')"
LOCAL_STAGE_ROOT="${LOCAL_STAGE_ROOT:-/tmp/${USER:-$(id -un)}/mfhq120k-${SLURM_JOB_ID}}"
export CHECKPOINT_BASENAME CHECKPOINT_BYTES DATASET_BYTES LOCAL_STAGE_ROOT STAGE_DATASET_LOCAL

mapfile -t allocated_nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
MASTER_NODE="${allocated_nodes[0]:-}"
if [[ -z "$MASTER_NODE" ]]; then
  echo "Unable to resolve a rendezvous host from: $SLURM_JOB_NODELIST"
  exit 1
fi
MASTER_ADDR="$(srun --nodes=1 --ntasks=1 --nodelist="$MASTER_NODE" bash -c "ip -o -4 addr show dev eno1 2>/dev/null | cut -d' ' -f7 | cut -d/ -f1 | head -n 1")"
if [[ -z "$MASTER_ADDR" ]]; then
  echo "Unable to determine a rendezvous address on $MASTER_NODE:eno1"
  exit 1
fi
MASTER_PORT="${MASTER_PORT:-$((20000 + SLURM_JOB_ID % 20000))}"

export WORKDIR DATASET_DIR BASE_CHECKPOINT OUT_DIR TRAIN_SCRIPT PYTHON_BIN CONDA_ROOT CONDA_ENV
export GPUS_PER_NODE MASTER_ADDR MASTER_PORT
export BATCH_SIZE_PER_GPU SAMPLES_PER_EPOCH TOTAL_EPOCHS NUM_WORKERS DATASET_CACHE_SIZE
export LOG_EVERY SAVE_STEP_EVERY VAL_EVERY VAL_MAX_BATCHES
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export DDP_TIMEOUT_SECONDS="${DDP_TIMEOUT_SECONDS:-3600}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export RCCL_DEBUG="${RCCL_DEBUG:-WARN}"
export UCX_NET_DEVICES="${UCX_NET_DEVICES:-mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1}"
export UCX_TLS="${UCX_TLS:-shm,self,rc}"
export HSA_FORCE_FINE_GRAIN_PCIE="${HSA_FORCE_FINE_GRAIN_PCIE:-1}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-eno1,eno2}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eno1,eno2}"
export RCCL_SOCKET_IFNAME="${RCCL_SOCKET_IFNAME:-eno1,eno2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

echo "Starting job $SLURM_JOB_ID on ${SLURM_NNODES} nodes x ${GPUS_PER_NODE} GPUs = $EXPECTED_WORLD_SIZE ranks"
echo "Master rendezvous: ${MASTER_ADDR}:${MASTER_PORT}"
echo "Dataset: $DATASET_DIR"
echo "Resume checkpoint: $BASE_CHECKPOINT"
echo "Output: $OUT_DIR"
echo "Per-GPU batch: $BATCH_SIZE_PER_GPU; global batch: $((BATCH_SIZE_PER_GPU * EXPECTED_WORLD_SIZE))"
echo "Local checkpoint cache: $LOCAL_STAGE_ROOT/node-<node_rank>/$CHECKPOINT_BASENAME"
echo "Stage dataset locally: $STAGE_DATASET_LOCAL"

srun --ntasks="$SLURM_NNODES" --ntasks-per-node=1 --kill-on-bad-exit=1 --label \
  bash -c '
    set -euo pipefail
    local_dir="$LOCAL_STAGE_ROOT/node-${SLURM_NODEID}"
    local_checkpoint="$local_dir/$CHECKPOINT_BASENAME"
    partial_checkpoint="$local_checkpoint.partial"
    local_dataset="$local_dir/dataset"
    partial_dataset="$local_dir/dataset.partial"
    required_bytes=$((CHECKPOINT_BYTES + 2 * 1024 * 1024 * 1024))
    if [[ "$STAGE_DATASET_LOCAL" == "1" ]]; then
      required_bytes=$((required_bytes + DATASET_BYTES))
    fi
    required_kb=$(((required_bytes + 1023) / 1024))
    mkdir -p "$local_dir"
    available_kb="$(df -Pk "$local_dir" | awk "NR==2 {print \$4}")"
    if (( available_kb < required_kb )); then
      echo "Insufficient local disk on $(hostname): need ${required_kb} KiB, have ${available_kb} KiB" >&2
      exit 1
    fi
    if [[ ! -f "$local_checkpoint" || "$(stat -c "%s" "$local_checkpoint")" -ne "$CHECKPOINT_BYTES" ]]; then
      rm -f "$partial_checkpoint"
      cp -- "$BASE_CHECKPOINT" "$partial_checkpoint"
      [[ "$(stat -c "%s" "$partial_checkpoint")" -eq "$CHECKPOINT_BYTES" ]]
      mv -f "$partial_checkpoint" "$local_checkpoint"
    fi
    echo "node=$(hostname) staged checkpoint=$local_checkpoint bytes=$CHECKPOINT_BYTES"
    if [[ "$STAGE_DATASET_LOCAL" == "1" ]]; then
      rm -rf "$partial_dataset"
      mkdir -p "$partial_dataset"
      cp -a -- "$DATASET_DIR/." "$partial_dataset/"
      [[ -f "$partial_dataset/metadata.json" ]]
      mv "$partial_dataset" "$local_dataset"
      echo "node=$(hostname) staged dataset=$local_dataset bytes=$DATASET_BYTES"
    fi
  '

# Slurm assigns the eight DCUs local to each node.  Do not set CUDA_VISIBLE_DEVICES:
# torchrun must see all local devices in every one of the five launcher tasks.
srun --ntasks="$SLURM_NNODES" --ntasks-per-node=1 --kill-on-bad-exit=1 --label \
  bash -c '
    set -euo pipefail
    cd "$WORKDIR"
    module purge
    module load \
      compiler/dtk/25.04.2 \
      mpi/ucx/1.18.0/dtk-25.04/mlnx \
      mpi/openmpi/5.0.3/gcc-8.5.0/mlnx \
      app/rccl/dtk-25.04/26.01.1 \
      app/rccl/topos/default
    source "$CONDA_ROOT/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
    export PYTHONNOUSERSITE=1
    export LD_LIBRARY_PATH="$CONDA_ROOT/envs/$CONDA_ENV/lib:${LD_LIBRARY_PATH:-}"
    ulimit -u 2000000 || true
    NODE_LOCAL_CHECKPOINT="$LOCAL_STAGE_ROOT/node-${SLURM_NODEID}/$CHECKPOINT_BASENAME"
    NODE_DATASET_DIR="$DATASET_DIR"
    if [[ "$STAGE_DATASET_LOCAL" == "1" ]]; then
      NODE_DATASET_DIR="$LOCAL_STAGE_ROOT/node-${SLURM_NODEID}/dataset"
    fi
    if [[ ! -f "$NODE_LOCAL_CHECKPOINT" ]]; then
      echo "Missing staged checkpoint on $(hostname): $NODE_LOCAL_CHECKPOINT" >&2
      exit 1
    fi
    if [[ ! -f "$NODE_DATASET_DIR/metadata.json" ]]; then
      echo "Missing staged dataset on $(hostname): $NODE_DATASET_DIR" >&2
      exit 1
    fi
    export REQUIRED_DEVICES="$GPUS_PER_NODE"
    "$PYTHON_BIN" -c "import os, sys, torch; count = torch.cuda.device_count(); print(\"node={} node_rank={} torch={} cuda_available={} cuda_device_count={}\".format(os.uname().nodename, os.environ.get(\"SLURM_PROCID\"), torch.__version__, torch.cuda.is_available(), count), flush=True); sys.exit(0 if torch.cuda.is_available() and count >= int(os.environ[\"REQUIRED_DEVICES\"]) else 1)"
    exec "$PYTHON_BIN" -m torch.distributed.run \
      --nnodes="$SLURM_NNODES" \
      --nproc_per_node="$GPUS_PER_NODE" \
      --node_rank="${SLURM_NODEID:-$SLURM_PROCID}" \
      --master_addr="$MASTER_ADDR" \
      --master_port="$MASTER_PORT" \
      "$TRAIN_SCRIPT" train \
      --dataset-dir "$NODE_DATASET_DIR" \
      --train-split train \
      --val-split val \
      --out-dir "$OUT_DIR" \
      --resume-checkpoint "$NODE_LOCAL_CHECKPOINT" \
      --resume-reset-optimizer \
      --resume-reset-best-metric \
      --epochs "$TOTAL_EPOCHS" \
      --batch-size "$BATCH_SIZE_PER_GPU" \
      --num-workers "$NUM_WORKERS" \
      --cache-size "$DATASET_CACHE_SIZE" \
      --simple-replay-ratio 1.0 \
      --samples-per-epoch "$SAMPLES_PER_EPOCH" \
      --mask-resolution 512 \
      --target-resolution 192 \
      --override-inter-num 0 \
      --forward-model asf_linear_convolution_fft \
      --fft-inter-num 10 \
      --val-fft-inter-num 10 \
      --train-stage direct \
      --prediction-type x0 \
      --val-mode direct \
      --base-channels 128 \
      --channel-mults 1,2,4,8 \
      --emb-dim 256 \
      --use-fourier \
      --diffusion-steps 1000 \
      --beta-start 0.0001 \
      --beta-end 0.02 \
      --amp \
      --amp-dtype bf16 \
      --allow-tf32 \
      --lr "$LR" \
      --weight-decay 1e-5 \
      --lr-scheduler score_plateau \
      --warmup-steps 20 \
      --plateau-factor 0.5 \
      --plateau-patience 2 \
      --plateau-threshold 0.001 \
      --plateau-cooldown 1 \
      --min-lr 3e-7 \
      --denoise-weight "${DENOISE_WEIGHT:-0.025}" \
      --bce-weight "${BCE_WEIGHT:-0.10}" \
      --physical-weight "${PHYSICAL_WEIGHT:-1.0}" \
      --physical-cosine-weight "${PHYSICAL_COSINE_WEIGHT:-1.0}" \
      --physical-centered-cosine-weight "${PHYSICAL_CENTERED_COSINE_WEIGHT:-0.0}" \
      --physical-l1-weight "${PHYSICAL_L1_WEIGHT:-0.25}" \
      --physical-highpass-cosine-weight "${PHYSICAL_HIGHPASS_COSINE_WEIGHT:-0.30}" \
      --physical-highpass-l1-weight "${PHYSICAL_HIGHPASS_L1_WEIGHT:-0.15}" \
      --physical-gradient-cosine-weight "${PHYSICAL_GRADIENT_COSINE_WEIGHT:-0.20}" \
      --cosine-score-loss-weight "${COSINE_SCORE_LOSS_WEIGHT:-0.0}" \
      --background-weight "${BACKGROUND_WEIGHT:-0.0}" \
      --foreground-weight "${FOREGROUND_WEIGHT:-0.0}" \
      --row-projection-weight "${ROW_PROJECTION_WEIGHT:-0.0}" \
      --col-projection-weight "${COL_PROJECTION_WEIGHT:-0.0}" \
      --bottom-supervision-weight "${BOTTOM_SUPERVISION_WEIGHT:-0.0}" \
      --bottom-foreground-weight "${BOTTOM_FOREGROUND_WEIGHT:-0.0}" \
      --bottom-background-weight "${BOTTOM_BACKGROUND_WEIGHT:-0.0}" \
      --bottom-endpoint-weight "${BOTTOM_ENDPOINT_WEIGHT:-0.0}" \
      --bottom-continuity-weight "${BOTTOM_CONTINUITY_WEIGHT:-0.0}" \
      --full-continuity-weight "${FULL_CONTINUITY_WEIGHT:-0.0}" \
      --visual-loss-weight "${VISUAL_LOSS_WEIGHT:-0.0}" \
      --binary-print-loss-weight "${BINARY_PRINT_LOSS_WEIGHT:-0.25}" \
      --binary-print-threshold "${BINARY_PRINT_THRESHOLD:-0.57}" \
      --binary-print-temperature "${BINARY_PRINT_TEMPERATURE:-0.04}" \
      --binary-print-dice-weight "${BINARY_PRINT_DICE_WEIGHT:-1.0}" \
      --binary-print-false-positive-weight "${BINARY_PRINT_FALSE_POSITIVE_WEIGHT:-0.50}" \
      --binary-print-false-negative-weight "${BINARY_PRINT_FALSE_NEGATIVE_WEIGHT:-0.50}" \
      --binary-print-background-tail-weight "${BINARY_PRINT_BACKGROUND_TAIL_WEIGHT:-0.10}" \
      --binary-print-pair-continuity-weight "${BINARY_PRINT_PAIR_CONTINUITY_WEIGHT:-0.0}" \
      --binary-print-window-continuity-weight "${BINARY_PRINT_WINDOW_CONTINUITY_WEIGHT:-0.05}" \
      --binary-print-endpoint-weight "${BINARY_PRINT_ENDPOINT_WEIGHT:-0.0}" \
      --mask-mean-weight "${MASK_MEAN_WEIGHT:-0.10}" \
      --fft-cache-size 1 \
      --grad-clip 1.0 \
      --best-metric cosine_score \
      --log-every "$LOG_EVERY" \
      --save-step-every "$SAVE_STEP_EVERY" \
      --val-every "$VAL_EVERY" \
      --val-max-batches "$VAL_MAX_BATCHES"
  '
