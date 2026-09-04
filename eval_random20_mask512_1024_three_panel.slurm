#!/usr/bin/env bash
# Independent one-DPU evaluation; this job only reads the active checkpoint.
#SBATCH --job-name=eval-mask512-1024
#SBATCH --partition=hx1hdnormal01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=dcu:Hygon:1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
WORKDIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
DATASET_DIR="${DATASET_DIR:-$WORKDIR/real_circuit_manhattan_expert_30k_diverse_correct_v1}"
CHECKPOINT="${CHECKPOINT:-$WORKDIR/runs/manhattan30k_score_only_linear_asf10_lr2.5e-6_epoch200/checkpoints/latest.pt}"
OUT_DIR="${OUT_DIR:-$WORKDIR/runs/eval_random20_mask512_1024_$(date +%Y%m%d_%H%M%S)}"
PYTHON_BIN="${PYTHON_BIN:-$WORKDIR/../miniconda3/envs/3torch251_py310_dtk25042/bin/python}"
CONDA_ROOT="${CONDA_ROOT:-$WORKDIR/../miniconda3}"
CONDA_ENV="${CONDA_ENV:-3torch251_py310_dtk25042}"
cd "$WORKDIR"
module purge
module load compiler/dtk/25.04.2 mpi/ucx/1.18.0/dtk-25.04/mlnx mpi/openmpi/5.0.3/gcc-8.5.0/mlnx app/rccl/dtk-25.04/26.01.1 app/rccl/topos/default
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$CONDA_ROOT/envs/$CONDA_ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
[[ -f "$CHECKPOINT" ]] || { echo "Missing checkpoint: $CHECKPOINT" >&2; exit 1; }
[[ ! -e "$OUT_DIR" ]] || { echo "Refusing to overwrite output: $OUT_DIR" >&2; exit 1; }
exec "$PYTHON_BIN" eval_random20_mask512_1024_three_panel.py \
  --dataset-dir "$DATASET_DIR" --split val --checkpoint "$CHECKPOINT" --out-dir "$OUT_DIR" \
  --mask-threshold 0.5 --device cuda --amp --amp-dtype bf16
