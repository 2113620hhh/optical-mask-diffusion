#!/usr/bin/env bash
# Score-only continuation on the 30K Manhattan dataset.
# This wrapper keeps the generic 5-node x 8-DPU launcher and writes to a new
# output directory, preserving the previous high-quality base run.
#SBATCH --job-name=manhattan30k-score-only
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
export DATASET_DIR="${DATASET_DIR:-$SCRIPT_DIR/real_circuit_manhattan_expert_30k_diverse_correct_v1}"
export BASE_CHECKPOINT="${BASE_CHECKPOINT:-$SCRIPT_DIR/runs/manhattan30k_field_print_linear_asf10_40gpu_lr5e-6_v2/checkpoints/latest.pt}"
export OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/runs/manhattan30k_score_only_linear_asf10_lr2.5e-6_epoch200}"
export LR="${LR:-2.5e-6}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-200}"
export SAMPLES_PER_EPOCH="${SAMPLES_PER_EPOCH:-96000}"

# Preserve the existing expert-mask stabilizers.  Disable only the disputed
# binary printing/Dice objective and slightly favor the global field score.
export DENOISE_WEIGHT=0.025
export BCE_WEIGHT=0.10
export PHYSICAL_WEIGHT=1.0
export PHYSICAL_COSINE_WEIGHT=1.20
export PHYSICAL_CENTERED_COSINE_WEIGHT=0.0
export PHYSICAL_L1_WEIGHT=0.25
export PHYSICAL_HIGHPASS_COSINE_WEIGHT=0.30
export PHYSICAL_HIGHPASS_L1_WEIGHT=0.15
export PHYSICAL_GRADIENT_COSINE_WEIGHT=0.20
export COSINE_SCORE_LOSS_WEIGHT=0.0
export BINARY_PRINT_LOSS_WEIGHT=0.0
export BINARY_PRINT_DICE_WEIGHT=0.0
export BINARY_PRINT_FALSE_POSITIVE_WEIGHT=0.0
export BINARY_PRINT_FALSE_NEGATIVE_WEIGHT=0.0
export BINARY_PRINT_BACKGROUND_TAIL_WEIGHT=0.0
export BINARY_PRINT_PAIR_CONTINUITY_WEIGHT=0.0
export BINARY_PRINT_WINDOW_CONTINUITY_WEIGHT=0.0
export BINARY_PRINT_ENDPOINT_WEIGHT=0.0
export MASK_MEAN_WEIGHT=0.10
export BACKGROUND_WEIGHT=0.0 FOREGROUND_WEIGHT=0.0
export ROW_PROJECTION_WEIGHT=0.0 COL_PROJECTION_WEIGHT=0.0
export BOTTOM_SUPERVISION_WEIGHT=0.0 BOTTOM_FOREGROUND_WEIGHT=0.0
export BOTTOM_BACKGROUND_WEIGHT=0.0 BOTTOM_ENDPOINT_WEIGHT=0.0
export BOTTOM_CONTINUITY_WEIGHT=0.0 FULL_CONTINUITY_WEIGHT=0.0
export VISUAL_LOSS_WEIGHT=0.0

exec bash "$SCRIPT_DIR/resume_mfhq120k_score_focus_40gpu.slurm"
