#!/bin/bash
#SBATCH --account=team-ai
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=14:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

set -x
ulimit -0

export PYTHONBUFFERED=1

source .venv/bin/activate

srun python -u preprocess_videos.py \
    --input_folder "$SCRATCH/gesture/data/wlaslvideos" \
    --output_folder "$SCRATCH/gesture/data/wlaslvideos_processed" \
    --dataset_json "data/WLASL_v0.3.json" \
    --include_pose \
    --no_face \
    --include_hands \
    --min_detection_confidence 0.5 \
    --min_tracking_confidence 0.5 \
    --coordinate_systems shoulder_centered \
    --num_workers 20