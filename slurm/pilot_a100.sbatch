#!/bin/bash
#SBATCH --job-name=lv-pilot
#SBATCH --account=csci_ga_3033-134-2026fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
# Pilot: ViT-B/16, every matrix in every layer, coarse rho grid, 5k selection images.
# Resumable: resubmitting skips finished rows in the results file.
set -euo pipefail
source "$HOME/laser-vision/slurm/env.sh"
nvidia-smi
incontainer python scripts/sweep.py --model vit_b16 \
    --data "$WORK/prepared/imagenet_val/s224_c0.9_bicubic" --split select --limit 5000 \
    --rhos 0.5,0.2,0.1,0.05,0.01 --modes top \
    --out "$WORK/results/pilot_vit.jsonl"
