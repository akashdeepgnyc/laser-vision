#!/bin/bash
#SBATCH --job-name=lv-smoke
#SBATCH --account=csci_ga_3033-134-2026fa
#SBATCH --partition=g2-standard-12
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=%x-%j.out
# Ten-minute check on a cheap L4: tests pass, CUDA works, a tiny real sweep writes results.
set -euo pipefail
source "$HOME/laser-vision/slurm/env.sh"
nvidia-smi
incontainer python -m pytest -q
incontainer python scripts/sweep.py --model vit_b16 \
    --data "$WORK/prepared/imagenet_val/s224_c0.9_bicubic" --split select --limit 1000 \
    --rhos 0.1 --modes top,bottom,random --layers 11 --mtypes mlp_in \
    --out "$WORK/results/smoke.jsonl"
