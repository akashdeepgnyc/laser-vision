#!/bin/bash
#SBATCH --job-name=lv-prep
#SBATCH --account=csci_ga_3033-134-2026fa
#SBATCH --partition=n2c48m24
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out
# CPU only: decode ImageNet-val once per preprocessing recipe.
# Set SRC to an ImageFolder copy of ImageNet-val (1000 class subfolders) before submitting:
#   sbatch --export=ALL,SRC=$SCRATCH/laser-vision/data/imagenet/val slurm/prep_cpu.sbatch
set -euo pipefail
source "$HOME/laser-vision/slurm/env.sh"
: "${SRC:?set SRC to the ImageNet-val ImageFolder root}"
for M in vit_b16 resnet50 clip_b32; do
  incontainer python scripts/prep_images.py --src "$SRC" --model $M \
      --out "$WORK/prepared/imagenet_val" --workers 22
done
ls -lh "$WORK/prepared/imagenet_val"/*
