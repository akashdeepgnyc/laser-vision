# LASER for Vision Models

Does layer-selective rank reduction (Sharma, Ash & Misra, ICLR 2024) improve pretrained
vision models, and where (attention vs MLP, which depth, which classes)?

## What's here

| Path | Purpose |
|---|---|
| `laser/edit.py` | SVD edit: keep top-k (`top`), plus controls `bottom` (only the discarded part) and `random` (random k-dim subspace). k = floor(rho * min(m, n)). |
| `laser/models.py` | Loads `vit_b16` (timm `vit_base_patch16_224.augreg2_in21k_ft_in1k`), `resnet50` (timm `resnet50.a1_in1k`), `clip_b32` (open_clip ViT-B-32 / openai, zero-shot, image encoder only). Lists editable matrices: `attn_q/k/v` (split from fused qkv), `attn_out`, `mlp_in`, `mlp_out`; ResNet `conv_a/b/c` reshaped to out x (in*k*k), `fc`. |
| `laser/data.py` | Decodes an ImageFolder once into `images.npy` (uint8) using the model's own eval preprocessing; writes a fixed class-stratified `select`/`report` split. |
| `laser/evaluate.py` | Accuracy, loss, ECE; paired bootstrap CI. |
| `scripts/sweep.py` | Baseline + every (matrix, rho, mode). One JSONL row per run, predictions saved per run. **Resumable.** |
| `slurm/` | Course Cloud Bursting setup and jobs (account `csci_ga_3033-134-2026fa`). |

## Running on NYU HPC Cloud Bursting

Everything goes through Open OnDemand: <https://ood.burst.hpc.nyu.edu/> (NYU VPN off campus). **Not Torch.**

1. **CPU JupyterLab session**: Interactive Apps → Jupyter Notebook, 0 GPUs, partition `n2c48m24`,
   account `csci_ga_3033-134-2026fa`, optional options `--cpus-per-task=4 --mem=16G`
   (add `--export=ALL` if you see "user environment retrieval failed").
2. In a JupyterLab terminal:
   ```bash
   git clone <your-github-url> ~/laser-vision
   bash ~/laser-vision/slurm/setup_env.sh      # overlay + uv + tests + weight download -> SETUP_OK
   ```
3. **Get ImageNet-val** into an ImageFolder layout (1000 class subfolders named by wnid) under
   `$SCRATCH/laser-vision/data/imagenet/val`. Ask course staff whether a shared copy exists first.
   The official `ILSVRC2012_img_val.tar` is flat; it must be sorted into class folders.
4. **Decode once (CPU job)**, about 7.5 GB per preprocessing recipe, three recipes:
   ```bash
   cd ~/laser-vision
   sbatch --export=ALL,SRC=$SCRATCH/laser-vision/data/imagenet/val slurm/prep_cpu.sbatch
   ```
5. **GPU smoke test on an L4** (about 10 min): `sbatch slurm/smoke_gpu.sbatch`
6. **Pilot on an A100** (ViT-B/16, all 72 matrices x 5 rhos, 5k images): `sbatch slurm/pilot_a100.sbatch`

Check jobs with `squeue --me` and `sacct -j <JOB_ID> --format=JobID,State,ExitCode,Elapsed`.
A job passed only when `sacct` shows `COMPLETED 0:0` and the log ends with `SWEEP_OK`.
Results land in `$SCRATCH/laser-vision/results/`.

## Cluster rules that shaped the design

- GPU instances with low utilization for 20 minutes are shut down: all decoding and weight
  downloads happen on CPU; GPU jobs hold the uint8 images on the GPU and only run forward passes.
- Release resources: `scancel <JOB_ID>`, end OOD sessions from *My Interactive Sessions*.
- Budget: 300 GPU-hours per student. Estimated need: 20-40.

## Local tests (CPU, random-weight tiny models)

```bash
uv sync --python 3.12 && uv run pytest -q
```

## Protocol

Select edits on `select` (half of ImageNet-val), report once on `report`, ImageNet-V2, -R, -Sketch.
Controls (`bottom`, `random`) must not reproduce a gain for it to count.
