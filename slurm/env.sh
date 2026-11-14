# Shared settings for all jobs. Edit REPO if you cloned elsewhere.
export ACCOUNT=csci_ga_3033-134-2026fa
export REPO="$HOME/laser-vision"
export WORK="$SCRATCH/laser-vision"                 # data, prepared arrays, results, caches
export OVERLAY="$SCRATCH/containers/overlay-25GB-500K.ext3"
export IMAGE=/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
export HF_HOME="$WORK/hf_cache"                     # pretrained weights (timm / open_clip)
mkdir -p "$WORK"/{logs,results,prepared,hf_cache}

# Run a command inside the container with the prepared overlay mounted read-only.
#   incontainer python scripts/sweep.py ...
incontainer() {
  singularity exec --nv \
    --overlay "$OVERLAY":ro "$IMAGE" \
    /bin/bash -c "export HF_HOME='$HF_HOME'; cd '$REPO'; /ext3/.venv/bin/$*"
}
