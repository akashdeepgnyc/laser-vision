#!/bin/bash
# ONE-TIME setup, run from a CPU JupyterLab terminal on Cloud Bursting (partition n2c48m24).
# Follows the course Connect Guide, section 5: Singularity image + writable overlay + uv.
set -euo pipefail
source "$(dirname "$0")/env.sh"

if [ ! -f "$OVERLAY" ]; then
  mkdir -p "$(dirname "$OVERLAY")"
  cp /share/apps/overlay-fs-ext3/overlay-25GB-500K.ext3.gz "$(dirname "$OVERLAY")/"
  gunzip "$OVERLAY.gz"
fi

singularity exec --overlay "$OVERLAY":rw "$IMAGE" /bin/bash -c "
set -euo pipefail
export UV_INSTALL_DIR=/ext3/.uv UV_NO_MODIFY_PATH=1 UV_PYTHON_INSTALL_DIR=/ext3/.uv/python
export UV_PROJECT_ENVIRONMENT=/ext3/.venv UV_CACHE_DIR=/ext3/.cache/uv UV_LINK_MODE=copy
[ -x /ext3/.uv/uv ] || { curl -fLsS https://astral.sh/uv/install.sh -o /ext3/install-uv.sh && sh /ext3/install-uv.sh; }
cd '$REPO'
/ext3/.uv/uv sync --frozen --python 3.12   # exact versions from uv.lock
/ext3/.venv/bin/python -c 'import torch, timm, open_clip; print(torch.__version__, timm.__version__)'
/ext3/.venv/bin/python -m pytest -q
export HF_HOME='$HF_HOME'
/ext3/.venv/bin/python scripts/fetch_weights.py   # download weights now, not on a GPU node
"
echo SETUP_OK
