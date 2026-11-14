"""Download pretrained weights into $HF_HOME (run on CPU, so GPU jobs don't sit idle downloading)."""
from laser import models

for name in ("vit_b16", "resnet50", "clip_b32"):
    b = models.load(name, classnames=["x"], templates=[lambda c: c])
    print(name, b.preprocess.key, len(b.targets), "targets")
print("WEIGHTS_OK")
