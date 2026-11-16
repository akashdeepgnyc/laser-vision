"""CPU job: decode an ImageFolder dataset into uint8 arrays using a model's eval preprocessing.

Example (ImageNet-val, ViT-B/16 recipe):
  python scripts/prep_images.py --src $DATA/imagenet/val --model vit_b16 --out $DATA/prepared/imagenet_val
The output goes to <out>/<preprocess_key>, so models sharing a recipe share one copy.
"""
import argparse
from pathlib import Path

from laser import data, models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="ImageFolder root (one subfolder per class)")
    ap.add_argument("--model", required=True, choices=list(models.MODELS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="only the first N images (testing)")
    ap.add_argument("--per-class", type=int, default=None, help="k random images from every class (laptop pilots)")
    a = ap.parse_args()

    # Only the preprocessing recipe is needed; CLIP's text tower is skipped by passing one class.
    b = models.load(a.model, classnames=["x"], templates=[lambda c: c])
    dest = Path(a.out) / (b.preprocess.key + (f"_pc{a.per_class}" if a.per_class else ""))
    data.prepare_imagefolder(a.src, str(dest), b.preprocess, workers=a.workers, limit=a.limit,
                            per_class=a.per_class)
    print("PREP_OK", dest)


if __name__ == "__main__":  # required on macOS: DataLoader workers re-import this file
    main()
