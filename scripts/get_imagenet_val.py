"""Download ImageNet-1k validation (50,000 images) from Hugging Face into an ImageFolder layout.

Needs approved access to the gated dataset ILSVRC/imagenet-1k and a login:
  uv run hf auth login          (paste a READ token once; or set HF_TOKEN in the environment)
  uv run python scripts/get_imagenet_val.py --out ~/data/laser-vision/imagenet/val

The repo has shipped validation data either as parquet shards (image bytes + integer label) or as
a val_images tar (file names end in the class id, e.g. ..._n01751748.JPEG). Both are handled.
Class folders sort into the standard ImageNet class order (0000..0999, or sorted WordNet ids),
which is the label order the pretrained models use.
"""
import argparse
import shutil
import tarfile
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files

REPO = "ILSVRC/imagenet-1k"

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True, help="destination ImageFolder root")
ap.add_argument("--keep-raw", action="store_true", help="keep the downloaded shards (about 6.5 GB)")
a = ap.parse_args()
out = Path(a.out).expanduser()
raw = out.parent / "_raw_download"
out.mkdir(parents=True, exist_ok=True)

files = list_repo_files(REPO, repo_type="dataset")
parquets = sorted(f for f in files if f.endswith(".parquet") and "validation" in f)
tars = sorted(f for f in files if "val_images" in f and (f.endswith(".tar.gz") or f.endswith(".tar")))
print(f"{len(parquets)} validation parquet shards, {len(tars)} validation tars in {REPO}")

n = 0
if parquets:
    import pyarrow.parquet as pq
    for f in parquets:
        path = hf_hub_download(REPO, f, repo_type="dataset", local_dir=raw)
        for batch in pq.ParquetFile(path).iter_batches(batch_size=512, columns=["image", "label"]):
            for img, label in zip(batch.column("image").to_pylist(), batch.column("label").to_pylist()):
                d = out / f"{int(label):04d}"
                d.mkdir(exist_ok=True)
                (d / f"val_{n:05d}.JPEG").write_bytes(img["bytes"])
                n += 1
        print(f"{f}: {n} images so far", flush=True)
elif tars:
    for f in tars:
        path = hf_hub_download(REPO, f, repo_type="dataset", local_dir=raw)
        with tarfile.open(path) as t:
            for m in t:
                if not m.isfile() or not m.name.upper().endswith((".JPEG", ".JPG")):
                    continue
                wnid = Path(m.name).stem.split("_")[-1]  # e.g. ILSVRC2012_val_00000001_n01751748
                d = out / wnid
                d.mkdir(exist_ok=True)
                (d / Path(m.name).name).write_bytes(t.extractfile(m).read())
                n += 1
                if n % 5000 == 0:
                    print(f"{n} images", flush=True)
else:
    raise SystemExit("No validation files found; check that your access request was approved.")

if not a.keep_raw:
    shutil.rmtree(raw, ignore_errors=True)

classes = sorted(p.name for p in out.iterdir() if p.is_dir())
counts = [len(list((out / c).iterdir())) for c in classes]
print(f"{len(classes)} classes, {sum(counts)} images (min {min(counts)}, max {max(counts)} per class)")
ok = len(classes) == 1000 and sum(counts) == 50000
print("IMAGENET_VAL_OK" if ok else "WARNING: expected 1000 classes and 50000 images", out)
