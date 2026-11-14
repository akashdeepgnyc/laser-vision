"""What do the strong vs weak singular directions MEAN? No image data needed.

  uv run python scripts/semantics.py --out results/semantics

Part A - class space. Each classifier is a matrix with one row (or column) per ImageNet class:
  resnet50 fc (1000 x 2048), vit_b16 head (1000 x 768), CLIP zero-shot text weights (512 x 1000).
  Its SVD gives directions over the 1000 classes. For each direction we list the classes at the
  two ends (singular vectors have arbitrary sign, so both ends matter). Strong directions should
  separate broad groups (animals vs objects, dogs vs rest); weak ones should look arbitrary.

Part B - naming CLIP directions with words. CLIP's visual.proj (768 x 512) maps image features
  into the shared image-text space. Its right singular vectors live in that space, so we can find
  the nearest text embeddings from a vocabulary (ImageNet classes + generic visual words).

Outputs: report.md (readable), class_directions.csv, clip_words.csv, class_plane_<model>.png
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open_clip
import pandas as pd
import torch
import torch.nn.functional as F

from laser import models

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="results/semantics")
ap.add_argument("--test", action="store_true", help="random-weight tiny models (CI)")
ap.add_argument("--k", type=int, default=6, help="classes / words shown per end")
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)

CLASSES = list(open_clip.IMAGENET_CLASSNAMES)
# Generic visual words, to see whether CLIP directions encode style / colour / scene, not only objects.
WORDS = ("red orange yellow green blue purple pink brown black white gray colorful dark bright "
         "wood metal glass plastic fabric fur feathers scales stone water snow grass sand sky "
         "indoors outdoors forest ocean beach mountain city street kitchen office field desert night "
         "close-up blurry drawing painting sketch cartoon text logo pattern stripes dots texture "
         "toy food vehicle building animal person face hand crowd tool machine plant flower").split()

# Coarse ImageNet groups by standard class index (used only to colour the class-plane plot).
def group(i: int) -> str:
    if 151 <= i <= 268:
        return "dogs"
    if i <= 397:
        return "other animals"
    return "objects & scenes"

GROUP_COLORS = {"dogs": "#2a78d6", "other animals": "#eb6834", "objects & scenes": "#1baf7a"}
GROUP_MARKERS = {"dogs": "o", "other animals": "^", "objects & scenes": "s"}


def class_matrices():
    """name -> (matrix with classes on the rows, class names)."""
    if a.test:
        names = ["cat", "dog", "car", "tree", "boat", "bird"]
        b = models.load("clip_test", classnames=names, templates=[lambda c: f"a photo of a {c}."])
        r = models.load("resnet18_test")
        return {"resnet18_test": (r.model.fc.weight[: len(names)], names),
                "clip_test": (b.model.text_weights.T, names)}
    out_ = {}
    for name, head in (("resnet50", lambda m: m.fc.weight), ("vit_b16", lambda m: m.head.weight)):
        out_[name] = (head(models.load(name).model), CLASSES)
    clip = models.load("clip_b32")
    out_["clip_b32"] = (clip.model.text_weights.T, CLASSES)
    return out_


rows, report = [], ["# Semantics of singular directions\n"]
report.append("## A. Class-space directions\n\nFor each classifier, U from W = U S Vᵀ gives directions over classes. "
              "Shown: the classes with the most positive and most negative scores (sign is arbitrary).\n")
for name, (W, names) in class_matrices().items():
    W = W.detach().float().cpu()
    W = W - W.mean(0, keepdim=True)  # remove the shared bias-like direction common to all classes
    U, S, _ = torch.linalg.svd(W, full_matrices=False)
    r = len(S)
    picks = list(range(min(5, r))) + sorted({r // 2, r - 2, r - 1} - set(range(5)))
    report.append(f"\n### {name}  (rank {r})\n\n| Direction | σ / σ₁ | One end | Other end |\n|---|---|---|---|")
    for i in picks:
        u = U[:, i].numpy()
        hi = [names[j] for j in np.argsort(-u)[: a.k]]
        lo = [names[j] for j in np.argsort(u)[: a.k]]
        rel = float(S[i] / S[0])
        rows.append(dict(model=name, direction=i + 1, sigma_rel=rel, end_pos="; ".join(hi), end_neg="; ".join(lo)))
        report.append(f"| {i + 1} | {rel:.3f} | {', '.join(hi)} | {', '.join(lo)} |")
    if len(names) == 1000:  # plot classes on the two strongest directions
        fig, ax = plt.subplots(figsize=(6, 5))
        g = np.array([group(i) for i in range(1000)])
        for lab, col in GROUP_COLORS.items():
            m = g == lab
            ax.scatter(U[m, 0], U[m, 1], s=10, color=col, marker=GROUP_MARKERS[lab], alpha=0.8, label=lab,
                       edgecolors="white", linewidths=0.3)
        ax.set_xlabel("Direction 1 score")
        ax.set_ylabel("Direction 2 score")
        ax.legend(frameon=False)
        ax.set_title(f"{name}: 1000 classes on the two strongest directions", loc="left", fontsize=10)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        fig.tight_layout()
        fig.savefig(out / f"class_plane_{name}.png", dpi=200)
        plt.close(fig)
pd.DataFrame(rows).to_csv(out / "class_directions.csv", index=False)

# Part B: name CLIP's visual.proj directions with words
arch, pre = ("ViT-B-32", None) if a.test else (models.MODELS["clip_b32"]["arch"], "openai")
clip, _, _ = open_clip.create_model_and_transforms(arch, pretrained=pre)
tok = open_clip.get_tokenizer(arch)
vocab = (["cat", "dog", "car"] if a.test else CLASSES) + WORDS
with torch.no_grad():
    T = F.normalize(clip.encode_text(tok([f"a photo of {w}." for w in vocab])).float(), dim=-1)  # (V, 512)
    P = clip.visual.proj.detach().float()  # (768, 512): image features -> shared space
    U, S, Vh = torch.linalg.svd(P, full_matrices=False)
r = len(S)
picks = list(range(min(8, r))) + sorted({r // 2, r - 2, r - 1} - set(range(8)))
wrows = []
report.append("\n## B. CLIP visual.proj directions, named by nearest words\n\n"
              "Each right singular vector of the 768×512 projection is a direction in CLIP's shared image-text "
              "space. Words with the highest and lowest cosine to it are listed.\n\n"
              "| Direction | σ / σ₁ | Toward | Away |\n|---|---|---|---|")
for i in picks:
    cos = (T @ Vh[i]).numpy()
    hi = [vocab[j] for j in np.argsort(-cos)[: a.k]]
    lo = [vocab[j] for j in np.argsort(cos)[: a.k]]
    rel = float(S[i] / S[0])
    wrows.append(dict(direction=i + 1, sigma_rel=rel, toward="; ".join(hi), away="; ".join(lo)))
    report.append(f"| {i + 1} | {rel:.3f} | {', '.join(hi)} | {', '.join(lo)} |")
pd.DataFrame(wrows).to_csv(out / "clip_words.csv", index=False)
(out / "report.md").write_text("\n".join(report) + "\n")
print("\n".join(report))
print("SEMANTICS_OK", out)
