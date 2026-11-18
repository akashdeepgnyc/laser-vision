"""Model loading and the list of editable matrices for each model family.

Matrix types (mtype) follow LASER's tau where possible:
  ViT / CLIP : attn_q, attn_k, attn_v (rows of the fused qkv matrix), attn_out, mlp_in, mlp_out
  ResNet     : conv_a (1x1 reduce), conv_b (3x3), conv_c (1x1 expand), fc
Conv kernels (out, in, kh, kw) are edited as matrices of shape out x (in*kh*kw).
"""
from __future__ import annotations

from dataclasses import dataclass, field


import torch
import torch.nn as nn
import torch.nn.functional as F

def pick_device() -> str:
    """NVIDIA GPU, else Apple-Silicon GPU (MPS), else CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# name -> how to build it. "*_test" entries use random weights (no download) for CPU tests.
MODELS = {
    "vit_b16": dict(family="timm_vit", timm_name="vit_base_patch16_224.augreg2_in21k_ft_in1k"),
    "resnet50": dict(family="timm_resnet", timm_name="resnet50.a1_in1k"),
    "clip_b32": dict(family="openclip_vit", arch="ViT-B-32-quickgelu", pretrained="openai"),  # OpenAI weights use QuickGELU
    "vit_tiny_test": dict(family="timm_vit", timm_name="vit_tiny_patch16_224", pretrained=False),
    "resnet18_test": dict(family="timm_resnet", timm_name="resnet18", pretrained=False),
    "clip_test": dict(family="openclip_vit", arch="ViT-B-32", pretrained=None),
}


@dataclass
class Preprocess:
    """Eval transform: resize shorter side to floor(size / crop_pct), center-crop `size`."""
    size: int
    crop_pct: float
    interpolation: str
    mean: tuple
    std: tuple

    @property
    def key(self) -> str:
        return f"s{self.size}_c{self.crop_pct:g}_{self.interpolation}"


@dataclass
class Target:
    name: str          # e.g. "blocks.11.mlp_in"
    layer: int         # depth index (0 = first block)
    mtype: str
    module: nn.Module
    param_name: str = "weight"
    rows: slice | None = None   # sub-block of a fused matrix (qkv)

    def parameter(self) -> nn.Parameter:
        return getattr(self.module, self.param_name)

    def read(self, data: torch.Tensor) -> torch.Tensor:
        W = data[self.rows] if self.rows is not None else data
        return W.reshape(W.shape[0], -1)

    def write(self, data: torch.Tensor, W: torch.Tensor) -> None:
        if self.rows is not None:
            data[self.rows] = W.to(data.dtype)
        else:
            data.copy_(W.reshape(data.shape).to(data.dtype))


@dataclass
class Bundle:
    name: str
    model: nn.Module      # maps normalized images -> class logits
    preprocess: Preprocess
    targets: list[Target] = field(default_factory=list)


class ClipZeroShot(nn.Module):
    """CLIP image encoder + fixed zero-shot text classifier. Only `visual` is edited."""

    def __init__(self, visual: nn.Module, text_weights: torch.Tensor, logit_scale: float):
        super().__init__()
        self.visual = visual
        self.register_buffer("text_weights", text_weights)  # (dim, n_classes), unit columns
        self.logit_scale = logit_scale

    def forward(self, x):
        f = F.normalize(self.visual(x), dim=-1)
        return self.logit_scale * f @ self.text_weights


def _qkv_targets(prefix, i, module, param_name, d):
    return [Target(f"{prefix}.{i}.attn_{n}", i, f"attn_{n}", module, param_name, slice(j * d, (j + 1) * d))
            for j, n in enumerate("qkv")]


def _vit_targets(model) -> list[Target]:
    t = []
    for i, b in enumerate(model.blocks):
        d = b.attn.proj.weight.shape[0]
        t += _qkv_targets("blocks", i, b.attn.qkv, "weight", d)
        t += [Target(f"blocks.{i}.attn_out", i, "attn_out", b.attn.proj),
              Target(f"blocks.{i}.mlp_in", i, "mlp_in", b.mlp.fc1),
              Target(f"blocks.{i}.mlp_out", i, "mlp_out", b.mlp.fc2)]
    return t


def _clip_targets(visual) -> list[Target]:
    t = []
    for i, b in enumerate(visual.transformer.resblocks):
        d = b.attn.out_proj.weight.shape[0]
        t += _qkv_targets("resblocks", i, b.attn, "in_proj_weight", d)
        t += [Target(f"resblocks.{i}.attn_out", i, "attn_out", b.attn.out_proj),
              Target(f"resblocks.{i}.mlp_in", i, "mlp_in", b.mlp.c_fc),
              Target(f"resblocks.{i}.mlp_out", i, "mlp_out", b.mlp.c_proj)]
    return t


def _resnet_targets(model) -> list[Target]:
    t, i = [], 0
    for stage in ("layer1", "layer2", "layer3", "layer4"):
        for j, block in enumerate(getattr(model, stage)):
            convs = [("conv_a", block.conv1), ("conv_b", block.conv2)]
            if hasattr(block, "conv3"):  # bottleneck (ResNet-50); basic blocks have two convs
                convs.append(("conv_c", block.conv3))
            t += [Target(f"{stage}.{j}.{n}", i, n, m) for n, m in convs]
            i += 1
    t.append(Target("fc", i, "fc", model.fc))
    return t


@torch.no_grad()
def _clip_text_weights(model, tokenizer, classnames, templates, device):
    model = model.to(device)
    cols = []
    for c in classnames:
        tokens = tokenizer([tpl(c) for tpl in templates]).to(device)
        e = F.normalize(model.encode_text(tokens), dim=-1).mean(0)
        cols.append(F.normalize(e, dim=0))
    return torch.stack(cols, dim=1).cpu()


def load(name: str, device: str = "cpu", classnames=None, templates=None, random_init: bool = False) -> Bundle:
    """random_init=True builds the same architecture with fresh initial weights (a reference
    for what training changed). Note: this is the library's init, not the original run's."""
    spec = MODELS[name]
    fam = spec["family"]
    if fam in ("timm_vit", "timm_resnet"):
        import timm
        model = timm.create_model(spec["timm_name"], pretrained=spec.get("pretrained", True) and not random_init)
        cfg = timm.data.resolve_data_config({}, model=model)
        pre = Preprocess(cfg["input_size"][-1], cfg["crop_pct"], cfg["interpolation"],
                         tuple(cfg["mean"]), tuple(cfg["std"]))
        targets = _vit_targets(model) if fam == "timm_vit" else _resnet_targets(model)
    elif fam == "openclip_vit":
        import open_clip
        clip, _, _ = open_clip.create_model_and_transforms(spec["arch"], pretrained=None if random_init else spec["pretrained"])
        tok = open_clip.get_tokenizer(spec["arch"])
        classnames = classnames or open_clip.IMAGENET_CLASSNAMES
        templates = templates or open_clip.OPENAI_IMAGENET_TEMPLATES
        W = _clip_text_weights(clip.eval(), tok, classnames, templates, device)
        model = ClipZeroShot(clip.visual, W, clip.logit_scale.detach().exp().item())
        v = clip.visual
        pre = Preprocess(v.image_size[0] if isinstance(v.image_size, tuple) else v.image_size,
                         1.0, "bicubic", tuple(v.image_mean or open_clip.OPENAI_DATASET_MEAN),
                         tuple(v.image_std or open_clip.OPENAI_DATASET_STD))
        targets = _clip_targets(v)
    else:
        raise ValueError(fam)
    return Bundle(name, model.eval().to(device), pre, targets)


def load_checkpoint(path: str, device: str = "cpu") -> Bundle:
    """A fine-tuned timm ViT saved by scripts/finetune_lt.py."""
    import timm
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = timm.create_model(ck["timm_name"], pretrained=False, num_classes=ck["num_classes"])
    model.load_state_dict(ck["state_dict"])
    pre = Preprocess(**ck["preprocess"])
    return Bundle(f"ckpt:{path}", model.eval().to(device), pre, _vit_targets(model))
