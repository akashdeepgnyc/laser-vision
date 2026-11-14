import torch

from laser.edit import edited
from laser.models import load


def _check(b, n_targets, n_classes):
    assert len(b.targets) == n_targets
    for t in b.targets:
        W = t.read(t.parameter().data)
        assert W.dim() == 2
    x = torch.randn(2, 3, b.preprocess.size, b.preprocess.size)
    with torch.no_grad():
        assert b.model(x).shape == (2, n_classes)


def test_vit_targets():
    b = load("vit_tiny_test")
    _check(b, 12 * 6, 1000)
    assert {t.mtype for t in b.targets} == {"attn_q", "attn_k", "attn_v", "attn_out", "mlp_in", "mlp_out"}


def test_resnet_conv_reshape_roundtrip():
    b = load("resnet18_test")
    _check(b, 8 * 2 + 1, 1000)  # basic blocks: two convs each, plus fc
    conv = next(t for t in b.targets if t.mtype == "conv_b")
    out, inp, kh, kw = conv.parameter().shape
    assert conv.read(conv.parameter().data).shape == (out, inp * kh * kw)
    before = conv.parameter().data.clone()
    with edited(conv, 1.0, "top"):  # full rank keeps the kernel (up to float error)
        assert torch.allclose(conv.parameter().data, before, atol=1e-4)


def test_clip_zero_shot():
    b = load("clip_test", classnames=["cat", "dog", "car"], templates=[lambda c: f"a photo of a {c}."])
    _check(b, 12 * 6, 3)
    assert b.preprocess.crop_pct == 1.0
