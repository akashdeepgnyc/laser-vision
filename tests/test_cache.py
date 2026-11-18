import numpy as np
import torch

from laser import cache, evaluate
from laser.edit import edited
from laser.models import load


def test_layer_cache_matches_full_forward_with_edits():
    torch.manual_seed(0)
    b = load("vit_tiny_test")
    imgs = np.random.default_rng(0).integers(0, 255, size=(6, 3, 224, 224), dtype=np.uint8)
    data = evaluate.GPUImages(imgs, b.preprocess.mean, b.preprocess.std, "cpu")
    c = cache.BlockCache(b.model, data, bs=4, amp=False)
    assert cache.supported(b.model)
    for name in ("blocks.3.mlp_in", "blocks.3.attn_q", "blocks.7.mlp_out", "blocks.11.attn_out"):
        t = next(x for x in b.targets if x.name == name)
        c.to_layer(t.layer)
        with edited(t, 0.1, "top"):
            full = evaluate.logits(b.model, data, bs=4, amp=False)
            fast = c.logits()
        assert torch.allclose(full, fast, atol=1e-4), name
