import torch

from laser.edit import edited, kept_rank, low_rank
from laser.models import load


def test_top_plus_bottom_is_identity():
    W = torch.randn(48, 20)
    assert torch.allclose(low_rank(W, 0.3, "top") + low_rank(W, 0.3, "bottom"), W, atol=1e-4)


def test_rank_and_full_keep():
    W = torch.randn(30, 50)
    k = kept_rank(W.shape, 0.2)
    assert k == 6
    assert torch.linalg.matrix_rank(low_rank(W, 0.2, "top")) == k
    assert torch.linalg.matrix_rank(low_rank(W, 0.2, "random")) == k
    assert torch.allclose(low_rank(W, 1.0, "top"), W, atol=1e-4)


def test_top_beats_random_in_frobenius_error():  # Eckart-Young
    W = torch.randn(64, 64)
    e_top = (W - low_rank(W, 0.25, "top")).norm()
    e_rand = (W - low_rank(W, 0.25, "random", seed=1)).norm()
    assert e_top < e_rand


def test_edit_restores_and_touches_only_its_qkv_slice():
    b = load("vit_tiny_test")
    q = next(t for t in b.targets if t.name == "blocks.3.attn_q")
    p = q.parameter()
    before = p.data.clone()
    with edited(q, 0.1, "top") as k:
        assert k == max(1, int(0.1 * 192))
        d = 192
        assert not torch.equal(p.data[:d], before[:d])        # q rows changed
        assert torch.equal(p.data[d:], before[d:])            # k, v rows untouched
    assert torch.equal(p.data, before)                        # restored


def test_shrink_matches_top_norm_and_keeps_directions():
    W = torch.randn(40, 70)
    top, sh = low_rank(W, 0.1, "top"), low_rank(W, 0.1, "shrink")
    assert torch.allclose(sh.norm(), top.norm(), rtol=1e-4)
    assert torch.linalg.matrix_rank(sh) == torch.linalg.matrix_rank(W)


def test_prune_budget_and_rank():
    W = torch.randn(64, 96)
    P = low_rank(W, 0.1, "prune")                      # k = 6 -> budget 6 * (64 + 96) = 960 entries
    assert int((P != 0).sum()) == 960
    kept = W.abs()[P != 0].min()
    assert kept >= W.abs()[P == 0].max()               # keeps exactly the largest-magnitude entries
    assert torch.linalg.matrix_rank(P) > 6             # sparse, but far higher rank than LASER's 6
