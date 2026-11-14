"""Offline tests for genroofs.metrics."""
import math

import numpy as np
import pytest
import torch
from torch import nn

from conftest import make_squares
from genroofs import metrics
from genroofs.models import RoofAE, RoofVAE


class ZeroModel(nn.Module):
    """Copies every roof as flat grey (all zeros in the -1..1 scale), so the error is easy to compute by hand."""

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return torch.zeros_like(x) + 0 * self.w


def test_psnr_values():
    a = torch.zeros(2, 3, 64, 64)
    assert torch.isinf(metrics.psnr(a, a)).all()
    assert metrics.psnr(a, a + 0.2)[0].item() == pytest.approx(20.0, abs=1e-4)      # 10 log10(4 / 0.04)
    u = make_squares(4)
    assert metrics.psnr(u, u).shape == (4,)
    with pytest.raises(ValueError, match="same shape"):
        metrics.psnr(a, torch.zeros(2, 3, 32, 32))


def test_reconstruction_errors_by_hand():
    x = make_squares(10)
    model = ZeroModel()
    model.train()
    errors = metrics.reconstruction_errors(model, x, batch_size=3)
    want = (x.float() / 127.5 - 1).pow(2).flatten(1).sum(1)
    assert torch.allclose(errors, want, rtol=1e-5)
    assert model.training, "the model's mode is restored"
    assert torch.equal(errors, metrics.reconstruction_errors(model, x, batch_size=100))


def test_vae_errors_use_the_mean_and_are_repeatable():
    vae = RoofVAE()
    x = make_squares(6)
    assert torch.equal(metrics.reconstruction_errors(vae, x), metrics.reconstruction_errors(vae, x))
    assert metrics.reconstruction_errors(RoofAE(), x).shape == (6,)


def test_strangest():
    errors = torch.tensor([3.0, 9.0, 1.0, 7.0])
    assert metrics.strangest(errors, 2) == [1, 3]
    assert metrics.strangest(errors, 10) == [1, 3, 0, 2]
    assert metrics.strangest([], 3) == []


def brute_force_auc(normal, unusual) -> float:
    wins = sum((u > n) + 0.5 * (u == n) for u in unusual for n in normal)
    return wins / (len(normal) * len(unusual))


def test_auc():
    assert metrics.unusual_score_auc([1, 2, 3], [4, 5]) == 1.0
    assert metrics.unusual_score_auc([4, 5], [1, 2, 3]) == 0.0
    assert metrics.unusual_score_auc([2, 2, 2], [2, 2]) == 0.5
    rng = np.random.default_rng(0)
    normal, unusual = rng.integers(0, 6, 40), rng.integers(1, 8, 25)      # many ties on purpose
    assert metrics.unusual_score_auc(normal, unusual) == pytest.approx(brute_force_auc(normal, unusual))
    assert metrics.unusual_score_auc(torch.tensor([0.1, 0.4]), torch.tensor([0.3])) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        metrics.unusual_score_auc([], [1.0])


def test_sample_spread():
    same = torch.zeros(5, 3, 64, 64)
    assert metrics.sample_spread(same) == pytest.approx(0.0, abs=1e-5)
    two = torch.stack([torch.full((3, 64, 64), -1.0), torch.full((3, 64, 64), 1.0)])
    assert metrics.sample_spread(two) == pytest.approx(2.0, rel=1e-5)
    many = make_squares(40)
    assert metrics.sample_spread(many, max_images=10, seed=1) == metrics.sample_spread(many, max_images=10, seed=1)
    assert metrics.sample_spread(many) > 0.1
    with pytest.raises(ValueError):
        metrics.sample_spread(same[:1])


def test_nearest_real_finds_copies():
    train = make_squares(20)
    fakes = train[[4, 17, 9]]
    idx, dist = metrics.nearest_real(fakes, train, chunk=6)
    assert idx.tolist() == [4, 17, 9]
    assert dist.max().item() < 1e-3
    other = make_squares(3, seed=5)
    idx2, dist2 = metrics.nearest_real(other, train, chunk=6)
    idx3, dist3 = metrics.nearest_real(other, train, chunk=1000)
    assert torch.equal(idx2, idx3) and torch.allclose(dist2, dist3)
    assert (dist2 > 0.01).all()
    assert not math.isinf(dist2.max().item())
