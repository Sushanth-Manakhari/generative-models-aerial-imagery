"""Offline tests for genroofs.utils: device choice, seeds and the stopwatch."""
import random
import time

import numpy as np
import pytest
import torch

from genroofs import utils


def machine(monkeypatch, cuda: bool, mps: bool) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)


@pytest.mark.parametrize("cuda, mps, want", [(True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")])
def test_pick_device_order(monkeypatch, cuda, mps, want):
    machine(monkeypatch, cuda, mps)
    assert utils.pick_device().type == want


def test_pick_device_prefer(monkeypatch, capsys):
    machine(monkeypatch, cuda=True, mps=False)
    assert utils.pick_device("cpu").type == "cpu"
    assert utils.pick_device("CUDA").type == "cuda"
    assert utils.pick_device("mps").type == "cuda"
    assert "mps is not available" in capsys.readouterr().out
    with pytest.raises(ValueError, match="cuda, mps, cpu"):
        utils.pick_device("tpu")


def test_set_seed_repeats_every_generator():
    def draw():
        return random.random(), np.random.rand(), torch.rand(3).tolist()
    assert utils.set_seed(123) == 123
    first = draw()
    utils.set_seed(123)
    assert draw() == first
    utils.set_seed(124)
    assert draw() != first


def test_timer():
    with utils.Timer() as t:
        time.sleep(0.02)
        assert t.elapsed > 0
    assert 0.015 < t.seconds < 1.0
    assert utils.Timer().elapsed == 0.0
