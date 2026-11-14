"""Offline tests for genroofs.app: the page logic without Gradio, and building the page when Gradio is present."""
import sys

import numpy as np
import pytest
import torch

from conftest import make_squares
from genroofs import app
from genroofs.models import Generator, RoofAE, RoofVAE


@pytest.fixture
def logic():
    x = make_squares(12)
    manifest = [{"place": "Gorham"}] * 12
    return app.RoofApp(RoofAE(latent=8), RoofVAE(), Generator(), x, manifest)


def test_inspect(logic):
    original, copy, text = logic.inspect(3)
    assert original.shape == copy.shape == (256, 256, 3) and original.dtype == np.uint8
    assert np.array_equal(original[::4, ::4], logic.images[3].permute(1, 2, 0).numpy())
    assert text.startswith("**Roof 3 from Gorham**") and "unusual score" in text
    assert logic.inspect(99)[2].startswith("**Roof 11"), "out-of-range picks are clamped to the test set"
    assert logic.inspect(-5)[2].startswith("**Roof 0")


def test_morph_and_new_roof(logic):
    assert logic.morph(0, 5, 0.0).shape == (256, 256, 3)
    assert not np.array_equal(logic.morph(0, 5, 0.0), logic.morph(0, 5, 1.0))
    assert np.array_equal(logic.new_roof(4), logic.new_roof(4))
    assert not np.array_equal(logic.new_roof(4), logic.new_roof(5))


def test_only_test_roofs_are_accepted():
    with pytest.raises(TypeError, match="uint8 test roofs"):
        app.RoofApp(RoofAE(), RoofVAE(), Generator(), torch.zeros(3, 3, 64, 64))


def test_no_public_links():
    with pytest.raises(ValueError, match="share links are off"):
        app.launch(None, None, None, None, share=True)


def test_gradio_is_optional(monkeypatch):
    monkeypatch.setitem(sys.modules, "gradio", None)
    with pytest.raises(ImportError, match='pip install "genroofs\\[app\\]"'):
        app.build_app(RoofAE(), RoofVAE(), Generator(), make_squares(4))


def test_page_builds_when_gradio_is_installed():
    gr = pytest.importorskip("gradio")
    page = app.build_app(RoofAE(latent=8), RoofVAE(), Generator(), make_squares(6), [{"place": "York"}] * 6)
    assert isinstance(page, gr.Blocks)
