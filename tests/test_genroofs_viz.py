"""Offline tests for genroofs.viz (drawn off-screen)."""
import warnings

import torch

from conftest import make_squares
from genroofs import train, viz
from genroofs.data import to_bytes, to_float
from genroofs.models import RoofAE, RoofVAE


def image_axes(fig):
    return [ax for ax in fig.axes if ax.get_images()]


def test_show_grid():
    x = make_squares(10)
    fig = viz.show_grid(x, ncol=4, title="t", labels=list("abcdefghij"), show=False)
    assert len(image_axes(fig)) == 10 and len(fig.axes) == 12
    assert [ax.get_title() for ax in image_axes(fig)][:3] == ["a", "b", "c"]
    floats = viz.show_grid(to_float(x[:2]), show=False)
    assert len(image_axes(floats)) == 2
    single = viz.show_grid(x[0], show=False)
    assert len(image_axes(single)) == 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")              # matplotlib warns that Agg can't open a window
        assert viz.show_grid(x[:3]) is None


def test_show_rows_and_pairs():
    x = make_squares(6)
    fig = viz.show_rows({"original": x, "copy": x[:4]}, labels=list("ABCDEF"), show=False)
    assert len(image_axes(fig)) == 10
    assert fig.axes[0].get_ylabel() == "original" and fig.axes[6].get_ylabel() == "copy"
    assert len(image_axes(viz.show_pairs(x[:3], x[:3], show=False))) == 6


def test_morph_images():
    x = make_squares(4)
    ae = RoofAE(latent=16).eval()
    frames = viz.morph_images(ae, x[0], x[1], steps=5)
    assert frames.shape == (5, 3, 64, 64) and frames.dtype == torch.uint8
    with torch.no_grad():
        first = to_bytes(ae(to_float(x[:1])))
    assert torch.equal(frames[:1], first), "t = 0 is roof A's own copy"
    vae = RoofVAE().train()
    vf = viz.morph_images(vae, x[2], x[3], steps=3)
    assert vf.shape == (3, 3, 64, 64) and vae.training
    fig = viz.morph(vae, x[2], x[3], steps=3, show=False)
    assert len(image_axes(fig)) == 5


def test_latent_walk():
    vae = RoofVAE()
    images = viz.latent_walk_images(vae, steps=4)
    assert images.shape == (16, 3, 64, 64)
    around = viz.latent_walk_images(vae, roof=make_squares(1)[0], dims=(3, 7), steps=3, span=1.0)
    assert around.shape == (9, 3, 64, 64)
    assert len(image_axes(viz.latent_walk(vae, steps=3, show=False))) == 9


def test_history_pictures():
    from genroofs.models import Discriminator, Generator
    x = make_squares(32)
    runs = [train.train_ae(RoofAE(), x, epochs=2, batch_size=16, device="cpu", snapshots=2, verbose=False),
            train.train_vae(RoofVAE(), x, epochs=1, batch_size=16, device="cpu", snapshots=1, verbose=False),
            train.train_gan(Generator(), Discriminator(), x, epochs=1, batch_size=16, device="cpu", watch_n=4,
                            snapshots=1, verbose=False)]
    panels = {"ae": 1, "vae": 2, "gan": 2}
    for h in runs:
        fig = viz.plot_history(h, show=False)
        assert len(fig.axes) == panels[h.kind]
        snaps = viz.show_snapshots(h, show=False)
        rows = len(h.snapshots) + (1 if h.kind != "gan" else 0)
        assert len(image_axes(snaps)) == rows * len(h.watch)
