"""Pictures for the notebook: grids of roofs, originals next to copies, morphs, latent walks and loss curves.

Every function draws with matplotlib. With show=True (the default) the picture appears in the notebook and the
function returns None, so it isn't drawn twice; with show=False it returns the matplotlib Figure instead.
"""
import math

import numpy as np
import torch

from .data import to_bytes, to_float

__all__ = ["show_grid", "show_rows", "show_pairs", "morph_images", "morph", "latent_walk_images", "latent_walk",
           "plot_history", "show_snapshots"]


def _plt():
    import matplotlib.pyplot as plt
    return plt


def _hwc(images) -> np.ndarray:
    """N x 3 x H x W (uint8, or floats in -1..1) -> N x H x W x 3 uint8 for imshow."""
    if isinstance(images, np.ndarray):
        images = torch.from_numpy(images)
    images = images.detach().cpu()
    if images.dim() == 3:
        images = images.unsqueeze(0)
    if images.dtype != torch.uint8:
        images = to_bytes(images)
    return images.permute(0, 2, 3, 1).numpy()


def _finish(fig, show: bool):
    if show:
        _plt().show()
        return None
    return fig


def show_grid(images, ncol: int = 8, title: str | None = None, labels=None, size: float = 1.6, show: bool = True):
    """A grid of roofs, `ncol` per row, with an optional label under each."""
    plt = _plt()
    imgs = _hwc(images)
    n = len(imgs)
    ncol = max(1, min(ncol, n))
    nrow = math.ceil(n / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(size * ncol, size * nrow + (0.5 if title else 0.1)),
                             squeeze=False)
    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i < n:
            ax.imshow(imgs[i])
            if labels is not None:
                ax.set_title(str(labels[i]), fontsize=8)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return _finish(fig, show)


def show_rows(rows: dict, title: str | None = None, labels=None, size: float = 1.6, show: bool = True):
    """Several rows of roofs with a name on the left of each row, for example original / copy / smaller copy."""
    plt = _plt()
    names = list(rows)
    data = [_hwc(rows[k]) for k in names]
    ncol = max(len(d) for d in data)
    fig, axes = plt.subplots(len(names), ncol, figsize=(size * ncol + 0.8, size * len(names) + (0.5 if title else 0.1)),
                             squeeze=False)
    for r, (name, imgs) in enumerate(zip(names, data)):
        for c in range(ncol):
            ax = axes[r, c]
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if c < len(imgs):
                ax.imshow(imgs[c])
                if r == 0 and labels is not None:
                    ax.set_title(str(labels[c]), fontsize=8)
        axes[r, 0].set_ylabel(name, fontsize=9)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return _finish(fig, show)


def show_pairs(originals, copies, title: str | None = None, labels=None, show: bool = True):
    """Originals on top, the model's copies underneath."""
    return show_rows({"original": originals, "copy": copies}, title=title, labels=labels, show=show)


def _device(model) -> torch.device:
    return next(model.parameters()).device


@torch.no_grad()
def _encode(model, x: torch.Tensor) -> torch.Tensor:
    z = model.encode(x)
    return z[0] if isinstance(z, tuple) else z          # a VAE gives (mu, logvar): use the mean


def _one(image) -> torch.Tensor:
    """One roof as a batch of one (accepts 3 x H x W or N x 3 x H x W, keeps the first)."""
    image = torch.as_tensor(image)
    return image.unsqueeze(0) if image.dim() == 3 else image[:1]


def _model_input(image, device) -> torch.Tensor:
    x = _one(image).to(device)
    return to_float(x) if x.dtype == torch.uint8 else x.float()


def _picture(image) -> torch.Tensor:
    x = _one(image).detach().cpu()
    return x if x.dtype == torch.uint8 else to_bytes(x)


@torch.no_grad()
def morph_images(model, roof_a, roof_b, steps: int = 8) -> torch.Tensor:
    """Decode evenly spaced points on the straight line from roof A's latent point to roof B's.
    Works with RoofVAE and RoofAE (comparing the two shows why the VAE's latent space is called smooth)."""
    was_training = model.training
    model.eval()
    try:
        dev = _device(model)
        a, b = _encode(model, _model_input(roof_a, dev)), _encode(model, _model_input(roof_b, dev))
        t = torch.linspace(0, 1, steps, device=dev).view(-1, 1)
        return to_bytes(model.decode((1 - t) * a + t * b)).cpu()
    finally:
        model.train(was_training)


def morph(model, roof_a, roof_b, steps: int = 8, title: str | None = None, show: bool = True):
    """Show the morph from roof A (left) to roof B (right), with the originals at both ends."""
    frames = morph_images(model, roof_a, roof_b, steps)
    row = torch.cat([_picture(roof_a), frames, _picture(roof_b)])
    labels = ["A"] + [f"t={t:.2f}" for t in np.linspace(0, 1, steps)] + ["B"]
    return show_grid(row, ncol=len(row), title=title or f"Morph through the latent space ({type(model).__name__})",
                     labels=labels, show=show)


@torch.no_grad()
def latent_walk_images(model, roof=None, dims: tuple = (0, 1), steps: int = 7, span: float = 2.5) -> torch.Tensor:
    """Decode a steps x steps grid: two latent numbers move from -span to +span around a starting point
    (the roof's latent point, or the centre of the space when roof is None)."""
    was_training = model.training
    model.eval()
    try:
        dev = _device(model)
        latent = model.latent
        base = torch.zeros(1, latent, device=dev) if roof is None else _encode(model, _model_input(roof, dev))
        values = torch.linspace(-span, span, steps, device=dev)
        z = base.repeat(steps * steps, 1)
        z[:, dims[0]] += values.repeat_interleave(steps)
        z[:, dims[1]] += values.repeat(steps)
        return to_bytes(model.decode(z)).cpu()
    finally:
        model.train(was_training)


def latent_walk(model, roof=None, dims: tuple = (0, 1), steps: int = 7, span: float = 2.5, show: bool = True):
    """Show a latent walk: down the rows latent number dims[0] changes, across the columns dims[1] changes."""
    images = latent_walk_images(model, roof, dims, steps, span)
    return show_grid(images, ncol=steps, size=1.2, show=show,
                     title=f"Latent walk: number {dims[0]} changes down the rows, number {dims[1]} across "
                           f"(from {-span:g} to {span:g})")


def _smooth(values, window: int) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    if window <= 1 or len(v) < window:
        return v
    return np.convolve(v, np.ones(window) / window, mode="valid")


def plot_history(history, show: bool = True):
    """Loss curves of one training call: every step (light) and a moving average (dark)."""
    plt = _plt()
    window = max(1, history.steps // 50)
    if history.kind == "gan":
        panels = [("Losses", ["d_loss", "g_loss"], False), ("Critic's average probability", ["d_real", "d_fake"], False)]
    elif history.kind == "vae":
        panels = [("Reconstruction", ["recon"], True), ("KL term", ["kl"], False)]
    else:
        panels = [("Reconstruction loss", ["loss"], True)]
    fig, axes = plt.subplots(1, len(panels), figsize=(5.5 * len(panels), 3.4), squeeze=False)
    for ax, (name, keys, log) in zip(axes[0], panels):
        for key in keys:
            values = history.losses[key]
            line, = ax.plot(values, alpha=0.25, linewidth=0.8)
            sm = _smooth(values, window)
            ax.plot(np.arange(len(sm)) + (len(values) - len(sm)), sm, color=line.get_color(), label=key)
        if history.kind == "gan" and keys == ["d_real", "d_fake"]:
            ax.axhline(0.5, color="grey", linestyle=":", linewidth=1)
            ax.set_ylim(0, 1)
        if log:
            ax.set_yscale("log")
        ax.set_title(name)
        ax.set_xlabel("step")
        ax.legend(fontsize=8)
    fig.suptitle(f"{history.kind.upper()}: {history.passes:.1f} passes in {history.seconds / 60:.1f} min")
    fig.tight_layout()
    return _finish(fig, show)


def show_snapshots(history, show: bool = True):
    """The same roofs (or, for the GAN, the same noise vectors) pictured as training went on."""
    rows = {}
    if history.kind in ("ae", "vae"):
        rows["original"] = history.watch
    for step, images in history.snapshots:
        rows[f"step {step}"] = images
    return show_rows(rows, title=f"{history.kind.upper()} pictures during training", size=1.2, show=show)
