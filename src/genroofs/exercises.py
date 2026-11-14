"""The five in-class exercises, one function each, sized to finish in a few minutes on a Colab T4.

    1. bottleneck       How much does the size of the latent space matter?            (ROOF-301)
    2. strangest_roofs  Which roofs would the autoencoder send to an inspector?        (ROOF-301)
    3. morph_pair       Walk from one roof to another through the VAE's latent space.  (ROOF-302)
    4. break_the_gan    Change one setting and watch the GAN fail.                     (ROOF-303)
    5. real_or_fake     Can the class tell generated roofs from real ones?             (ROOF-303)

Each function prints what to look at and returns a small Result (or, for the quiz, a Quiz).
"""
import numpy as np
import torch

from .data import to_bytes, to_float
from .metrics import psnr, reconstruction_errors, sample_spread, strangest, unusual_score_auc
from .models import Discriminator, Generator, RoofAE
from .train import train_ae, train_gan
from .utils import set_seed
from .viz import morph_images, plot_history, show_grid, show_rows

__all__ = ["Result", "bottleneck", "strangest_roofs", "morph_pair", "break_the_gan", "GAN_CHANGES", "Quiz",
           "real_or_fake"]


class Result:
    """What an exercise found. Attributes hold the details; printing it shows only the short numbers."""

    def __init__(self, **values):
        self.__dict__.update(values)

    def __repr__(self) -> str:
        short = []
        for k, v in self.__dict__.items():
            if isinstance(v, (int, float, str)) or v is None:
                short.append(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v!r}")
        return f"Result({', '.join(short)})"


@torch.no_grad()
def _copies(model, images: torch.Tensor) -> torch.Tensor:
    was_training = model.training
    model.eval()
    try:
        x = to_float(images.to(next(model.parameters()).device))
        out = model.decode(model.encode(x)[0]) if hasattr(model, "reparameterize") else model(x)
        return to_bytes(out).cpu()
    finally:
        model.train(was_training)


def _mean_psnr(model, images: torch.Tensor) -> float:
    return float(psnr(images, _copies(model, images)).mean())


# --------------------------------------------------------------------------------------- 1. bottleneck

def bottleneck(trained_ae, train_images: torch.Tensor, test_images: torch.Tensor, latent: int = 8,
               minutes: float = 1.0, n_show: int = 8, batch_size: int = 128, seed: int = 7, device=None,
               show: bool = True) -> Result:
    """Train a second autoencoder with only `latent` numbers in its bottleneck for `minutes`, then compare its
    copies with the full model's. Fewer numbers means less detail survives the squeeze."""
    set_seed(seed)
    small = RoofAE(latent=latent)
    history = train_ae(small, train_images, minutes=minutes, batch_size=batch_size, seed=seed, device=device,
                       snapshots=0, verbose=False, watch=test_images[:n_show])
    sample = test_images[:500]
    big_db, small_db = _mean_psnr(trained_ae, sample), _mean_psnr(small, sample)
    print(f"Latent {trained_ae.latent}: average PSNR {big_db:.1f} dB.  Latent {latent}: {small_db:.1f} dB "
          f"(on {len(sample)} test roofs; higher is closer to the original).")
    print(f"The latent-{latent} model trained for {history.passes:.1f} passes. What detail is lost first?")
    if show:
        show_rows({"original": test_images[:n_show], f"latent {trained_ae.latent}": _copies(trained_ae, test_images[:n_show]),
                   f"latent {latent}": _copies(small, test_images[:n_show])},
                  title="Bottleneck: the same roofs squeezed through 128 numbers and through "
                        f"{latent}")
    return Result(latent=latent, psnr_full=big_db, psnr_small=small_db, model=small, history=history)


# ---------------------------------------------------------------------------------- 2. strangest roofs

def strangest_roofs(ae, test_images: torch.Tensor, test_manifest: list, k: int = 10, unusual_images=None,
                    ordinary: bool = True, show: bool = True) -> Result:
    """Score every test roof by how badly the autoencoder copies it and show the k strangest: an inspection
    queue. With unusual_images (for example the variety pack), also measure how well the score separates them.

    ordinary: also show the k lowest scores. That end holds plain lawns, woods and empty lots: images with no house
    are easy to copy, so this kind of score does not catch "no house here". A limit worth discussing.
    """
    errors = reconstruction_errors(ae, test_images)
    top = strangest(errors, k)
    bottom = torch.argsort(errors)[:k].tolist()
    order = torch.argsort(errors)
    percentile = torch.empty_like(errors)
    percentile[order] = torch.arange(len(errors), dtype=errors.dtype) / max(1, len(errors) - 1) * 100
    print(f"{'rank':>4}  {'chip':>6}  {'town':<18} {'roof m2':>7}  {'score':>8}  {'percentile':>10}")
    for rank, i in enumerate(top, 1):
        row = test_manifest[i]
        print(f"{rank:>4}  {row['chip_id']:>6}  {row['place']:<18} {row['area_m2']:>7.0f}  {errors[i]:>8.0f}  "
              f"{percentile[i]:>9.1f}%")
    auc = None
    if unusual_images is not None:
        unusual = reconstruction_errors(ae, unusual_images)
        auc = unusual_score_auc(errors, unusual)
        print(f"\nA roof from the other pack scores higher than a Maine test roof {auc * 100:.0f}% of the time "
              f"(AUC {auc:.2f}; 0.50 would be a coin toss).")
    print("\nLook at the top of the queue: clutter, trees over the roof, unusual shapes or colours? Would an "
          "inspector agree?")
    if ordinary:
        print(f"At the other end, the {len(bottom)} lowest scores: if they show lawns, woods or empty lots, the score "
              "missed 'no house here', because empty ground is easy to copy.")
    if show:
        show_rows({"roof": test_images[top], "copy": _copies(ae, test_images[top])},
                  labels=[f"#{r} {test_manifest[i]['place']}" for r, i in enumerate(top, 1)],
                  title=f"The {len(top)} strangest test roofs: the autoencoder's inspection queue")
        if ordinary:
            show_grid(test_images[bottom], ncol=len(bottom), labels=[test_manifest[i]["place"] for i in bottom],
                      title=f"The {len(bottom)} most ordinary test roofs (lowest scores)")
    return Result(k=len(top), indices=top, ordinary_indices=bottom if ordinary else [], errors=errors, auc=auc)


# ---------------------------------------------------------------------------------------------- 3. morph

def morph_pair(vae, images: torch.Tensor, a: int | None = None, b: int | None = None, steps: int = 8, ae=None,
               seed: int = 7, show: bool = True) -> Result:
    """Blend roof `a` into roof `b` through the VAE's latent space (random roofs when a or b is None).
    Pass the autoencoder as `ae` to see its morph underneath: the VAE's path stays roof-like, the AE's often not."""
    rng = np.random.default_rng(seed)
    a = int(rng.integers(len(images))) if a is None else int(a)
    b = int(rng.integers(len(images))) if b is None else int(b)
    rows = {"VAE": torch.cat([images[a:a + 1].cpu(), morph_images(vae, images[a], images[b], steps),
                              images[b:b + 1].cpu()])}
    if ae is not None:
        rows["AE"] = torch.cat([images[a:a + 1].cpu(), morph_images(ae, images[a], images[b], steps),
                                images[b:b + 1].cpu()])
    print(f"Morphing roof {a} into roof {b} in {steps} steps. Does every in-between picture still look like a roof?")
    if show:
        labels = ["A"] + [f"{t:.2f}" for t in np.linspace(0, 1, steps)] + ["B"]
        show_rows(rows, labels=labels, title="Morph: straight line between two latent points")
    return Result(a=a, b=b, steps=steps)


# ----------------------------------------------------------------------------------------- 4. break it

GAN_CHANGES = {
    "none": ({}, {}, "the normal settings"),
    "critic_lr_x10": ({}, {"lr_d": 2e-3}, "the critic learns 10 times faster than the generator"),
    "no_smoothing": ({}, {"label_smooth": 1.0}, "no label smoothing: the critic is told real roofs are exactly 1"),
    "tiny_noise": ({"z_dim": 2}, {}, "the generator gets only 2 random numbers instead of 100"),
}


def break_the_gan(train_images: torch.Tensor, change: str = "critic_lr_x10", minutes: float = 1.5,
                  real_images=None, batch_size: int = 128, seed: int = 7, device=None, show: bool = True) -> Result:
    """Train a fresh GAN for `minutes` with one setting changed, then look at its roofs and their spread.
    Changes: "none", "critic_lr_x10", "no_smoothing", "tiny_noise" (see GAN_CHANGES)."""
    if change not in GAN_CHANGES:
        raise ValueError(f"change must be one of {', '.join(GAN_CHANGES)}, not {change!r}.")
    model_kw, train_kw, what = GAN_CHANGES[change]
    set_seed(seed)
    g, d = Generator(**model_kw), Discriminator()
    history = train_gan(g, d, train_images, minutes=minutes, batch_size=batch_size, seed=seed, device=device,
                        snapshots=0, verbose=False, **train_kw)
    samples = g.sample(64, seed=seed)
    spread = sample_spread(samples)
    real_spread = sample_spread(real_images[:512]) if real_images is not None else None
    print(f"Change: {what}. Trained {history.passes:.1f} passes in {history.seconds / 60:.1f} min.")
    line = f"Spread of 64 generated roofs: {spread:.3f}"
    if real_spread is not None:
        line += f" (real roofs: {real_spread:.3f}; much lower means the GAN keeps drawing the same roof)"
    print(line + ".")
    print(f"Critic's last average probability: real {history.last('d_real'):.2f}, generated "
          f"{history.last('d_fake'):.2f}.")
    if show:
        show_grid(samples[:16], ncol=8, title=f"Break the GAN: {change}")
        plot_history(history)
    return Result(change=change, spread=spread, real_spread=real_spread, generator=g, critic=d, history=history)


# ------------------------------------------------------------------------------------------ 5. the quiz

class Quiz:
    """A numbered grid of real and generated roofs. The answer key stays hidden until reveal()."""

    def __init__(self, images: torch.Tensor, is_fake: list):
        self.images = images
        self._is_fake = list(is_fake)

    def __len__(self) -> int:
        return len(self._is_fake)

    def __repr__(self) -> str:
        return f"Quiz({len(self)} roofs; the answer key is hidden until quiz.reveal())"

    def show(self) -> None:
        show_grid(self.images, ncol=min(10, len(self)), labels=[str(i + 1) for i in range(len(self))],
                  title="Real or generated? Write R or G for each number")

    def score(self, guesses) -> int:
        """Count correct guesses, given as a string like "RGGR..." or a list; R = real, G (or F) = generated."""
        g = [c for c in (guesses if isinstance(guesses, (list, tuple)) else str(guesses).replace(" ", "").upper())]
        if len(g) != len(self):
            raise ValueError(f"Give one guess per roof: {len(self)} guesses, got {len(g)}.")
        return sum((str(x).upper() in ("G", "F")) == fake for x, fake in zip(g, self._is_fake))

    def reveal(self) -> None:
        key = "".join("G" if f else "R" for f in self._is_fake)
        print(f"Answer key: {key}   ({sum(self._is_fake)} generated, {len(self) - sum(self._is_fake)} real)")
        show_grid(self.images, ncol=min(10, len(self)),
                  labels=[f"{i + 1}: {'generated' if f else 'real'}" for i, f in enumerate(self._is_fake)],
                  title="Answers")


def real_or_fake(real_images: torch.Tensor, generator, n: int = 10, seed: int = 7, show: bool = True) -> Quiz:
    """Mix n roofs, some real (from real_images) and some generated, in a random order the class can't see."""
    if n < 4:
        raise ValueError("Use at least 4 roofs.")
    rng = np.random.default_rng(seed)
    n_fake = int(rng.integers(n // 2 - 1, n // 2 + 2))
    real_idx = rng.choice(len(real_images), n - n_fake, replace=False)
    real = real_images[torch.from_numpy(real_idx).to(real_images.device)].cpu()
    fake = to_bytes(generator.sample(n_fake, seed=seed)).cpu()
    is_fake = np.array([False] * len(real) + [True] * n_fake)
    order = rng.permutation(n)
    quiz = Quiz(torch.cat([real, fake])[torch.from_numpy(order)], is_fake[order].tolist())
    if show:
        quiz.show()
    return quiz
