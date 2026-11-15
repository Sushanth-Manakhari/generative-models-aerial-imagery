"""Training with a time limit, the three losses, and saving and loading weights.

Every loop stops at whichever comes first: `minutes` of wall-clock time or `epochs` full passes over the images.
The clock is checked before every batch, so a run ends at most one batch late. With neither given, the in-class
budgets apply (AE 3 minutes, VAE 3 minutes, GAN 6 minutes); the long runs before the lecture pass `epochs`.

    from genroofs.models import RoofAE
    from genroofs.train import train_ae
    ae = RoofAE()
    history = train_ae(ae, train.images, watch=test.images[:8])     # 3 minutes on the class slice

Each call returns a History: the loss of every step, the seconds of every pass, and pictures of the same roofs
(or, for the GAN, the same noise vectors) taken as training goes, so the class can watch one fixed set improve.
"""
import json
import math
import time
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import __version__
from .data import REPO, PackError, batches, cache_dir, fetch_checked, to_bytes, to_float
from .models import MODELS, count_parameters
from .utils import pick_device, set_seed

__all__ = ["History", "reconstruction_loss", "kl_divergence", "vae_loss", "critic_loss", "generator_loss",
           "train_ae", "train_vae", "train_gan", "save_weights", "load_weights", "weights_note", "load_pretrained",
           "WeightsNotPublished", "CLASS_MINUTES"]

CLASS_MINUTES = {"ae": 3.0, "vae": 3.0, "gan": 6.0}   # in-class budgets measured on a Colab T4 in G3
WEIGHTS_TAG = "models-v1"
WEIGHTS_URL = f"https://github.com/{REPO}/releases/download/{WEIGHTS_TAG}/{{file}}"
_WEIGHTS_INFO = None


def _now() -> float:
    return time.perf_counter()


# --------------------------------------------------------------------------------------------------- losses

def reconstruction_loss(copy: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Squared error summed over every pixel of a roof, averaged over the batch: (1/B) * sum ||x - x_hat||^2."""
    return F.mse_loss(copy, x, reduction="sum") / x.shape[0]


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL(q(z|x) || N(0, I)) = -1/2 * sum(1 + log sigma^2 - mu^2 - sigma^2), averaged over the batch.
    Zero when every mu is 0 and every sigma is 1; positive otherwise."""
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.shape[0]


def vae_loss(copy, x, mu, logvar, beta: float = 1.0) -> tuple:
    """Returns (total, reconstruction, kl) with total = reconstruction + beta * kl."""
    rec, kl = reconstruction_loss(copy, x), kl_divergence(mu, logvar)
    return rec + beta * kl, rec, kl


def critic_loss(real_scores: torch.Tensor, fake_scores: torch.Tensor, real_label: float = 0.9) -> torch.Tensor:
    """The critic should call real roofs `real_label` (0.9: one-sided label smoothing) and generated roofs 0."""
    return (F.binary_cross_entropy_with_logits(real_scores, torch.full_like(real_scores, real_label))
            + F.binary_cross_entropy_with_logits(fake_scores, torch.zeros_like(fake_scores)))


def generator_loss(fake_scores: torch.Tensor) -> torch.Tensor:
    """The generator is rewarded when the critic calls its roofs real (1)."""
    return F.binary_cross_entropy_with_logits(fake_scores, torch.ones_like(fake_scores))


# -------------------------------------------------------------------------------------------------- history

@dataclass
class History:
    """What happened during one training call."""
    kind: str                                      # "ae", "vae" or "gan"
    settings: dict
    losses: dict                                   # name -> one value per step
    watch: torch.Tensor                            # the roofs (AE, VAE) or noise vectors (GAN) pictured
    snapshots: list = field(default_factory=list)  # (step, uint8 images N x 3 x 64 x 64)
    pass_seconds: list = field(default_factory=list)
    steps: int = 0
    batches_per_pass: int = 0
    seconds: float = 0.0
    stopped_by: str = ""                           # "minutes", "epochs" or "nan"

    @property
    def passes(self) -> float:
        """Passes over the images, counting a pass cut short by the clock as a fraction."""
        return self.steps / self.batches_per_pass if self.batches_per_pass else 0.0

    def last(self, name: str, n: int = 50) -> float:
        """Average of the last n values of one loss (n smooths out batch-to-batch noise)."""
        values = self.losses[name][-n:]
        return float(np.mean(values)) if values else float("nan")

    def summary(self) -> dict:
        """A small JSON-friendly summary, for example to store next to saved weights."""
        return {"kind": self.kind, "steps": self.steps, "passes": round(self.passes, 2),
                "seconds": round(self.seconds, 1), "stopped_by": self.stopped_by,
                "last": {k: round(self.last(k), 4) for k in self.losses}, "settings": self.settings}

    def __repr__(self) -> str:
        last = ", ".join(f"{k} {self.last(k):.4g}" for k in self.losses)
        return (f"History({self.kind}: {self.passes:.1f} passes, {self.steps} steps, {_clock(self.seconds)}, "
                f"stopped by {self.stopped_by or '-'}; last {last})")


def _clock(seconds: float) -> str:
    m, s = divmod(int(round(seconds)), 60)
    return f"{m}:{s:02d}"


# ---------------------------------------------------------------------------------------------- the loop

def _run(hist: History, step, picture, images: torch.Tensor, *, minutes, epochs, batch_size: int, seed: int,
         device, snapshots: int, verbose: bool) -> History:
    """The shared loop: limits, snapshots, progress lines. `step(x)` trains on one batch and returns its losses."""
    n, batch_size = len(images), int(batch_size)
    per_pass = n // batch_size
    if per_pass == 0:
        raise ValueError(f"Only {n} roofs, fewer than one batch of {batch_size}. Use more roofs or a smaller "
                         f"batch_size.")
    limit_s = None if minutes is None else float(minutes) * 60.0
    total_steps = None if epochs is None else int(epochs) * per_pass
    hist.batches_per_pass = per_pass
    x_all = images.to(device)
    marks = [k / snapshots for k in range(1, snapshots + 1)] if snapshots > 0 else []
    hist.snapshots.append((0, picture()))
    start, stop, pass_index = _now(), None, 0
    while stop is None:
        if total_steps is not None and hist.steps >= total_steps:
            stop = "epochs"
            break
        pass_start, whole_pass = _now(), True
        for x in batches(x_all, batch_size, shuffle=True, seed=seed + pass_index):
            elapsed = _now() - start
            if limit_s is not None and elapsed >= limit_s:
                stop, whole_pass = "minutes", False
                break
            values = step(x)
            for k, v in values.items():
                hist.losses[k].append(v)
            hist.steps += 1
            if not all(math.isfinite(v) for v in values.values()):
                stop, whole_pass = "nan", False
                break
            progress = max(elapsed / limit_s if limit_s else 0.0,
                           hist.steps / total_steps if total_steps else 0.0)
            while marks and progress >= marks[0]:
                marks.pop(0)
                hist.snapshots.append((hist.steps, picture()))
                if verbose:
                    losses = "  ".join(f"{k} {hist.last(k, 20):.4g}" for k in hist.losses)
                    print(f"{hist.kind.upper():4s} pass {hist.steps / per_pass:6.1f}  {_clock(_now() - start)}  "
                          f"{losses}")
        if whole_pass:
            hist.pass_seconds.append(_now() - pass_start)
            pass_index += 1
    hist.seconds = _now() - start
    hist.stopped_by = stop
    if hist.snapshots[-1][0] != hist.steps:
        hist.snapshots.append((hist.steps, picture()))
    if verbose:
        if stop == "minutes":
            why = f"the {float(minutes):g}-minute limit"
        elif stop == "epochs":
            why = f"the {int(epochs)}-pass limit"
        else:
            why = "a loss that became NaN or infinite (try a smaller learning rate)"
        print(f"{hist.kind.upper()} trained {hist.passes:.1f} passes ({hist.steps:,} steps) in "
              f"{_clock(hist.seconds)} on {device.type}; stopped by {why}.")
    return hist


def _limits(kind: str, minutes, epochs) -> tuple:
    if minutes is None and epochs is None:
        minutes = CLASS_MINUTES[kind]
    if minutes is not None and float(minutes) <= 0:
        raise ValueError("minutes must be positive.")
    if epochs is not None and int(epochs) <= 0:
        raise ValueError("epochs must be a positive whole number.")
    return minutes, epochs


def _watch_roofs(images: torch.Tensor, watch, seed: int) -> torch.Tensor:
    if watch is None:
        idx = np.sort(np.random.default_rng(seed).choice(len(images), min(8, len(images)), replace=False))
        watch = images[torch.from_numpy(idx).to(images.device)]
    if watch.dtype != torch.uint8:
        raise TypeError("watch must be uint8 roof images, for example test.images[:8].")
    return watch.detach().cpu()


def _base_settings(kind, images, minutes, epochs, batch_size, seed, device, **extra) -> dict:
    return {"kind": kind, "roofs": len(images), "minutes": minutes, "epochs": epochs, "batch_size": int(batch_size),
            "seed": int(seed), "device": device.type, **extra}


def train_ae(model, images: torch.Tensor, minutes: float | None = None, epochs: int | None = None,
             batch_size: int = 128, lr: float = 1e-3, seed: int = 7, device=None, watch=None,
             snapshots: int = 8, verbose: bool = True) -> History:
    """Train a RoofAE on uint8 roof images. Adam, learning rate 1e-3; loss = reconstruction_loss.

    watch: up to 16 uint8 roofs to picture as training goes (default: 8 picked from `images`).
    The model ends in eval mode, ready to use.
    """
    minutes, epochs = _limits("ae", minutes, epochs)
    device = torch.device(device) if device is not None else pick_device()
    set_seed(seed)
    model.to(device).train()
    watch = _watch_roofs(images, watch, seed)
    watch_x = to_float(watch.to(device))
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    def step(x):
        loss = reconstruction_loss(model(x), x)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        return {"loss": loss.item()}

    @torch.no_grad()
    def picture():
        model.eval()
        out = to_bytes(model(watch_x)).cpu()
        model.train()
        return out

    hist = History("ae", _base_settings("ae", images, minutes, epochs, batch_size, seed, device, lr=lr,
                                        model=model.settings(), parameters=count_parameters(model)),
                   {"loss": []}, watch)
    _run(hist, step, picture, images, minutes=minutes, epochs=epochs, batch_size=batch_size, seed=seed,
         device=device, snapshots=snapshots, verbose=verbose)
    model.eval()
    return hist


def train_vae(model, images: torch.Tensor, minutes: float | None = None, epochs: int | None = None,
              batch_size: int = 128, lr: float = 1e-3, beta: float = 1.0, seed: int = 7, device=None, watch=None,
              snapshots: int = 8, verbose: bool = True) -> History:
    """Train a RoofVAE. Adam, learning rate 1e-3; loss = reconstruction + beta * KL (beta = 1 by default).

    The history records the total ("loss") and both parts ("recon", "kl"). Pictures decode the mean mu, so they
    show the model, not the random draw.
    """
    minutes, epochs = _limits("vae", minutes, epochs)
    device = torch.device(device) if device is not None else pick_device()
    set_seed(seed)
    model.to(device).train()
    watch = _watch_roofs(images, watch, seed)
    watch_x = to_float(watch.to(device))
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    def step(x):
        copy, mu, logvar = model(x)
        total, rec, kl = vae_loss(copy, x, mu, logvar, beta)
        opt.zero_grad(set_to_none=True)
        total.backward()
        opt.step()
        return {"loss": total.item(), "recon": rec.item(), "kl": kl.item()}

    @torch.no_grad()
    def picture():
        model.eval()
        out = to_bytes(model.decode(model.encode(watch_x)[0])).cpu()
        model.train()
        return out

    hist = History("vae", _base_settings("vae", images, minutes, epochs, batch_size, seed, device, lr=lr,
                                         beta=float(beta), model=model.settings(),
                                         parameters=count_parameters(model)),
                   {"loss": [], "recon": [], "kl": []}, watch)
    _run(hist, step, picture, images, minutes=minutes, epochs=epochs, batch_size=batch_size, seed=seed,
         device=device, snapshots=snapshots, verbose=verbose)
    model.eval()
    return hist


def train_gan(generator, critic, images: torch.Tensor, minutes: float | None = None, epochs: int | None = None,
              batch_size: int = 128, lr_g: float = 2e-4, lr_d: float = 2e-4, betas: tuple = (0.5, 0.999),
              label_smooth: float = 0.9, seed: int = 7, device=None, watch_n: int = 16, snapshots: int = 8,
              verbose: bool = True) -> History:
    """Train a Generator against a Discriminator (the critic). Adam 2e-4 with betas (0.5, 0.999) for both.

    Each step: the critic learns to score real roofs `label_smooth` (0.9) and generated roofs 0, then the generator
    learns to make the critic say 1. The history records both losses and the critic's average probability for
    real ("d_real") and generated ("d_fake") roofs. Pictures use the same `watch_n` noise vectors throughout.
    """
    minutes, epochs = _limits("gan", minutes, epochs)
    device = torch.device(device) if device is not None else pick_device()
    set_seed(seed)
    generator.to(device).train()
    critic.to(device).train()
    noise = generator.noise(watch_n, seed=seed)
    opt_g = torch.optim.Adam(generator.parameters(), lr=lr_g, betas=tuple(betas))
    opt_d = torch.optim.Adam(critic.parameters(), lr=lr_d, betas=tuple(betas))

    def step(x):
        fake = generator(torch.randn(x.shape[0], generator.z_dim, device=device))
        real_scores, fake_scores = critic(x), critic(fake.detach())
        loss_d = critic_loss(real_scores, fake_scores, label_smooth)
        opt_d.zero_grad(set_to_none=True)
        loss_d.backward()
        opt_d.step()
        loss_g = generator_loss(critic(fake))
        opt_g.zero_grad(set_to_none=True)
        loss_g.backward()
        opt_g.step()
        return {"d_loss": loss_d.item(), "g_loss": loss_g.item(),
                "d_real": torch.sigmoid(real_scores).mean().item(), "d_fake": torch.sigmoid(fake_scores).mean().item()}

    @torch.no_grad()
    def picture():
        generator.eval()
        out = to_bytes(generator(noise)).cpu()
        generator.train()
        return out

    hist = History("gan", _base_settings("gan", images, minutes, epochs, batch_size, seed, device, lr_g=lr_g,
                                         lr_d=lr_d, betas=list(betas), label_smooth=float(label_smooth),
                                         generator=generator.settings(), critic=critic.settings(),
                                         parameters=count_parameters(generator) + count_parameters(critic)),
                   {"d_loss": [], "g_loss": [], "d_real": [], "d_fake": []}, noise.detach().cpu())
    _run(hist, step, picture, images, minutes=minutes, epochs=epochs, batch_size=batch_size, seed=seed,
         device=device, snapshots=snapshots, verbose=verbose)
    generator.eval()
    critic.eval()
    return hist


# -------------------------------------------------------------------------------------------------- weights

def save_weights(model, path, history: History | None = None, notes: dict | None = None) -> Path:
    """Save a model as .safetensors (plain numbers, no pickled code, so loading can't run anything) plus a small
    .json note beside it: model class, settings, parameter count, versions, and any notes or training summary."""
    from safetensors.torch import save_file
    if type(model).__name__ not in MODELS:
        raise TypeError(f"save_weights saves the genroofs models ({', '.join(MODELS)}), not {type(model).__name__}.")
    path = Path(path)
    if path.suffix != ".safetensors":
        path = path.with_name(path.name + ".safetensors")
    note = {"class": type(model).__name__, "settings": model.settings(), "parameters": count_parameters(model),
            "torch": torch.__version__, "genroofs": __version__, "notes": dict(notes or {})}
    if history is not None:
        note["training"] = history.summary()
    try:
        text = json.dumps(note)
    except TypeError as err:
        raise TypeError(f"notes must hold plain values (text, numbers, lists, dicts): {err}") from None
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    save_file(state, str(path), metadata={"genroofs": text})
    path.with_suffix(".json").write_text(json.dumps(note, indent=2) + "\n")
    return path


def weights_note(path) -> dict:
    """The note stored inside a weights file (class, settings, versions, notes, training summary)."""
    from safetensors import safe_open
    try:
        with safe_open(str(path), framework="pt") as fh:
            raw = (fh.metadata() or {}).get("genroofs")
    except Exception as err:      # safetensors raises its own error types for damaged files
        raise ValueError(f"{Path(path).name} is not a readable .safetensors file ({err}).") from None
    if raw is None:
        raise ValueError(f"{Path(path).name} was not saved by genroofs.save_weights (no genroofs note inside).")
    return json.loads(raw)


def load_weights(path, model=None, device=None):
    """Load saved weights. With no model, the right class is rebuilt from the note inside the file.
    The model comes back in eval mode on `device` (default: the best device here)."""
    from safetensors.torch import load_file
    path = Path(path)
    note = weights_note(path)
    cls = MODELS.get(note["class"])
    if cls is None:
        raise ValueError(f"{path.name} holds an unknown model class {note['class']!r}.")
    if model is None:
        model = cls(**note["settings"])
    elif type(model).__name__ != note["class"]:
        raise ValueError(f"{path.name} holds a {note['class']}, not a {type(model).__name__}.")
    try:
        model.load_state_dict(load_file(str(path)), strict=True)
    except RuntimeError:
        raise ValueError(f"{path.name} does not fit this {note['class']}: the file was saved with settings "
                         f"{note['settings']}, this model has {model.settings()}.") from None
    model.to(torch.device(device) if device is not None else pick_device()).eval()
    return model


class WeightsNotPublished(PackError):
    """The trained weights asked for are not on release models-v1 yet."""


def _weights_info() -> dict:
    global _WEIGHTS_INFO
    if _WEIGHTS_INFO is None:
        _WEIGHTS_INFO = json.loads(resources.files("genroofs").joinpath("weights_info.json")
                                   .read_text(encoding="utf-8"))
    return _WEIGHTS_INFO


def load_pretrained(kind: str, cache=None, device=None, quiet: bool = False):
    """Weights from the long training runs before the lecture (release models-v1): "ae", "vae" or "gan"
    (the GAN's generator). The download is checked against its recorded size and SHA-256, like the packs."""
    if kind not in CLASS_MINUTES:
        raise ValueError(f"kind must be one of {', '.join(CLASS_MINUTES)}, not {kind!r}.")
    entry = _weights_info().get(kind)
    if entry is None:
        raise WeightsNotPublished(f"Pretrained {kind} weights are not published yet: they come from the long "
                                  f"training runs before the lecture (release {WEIGHTS_TAG}). For now, train your "
                                  f"own with train_{kind}(...).")
    folder = (Path(cache).expanduser() if cache is not None else cache_dir()) / "weights"
    path = fetch_checked(entry, WEIGHTS_URL.format(file=entry["file"]), folder, WEIGHTS_TAG, quiet)
    return load_weights(path, device=device)
