"""The numbers the class discusses: how close a copy is, how unusual a roof is, and whether a GAN collapsed or copied.

Images can be uint8 (0..255) or floats in -1..1; every function converts as needed.
"""
import numpy as np
import torch

from .data import to_float

__all__ = ["psnr", "reconstruction_errors", "strangest", "unusual_score_auc", "sample_spread", "nearest_real"]


def _as_float(images: torch.Tensor) -> torch.Tensor:
    return to_float(images) if images.dtype == torch.uint8 else images.float()


def psnr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Peak signal-to-noise ratio of each image pair, in decibels; higher means a closer copy.

    PSNR = 10 * log10(range^2 / MSE), with range = 2 for pixels in -1..1. Identical images give infinity.
    Typical values: about 20 dB looks blurry, about 30 dB is hard to tell apart.
    """
    a, b = _as_float(a), _as_float(b)
    if a.shape != b.shape:
        raise ValueError(f"psnr needs images of the same shape, got {tuple(a.shape)} and {tuple(b.shape)}.")
    mse = (a - b).pow(2).flatten(1).mean(1)
    return 10.0 * torch.log10(4.0 / mse)


def _device_of(model) -> torch.device:
    return next(model.parameters()).device


@torch.no_grad()
def _copies(model, x: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "reparameterize"):          # a VAE: decode the mean, so the score has no random part
        return model.decode(model.encode(x)[0])
    return model(x)


@torch.no_grad()
def reconstruction_errors(model, images: torch.Tensor, batch_size: int = 256) -> torch.Tensor:
    """The autoencoder's "unusual" score for each roof: squared error summed over its pixels (the same scale as
    the training loss). Works with RoofAE and RoofVAE. Returns a float tensor on the CPU, one value per roof."""
    was_training = model.training
    model.eval()
    device = _device_of(model)
    out = []
    try:
        for i in range(0, len(images), batch_size):
            x = _as_float(images[i:i + batch_size].to(device))
            out.append((_copies(model, x) - x).pow(2).flatten(1).sum(1).cpu())
    finally:
        model.train(was_training)
    return torch.cat(out) if out else torch.zeros(0)


def strangest(errors, k: int = 10) -> list:
    """Positions of the k roofs with the highest scores, highest first: an inspection queue."""
    errors = torch.as_tensor(errors, dtype=torch.float64)
    k = min(int(k), len(errors))
    return torch.topk(errors, k).indices.tolist() if k > 0 else []


def _ranks(values: np.ndarray) -> np.ndarray:
    """1-based ranks with ties sharing their average rank."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_vals = values[order]
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def unusual_score_auc(normal_scores, unusual_scores) -> float:
    """The chance that a random unusual roof scores higher than a random normal one (area under the ROC curve).

    0.5 is a coin toss, 1.0 means every unusual roof scores above every normal one. Computed from ranks
    (the Mann-Whitney form) with ties counted as half, so no extra library is needed.
    """
    normal = np.asarray(torch.as_tensor(normal_scores, dtype=torch.float64), dtype=np.float64).ravel()
    unusual = np.asarray(torch.as_tensor(unusual_scores, dtype=torch.float64), dtype=np.float64).ravel()
    if len(normal) == 0 or len(unusual) == 0:
        raise ValueError("Both groups need at least one score.")
    ranks = _ranks(np.concatenate([normal, unusual]))
    r_unusual = ranks[len(normal):].sum()
    n_u, n_n = len(unusual), len(normal)
    return float((r_unusual - n_u * (n_u + 1) / 2.0) / (n_u * n_n))


def _rms_distances(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Root-mean-square pixel difference between every image in a and every image in b (in the -1..1 scale)."""
    return torch.cdist(a.flatten(1), b.flatten(1)) / (a[0].numel() ** 0.5)


@torch.no_grad()
def sample_spread(images: torch.Tensor, max_images: int = 512, seed: int = 0) -> float:
    """Average RMS pixel difference between pairs of roofs. Compare generated roofs with real ones: a sharp drop
    for the generated set means mode collapse, the GAN drawing nearly the same roof again and again."""
    if len(images) < 2:
        raise ValueError("sample_spread needs at least two images.")
    if len(images) > max_images:
        idx = torch.from_numpy(np.random.default_rng(seed).choice(len(images), max_images, replace=False))
        images = images[idx.to(images.device)]
    x = _as_float(images)
    d = _rms_distances(x, x)
    n = len(x)
    return float(d.sum() / (n * (n - 1)))


@torch.no_grad()
def nearest_real(fakes: torch.Tensor, train_images: torch.Tensor, chunk: int = 2048) -> tuple:
    """For each generated roof, the closest training roof: returns (positions, distances).

    Distance is the RMS pixel difference in the -1..1 scale. Near zero means the GAN reproduced a real home from
    its training set, which is the memorisation question in the ethics part.
    """
    f = _as_float(fakes)
    best_d = torch.full((len(f),), float("inf"), device=f.device)
    best_i = torch.zeros(len(f), dtype=torch.long, device=f.device)
    for start in range(0, len(train_images), chunk):
        t = _as_float(train_images[start:start + chunk].to(f.device))
        d = _rms_distances(f, t)
        d_min, i_min = d.min(dim=1)
        better = d_min < best_d
        best_d = torch.where(better, d_min, best_d)
        best_i = torch.where(better, i_min + start, best_i)
    return best_i.cpu(), best_d.cpu()
