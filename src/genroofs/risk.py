"""After the storm (Part 6, ROOF-305): turn model scores into an inspection order and check it against FEMA's labels.

The storm pack holds roofs photographed by NOAA the day after Hurricane Michael (2018). Each roof carries FEMA's
public damage level for that home: "Affected" (some damage), "Destroyed", or "none recorded". These helpers stay
small on purpose, so every step can be read in a minute:

    damaged(manifest)                 True where FEMA recorded damage
    spread(mask, n)                   n examples spread evenly through the pack (so every town shows up)
    middle_numbers(model, images)     the autoencoder's bottleneck numbers for each roof (its summary of the roof)
    labelled_split(labels, 50)        a few labelled roofs to learn from, the rest to test on
    fit_probe(features, labels)       logistic regression: a small model on top of those numbers
    random_split(n, 0.3)              a random 70 / 30 split: roofs to learn from, roofs to test on
    train_damage_model(images, labels)  a small network trained for the job, the way a GAN's critic is trained
    damage_scores(model, images)      its chance of damage for each roof
    quiz_roofs(labels, 12) / reveal(...)  Exercise 6: be the adjuster, then compare with FEMA and the model
    triage_table(scores, labels)      if adjusters visit the top-k homes first, how many have damage?
    share_by_place(probs, manifest)   the expected share of damaged homes per town, against FEMA's count
    show_queue(images, ...)           the inspection queue as an adjuster would see it

No extra library is needed: everything is plain PyTorch.
"""
import math
import time

import numpy as np
import torch

from torch import nn

from .data import to_float
from .metrics import unusual_score_auc
from .utils import pick_device

__all__ = ["STORM_PACK", "DAMAGED", "damaged", "spread", "middle_numbers", "labelled_split", "Probe", "fit_probe",
           "precision_at_k", "recall_at_k", "triage_table", "share_by_place", "show_queue", "random_split",
           "DamageNet", "train_damage_model", "damage_scores", "quiz_roofs", "reveal"]

STORM_PACK = "roofs_storm_michael_2018"
DAMAGED = ("Affected", "Minor", "Major", "Destroyed")   # FEMA's levels that mean "some damage was recorded"


def damaged(manifest: list) -> torch.Tensor:
    """True for each roof whose home FEMA recorded as damaged (Affected, Minor, Major or Destroyed)."""
    if manifest and "damage" not in manifest[0]:
        raise ValueError("These roofs have no damage labels. Load the storm pack: data.load_roofs(risk.STORM_PACK).")
    return torch.tensor([row["damage"] in DAMAGED for row in manifest], dtype=torch.bool)


def spread(mask, n: int = 8) -> torch.Tensor:
    """Positions of n roofs where mask is True, spread evenly through the pack (the pack is ordered by town)."""
    idx = torch.nonzero(torch.as_tensor(mask, dtype=torch.bool)).flatten()
    if len(idx) == 0:
        return idx
    return idx[torch.linspace(0, len(idx) - 1, min(int(n), len(idx))).round().long()]


@torch.no_grad()
def middle_numbers(model, images: torch.Tensor, batch_size: int = 256) -> torch.Tensor:
    """The bottleneck numbers (latent code) of each roof: N x latent floats on the CPU.

    For a VAE this is the centre of each roof's range (mu), so the result has no random part."""
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    out = []
    try:
        for i in range(0, len(images), batch_size):
            x = images[i:i + batch_size].to(device)
            x = to_float(x) if x.dtype == torch.uint8 else x.float()
            z = model.encode(x)
            out.append((z[0] if isinstance(z, tuple) else z).float().cpu())
    finally:
        model.train(was_training)
    return torch.cat(out) if out else torch.zeros(0)


def labelled_split(labels, per_class: int = 50, seed: int = 7) -> tuple:
    """(learn, test): positions of `per_class` damaged and `per_class` undamaged roofs to learn from, and of every
    other roof to test on. Like asking adjusters to label a small sample before trusting a model on the rest."""
    labels = torch.as_tensor(labels, dtype=torch.bool)
    gen = torch.Generator().manual_seed(int(seed))
    learn = []
    for value in (True, False):
        idx = torch.nonzero(labels == value).flatten()
        if len(idx) < per_class + 1:
            raise ValueError(f"Only {len(idx)} roofs with damaged={value}; choose a smaller per_class.")
        learn.append(idx[torch.randperm(len(idx), generator=gen)[:per_class]])
    learn = torch.cat(learn).sort().values
    test = torch.ones(len(labels), dtype=torch.bool)
    test[learn] = False
    return learn, torch.nonzero(test).flatten()


class Probe:
    """A fitted logistic regression: p = sigmoid(w . z + b), with z standardised by the learning roofs."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor, w: torch.Tensor, b: torch.Tensor):
        self.mean, self.std, self.w, self.b = mean, std, w, b

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        """Chance of damage for each roof, between 0 and 1."""
        z = (torch.as_tensor(features, dtype=torch.float32) - self.mean) / self.std
        return torch.sigmoid(z @ self.w + self.b)


def fit_probe(features, labels, steps: int = 400, l2: float = 0.01, seed: int = 7) -> Probe:
    """Train a logistic regression on a few labelled roofs.

    The loss is the average cross-entropy, -[y log p + (1 - y) log(1 - p)], plus l2 times the squared weights,
    a small penalty that keeps the model from memorising 100 roofs. A few hundred full-batch steps take well under
    a second."""
    x = torch.as_tensor(features, dtype=torch.float32)
    y = torch.as_tensor(labels, dtype=torch.float32)
    if len(x) != len(y) or y.min() == y.max():
        raise ValueError("fit_probe needs one label per roof, with both damaged and undamaged roofs.")
    mean, std = x.mean(0), x.std(0).clamp_min(1e-6)
    z = (x - mean) / std
    torch.manual_seed(int(seed))
    w = torch.zeros(z.shape[1], requires_grad=True)
    b = torch.zeros((), requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=0.5, max_iter=int(steps), line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z @ w + b, y) + l2 * w.pow(2).sum()
        loss.backward()
        return loss

    with torch.enable_grad():
        opt.step(closure)
    return Probe(mean, std, w.detach(), b.detach())


def _order(scores) -> np.ndarray:
    s = np.asarray(torch.as_tensor(scores, dtype=torch.float64)).ravel()
    return np.argsort(-s, kind="mergesort")          # highest first; ties keep their order


def precision_at_k(scores, labels, k: int) -> float:
    """Of the k homes with the highest scores, the share where FEMA recorded damage."""
    lab = np.asarray(torch.as_tensor(labels, dtype=torch.bool)).ravel()
    k = min(int(k), len(lab))
    return float(lab[_order(scores)[:k]].mean()) if k > 0 else float("nan")


def recall_at_k(scores, labels, k: int) -> float:
    """Of all damaged homes, the share found among the k highest scores."""
    lab = np.asarray(torch.as_tensor(labels, dtype=torch.bool)).ravel()
    total = int(lab.sum())
    return float(lab[_order(scores)[:int(k)]].sum() / total) if total else float("nan")


def triage_table(scores, labels, ks=(25, 50, 100), name: str = "score", show: bool = True) -> dict:
    """How good is this score as an inspection order? Prints (show=True) and returns a summary.

    precision@k: of the first k homes visited, the share with recorded damage.
    base rate:   the share with damage overall, what visiting homes in random order would find.
    lift:        precision@k divided by the base rate (2.0 = twice as many damaged homes per visit as random).
    recall@k:    the share of all damaged homes found in the first k visits.
    AUC:         the chance that a random damaged home scores above a random undamaged one (0.5 = coin toss)."""
    lab = torch.as_tensor(labels, dtype=torch.bool)
    s = torch.as_tensor(scores, dtype=torch.float64)
    base = float(lab.float().mean())
    auc = unusual_score_auc(s[~lab], s[lab])
    rows = []
    for k in ks:
        p = precision_at_k(s, lab, k)
        rows.append({"k": int(k), "precision": p, "lift": p / base if base else float("nan"),
                     "recall": recall_at_k(s, lab, k)})
    out = {"name": name, "roofs": len(lab), "damaged": int(lab.sum()), "base_rate": base, "auc": auc, "rows": rows}
    if show:
        print(f"{name}: {len(lab):,} roofs, {int(lab.sum()):,} with damage recorded (base rate {base:.0%}), "
              f"AUC {auc:.2f}")
        print(f"   {'visit first':>12}  {'damaged':>8}  {'lift':>5}  {'found':>6}")
        for r in rows:
            print(f"   {r['k']:>6} homes  {r['precision']:>8.0%}  {r['lift']:>5.1f}  {r['recall']:>6.0%}")
    return out


def share_by_place(probs, manifest: list, labels=None, show: bool = True) -> list:
    """Expected share of damaged homes per town (the average chance of damage), next to FEMA's share.

    After a storm an insurer needs this before any claim is filed: how many adjusters to send where, and how much
    money to set aside."""
    p = torch.as_tensor(probs, dtype=torch.float64)
    lab = torch.as_tensor(labels, dtype=torch.bool) if labels is not None else None
    rows = []
    for place in sorted({row["place"] for row in manifest}):
        idx = torch.tensor([i for i, row in enumerate(manifest) if row["place"] == place])
        r = {"place": place, "roofs": len(idx), "expected": float(p[idx].mean())}
        if lab is not None:
            r["fema"] = float(lab[idx].float().mean())
        rows.append(r)
    if show:
        print(f"   {'town':<22}{'roofs':>6}  {'model expects':>13}" + ("  FEMA recorded" if lab is not None else ""))
        for r in rows:
            print(f"   {r['place']:<22}{r['roofs']:>6}  {r['expected']:>13.0%}"
                  + (f"  {r['fema']:>13.0%}" if lab is not None else ""))
    return rows


def show_queue(images, labels, scores, k: int = 12, ncol: int = 6, title: str | None = None, size: float = 1.8,
               show: bool = True):
    """The k highest-scoring roofs in visiting order. A red frame means FEMA recorded damage; grey means none
    recorded. Each label shows the visit number and the score."""
    from .viz import _finish, _hwc, _plt
    plt = _plt()
    order = _order(scores)[:int(k)]
    lab = torch.as_tensor(labels, dtype=torch.bool)
    s = torch.as_tensor(scores, dtype=torch.float64)
    imgs = _hwc(torch.as_tensor(images)[torch.as_tensor(order)])
    n = len(order)
    ncol = max(1, min(ncol, n))
    nrow = math.ceil(n / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(size * ncol, (size + 0.3) * nrow + 0.6), squeeze=False)
    for i, ax in enumerate(axes.flat):
        ax.set_xticks([])
        ax.set_yticks([])
        if i >= n:
            ax.axis("off")
            continue
        j = int(order[i])
        hit = bool(lab[j])
        ax.imshow(imgs[i])
        for spine in ax.spines.values():
            spine.set_edgecolor("#c0392b" if hit else "#9e9e9e")
            spine.set_linewidth(4 if hit else 2)
        ax.set_title(f"#{i + 1} · {float(s[j]):.2f}" + (" · damage" if hit else ""), fontsize=8,
                     color="#c0392b" if hit else "#555555")
    found = int(lab[torch.as_tensor(order)].sum())
    fig.suptitle(title or f"Inspection queue: the first {n} visits ({found} with damage recorded by FEMA)")
    fig.tight_layout()
    return _finish(fig, show)


def random_split(n: int, test_share: float = 0.3, seed: int = 0) -> tuple:
    """(learn, test): a random split of n roofs, test_share of them kept aside to test on."""
    perm = torch.randperm(int(n), generator=torch.Generator().manual_seed(int(seed)))
    k = int(round(n * test_share))
    return perm[k:].sort().values, perm[:k].sort().values


class DamageNet(nn.Module):
    """A small convolutional network for 128-pixel roofs: four layers that each halve the picture and learn what to
    look for, an average over the whole roof (128 numbers), then the same last step as logistic regression and as
    the GAN's critic, w . h + b. It returns that number before the sigmoid (a 'logit')."""

    def __init__(self):
        super().__init__()
        def block(n_in, n_out, norm=True):
            return [nn.Conv2d(n_in, n_out, 3, stride=2, padding=1)] + ([nn.BatchNorm2d(n_out)] if norm else []) + \
                   [nn.ReLU()]
        self.features = nn.Sequential(*block(3, 32, norm=False), *block(32, 64), *block(64, 64), *block(64, 128),
                                      nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3))
        self.head = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x)).squeeze(1)


def _as_float(images: torch.Tensor) -> torch.Tensor:
    return to_float(images) if images.dtype == torch.uint8 else images.float()


def train_damage_model(images, labels, epochs: int = 20, batch_size: int = 64, lr: float = 1e-3, seed: int = 7,
                       device=None, quiet: bool = False) -> DamageNet:
    """Train DamageNet on labelled roofs (uint8 N x 3 x 128 x 128) with the binary cross-entropy loss,
    -[y log p + (1 - y) log(1 - p)]: the loss of the GAN's critic, with 'damaged' in place of 'real'.
    Each pass shows every roof once, flipped and turned at random (a roof seen from above has no 'up').
    About 20 seconds on a T4 for 1,400 roofs; a few minutes on a laptop CPU."""
    x = torch.as_tensor(images)
    y = torch.as_tensor(labels, dtype=torch.float32)
    if x.ndim != 4 or x.shape[1] != 3 or len(x) != len(y):
        raise ValueError("train_damage_model needs N x 3 x H x W roofs and one label per roof.")
    device = torch.device(device) if device is not None else pick_device()
    torch.manual_seed(int(seed))
    gen = torch.Generator().manual_seed(int(seed))
    net = DamageNet().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-3)
    steps = int(epochs) * ((len(x) + batch_size - 1) // batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.2)  # warm up, then ease off
    t0 = time.perf_counter()
    for epoch in range(1, int(epochs) + 1):
        net.train()
        order, total = torch.randperm(len(x), generator=gen), 0.0
        for i in range(0, len(x), batch_size):
            idx = order[i:i + batch_size]
            xb, yb = _as_float(x[idx].to(device)), y[idx].to(device)
            if torch.rand(1, generator=gen) < 0.5:
                xb = xb.flip(3)
            if torch.rand(1, generator=gen) < 0.5:
                xb = xb.flip(2)
            xb = torch.rot90(xb, int(torch.randint(0, 4, (1,), generator=gen)), (2, 3))
            loss = nn.functional.binary_cross_entropy_with_logits(net(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(idx)
        if not quiet and (epoch % 5 == 0 or epoch == epochs):
            print(f"   pass {epoch:2d}/{epochs}: average loss {total / len(x):.3f}  ({time.perf_counter() - t0:.0f} s)")
    return net.eval()


@torch.no_grad()
def damage_scores(model: nn.Module, images, batch_size: int = 256) -> torch.Tensor:
    """The model's chance of damage for each roof, between 0 and 1, on the CPU."""
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    x = torch.as_tensor(images)
    out = [torch.sigmoid(model(_as_float(x[i:i + batch_size].to(device)))).cpu() for i in range(0, len(x), batch_size)]
    model.train(was_training)
    return torch.cat(out) if out else torch.zeros(0)


def quiz_roofs(labels, n: int = 12, seed: int = 11) -> torch.Tensor:
    """Positions of n roofs for Exercise 6, half with damage recorded and half without, in a mixed order."""
    lab = torch.as_tensor(labels, dtype=torch.bool)
    gen = torch.Generator().manual_seed(int(seed))
    pick = []
    for value, k in ((True, n // 2), (False, n - n // 2)):
        idx = torch.nonzero(lab == value).flatten()
        pick.append(idx[torch.randperm(len(idx), generator=gen)[:k]])
    pick = torch.cat(pick)
    return pick[torch.randperm(len(pick), generator=gen)]


def reveal(quiz, labels, probs=None, show: bool = True) -> list:
    """FEMA's label (and the model's chance, if given) for each quiz roof, numbered as on the quiz page."""
    lab = torch.as_tensor(labels, dtype=torch.bool)
    rows = []
    for k, i in enumerate(torch.as_tensor(quiz).tolist(), 1):
        row = {"roof": k, "fema": "damage recorded" if bool(lab[i]) else "none recorded"}
        if probs is not None:
            row["model"] = float(torch.as_tensor(probs)[i])
        rows.append(row)
    if show:
        print(f"   {'roof':>4}  {'FEMA':<16}" + ("  model's chance of damage" if probs is not None else ""))
        for r in rows:
            print(f"   {r['roof']:>4}  {r['fema']:<16}" + (f"  {r['model']:>6.0%}" if "model" in r else ""))
    return rows
