"""Small helpers shared by every part of the lab: which device to train on, seeds, and a stopwatch."""
import os
import random
import time

import numpy as np
import torch

DEVICE_ORDER = ("cuda", "mps", "cpu")


def _available(kind: str) -> bool:
    if kind == "cuda":
        return torch.cuda.is_available()
    if kind == "mps":
        mps = getattr(torch.backends, "mps", None)
        return bool(mps is not None and mps.is_available())
    return kind == "cpu"


def pick_device(prefer: str | None = None) -> torch.device:
    """The fastest device this machine has: an NVIDIA GPU (cuda), then an Apple GPU (mps), then the CPU.

    prefer: ask for one kind ("cuda", "mps" or "cpu"). If it is not available you get the usual choice and a note.
    """
    if prefer is not None:
        prefer = prefer.lower()
        if prefer not in DEVICE_ORDER:
            raise ValueError(f"prefer must be one of {', '.join(DEVICE_ORDER)}, not {prefer!r}")
        if _available(prefer):
            return torch.device(prefer)
        print(f"Note: {prefer} is not available on this machine, picking the next best device.")
    for kind in DEVICE_ORDER:
        if _available(kind):
            return torch.device(kind)
    return torch.device("cpu")


def set_seed(seed: int = 7) -> int:
    """Seed Python, NumPy and torch so a run can be repeated. Returns the seed.

    GPU runs stay close between repeats but are not bit-identical; that is normal for cuDNN.
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return seed


class Timer:
    """A stopwatch for `with` blocks.

        with Timer() as t:
            ...
        print(t.seconds)
    """

    def __init__(self):
        self.start = None
        self.seconds = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.start
        return False

    @property
    def elapsed(self) -> float:
        """Seconds so far, also while the block is still running."""
        if self.start is None:
            return 0.0
        return time.perf_counter() - self.start
