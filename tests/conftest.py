"""Shared test fixtures.

- Any real network call fails loudly (DNS lookups and socket connects raise), so no test can pass by luck online.
- `fake_pack` builds a tiny roof pack in memory with the same layout as the real zips.
- `fake_env` registers that pack with genroofs.data, points the cache at a temporary folder and serves the zip
  from a fake release server whose behaviour each test can change (corrupt file, dropped connection, 404, ...).
"""
import csv
import hashlib
import io
import json
import os
import socket
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

os.environ.setdefault("MPLBACKEND", "Agg")      # draw pictures off-screen during tests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

FAKE_NAME = "roofs_fake_test"
FAKE_SOURCES = {           # source id: (place, number of roofs); deliberately uneven
    "me_alpha_2022": ("Alpha", 16),
    "me_beta_2022": ("Beta", 12),
    "me_gamma_2022": ("Gamma", 12),
    "me_delta_2022": ("Delta", 8),
}
FAKE_ANGLES = [-90.0, 90.0, 0.0, 45.5, -45.5, -89.9, 12.3]
MANIFEST_FIELDS = ["chip_id", "file", "source", "region", "place", "year", "area_m2", "long_m", "short_m",
                   "angle_deg", "use"]


def marker_value(i: int) -> int:
    """Grey level painted in the top-left 16x16 corner of fake roof i, so tests can tell images apart."""
    return 20 + 4 * i


def fake_image(i: int) -> Image.Image:
    img = np.zeros((128, 128, 3), np.uint8)
    img[:] = (70, 110, 60)                                   # lawn
    half = 20 + (i % 5) * 6
    img[64 - half // 2:64 + half // 2, 64 - half:64 + half] = (150 + i, 60, 50)   # a roof
    img[0:16, 0:16] = marker_value(i)
    return Image.fromarray(img)


def attribution_text() -> str:
    return "\n".join([
        "Fake test pack: roofs drawn by the test suite",
        "Built for an educational guest lecture on generative models.",
        "",
        "Aerial imagery",
        "  Imagery: Test Imagery Office, Fake Orthoimagery - Alpha 2022",
        "    https://example.invalid/alpha",
        "    Licence: public domain",
        "  Imagery: Test Imagery Office, Fake Orthoimagery - Beta 2022",
        "    https://example.invalid/beta",
        "    Licence: public domain",
        "  Imagery: Second Office, Gamma and Delta 2022",
        "    https://example.invalid/gamma",
        "    Licence: CC BY 4.0",
        "",
        "Building outlines (used to find, turn and centre each roof; NOT included in this pack)",
        "  Fake Outlines Project",
        "    https://example.invalid/outlines",
        "    Licence: CC BY 4.0",
        "",
        "This pack contains no addresses, coordinates, building IDs or owner information.",
        "",
    ])


def build_zip(rows: list, images: dict, name: str = FAKE_NAME, skip=(), manifest_fields=None) -> bytes:
    """Zip bytes laid out like the real packs: <name>/manifest.csv, <name>/images/*.jpg, <name>/ATTRIBUTION.txt."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
        out = io.StringIO()
        writer = csv.DictWriter(out, fieldnames=manifest_fields or MANIFEST_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        z.writestr(f"{name}/manifest.csv", out.getvalue())
        z.writestr(f"{name}/ATTRIBUTION.txt", attribution_text())
        z.writestr(f"{name}/README.txt", "Fake pack for tests.\n")
        for fname, data in images.items():
            if fname not in skip:
                z.writestr(f"{name}/images/{fname}", data)
    return buf.getvalue()


@dataclass
class FakePack:
    name: str
    rows: list
    images: dict                      # file name -> JPEG bytes
    zip_bytes: bytes
    info: dict = field(default_factory=dict)


def _make_fake_pack() -> FakePack:
    rows, images, i = [], {}, 0
    for source, (place, count) in FAKE_SOURCES.items():
        for _ in range(count):
            chip_id = f"{i + 1:06d}"
            fname = f"{chip_id}.jpg"
            buf = io.BytesIO()
            fake_image(i).save(buf, "JPEG", quality=95, subsampling=0)
            images[fname] = buf.getvalue()
            rows.append({"chip_id": chip_id, "file": fname, "source": source, "region": "Maine", "place": place,
                         "year": 2022, "area_m2": 100.0 + i, "long_m": 12.5, "short_m": 8.0,
                         "angle_deg": FAKE_ANGLES[i % len(FAKE_ANGLES)], "use": "RES"})
            i += 1
    zip_bytes = build_zip(rows, images)
    info = {"name": FAKE_NAME, "file": f"{FAKE_NAME}.zip", "bytes": len(zip_bytes),
            "sha256": hashlib.sha256(zip_bytes).hexdigest(), "chips": len(rows), "chip_px": 128,
            "window_m": 24.0, "per_source": {s: c for s, (_, c) in FAKE_SOURCES.items()},
            "built": "2026-11-13", "docs_updated": "2026-11-13"}
    return FakePack(FAKE_NAME, rows, images, zip_bytes, info)


@pytest.fixture(scope="session")
def fake_pack() -> FakePack:
    return _make_fake_pack()


def make_squares(n: int = 32, seed: int = 0):
    """n simple uint8 'roofs' (torch tensor N x 3 x 64 x 64): a coloured square on a plain background."""
    import torch
    rng = np.random.default_rng(seed)
    out = np.zeros((n, 3, 64, 64), np.uint8)
    for i in range(n):
        out[i] = rng.integers(40, 120, size=(3, 1, 1))
        y, x, s = rng.integers(8, 32), rng.integers(8, 32), rng.integers(12, 24)
        out[i, :, y:y + s, x:x + s] = rng.integers(140, 250, size=(3, 1, 1))
    return torch.from_numpy(out)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    if "matplotlib.pyplot" in sys.modules:
        sys.modules["matplotlib.pyplot"].close("all")


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Make every real network call fail loudly."""
    def blocked(*args, **kwargs):
        raise RuntimeError("network access is blocked in tests")
    monkeypatch.setattr(socket, "getaddrinfo", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)


class FakeResponse:
    def __init__(self, body: bytes, length=None, drop_after=None):
        self._buf = io.BytesIO(body)
        self.headers = {} if length is None else {"Content-Length": str(length)}
        self._drop_after = drop_after
        self._sent = 0

    def read(self, n=-1):
        if self._drop_after is not None and self._sent >= self._drop_after:
            raise ConnectionResetError("connection reset by peer (fake)")
        chunk = self._buf.read(n if n and n > 0 else -1)
        if self._drop_after is not None:
            chunk = chunk[: max(0, self._drop_after - self._sent)]
        self._sent += len(chunk)
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeServer:
    """Serves the registered pack. mode: ok, corrupt, wrong_length, drop_once, drop_always, http404, short."""

    def __init__(self, body: bytes):
        self.body = body
        self.mode = "ok"
        self.calls = []

    def open(self, url, timeout, headers=None):
        import urllib.error
        self.calls.append(url)
        self.headers = headers
        n = len(self.body)
        if self.mode == "http404":
            raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)
        if self.mode == "corrupt":
            bad = bytearray(self.body)
            bad[n // 2] ^= 0xFF
            return FakeResponse(bytes(bad), length=n)
        if self.mode == "wrong_length":
            return FakeResponse(self.body + b"extra", length=n + 5)
        if self.mode == "short":          # no Content-Length and the connection ends early
            return FakeResponse(self.body[: n // 2])
        if self.mode == "drop_always" or (self.mode == "drop_once" and len(self.calls) == 1):
            return FakeResponse(self.body, length=n, drop_after=n // 3)
        return FakeResponse(self.body, length=n)


@dataclass
class FakeEnv:
    pack: FakePack
    server: FakeServer
    cache: Path
    data: object

    @property
    def zip_path(self) -> Path:
        return self.cache / self.pack.info["file"]

    def install(self, zip_bytes: bytes, chips: int | None = None) -> None:
        """Replace the served pack (and its recorded size and SHA-256) with other zip bytes."""
        info = dict(self.pack.info, bytes=len(zip_bytes), sha256=hashlib.sha256(zip_bytes).hexdigest())
        if chips is not None:
            info["chips"] = chips
        self.data._PACK_INFO[self.pack.name] = info
        self.server.body = zip_bytes


@pytest.fixture
def fake_env(monkeypatch, tmp_path, fake_pack):
    from genroofs import data
    info = json.loads((ROOT / "src" / "genroofs" / "pack_info.json").read_text())
    info[fake_pack.name] = dict(fake_pack.info)
    monkeypatch.setattr(data, "_PACK_INFO", info)
    monkeypatch.setattr(data, "_ATTRIBUTED", set())
    cache = tmp_path / "cache"
    monkeypatch.setenv("GENROOFS_DATA", str(cache))
    server = FakeServer(fake_pack.zip_bytes)
    monkeypatch.setattr(data, "_open_url", server.open)
    monkeypatch.setattr(data, "_pause", lambda s: None)
    return FakeEnv(fake_pack, server, cache, data)
