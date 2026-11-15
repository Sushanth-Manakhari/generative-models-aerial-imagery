"""Roof image packs: download, check, cache and load.

The two packs live on the public GitHub release data-v1 of Sushanth-Manakhari/generative-models-aerial-imagery.
Every download is checked against the size and SHA-256 in pack_info.json before anything uses it, and the
decoded images are kept next to the zip, so the slow decode happens once per machine.

The packs hold no addresses, coordinates or building IDs, and nothing in this module adds any.

    from genroofs.data import load_roofs, class_slice, train_test_split, batches
    images, manifest = load_roofs("roofs_maine_2022", size=64)
    images, manifest = class_slice(images, manifest, n=10_000)
    train, test = train_test_split(images, manifest)
    for x in batches(train.images, batch_size=128):
        ...  # x is a float batch in -1..1, shape 128 x 3 x 64 x 64
"""
import csv
import hashlib
import http.client
import io
import json
import os
import shutil
import ssl
import sys
import time
import urllib.error
import urllib.request
import zipfile
from importlib import resources
from pathlib import Path
from typing import Iterator, NamedTuple

import numpy as np
import torch
from PIL import Image

from . import __version__

__all__ = ["Roofs", "PackError", "pack_info", "available_packs", "cache_dir", "download_pack", "load_roofs",
           "attribution", "normalize_angle", "class_slice", "train_test_split", "to_float", "to_bytes", "batches",
           "to_pandas", "MANIFEST_FIELDS", "CLASS_SIZE", "USER_AGENT"]

REPO = "Sushanth-Manakhari/generative-models-aerial-imagery"
RELEASE_TAG = "data-v1"
RELEASE_URL = f"https://github.com/{REPO}/releases/download/{RELEASE_TAG}/{{file}}"
MANIFEST_FIELDS = ("chip_id", "file", "source", "region", "place", "year", "area_m2", "long_m", "short_m",
                   "angle_deg", "use")
FLOAT_FIELDS = ("area_m2", "long_m", "short_m", "angle_deg")
CLASS_SIZE = 10_000
CHUNK = 1 << 20
_BOX = getattr(Image, "Resampling", Image).BOX

_PACK_INFO = None          # filled from pack_info.json on first use
_ATTRIBUTED = set()        # packs whose credits were already printed in this session


class PackError(RuntimeError):
    """A pack could not be downloaded, checked or read. The message says what to do next."""


class Roofs(NamedTuple):
    """Roof images and their manifest rows, in the same order. Unpacks as `images, manifest = roofs`."""
    images: torch.Tensor   # uint8, N x 3 x size x size
    manifest: list         # one dict per roof


# ----------------------------------------------------------------------------------------------- pack info

def _all_info() -> dict:
    global _PACK_INFO
    if _PACK_INFO is None:
        text = resources.files("genroofs").joinpath("pack_info.json").read_text(encoding="utf-8")
        _PACK_INFO = json.loads(text)
    return _PACK_INFO


def available_packs() -> list:
    """Names of the packs this version of genroofs knows."""
    return sorted(_all_info())


def pack_info(name: str) -> dict:
    """File name, size, SHA-256 and roof counts of one pack (a copy; changing it changes nothing)."""
    info = _all_info()
    if name not in info:
        raise PackError(f"Unknown pack {name!r}. Available packs: {', '.join(available_packs())}.")
    return json.loads(json.dumps(info[name]))


# --------------------------------------------------------------------------------------------- cache folder

def _on_colab() -> bool:
    return "google.colab" in sys.modules or "COLAB_RELEASE_TAG" in os.environ


def _project_root(start: Path | None = None) -> Path | None:
    """The project folder when running inside it (it has packs/*.json and src/genroofs), else None."""
    here = Path(start or Path.cwd()).resolve()
    for folder in (here, *here.parents):
        if (folder / "packs" / "roofs_maine_2022.json").is_file() and (folder / "src" / "genroofs").is_dir():
            return folder
    return None


def cache_dir(create: bool = True) -> Path:
    """Where packs and decoded images are kept. The first match wins:

    1. the GENROOFS_DATA environment variable
    2. /content/genroofs_data on Colab
    3. data/cache inside the project folder, when running there (git ignores data/)
    4. ~/.cache/genroofs
    """
    env = os.environ.get("GENROOFS_DATA", "").strip()
    if env:
        folder = Path(env).expanduser()
    elif _on_colab():
        folder = Path("/content/genroofs_data")
    else:
        root = _project_root()
        folder = root / "data" / "cache" if root is not None else Path.home() / ".cache" / "genroofs"
    if create:
        folder.mkdir(parents=True, exist_ok=True)
    return folder


# ------------------------------------------------------------------------------------------------ download

def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _remove(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _marker(target: Path) -> Path:
    return target.with_name(target.name + ".ok")


def _write_marker(target: Path, info: dict) -> None:
    st = target.stat()
    _marker(target).write_text(json.dumps({"sha256": info["sha256"], "bytes": st.st_size,
                                           "mtime_ns": st.st_mtime_ns}))


def _verified(target: Path, info: dict) -> bool:
    """True when the file is the expected pack. A small .ok note next to it saves re-hashing every time."""
    st = target.stat()
    if st.st_size != info["bytes"]:
        return False
    try:
        note = json.loads(_marker(target).read_text())
        if note == {"sha256": info["sha256"], "bytes": st.st_size, "mtime_ns": st.st_mtime_ns}:
            return True
    except (OSError, ValueError):
        pass
    if sha256_of(target) != info["sha256"]:
        return False
    _write_marker(target, info)
    return True


def _check_space(folder: Path, need: int, what: str) -> None:
    free = shutil.disk_usage(folder).free
    if free < need * 1.05:
        raise PackError(f"Not enough free disk space in {folder} for {what}: need about {need / 1e6:,.0f} MB, "
                        f"{free / 1e6:,.0f} MB free. Free some space or set GENROOFS_DATA to another folder.")


USER_AGENT = f"genroofs/{__version__} (educational lecture; +https://github.com/{REPO})"


def _open_url(url: str, timeout: float, headers: dict | None = None):
    """Open a URL with a certificate bundle that works on Mac, Windows and Colab (certifi when present).
    Every request says what it is (User-Agent), as public services such as OpenStreetMap ask."""
    cafile = os.environ.get("SSL_CERT_FILE") or None
    if cafile is None:
        try:
            import certifi
            cafile = certifi.where()
        except ImportError:
            cafile = None
    ctx = ssl.create_default_context(cafile=cafile)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
    return urllib.request.urlopen(req, timeout=timeout, context=ctx)


def _pause(seconds: float) -> None:
    time.sleep(seconds)


class _ShortRead(OSError):
    """The connection closed before the whole file arrived."""


def _fetch(url: str, dest: Path, expected: int, quiet: bool, tag: str = RELEASE_TAG, tries: int = 3,
           timeout: float = 60) -> None:
    """Download url to dest, retrying a dropped connection. Raises PackError when it can't."""
    last = None
    for attempt in range(1, tries + 1):
        try:
            with _open_url(url, timeout) as resp, open(dest, "wb") as out:
                length = resp.headers.get("Content-Length")
                if length is not None and int(length) != expected:
                    raise PackError(f"The file on release {tag} is {int(length):,} bytes, but this version "
                                    f"of genroofs expects {expected:,}. Update genroofs (re-run the install cell) "
                                    f"and try again.")
                got, report_at, t0 = 0, 0.1, time.time()
                while True:
                    chunk = resp.read(CHUNK)
                    if not chunk:
                        break
                    out.write(chunk)
                    got += len(chunk)
                    if not quiet and got >= report_at * expected:
                        rate = got / max(time.time() - t0, 1e-6) / 1e6
                        print(f"\r   {got / 1e6:6.1f} of {expected / 1e6:.1f} MB  ({rate:.1f} MB/s)", end="",
                              flush=True)
                        report_at += 0.1
                if not quiet:
                    print()
                if got != expected:
                    raise _ShortRead(f"received {got:,} of {expected:,} bytes")
            return
        except (OSError, http.client.HTTPException) as err:   # URLError, HTTPError, resets, timeouts
            last = err
            if isinstance(err, urllib.error.HTTPError) and err.code in (403, 404):
                break
            if attempt < tries:
                if not quiet:
                    print(f"   connection problem ({_reason(err)}); trying again ({attempt + 1} of {tries})")
                _pause(2 * attempt)
    raise PackError(f"Could not download {dest.name.removesuffix('.part')} ({_reason(last)}). Check the internet "
                    f"connection and run the cell again.")


def _reason(err) -> str:
    if isinstance(err, urllib.error.HTTPError):
        return f"HTTP {err.code}"
    if isinstance(err, urllib.error.URLError):
        return str(err.reason)
    return str(err) or type(err).__name__


def download_pack(name: str = "roofs_maine_2022", cache: str | Path | None = None, quiet: bool = False) -> Path:
    """Fetch a pack from release data-v1 (once) and return the path of its checked zip.

    A file whose size or SHA-256 doesn't match is deleted, never used. A half-finished download is written
    under a temporary .part name and removed if anything goes wrong.
    """
    info = pack_info(name)
    folder = Path(cache).expanduser() if cache is not None else cache_dir()
    return fetch_checked(info, RELEASE_URL.format(file=info["file"]), folder, RELEASE_TAG, quiet)


def fetch_checked(info: dict, url: str, folder: Path, tag: str, quiet: bool = False) -> Path:
    """Download url into folder as info['file'] unless a copy with the right size and SHA-256 is already there.

    info needs 'file', 'bytes' and 'sha256'. Used for the packs (release data-v1) and the trained weights
    (release models-v1). Returns the checked file's path.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / info["file"]
    if target.exists():
        if _verified(target, info):
            return target
        if not quiet:
            print(f"{target.name} in {folder} does not match its checksum; downloading it again.")
        _remove(target)
        _remove(_marker(target))
    _check_space(folder, info["bytes"], info["file"])
    part = target.with_name(target.name + ".part")
    if not quiet:
        print(f"Downloading {info['file']} ({info['bytes'] / 1e6:.1f} MB) from release {tag} into {folder}")
    try:
        _fetch(url, part, info["bytes"], quiet, tag=tag)
        if sha256_of(part) != info["sha256"]:
            raise PackError(f"{info['file']} arrived, but its SHA-256 does not match the one recorded for "
                            f"release {tag}, so it was deleted. Run the cell again; if it happens again, "
                            f"tell the instructor.")
        os.replace(part, target)
        _write_marker(target, info)
    finally:
        _remove(part)
    return target


# ---------------------------------------------------------------------------------------------- manifest

def normalize_angle(angle: float) -> float:
    """A roof turned -90 degrees looks the same as one turned +90: keep every angle in (-90, 90]."""
    a = float(angle)
    while a <= -90.0:
        a += 180.0
    while a > 90.0:
        a -= 180.0
    return a


def _clean_row(row: dict) -> dict:
    out = {k: row[k] for k in MANIFEST_FIELDS}
    for k in FLOAT_FIELDS:
        out[k] = float(out[k])
    out["year"] = int(out["year"])
    out["angle_deg"] = normalize_angle(out["angle_deg"])
    return out


def _read_manifest(zf: zipfile.ZipFile, name: str) -> list:
    try:
        text = zf.read(f"{name}/manifest.csv").decode("utf-8")
    except KeyError:
        raise PackError(f"{name}: manifest.csv is missing from the zip. Delete the zip and download it again.")
    reader = csv.DictReader(io.StringIO(text))
    if tuple(reader.fieldnames or ()) != MANIFEST_FIELDS:
        raise PackError(f"{name}: manifest.csv has unexpected columns {reader.fieldnames}.")
    return [_clean_row(r) for r in reader]


def _credit_line(name: str, text: str) -> str:
    """One short credit from the pack's ATTRIBUTION.txt: title, imagery publishers, building outlines."""
    lines = [ln.strip() for ln in text.splitlines()]
    title = next((ln for ln in lines if ln), name)
    publishers = []
    for ln in lines:
        if ln.startswith("Imagery:"):
            who = ln[len("Imagery:"):].split(",")[0].strip()
            if who and who not in publishers:
                publishers.append(who)
    outlines = ""
    for i, ln in enumerate(lines):
        if ln.startswith("Building outlines"):
            outlines = next((x for x in lines[i + 1:] if x), "")
            break
    parts = [f"{name}: {title}"]
    if publishers:
        parts.append(f"Imagery: {'; '.join(publishers)}.")
    if outlines:
        parts.append(f"Building outlines (used only to find each roof, not included): {outlines}.")
    parts.append("Full credits and licences: ATTRIBUTION.txt inside the pack.")
    return "\n".join(parts)


def _attribution_text(zf: zipfile.ZipFile, name: str) -> str:
    try:
        return zf.read(f"{name}/ATTRIBUTION.txt").decode("utf-8")
    except KeyError:
        raise PackError(f"{name}: ATTRIBUTION.txt is missing from the zip. Delete the zip and download it again.")


def attribution(name: str = "roofs_maine_2022", cache: str | Path | None = None) -> str:
    """The full sources-and-licences text that ships inside the pack."""
    with zipfile.ZipFile(download_pack(name, cache, quiet=True)) as zf:
        return _attribution_text(zf, name)


# ---------------------------------------------------------------------------------------------- decoding

def _decode_all(zf: zipfile.ZipFile, name: str, manifest: list, size: int, quiet: bool) -> np.ndarray:
    n = len(manifest)
    out = np.empty((n, 3, size, size), np.uint8)
    t0, step = time.time(), max(1, n // 10)
    for i, row in enumerate(manifest):
        try:
            data = zf.read(f"{name}/images/{row['file']}")
        except KeyError:
            raise PackError(f"{name}: image {row['file']} is listed in the manifest but missing from the zip.")
        with Image.open(io.BytesIO(data)) as im:
            im = im.convert("RGB")
            if im.size != (size, size):
                im = im.resize((size, size), _BOX)
            out[i] = np.asarray(im, dtype=np.uint8).transpose(2, 0, 1)
        if not quiet and ((i + 1) % step == 0 or i + 1 == n):
            print(f"\r   decoded {i + 1:,} of {n:,} roofs ({time.time() - t0:.0f} s)", end="", flush=True)
    if not quiet:
        print()
    return out


def _decoded(zpath: Path, zf: zipfile.ZipFile, info: dict, manifest: list, size: int, quiet: bool) -> np.ndarray:
    """Decoded images at this size, read from the .npy kept next to the zip or decoded once and saved."""
    npy = zpath.with_name(f"{info['name']}_{size}px_{info['sha256'][:12]}.npy")
    shape = (len(manifest), 3, size, size)
    if npy.exists():
        try:
            arr = np.load(npy, allow_pickle=False)
            if arr.shape == shape and arr.dtype == np.uint8:
                return arr
        except (OSError, ValueError):
            pass
        if not quiet:
            print(f"{npy.name} is damaged; decoding again.")
        _remove(npy)
    if not quiet:
        print(f"Decoding {info['name']} to {size}x{size} (once per machine; later loads take seconds)")
    arr = _decode_all(zf, info["name"], manifest, size, quiet)
    try:
        _check_space(npy.parent, arr.nbytes, npy.name)
        tmp = npy.with_name(npy.name + ".part")
        with open(tmp, "wb") as fh:
            np.save(fh, arr, allow_pickle=False)
        os.replace(tmp, npy)
    except (OSError, PackError) as err:    # a full disk only costs speed next time, never correctness
        _remove(npy.with_name(npy.name + ".part"))
        if not quiet:
            print(f"Note: could not save the decoded copy ({err}); the next load will decode again.")
    return arr


# ------------------------------------------------------------------------------------------------- loading

def _subset(images: torch.Tensor, manifest: list, idx) -> Roofs:
    idx = np.asarray(idx, dtype=np.int64)
    return Roofs(images[torch.from_numpy(idx).to(images.device)], [manifest[i] for i in idx.tolist()])


def _split_args(images, manifest):
    if isinstance(images, Roofs) and manifest is None:
        images, manifest = images
    if manifest is None:
        raise TypeError("Pass the images and the manifest, or one Roofs value from load_roofs.")
    if len(images) != len(manifest):
        raise ValueError(f"{len(images)} images but {len(manifest)} manifest rows; they must match.")
    return images, manifest


def load_roofs(name: str = "roofs_maine_2022", size: int = 64, limit: int | None = None, seed: int = 7,
               cache: str | Path | None = None, quiet: bool = False) -> Roofs:
    """Load a pack as (images, manifest).

    images: uint8 tensor, N x 3 x size x size (bytes, not floats: 22,000 roofs at 64x64 take about 270 MB).
    manifest: one dict per roof in the same order, with the pack's manifest.csv fields.
    size: 8 to 128 pixels; the pack's 128x128 images are shrunk with area averaging.
    limit: keep a random `limit` roofs (picked with `seed`), for a quick run on a laptop.
    """
    info = pack_info(name)
    size = int(size)
    if not 8 <= size <= int(info["chip_px"]):
        raise ValueError(f"size must be between 8 and {info['chip_px']} pixels, not {size}.")
    zpath = download_pack(name, cache, quiet)
    with zipfile.ZipFile(zpath) as zf:
        manifest = _read_manifest(zf, name)
        if len(manifest) != int(info["chips"]):
            raise PackError(f"{name}: the manifest lists {len(manifest):,} roofs, expected {info['chips']:,}.")
        if not quiet and name not in _ATTRIBUTED:
            print(_credit_line(name, _attribution_text(zf, name)))
            _ATTRIBUTED.add(name)
        arr = _decoded(zpath, zf, info, manifest, size, quiet)
    roofs = Roofs(torch.from_numpy(arr), manifest)
    if limit is not None:
        limit = int(limit)
        if limit <= 0:
            raise ValueError("limit must be a positive number of roofs.")
        if limit < len(manifest):
            rng = np.random.default_rng(seed)
            roofs = _subset(roofs.images, manifest, np.sort(rng.choice(len(manifest), limit, replace=False)))
    if not quiet:
        mb = roofs.images.numel() / 1e6
        print(f"Loaded {len(roofs.manifest):,} roofs at {size}x{size} from {name} ({mb:,.0f} MB as bytes).")
    return roofs


def _groups(manifest: list) -> dict:
    groups = {}
    for i, row in enumerate(manifest):
        groups.setdefault(row["source"], []).append(i)
    return {s: groups[s] for s in sorted(groups)}


def _even_quotas(sizes: dict, n: int) -> dict:
    """Split n as evenly as possible over the groups; a group that runs out passes its share on."""
    quotas = {s: 0 for s in sizes}
    still_open, remaining = sorted(sizes), n
    while remaining > 0 and still_open:
        share, extra = divmod(remaining, len(still_open))
        next_open = []
        for i, s in enumerate(still_open):
            take = min(share + (1 if i < extra else 0), sizes[s] - quotas[s])
            quotas[s] += take
            remaining -= take
            if quotas[s] < sizes[s]:
                next_open.append(s)
        still_open = next_open
    return quotas


def class_slice(images, manifest=None, *, n: int = CLASS_SIZE, seed: int = 7) -> Roofs:
    """The same n roofs for everyone in the room, with an even share from each town (manifest 'source')."""
    images, manifest = _split_args(images, manifest)
    n = int(n)
    if n <= 0:
        raise ValueError("n must be a positive number of roofs.")
    if n >= len(manifest):
        return Roofs(images, list(manifest))
    groups = _groups(manifest)
    quotas = _even_quotas({s: len(ix) for s, ix in groups.items()}, n)
    rng = np.random.default_rng(seed)
    chosen = [rng.choice(np.asarray(ix), quotas[s], replace=False) for s, ix in groups.items()]
    return _subset(images, manifest, np.sort(np.concatenate(chosen)))


def train_test_split(images, manifest=None, *, test_frac: float = 0.1, seed: int = 7) -> tuple:
    """Split into (train, test), each a Roofs value. Each town keeps the same share in both parts.

    The split depends only on the manifest and the seed, so it is the same on every run and every machine.
    """
    images, manifest = _split_args(images, manifest)
    if not 0.0 < float(test_frac) < 1.0:
        raise ValueError("test_frac must be between 0 and 1, for example 0.1.")
    rng = np.random.default_rng(seed)
    test = [rng.choice(np.asarray(ix), int(round(len(ix) * test_frac)), replace=False)
            for ix in _groups(manifest).values()]
    test_idx = np.sort(np.concatenate(test)) if test else np.array([], np.int64)
    train_idx = np.setdiff1d(np.arange(len(manifest)), test_idx)
    return _subset(images, manifest, train_idx), _subset(images, manifest, test_idx)


# ---------------------------------------------------------------------------------------------- tensors

def to_float(images: torch.Tensor) -> torch.Tensor:
    """Bytes 0..255 to floats -1..1, the range of the tanh output of the decoder and the generator."""
    if images.dtype != torch.uint8:
        raise TypeError(f"to_float expects uint8 images, got {images.dtype}.")
    return images.float().div_(127.5).sub_(1.0)


def to_bytes(images: torch.Tensor) -> torch.Tensor:
    """Floats -1..1 back to bytes 0..255 for pictures. Values outside -1..1 are clipped."""
    return ((images.detach().float().clamp(-1.0, 1.0) + 1.0) * 127.5).round().to(torch.uint8)


def batches(images: torch.Tensor, batch_size: int = 128, shuffle: bool = True, seed: int | None = None,
            device=None, drop_last: bool = True) -> Iterator[torch.Tensor]:
    """Yield float batches in -1..1, one pass over the images.

    Bytes are moved to `device` one batch at a time and converted there, which keeps GPU memory low.
    With drop_last (the default, as in the G3 timing test) a final short batch is skipped.
    seed: fixes the shuffle order; without it the order follows torch's global seed (see utils.set_seed).
    """
    n, batch_size = len(images), int(batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if drop_last and n < batch_size:
        raise ValueError(f"Only {n} roofs, fewer than one batch of {batch_size}. Use a smaller batch_size.")
    if shuffle:
        order = (torch.randperm(n, generator=torch.Generator().manual_seed(int(seed))) if seed is not None
                 else torch.randperm(n))
    else:
        order = torch.arange(n)
    order = order.to(images.device)
    stop = n - batch_size + 1 if drop_last else n
    for i in range(0, stop, batch_size):
        batch = images[order[i:i + batch_size]]
        if device is not None:
            batch = batch.to(device, non_blocking=True)
        yield to_float(batch)


def to_pandas(manifest: list):
    """The manifest as a pandas DataFrame. pandas is optional: install it with `pip install pandas`."""
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("to_pandas needs pandas, which is optional. Install it with: pip install pandas") from None
    return pd.DataFrame(list(manifest), columns=list(MANIFEST_FIELDS))
