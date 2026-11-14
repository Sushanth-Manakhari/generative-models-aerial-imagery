"""Setup self-check. Run it with:   python -m genroofs.check

It prints the Python and torch versions, the device it picked, the cache folder and its free space, and whether
release data-v1 answers, then runs two training steps of a tiny autoencoder. It ends with ALL GOOD or a numbered
list of what to fix. It never prints the machine's user name: paths under your home folder are shown with ~.
"""
import json
import math
import shutil
import sys
import time
from importlib import resources
from pathlib import Path

REQUIRED = [("torch", "torch"), ("numpy", "numpy"), ("PIL", "Pillow"), ("matplotlib", "matplotlib"),
            ("safetensors", "safetensors"), ("certifi", "certifi")]
OPTIONAL = [("pandas", "pandas", "to_pandas()"), ("gradio", "gradio", "the web app")]
MIN_FREE_GB = 1.0


def tidy(path) -> str:
    """A path with the home folder shown as ~ so no user name is printed."""
    path, home = str(path), str(Path.home())
    return "~" + path[len(home):] if home and path.startswith(home) else path


def _version(module: str) -> str | None:
    try:
        mod = __import__(module)
    except Exception:
        return None
    return getattr(mod, "__version__", "installed")


def main(argv=None) -> int:
    problems, out = [], print
    out("genroofs setup check")
    out("-" * 20)

    v = sys.version_info
    out(f"Python      {v.major}.{v.minor}.{v.micro}")
    if (v.major, v.minor) < (3, 10):
        problems.append(f"Python {v.major}.{v.minor} is too old: install Python 3.10 or newer from python.org, "
                        "then run the setup script again.")

    missing = []
    for module, package in REQUIRED:
        ver = _version(module)
        out(f"{package:<11} {ver or 'MISSING'}")
        if ver is None:
            missing.append(package)
    if missing:
        problems.append(f"Missing packages: {', '.join(missing)}. Run the setup script again, or inside the "
                        f"project's .venv: pip install {' '.join(missing)}")
    for module, package, used_for in OPTIONAL:
        ver = _version(module)
        out(f"{package:<11} {ver or 'not installed'} (optional, for {used_for})")

    if "torch" in missing:
        return _end(problems, out)

    from . import __version__
    out(f"genroofs    {__version__}")
    try:
        for name in ("pack_info.json", "weights_info.json"):
            json.loads(resources.files("genroofs").joinpath(name).read_text(encoding="utf-8"))
    except Exception as err:
        problems.append(f"The package is missing its data files ({err}). Reinstall genroofs.")

    import torch
    from .utils import pick_device
    device = pick_device()
    if device.type == "cuda":
        name = torch.cuda.get_device_name(0)
    elif device.type == "mps":
        name = "Apple GPU (mps)"
    else:
        name = f"CPU, {torch.get_num_threads()} threads"
    out(f"Device      {device.type} ({name})")

    from . import data
    try:
        folder = data.cache_dir()
        free_gb = shutil.disk_usage(folder).free / 1e9
        out(f"Cache       {tidy(folder)} ({free_gb:,.1f} GB free)")
        if free_gb < MIN_FREE_GB:
            problems.append(f"Only {free_gb:.1f} GB free for the roof packs; they need about {MIN_FREE_GB:g} GB. "
                            "Free some space or set GENROOFS_DATA to a folder on a bigger disk.")
    except OSError as err:
        problems.append(f"Can't create the cache folder ({err}). Set GENROOFS_DATA to a folder you can write to.")

    info = data.pack_info("roofs_us_variety")
    url = data.RELEASE_URL.format(file=info["file"])
    try:
        with data._open_url(url, 20, headers={"Range": "bytes=0-0"}) as resp:
            resp.read(1)
        out(f"Release     {data.RELEASE_TAG} answers")
    except Exception as err:
        out(f"Release     {data.RELEASE_TAG} did NOT answer")
        problems.append(f"Can't reach the roof packs on GitHub release {data.RELEASE_TAG} "
                        f"({data._reason(err)}). Check the internet connection; on a work or campus network, "
                        "a proxy may need SSL_CERT_FILE set.")

    try:
        from .models import RoofAE
        from .train import reconstruction_loss
        torch.manual_seed(0)
        model = RoofAE(latent=8).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        x = torch.rand(4, 3, 64, 64, device=device) * 2 - 1
        t0, losses = time.perf_counter(), []
        for _ in range(2):
            loss = reconstruction_loss(model(x), x)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        out(f"Training    2 steps of a tiny autoencoder on {device.type} in {time.perf_counter() - t0:.1f} s")
        if not all(math.isfinite(value) for value in losses):
            problems.append("The test training produced NaN. Restart the runtime and run the check again.")
    except Exception as err:
        problems.append(f"The test training failed on {device.type} ({type(err).__name__}: {err}).")

    return _end(problems, out)


def _end(problems: list, out) -> int:
    out("-" * 20)
    if not problems:
        out("ALL GOOD")
        return 0
    out("Fix these, then run the check again:")
    for i, p in enumerate(problems, 1):
        out(f"  {i}. {p}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
