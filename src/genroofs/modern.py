"""Today's open enlargers, for the lecture's "modern fixes" part.

Our autoencoder and VAE blur, and our GAN draws sharp but strange roofs. Two free, open "super-resolution" models
show what later research does about blur: each takes a small picture and draws one 4 times bigger with believable
detail. Both were trained on millions of photo pairs (a sharp photo and the same photo made small, blurry and
noisy), and both learned with a critic, the GAN idea from ROOF-303.

- Real-ESRGAN x4plus (Wang et al., ICCV Workshops 2021): a GAN. BSD-3-Clause, (c) 2021 Xintao Wang.
- SwinIR-M, real-world x4 (Liang et al., ICCV Workshops 2021): a Swin Transformer trained the same way.
  Apache-2.0, (c) 2021 SwinIR Authors.

The weight files come from each project's official GitHub release. They are checked against a recorded size and
SHA-256 before use and opened with torch.load(..., weights_only=True), so no code inside a file can run. They load
through spandrel (MIT), the optional extra `modern`:  pip install "genroofs[modern] @ git+...".

An enlarger sharpens detail that is still in the picture; it cannot bring back detail a model threw away, and what
it adds is believable, not necessarily true.

The fix for blur is a better autoencoder. Stable Diffusion's own (`stabilityai/sd-vae-ft-mse`, MIT licence) keeps a
grid of numbers instead of one list, was scored on "does it look the same" and "does it look real" as well as pixel
by pixel, and learned from millions of photos. It loads through diffusers (Apache-2.0), also in the `modern` extra,
from one fixed revision and only from .safetensors files, which hold numbers and never code.
"""
import base64
import html as _html
import io
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .data import cache_dir, fetch_checked
from .utils import pick_device

__all__ = ["UPSCALERS", "SD_AUTOENCODER", "available_upscalers", "upscaler_info", "credit", "fetch_upscaler",
           "load_upscaler", "enlarge", "stretch", "load_sd_autoencoder", "sd_copies", "sd_numbers", "latent_numbers",
           "big_page", "show_big"]

UPSCALERS = {
    "realesrgan": {
        "title": "Real-ESRGAN x4plus",
        "kind": "a GAN",
        "file": "RealESRGAN_x4plus.pth",
        "bytes": 67_040_989,
        "sha256": "4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "release": "xinntao/Real-ESRGAN v0.1.0",
        "licence": "BSD-3-Clause",
        "copyright": "(c) 2021 Xintao Wang",
        "paper": "Wang, Xie, Dong and Shan, Real-ESRGAN, ICCV Workshops 2021",
    },
    "swinir": {
        "title": "SwinIR-M, real-world x4",
        "kind": "a Swin Transformer",
        "file": "003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth",
        "bytes": 67_129_861,
        "sha256": "b9afb61e65e04eb7f8aba5095d070bbe9af28df76acd0c9405aeb33b814bcfc6",
        "url": "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/"
               "003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth",
        "release": "JingyunLiang/SwinIR v0.0",
        "licence": "Apache-2.0",
        "copyright": "(c) 2021 SwinIR Authors",
        "paper": "Liang, Cao, Sun, Zhang, Van Gool and Timofte, SwinIR, ICCV Workshops 2021",
    },
}
SCALE = 4

SD_AUTOENCODER = {
    "title": "Stable Diffusion's autoencoder (sd-vae-ft-mse)",
    "kind": "an autoencoder trained with a \"looks real\" critic",
    "repo": "stabilityai/sd-vae-ft-mse",
    "revision": "31f26fdeee1355a5c34592e401dd41e45d25a493",
    "licence": "MIT",
    "copyright": "Stability AI",
    "paper": "Rombach, Blattmann, Lorenz, Esser and Ommer, Latent Diffusion Models, CVPR 2022",
    "megabytes": 335,
}
_SD_NAMES = ("sdautoencoder", "sdvae", "sdvaeftmse", "stablediffusion")


def available_upscalers() -> list:
    return list(UPSCALERS)


def upscaler_info(name: str) -> dict:
    key = str(name).lower().replace("-", "").replace(" ", "")
    if key not in UPSCALERS:
        raise ValueError(f"Unknown enlarger {name!r}; choose one of {', '.join(UPSCALERS)}.")
    return UPSCALERS[key]


def credit(name: str) -> str:
    """One line naming the model, its authors' paper, its licence and copyright, as the licences ask.
    Takes an enlarger's name or "sd-autoencoder"."""
    key = str(name).lower().replace("-", "").replace("_", "").replace(" ", "")
    if key in _SD_NAMES:
        info = SD_AUTOENCODER
        return (f"{info['title']} ({info['kind']}) · {info['paper']} · {info['licence']}, {info['copyright']} · "
                f"huggingface.co/{info['repo']}")
    info = upscaler_info(name)
    return f"{info['title']} ({info['kind']}) · {info['paper']} · {info['licence']}, {info['copyright']}"


def fetch_upscaler(name: str, cache=None, quiet: bool = False) -> Path:
    """Download the official weight file once into the cache folder's `upscalers/`, checked against its recorded
    size and SHA-256 (a file that does not match is deleted, never used)."""
    info = upscaler_info(name)
    folder = (Path(cache).expanduser() if cache is not None else cache_dir()) / "upscalers"
    return fetch_checked(info, info["url"], folder, info["release"], quiet)


def _spandrel():
    try:
        import spandrel
    except ImportError:
        raise ImportError("The modern enlargers need the optional package spandrel. On Colab, install genroofs "
                          "with the [modern] extra (the notebook's first cell does this); on a laptop, inside the "
                          "project's .venv:  pip install \"spandrel>=0.4.2,<0.5\"") from None
    return spandrel


def load_upscaler(name: str, cache=None, device=None, quiet: bool = False):
    """Download (once), check and load an enlarger. Returns a spandrel model: call it through `enlarge`.

    The file is opened with weights_only=True, so it can only hold numbers; spandrel then recognises the design
    (Real-ESRGAN's RRDBNet or SwinIR) from the names and shapes of those numbers.
    """
    spandrel = _spandrel()
    info = upscaler_info(name)
    path = fetch_upscaler(name, cache, quiet)
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict):
        state = state.get("params_ema", state.get("params", state))
    model = spandrel.ModelLoader().load_from_state_dict(state).eval()
    if getattr(model, "scale", SCALE) != SCALE:
        raise ValueError(f"{info['file']} enlarges {model.scale} times, not {SCALE}.")
    model.to(torch.device(device) if device is not None else pick_device())
    if not quiet:
        print(credit(name))
    return model


def _as_batch(images) -> torch.Tensor:
    if isinstance(images, np.ndarray):
        images = torch.from_numpy(images)
    images = images.detach().cpu()
    if images.dim() == 3:
        images = images.unsqueeze(0)
    if images.dim() != 4 or images.shape[1] != 3:
        raise ValueError(f"Expected roofs shaped N x 3 x H x W, got {tuple(images.shape)}.")
    if images.dtype != torch.uint8:                       # floats in -1..1, as the models give them
        images = ((images.float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
    return images


@torch.no_grad()
def enlarge(model, images, batch_size: int = 8) -> torch.Tensor:
    """Enlarge roofs 4 times with a model from `load_upscaler`: uint8 N x 3 x H x W -> uint8 N x 3 x 4H x 4W.
    Also takes one roof (3 x H x W) or floats in -1..1."""
    images = _as_batch(images)
    device = next(model.model.parameters()).device
    out = []
    for i in range(0, len(images), batch_size):
        x = images[i:i + batch_size].to(device).float() / 255
        out.append((model(x).clamp(0, 1) * 255).round().to(torch.uint8).cpu())
    return torch.cat(out)


@torch.no_grad()
def stretch(images, scale: int = SCALE) -> torch.Tensor:
    """"Simple stretching": the same 4 times enlargement with plain bicubic interpolation, for comparison."""
    images = _as_batch(images)
    big = F.interpolate(images.float(), scale_factor=scale, mode="bicubic", align_corners=False)
    return big.clamp(0, 255).round().to(torch.uint8)


def _diffusers():
    try:
        import diffusers
    except ImportError:
        raise ImportError("Stable Diffusion's autoencoder needs the optional package diffusers. On Colab, install "
                          "genroofs with the [modern] extra (the notebook's first cell does this); on a laptop, inside "
                          "the project's .venv:  pip install \"diffusers>=0.30,<1\"") from None
    return diffusers


def load_sd_autoencoder(device=None, quiet: bool = False):
    """Download (once, about 335 MB) and load Stable Diffusion's autoencoder from Hugging Face, at one fixed
    revision and only from .safetensors files (numbers, never code). Use it through `sd_copies`."""
    diffusers = _diffusers()
    info = SD_AUTOENCODER
    model = diffusers.AutoencoderKL.from_pretrained(info["repo"], revision=info["revision"], use_safetensors=True)
    model = model.eval().to(torch.device(device) if device is not None else pick_device())
    if not quiet:
        print(credit("sd-autoencoder"))
    return model


def latent_numbers(size: int) -> int:
    """How many numbers Stable Diffusion's autoencoder keeps for a size x size picture: a grid of 4 channels,
    each size/8 x size/8 (1,024 for a 128-pixel roof). Our autoencoder keeps 128 for a 64-pixel roof."""
    if size % 8:
        raise ValueError(f"Stable Diffusion's autoencoder needs a picture size divisible by 8, not {size}.")
    return 4 * (size // 8) ** 2


@torch.no_grad()
def sd_copies(model, images, batch_size: int = 8) -> torch.Tensor:
    """Copy roofs through Stable Diffusion's autoencoder (encode to its grid of numbers, decode back):
    uint8 N x 3 x H x W -> uint8 N x 3 x H x W, with H and W divisible by 8. Also takes floats in -1..1."""
    images = _as_batch(images)
    if images.shape[-1] % 8 or images.shape[-2] % 8:
        raise ValueError(f"Stable Diffusion's autoencoder needs sizes divisible by 8, got {tuple(images.shape[-2:])}.")
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    out = []
    for i in range(0, len(images), batch_size):
        x = (images[i:i + batch_size].to(device).float() / 127.5 - 1).to(dtype)
        latent = model.encode(x).latent_dist.mode()
        y = model.decode(latent).sample.float()
        out.append(((y.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).cpu())
    return torch.cat(out)


@torch.no_grad()
def sd_numbers(model, images, batch_size: int = 16) -> torch.Tensor:
    """Stable Diffusion's autoencoder as a feature maker: each roof's grid of numbers (the centre of its range),
    flattened. uint8 N x 3 x H x W -> float N x (4 * H/8 * W/8) on the CPU (1,024 numbers for a 128-pixel roof)."""
    images = _as_batch(images)
    if images.shape[-1] % 8 or images.shape[-2] % 8:
        raise ValueError(f"Stable Diffusion's autoencoder needs sizes divisible by 8, got {tuple(images.shape[-2:])}.")
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    out = []
    for i in range(0, len(images), batch_size):
        x = (images[i:i + batch_size].to(device).float() / 127.5 - 1).to(dtype)
        out.append(model.encode(x).latent_dist.mode().float().flatten(1).cpu())
    return torch.cat(out)


def _uri(img: torch.Tensor) -> str:
    buf = io.BytesIO()
    Image.fromarray(img.permute(1, 2, 0).numpy()).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def big_page(rows: dict, px: int = 256, title: str | None = None, note: str | None = None) -> str:
    """HTML for rows of roofs shown at px x px, one screen pixel per picture pixel, so an enlarger's sharpness is
    seen as it is (a matplotlib grid would shrink it again). Smaller pictures are stretched to px with bicubic
    interpolation and the row says so. Labels are escaped; the page holds no links and no script."""
    parts = ["<div style='font-family:sans-serif'>"]
    if title:
        parts.append(f"<h3 style='margin:12px 0 2px'>{_html.escape(title)}</h3>")
    if note:
        parts.append(f"<div style='opacity:.75;margin-bottom:6px'>{_html.escape(note)}</div>")
    parts.append("<div style='overflow-x:auto'><table style='border-spacing:4px'>")
    for label, images in rows.items():
        batch = _as_batch(images)
        stretched = batch.shape[-1] < px
        if batch.shape[-1] != px:
            batch = F.interpolate(batch.float(), size=(px, px), mode="bicubic" if stretched else "area",
                                  align_corners=False if stretched else None).clamp(0, 255).round().to(torch.uint8)
        name = _html.escape(str(label)) + (" <span style='opacity:.6'>(stretched)</span>" if stretched else "")
        cells = "".join(f"<td><img src='{_uri(im)}' width='{px}' height='{px}'></td>" for im in batch)
        parts.append(f"<tr><th style='text-align:right;padding-right:10px;width:170px'>{name}</th>{cells}</tr>")
    parts.append("</table></div></div>")
    return "".join(parts)


def show_big(rows: dict, px: int = 256, title: str | None = None, note: str | None = None):
    """Show `big_page` in the notebook. Outside a notebook it returns the HTML instead."""
    page = big_page(rows, px, title, note)
    try:
        from IPython.display import HTML, display
    except ImportError:
        return page
    display(HTML(page))
    return None
