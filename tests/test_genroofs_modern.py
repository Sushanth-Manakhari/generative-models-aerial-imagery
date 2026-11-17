"""Offline tests for genroofs.modern: the registry, checked downloads, safe loading, enlarging and the page."""
import builtins
import pickle
import re

import pytest
import torch

from genroofs import modern


def test_registry_is_complete_and_pinned():
    assert modern.available_upscalers() == ["realesrgan", "swinir"]
    for name, info in modern.UPSCALERS.items():
        assert re.fullmatch(r"[0-9a-f]{64}", info["sha256"]), name
        assert info["bytes"] > 60_000_000
        assert info["url"].startswith("https://github.com/") and "/releases/download/" in info["url"]
        assert info["url"].endswith("/" + info["file"])
        assert info["licence"] in ("BSD-3-Clause", "Apache-2.0")
        line = modern.credit(name)
        assert info["licence"] in line and info["copyright"] in line and "ICCV Workshops 2021" in line


def test_names_are_forgiving_but_unknown_names_fail():
    assert modern.upscaler_info("Real-ESRGAN") is modern.UPSCALERS["realesrgan"]
    assert modern.upscaler_info("SwinIR") is modern.UPSCALERS["swinir"]
    with pytest.raises(ValueError, match="realesrgan, swinir"):
        modern.upscaler_info("waifu2x")


def test_fetch_uses_the_checked_download(monkeypatch, tmp_path):
    seen = {}

    def fake_fetch(info, url, folder, tag, quiet):
        seen.update(info=info, url=url, folder=folder, tag=tag)
        return folder / info["file"]

    monkeypatch.setattr(modern, "fetch_checked", fake_fetch)
    path = modern.fetch_upscaler("swinir", cache=tmp_path, quiet=True)
    assert seen["folder"] == tmp_path / "upscalers"
    assert seen["url"] == modern.UPSCALERS["swinir"]["url"]
    assert seen["tag"] == "JingyunLiang/SwinIR v0.0"
    assert path.name == modern.UPSCALERS["swinir"]["file"]


def tiny_esrgan_file(tmp_path):
    arch = pytest.importorskip("spandrel.architectures.ESRGAN")
    torch.manual_seed(0)
    net = arch.ESRGAN(in_nc=3, out_nc=3, num_filters=8, num_blocks=1, scale=4)
    path = tmp_path / "tiny.pth"
    torch.save({"params_ema": net.state_dict()}, path)
    return path


def test_load_and_enlarge_a_tiny_model(monkeypatch, tmp_path, capsys):
    path = tiny_esrgan_file(tmp_path)
    monkeypatch.setattr(modern, "fetch_upscaler", lambda name, cache=None, quiet=False: path)
    model = modern.load_upscaler("realesrgan", device="cpu")
    assert "BSD-3-Clause" in capsys.readouterr().out
    roofs = torch.randint(0, 256, (3, 3, 16, 16), dtype=torch.uint8)
    big = modern.enlarge(model, roofs, batch_size=2)
    assert big.shape == (3, 3, 64, 64) and big.dtype == torch.uint8
    one = modern.enlarge(model, roofs[0])
    assert one.shape == (1, 3, 64, 64)
    assert torch.equal(one[0], big[0])
    floats = modern.enlarge(model, roofs.float() / 127.5 - 1)          # -1..1 floats are accepted too
    assert (floats.int() - big.int()).abs().max() <= 2


def test_loading_refuses_files_that_hold_code(monkeypatch, tmp_path):
    pytest.importorskip("spandrel")

    class Sneaky:
        def __reduce__(self):
            return (print, ("this would run on load",))

    path = tmp_path / "bad.pth"
    with open(path, "wb") as fh:
        pickle.dump({"params_ema": Sneaky()}, fh, protocol=2)
    monkeypatch.setattr(modern, "fetch_upscaler", lambda name, cache=None, quiet=False: path)
    with pytest.raises(pickle.UnpicklingError):
        modern.load_upscaler("realesrgan", device="cpu", quiet=True)


def test_a_missing_spandrel_says_how_to_install_it(monkeypatch):
    real_import = builtins.__import__

    def no_spandrel(name, *args, **kwargs):
        if name == "spandrel":
            raise ImportError("No module named 'spandrel'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_spandrel)
    with pytest.raises(ImportError, match=r"\[modern\]"):
        modern.load_upscaler("swinir")


def test_stretch_is_plain_bicubic():
    flat = torch.full((2, 3, 8, 8), 90, dtype=torch.uint8)
    big = modern.stretch(flat)
    assert big.shape == (2, 3, 32, 32) and big.dtype == torch.uint8
    assert torch.all(big == 90)
    with pytest.raises(ValueError, match="N x 3 x H x W"):
        modern.stretch(torch.zeros(2, 1, 8, 8, dtype=torch.uint8))


def test_big_page_shows_every_picture_at_full_size():
    small = torch.randint(0, 256, (4, 3, 64, 64), dtype=torch.uint8)
    large = torch.randint(0, 256, (4, 3, 256, 256), dtype=torch.uint8)
    huge = torch.randint(0, 256, (4, 3, 512, 512), dtype=torch.uint8)
    page = modern.big_page({"photo <b>": small, "SwinIR": large, "drawn": huge}, title="A & B", note="n")
    assert page.count("<img ") == 12
    assert page.count("width='256' height='256'") == 12
    assert page.count("(stretched)") == 1
    assert "photo &lt;b&gt;" in page and "A &amp; B" in page
    assert "<script" not in page and "<a " not in page and "http" not in page


def test_show_big_returns_html_outside_a_notebook(monkeypatch):
    real_import = builtins.__import__

    def no_ipython(name, *args, **kwargs):
        if name.startswith("IPython"):
            raise ImportError("no IPython")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_ipython)
    page = modern.show_big({"a": torch.zeros(1, 3, 16, 16, dtype=torch.uint8)}, px=32)
    assert page.count("<img ") == 1


# ------------------------------------------------------------------------------- Stable Diffusion's autoencoder

def _tiny_sd_autoencoder():
    diffusers = pytest.importorskip("diffusers")
    torch.manual_seed(0)
    return diffusers.AutoencoderKL(block_out_channels=[32, 64, 64, 64], in_channels=3, out_channels=3,
                                   latent_channels=4, down_block_types=["DownEncoderBlock2D"] * 4,
                                   up_block_types=["UpDecoderBlock2D"] * 4, norm_num_groups=32)


def test_sd_autoencoder_is_pinned_and_credited():
    info = modern.SD_AUTOENCODER
    assert info["repo"] == "stabilityai/sd-vae-ft-mse"
    assert re.fullmatch(r"[0-9a-f]{40}", info["revision"])
    assert info["licence"] == "MIT"
    for name in ("sd-autoencoder", "SD VAE", "sd_vae"):
        line = modern.credit(name)
        assert "MIT" in line and "CVPR 2022" in line and info["repo"] in line
    assert modern.latent_numbers(128) == 1024 and modern.latent_numbers(64) == 256
    with pytest.raises(ValueError, match="divisible by 8"):
        modern.latent_numbers(100)


def test_load_sd_autoencoder_asks_for_one_revision_and_safetensors_only(monkeypatch, capsys):
    diffusers = pytest.importorskip("diffusers")
    seen = {}

    def fake_from_pretrained(repo, **kw):
        seen.update(repo=repo, **kw)
        return _tiny_sd_autoencoder()

    monkeypatch.setattr(diffusers.AutoencoderKL, "from_pretrained", staticmethod(fake_from_pretrained))
    model = modern.load_sd_autoencoder(device="cpu")
    assert seen == {"repo": "stabilityai/sd-vae-ft-mse", "revision": modern.SD_AUTOENCODER["revision"],
                    "use_safetensors": True}
    assert not model.training
    assert "MIT" in capsys.readouterr().out


def test_sd_copies_keep_size_and_type():
    model = _tiny_sd_autoencoder().eval()
    roofs = torch.randint(0, 256, (3, 3, 32, 32), dtype=torch.uint8)
    copies = modern.sd_copies(model, roofs, batch_size=2)
    assert copies.shape == roofs.shape and copies.dtype == torch.uint8
    one = modern.sd_copies(model, roofs[0].float() / 127.5 - 1)            # one roof, floats in -1..1
    assert one.shape == (1, 3, 32, 32)
    with pytest.raises(ValueError, match="divisible by 8"):
        modern.sd_copies(model, torch.zeros(1, 3, 36, 36, dtype=torch.uint8))


def test_a_missing_diffusers_says_how_to_install_it(monkeypatch):
    real_import = builtins.__import__

    def no_diffusers(name, *args, **kwargs):
        if name == "diffusers":
            raise ModuleNotFoundError("No module named 'diffusers'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_diffusers)
    with pytest.raises(ImportError, match=r"\[modern\].*diffusers"):
        modern.load_sd_autoencoder()
