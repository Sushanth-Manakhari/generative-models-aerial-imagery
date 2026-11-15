"""Offline tests for genroofs.train: losses, time and pass limits, history, saving and loading weights."""
import json
import math

import numpy as np
import pytest
import torch

from conftest import FakeServer
from genroofs import data, train
from genroofs.models import Discriminator, Generator, RoofAE, RoofVAE
from genroofs.utils import set_seed


def squares(n: int = 64, seed: int = 0) -> torch.Tensor:
    """n simple uint8 'roofs': a coloured square on a plain background."""
    rng = np.random.default_rng(seed)
    out = np.zeros((n, 3, 64, 64), np.uint8)
    for i in range(n):
        out[i] = rng.integers(40, 120, size=(3, 1, 1))
        y, x, s = rng.integers(8, 32), rng.integers(8, 32), rng.integers(12, 24)
        out[i, :, y:y + s, x:x + s] = rng.integers(140, 250, size=(3, 1, 1))
    return torch.from_numpy(out)


class FakeClock:
    """Each reading moves time forward by `tick` seconds, so time limits can be tested without waiting."""

    def __init__(self, tick: float):
        self.t, self.tick = 0.0, tick

    def __call__(self) -> float:
        self.t += self.tick
        return self.t


# --------------------------------------------------------------------------------------------------- losses

def test_reconstruction_loss_is_summed_over_pixels_and_averaged_over_the_batch():
    x = torch.zeros(4, 3, 64, 64)
    copy = torch.full_like(x, 0.5)
    assert train.reconstruction_loss(copy, x).item() == pytest.approx(0.25 * 3 * 64 * 64)


def test_kl_divergence():
    zeros = torch.zeros(5, 128)
    assert train.kl_divergence(zeros, zeros).item() == 0.0
    assert train.kl_divergence(torch.ones(1, 1), torch.zeros(1, 1)).item() == pytest.approx(0.5)   # mu^2 / 2
    torch.manual_seed(0)
    assert train.kl_divergence(torch.randn(8, 16), torch.randn(8, 16)).item() > 0


def test_vae_loss_adds_beta_times_kl():
    x, copy = torch.zeros(2, 3, 64, 64), torch.full((2, 3, 64, 64), 0.1)
    mu, logvar = torch.ones(2, 4), torch.zeros(2, 4)
    total, rec, kl = train.vae_loss(copy, x, mu, logvar, beta=3.0)
    assert kl.item() == pytest.approx(2.0)
    assert total.item() == pytest.approx(rec.item() + 3.0 * 2.0)


def test_gan_losses():
    zero = torch.zeros(6, 1)
    assert train.generator_loss(zero).item() == pytest.approx(math.log(2))
    sure = torch.full((6, 1), 30.0)
    assert train.critic_loss(sure, -sure, real_label=1.0).item() == pytest.approx(0.0, abs=1e-6)
    smooth = train.critic_loss(sure, -sure, real_label=0.9).item()
    assert smooth > 1.0, "with label smoothing an over-confident critic is penalised"


# --------------------------------------------------------------------------------------------------- limits

def test_epochs_limit_counts_whole_passes():
    h = train.train_ae(RoofAE(), squares(64), epochs=2, batch_size=16, device="cpu", verbose=False)
    assert h.stopped_by == "epochs" and h.steps == 8 and h.batches_per_pass == 4 and h.passes == 2.0
    assert len(h.pass_seconds) == 2 and len(h.losses["loss"]) == 8


def test_minutes_limit_with_a_fake_clock(monkeypatch):
    monkeypatch.setattr(train, "_now", FakeClock(tick=1.0))
    h = train.train_ae(RoofAE(), squares(64), minutes=0.1, batch_size=16, device="cpu", verbose=False)
    assert h.stopped_by == "minutes"
    assert 1 <= h.steps <= 6                      # 6 seconds at one second per clock reading
    assert h.seconds >= 6.0


def test_class_budget_applies_when_no_limit_is_given(monkeypatch):
    monkeypatch.setattr(train, "_now", FakeClock(tick=40.0))
    h = train.train_gan(Generator(), Discriminator(), squares(32), batch_size=16, device="cpu", verbose=False)
    assert h.settings["minutes"] == 6.0 and h.settings["epochs"] is None and h.stopped_by == "minutes"
    h = train.train_vae(RoofVAE(), squares(32), batch_size=16, device="cpu", verbose=False)
    assert h.settings["minutes"] == 3.0


def test_epochs_alone_means_no_time_limit(monkeypatch):
    monkeypatch.setattr(train, "_now", FakeClock(tick=1000.0))
    h = train.train_ae(RoofAE(), squares(32), epochs=3, batch_size=16, device="cpu", verbose=False)
    assert h.stopped_by == "epochs" and h.passes == 3.0


def test_a_real_short_run_ends_on_time():
    h = train.train_ae(RoofAE(), squares(64), minutes=0.01, batch_size=16, device="cpu", verbose=False)
    assert h.stopped_by in ("minutes", "epochs")
    assert h.seconds < 0.6 + 2.0


def test_bad_limits_and_too_few_roofs():
    for kwargs in ({"minutes": 0}, {"minutes": -1}, {"epochs": 0}):
        with pytest.raises(ValueError):
            train.train_ae(RoofAE(), squares(32), batch_size=16, device="cpu", verbose=False, **kwargs)
    with pytest.raises(ValueError, match="fewer than one batch"):
        train.train_ae(RoofAE(), squares(10), epochs=1, batch_size=16, device="cpu", verbose=False)


# -------------------------------------------------------------------------------------------------- history

def test_autoencoder_learns_and_pictures_the_watched_roofs():
    roofs = squares(64)
    watch = roofs[:6]
    set_seed(0)
    h = train.train_ae(RoofAE(), roofs, epochs=6, batch_size=16, device="cpu", watch=watch, snapshots=3,
                       verbose=False)
    losses = h.losses["loss"]
    assert np.mean(losses[-4:]) < 0.5 * np.mean(losses[:4])
    assert torch.equal(h.watch, watch)
    assert [s for s, _ in h.snapshots] == [0, 8, 16, 24]
    assert all(img.shape == (6, 3, 64, 64) and img.dtype == torch.uint8 for _, img in h.snapshots)


def test_default_watch_is_eight_roofs():
    h = train.train_ae(RoofAE(), squares(32), epochs=1, batch_size=16, device="cpu", verbose=False)
    assert h.watch.shape == (8, 3, 64, 64)
    with pytest.raises(TypeError, match="uint8"):
        train.train_ae(RoofAE(), squares(32), epochs=1, batch_size=16, device="cpu", verbose=False,
                       watch=torch.zeros(2, 3, 64, 64))


def test_same_seed_same_run():
    runs = []
    for _ in range(2):
        set_seed(3)
        model = RoofAE()
        runs.append(train.train_ae(model, squares(48), epochs=2, batch_size=16, seed=11, device="cpu",
                                   verbose=False).losses["loss"])
    assert runs[0] == runs[1]


def test_vae_records_both_parts_and_beta():
    h = train.train_vae(RoofVAE(), squares(32), epochs=1, batch_size=16, beta=0.0, device="cpu", verbose=False)
    assert set(h.losses) == {"loss", "recon", "kl"} and h.settings["beta"] == 0.0
    assert h.losses["loss"] == h.losses["recon"]
    h = train.train_vae(RoofVAE(), squares(32), epochs=1, batch_size=16, beta=2.0, device="cpu", verbose=False)
    for total, rec, kl in zip(h.losses["loss"], h.losses["recon"], h.losses["kl"]):
        assert total == pytest.approx(rec + 2.0 * kl, rel=1e-5)


def test_gan_history():
    g, d = Generator(), Discriminator()
    h = train.train_gan(g, d, squares(32), epochs=2, batch_size=16, label_smooth=1.0, watch_n=4, snapshots=2,
                        device="cpu", verbose=False)
    assert set(h.losses) == {"d_loss", "g_loss", "d_real", "d_fake"}
    assert all(0.0 <= p <= 1.0 for p in h.losses["d_real"] + h.losses["d_fake"])
    assert h.watch.shape == (4, 100) and h.snapshots[-1][1].shape == (4, 3, 64, 64)
    assert h.settings["label_smooth"] == 1.0 and h.settings["lr_d"] == 2e-4 and h.settings["betas"] == [0.5, 0.999]
    assert not g.training and not d.training


def test_models_end_in_eval_mode():
    ae = RoofAE()
    train.train_ae(ae, squares(32), epochs=1, batch_size=16, device="cpu", verbose=False)
    assert not ae.training


def test_nan_stops_the_run(monkeypatch, capsys):
    real, calls = train.reconstruction_loss, []

    def goes_bad(copy, x):
        calls.append(1)
        loss = real(copy, x)
        return loss * float("nan") if len(calls) >= 3 else loss
    monkeypatch.setattr(train, "reconstruction_loss", goes_bad)
    h = train.train_ae(RoofAE(), squares(64), epochs=5, batch_size=16, device="cpu")
    assert h.stopped_by == "nan" and h.steps == 3
    assert "NaN or infinite" in capsys.readouterr().out


def test_progress_lines_and_summary(capsys):
    h = train.train_ae(RoofAE(), squares(32), epochs=2, batch_size=16, device="cpu", snapshots=2)
    out = capsys.readouterr().out
    assert "AE   pass    1.0" in out and "stopped by the 2-pass limit" in out
    summary = h.summary()
    json.dumps(summary)
    assert summary["passes"] == 2.0 and summary["stopped_by"] == "epochs" and "loss" in summary["last"]
    assert repr(h).startswith("History(ae: 2.0 passes, 4 steps")


# -------------------------------------------------------------------------------------------------- weights

@pytest.mark.parametrize("make", [lambda: RoofAE(latent=16), RoofVAE, Generator, Discriminator])
def test_save_and_load_give_identical_outputs(tmp_path, make):
    model = make()
    if isinstance(model, Generator):        # give the batch-norm running statistics non-default values
        model.train()
        model(model.noise(8, seed=1))
    model.eval()
    path = train.save_weights(model, tmp_path / "w" / "model", notes={"pack": "roofs_maine_2022"})
    assert path.name == "model.safetensors" and path.with_suffix(".json").is_file()
    assert path.read_bytes()[8:9] == b"{", "a .safetensors file starts with its JSON header, not pickled code"
    loaded = train.load_weights(path, device="cpu")
    assert type(loaded) is type(model) and not loaded.training
    if isinstance(model, Generator):
        x = model.noise(3, seed=2)
    else:
        torch.manual_seed(0)
        x = torch.rand(3, 3, 64, 64) * 2 - 1
    with torch.no_grad():
        if isinstance(model, RoofVAE):
            assert torch.equal(model.decode(model.encode(x)[0]), loaded.decode(loaded.encode(x)[0]))
        else:
            assert torch.equal(model(x), loaded(x))


def test_weights_note(tmp_path):
    ae = RoofAE()
    h = train.train_ae(ae, squares(32), epochs=1, batch_size=16, device="cpu", verbose=False)
    path = train.save_weights(ae, tmp_path / "ae.safetensors", history=h, notes={"pack": "roofs_maine_2022"})
    note = train.weights_note(path)
    assert note["class"] == "RoofAE" and note["settings"] == {"latent": 128} and note["parameters"] == 2_432_835
    assert note["notes"] == {"pack": "roofs_maine_2022"} and note["training"]["passes"] == 1.0
    assert json.loads(path.with_suffix(".json").read_text()) == note


def test_load_refuses_the_wrong_model(tmp_path):
    path = train.save_weights(RoofAE(latent=8), tmp_path / "small.safetensors")
    with pytest.raises(ValueError, match="holds a RoofAE, not a RoofVAE"):
        train.load_weights(path, model=RoofVAE(latent=8))
    with pytest.raises(ValueError, match=r"saved with settings \{'latent': 8\}, this model has \{'latent': 128\}"):
        train.load_weights(path, model=RoofAE(latent=128))
    assert train.load_weights(path, device="cpu").latent == 8


def test_load_refuses_files_not_saved_by_genroofs(tmp_path):
    from safetensors.torch import save_file
    other = tmp_path / "other.safetensors"
    save_file({"w": torch.zeros(2)}, str(other))
    with pytest.raises(ValueError, match="not saved by genroofs"):
        train.load_weights(other)
    junk = tmp_path / "junk.safetensors"
    junk.write_bytes(b"not a weights file")
    with pytest.raises(ValueError, match="not a readable"):
        train.load_weights(junk)
    with pytest.raises(TypeError, match="genroofs models"):
        train.save_weights(torch.nn.Linear(2, 2), tmp_path / "linear")
    with pytest.raises(TypeError, match="plain values"):
        train.save_weights(RoofAE(latent=8), tmp_path / "x", notes={"bad": object()})


# ------------------------------------------------------------------------------------------- pretrained

def test_pretrained_not_published_yet(monkeypatch):
    monkeypatch.setattr(train, "_WEIGHTS_INFO", None)
    assert train._weights_info() == {}
    with pytest.raises(train.WeightsNotPublished, match="not published yet.*train_ae"):
        train.load_pretrained("ae")
    with pytest.raises(ValueError, match="ae, vae, gan"):
        train.load_pretrained("diffusion")


def test_weights_url():
    assert train.WEIGHTS_URL.format(file="roof_ae.safetensors") == (
        "https://github.com/Sushanth-Manakhari/generative-models-aerial-imagery/releases/download/models-v1/"
        "roof_ae.safetensors")


@pytest.fixture
def weights_release(monkeypatch, tmp_path):
    """A fake models-v1 release serving one saved autoencoder."""
    import hashlib
    ae = RoofAE(latent=8).eval()
    path = train.save_weights(ae, tmp_path / "made" / "roof_ae.safetensors")
    body = path.read_bytes()
    server = FakeServer(body)
    monkeypatch.setattr(data, "_open_url", server.open)
    monkeypatch.setattr(data, "_pause", lambda s: None)
    monkeypatch.setenv("GENROOFS_DATA", str(tmp_path / "cache"))
    monkeypatch.setattr(train, "_WEIGHTS_INFO", {"ae": {"file": "roof_ae.safetensors", "bytes": len(body),
                                                        "sha256": hashlib.sha256(body).hexdigest()}})
    return ae, server, tmp_path / "cache" / "weights"


def test_load_pretrained_downloads_checks_and_loads(weights_release):
    ae, server, folder = weights_release
    loaded = train.load_pretrained("ae", device="cpu", quiet=True)
    assert isinstance(loaded, RoofAE) and loaded.latent == 8
    assert server.calls[0].endswith("/releases/download/models-v1/roof_ae.safetensors")
    assert (folder / "roof_ae.safetensors").is_file()
    x = torch.rand(2, 3, 64, 64) * 2 - 1
    with torch.no_grad():
        assert torch.equal(ae(x), loaded(x))
    train.load_pretrained("ae", device="cpu", quiet=True)
    assert len(server.calls) == 1


def test_load_pretrained_rejects_a_bad_download(weights_release):
    _, server, folder = weights_release
    server.mode = "corrupt"
    with pytest.raises(data.PackError, match="SHA-256 does not match.*models-v1"):
        train.load_pretrained("ae", device="cpu", quiet=True)
    assert not list(folder.glob("*"))
