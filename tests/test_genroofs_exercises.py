"""Offline tests for genroofs.exercises: each one runs end to end on tiny made-up roofs in seconds."""
import pytest
import torch

from conftest import make_squares
from genroofs import exercises
from genroofs.models import Generator, RoofAE, RoofVAE


def manifest(n: int) -> list:
    return [{"chip_id": f"{i + 1:06d}", "place": ["Gorham", "York"][i % 2], "area_m2": 100.0 + i} for i in range(n)]


def test_result_prints_only_short_values():
    r = exercises.Result(a=1, b=0.123456, model=RoofAE(latent=8), name="x")
    assert repr(r) == "Result(a=1, b=0.1235, name='x')"
    assert isinstance(r.model, RoofAE)


def test_bottleneck(capsys):
    x = make_squares(48)
    r = exercises.bottleneck(RoofAE(latent=16), x[:32], x[32:], latent=4, minutes=0.005, batch_size=16,
                             device="cpu", show=False)
    assert r.latent == 4 and r.model.latent == 4
    assert isinstance(r.psnr_full, float) and isinstance(r.psnr_small, float)
    out = capsys.readouterr().out
    assert "Latent 16: average PSNR" in out and "Latent 4:" in out


def test_bottleneck_draws(capsys):
    x = make_squares(40)
    exercises.bottleneck(RoofAE(latent=16), x[:32], x[32:], latent=4, minutes=0.003, batch_size=16, n_show=4,
                         device="cpu")


def test_strangest_roofs(capsys):
    x = make_squares(30)
    ae = RoofAE(latent=8)
    r = exercises.strangest_roofs(ae, x, manifest(30), k=5, unusual_images=make_squares(10, seed=3))
    assert len(r.indices) == 5 and r.errors.shape == (30,)
    assert [r.errors[i].item() for i in r.indices] == sorted((r.errors[i].item() for i in r.indices), reverse=True)
    assert 0.0 <= r.auc <= 1.0
    assert r.ordinary_indices == torch.argsort(r.errors)[:5].tolist()
    assert not set(r.ordinary_indices) & set(r.indices)
    out = capsys.readouterr().out
    assert "rank" in out and "percentile" in out and "AUC" in out and "lowest scores" in out
    plain = exercises.strangest_roofs(ae, x, manifest(30), k=3, ordinary=False, show=False)
    assert plain.auc is None and plain.ordinary_indices == []


def test_morph_pair(capsys):
    x = make_squares(12)
    r = exercises.morph_pair(RoofVAE(), x, steps=4, ae=RoofAE(latent=8), seed=1)
    assert 0 <= r.a < 12 and 0 <= r.b < 12
    assert exercises.morph_pair(RoofVAE(), x, a=2, b=5, show=False).a == 2
    assert "Morphing roof" in capsys.readouterr().out


@pytest.mark.parametrize("change", list(exercises.GAN_CHANGES))
def test_break_the_gan(change, capsys):
    x = make_squares(32)
    r = exercises.break_the_gan(x, change=change, minutes=0.003, batch_size=16, real_images=x, device="cpu",
                                show=(change == "none"))
    assert r.change == change and r.spread >= 0 and r.real_spread > 0
    assert r.generator.z_dim == (2 if change == "tiny_noise" else 100)
    assert "Spread of 64 generated roofs" in capsys.readouterr().out


def test_break_the_gan_rejects_unknown_changes():
    with pytest.raises(ValueError, match="critic_lr_x10"):
        exercises.break_the_gan(make_squares(32), change="bigger")


def test_real_or_fake_quiz(capsys):
    real = make_squares(30)
    quiz = exercises.real_or_fake(real, Generator(), n=10, seed=3, show=False)
    assert len(quiz) == 10 and quiz.images.shape == (10, 3, 64, 64) and quiz.images.dtype == torch.uint8
    key = "".join("G" if f else "R" for f in quiz._is_fake)
    assert "hidden" in repr(quiz) and key not in repr(quiz)
    assert 4 <= key.count("G") <= 6
    assert quiz.score(key) == 10
    assert quiz.score(key.replace("G", "F")) == 10
    assert quiz.score(list(key)) == 10
    flipped = "".join("R" if c == "G" else "G" for c in key)
    assert quiz.score(flipped) == 0
    with pytest.raises(ValueError, match="one guess per roof"):
        quiz.score("RG")
    again = exercises.real_or_fake(real, Generator(), n=10, seed=3, show=False)
    assert again._is_fake == quiz._is_fake
    quiz.reveal()
    assert f"Answer key: {key}" in capsys.readouterr().out
    exercises.real_or_fake(real, Generator(), n=6, seed=1)
    with pytest.raises(ValueError):
        exercises.real_or_fake(real, Generator(), n=3)
