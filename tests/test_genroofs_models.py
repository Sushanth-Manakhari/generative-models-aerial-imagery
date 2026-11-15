"""Offline tests for genroofs.models: the four networks match the designs timed on the Colab T4 in G3."""
import hashlib
import json

import pytest
import torch

from genroofs import models

# Parameter counts and a fingerprint of every layer's name and shape, taken from the G3 timing notebook.
G3 = {
    "RoofAE": (2_432_835, "c89d7ad5354e77e9"),
    "RoofVAE": (2_957_251, "03240842dfc843b1"),
    "Generator": (1_506_051, "86a3f03369a80f1c"),
    "Discriminator": (1_558_801, "a9424709971e0433"),
}


def layout_fingerprint(model) -> str:
    layout = [[k, list(v.shape)] for k, v in model.state_dict().items()]
    return hashlib.sha256(json.dumps(layout).encode()).hexdigest()[:16]


def images(n=4):
    torch.manual_seed(0)
    return torch.rand(n, 3, 64, 64) * 2 - 1


@pytest.mark.parametrize("name", list(G3))
def test_same_design_as_timed_in_g3(name):
    model = models.MODELS[name]()
    count, fingerprint = G3[name]
    assert models.count_parameters(model) == count
    assert layout_fingerprint(model) == fingerprint, f"{name} no longer matches the design timed in G3"


def test_autoencoder_shapes():
    ae = models.RoofAE()
    x = images()
    assert ae.encode(x).shape == (4, 128)
    out = ae(x)
    assert out.shape == x.shape and out.min() >= -1 and out.max() <= 1
    small = models.RoofAE(latent=8)
    assert small.encode(x).shape == (4, 8) and small.settings() == {"latent": 8}


def test_vae_shapes_and_sampling():
    vae = models.RoofVAE()
    x = images()
    copy, mu, logvar = vae(x)
    assert copy.shape == x.shape and mu.shape == logvar.shape == (4, 128)
    assert torch.equal(vae.encode(x)[0], mu)
    a, b = vae.sample(3, seed=5), vae.sample(3, seed=5)
    assert a.shape == (3, 3, 64, 64) and torch.equal(a, b)
    assert not torch.equal(a, vae.sample(3, seed=6))


def test_reparameterize_is_mu_plus_sigma_times_noise():
    mu = torch.full((2000, 4), 3.0)
    logvar = torch.full((2000, 4), 2 * torch.log(torch.tensor(0.5)).item())   # sigma = 0.5
    torch.manual_seed(1)
    z = models.RoofVAE.reparameterize(mu, logvar)
    assert z.mean().item() == pytest.approx(3.0, abs=0.03)
    assert z.std().item() == pytest.approx(0.5, abs=0.03)


def test_generator_and_critic_shapes():
    g, d = models.Generator(), models.Discriminator()
    z = g.noise(5, seed=3)
    assert z.shape == (5, 100) and torch.equal(z, g.noise(5, seed=3))
    fake = g(z)
    assert fake.shape == (5, 3, 64, 64) and fake.min() >= -1 and fake.max() <= 1
    assert d(fake).shape == (5, 1)
    assert g.settings() == {"z_dim": 100, "ngf": 48} and d.settings() == {"ndf": 48}


def test_generator_sample_keeps_the_training_mode():
    g = models.Generator()
    g.train()
    g.sample(2)
    assert g.training
    g.eval()
    g.sample(2)
    assert not g.training


def test_clear_errors_for_wrong_shapes():
    with pytest.raises(ValueError, match="size=64"):
        models.RoofAE()(torch.zeros(2, 3, 32, 32))
    with pytest.raises(ValueError, match="N x 3 x 64 x 64"):
        models.Discriminator()(torch.zeros(3, 64, 64))
    with pytest.raises(ValueError, match="N x 100"):
        models.Generator()(torch.zeros(2, 64))
