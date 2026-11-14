"""The four networks of the lab, for 64 x 64 colour roof images with pixels in -1..1.

These are the exact designs timed on a free Colab T4 GPU before the lecture, so the in-class time budgets hold.

    RoofAE         autoencoder: squeeze a roof into `latent` numbers and rebuild it        2.43 M parameters
    RoofVAE        variational autoencoder: the same, but the latent space is smooth       2.96 M parameters
    Generator      GAN generator: 100 random numbers -> a new roof                         1.51 M parameters
    Discriminator  GAN critic: a roof -> one score, "real" (high) or "generated" (low)     1.56 M parameters
"""
import torch
from torch import nn

IMAGE_SIZE = 64        # pixels per side
CHANNELS = 3           # red, green, blue
_BOTTOM = 256 * 4 * 4  # the encoder's last feature map, 256 channels of 4 x 4, as one row of numbers

__all__ = ["RoofAE", "RoofVAE", "Generator", "Discriminator", "count_parameters", "IMAGE_SIZE", "MODELS"]


def count_parameters(model: nn.Module) -> int:
    """How many numbers the model learns."""
    return sum(p.numel() for p in model.parameters())


def _check_images(x: torch.Tensor) -> None:
    if x.dim() != 4 or tuple(x.shape[1:]) != (CHANNELS, IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(f"Expected a batch of images shaped N x {CHANNELS} x {IMAGE_SIZE} x {IMAGE_SIZE}, got "
                         f"{tuple(x.shape)}. Load the pack with size={IMAGE_SIZE} and convert with to_float.")


def _encoder() -> nn.Sequential:
    """64 -> 32 -> 16 -> 8 -> 4 pixels per side; each 4x4 convolution with stride 2 halves the size."""
    return nn.Sequential(
        nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(),
        nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
        nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),
        nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU(),
        nn.Flatten())                                  # 256 x 4 x 4 -> 4096 numbers


def _decoder() -> nn.Sequential:
    """The mirror image: 4 -> 8 -> 16 -> 32 -> 64 pixels per side, ending in tanh so pixels land in -1..1."""
    return nn.Sequential(
        nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(),
        nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
        nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
        nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Tanh())


class RoofAE(nn.Module):
    """Autoencoder. encode: roof -> `latent` numbers (the bottleneck). decode: those numbers -> a copy of the roof.

    A roof unlike the training roofs comes back as a poor copy; that error is the "unusual" score.
    """

    def __init__(self, latent: int = 128):
        super().__init__()
        self.latent = int(latent)
        self.enc = _encoder()
        self.to_z = nn.Linear(_BOTTOM, self.latent)
        self.from_z = nn.Linear(self.latent, _BOTTOM)
        self.dec = _decoder()

    def settings(self) -> dict:
        return {"latent": self.latent}

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        _check_images(x)
        return self.to_z(self.enc(x))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.dec(self.from_z(z).view(-1, 256, 4, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


class RoofVAE(nn.Module):
    """Variational autoencoder. The encoder gives a mean (mu) and a log-variance for each latent number;
    z is drawn from them, so nearby points in the latent space decode to similar roofs.
    """

    def __init__(self, latent: int = 128):
        super().__init__()
        self.latent = int(latent)
        self.enc = _encoder()
        self.to_z = nn.Linear(_BOTTOM, 2 * self.latent)   # mean and log-variance side by side
        self.from_z = nn.Linear(self.latent, _BOTTOM)
        self.dec = _decoder()

    def settings(self) -> dict:
        return {"latent": self.latent}

    def encode(self, x: torch.Tensor) -> tuple:
        """Returns (mu, logvar), each N x latent."""
        _check_images(x)
        mu, logvar = self.to_z(self.enc(x)).chunk(2, dim=1)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """z = mu + sigma * eps, with sigma = exp(logvar / 2) and eps ~ N(0, 1).
        Writing the random draw this way lets the gradient flow through mu and sigma."""
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.dec(self.from_z(z).view(-1, 256, 4, 4))

    def forward(self, x: torch.Tensor) -> tuple:
        """Returns (copy, mu, logvar)."""
        mu, logvar = self.encode(x)
        return self.decode(self.reparameterize(mu, logvar)), mu, logvar

    @torch.no_grad()
    def sample(self, n: int, seed: int | None = None) -> torch.Tensor:
        """n new roofs decoded from random points z ~ N(0, 1)."""
        device = next(self.parameters()).device
        gen = torch.Generator().manual_seed(int(seed)) if seed is not None else None
        z = torch.randn(n, self.latent, generator=gen).to(device)
        return self.decode(z)


def _up(c_in: int, c_out: int) -> nn.Sequential:
    """Double the size, then a 3x3 convolution. Avoids the checkerboard pattern of transposed convolutions."""
    return nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(c_in, c_out, 3, 1, 1),
                         nn.BatchNorm2d(c_out), nn.ReLU(True))


class Generator(nn.Module):
    """GAN generator: z_dim random numbers -> (ngf*8) x 4 x 4 -> 8 -> 16 -> 32 -> a 3 x 64 x 64 roof in -1..1."""

    def __init__(self, z_dim: int = 100, ngf: int = 48):
        super().__init__()
        self.z_dim, self.ngf = int(z_dim), int(ngf)
        self.fc = nn.Sequential(nn.Linear(self.z_dim, self.ngf * 8 * 16), nn.BatchNorm1d(self.ngf * 8 * 16),
                                nn.ReLU(True))
        self.net = nn.Sequential(_up(self.ngf * 8, self.ngf * 4), _up(self.ngf * 4, self.ngf * 2),
                                 _up(self.ngf * 2, self.ngf),
                                 nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(self.ngf, 3, 3, 1, 1),
                                 nn.Tanh())

    def settings(self) -> dict:
        return {"z_dim": self.z_dim, "ngf": self.ngf}

    def noise(self, n: int, seed: int | None = None) -> torch.Tensor:
        """n random input vectors on the model's device; the same seed gives the same vectors."""
        device = next(self.parameters()).device
        gen = torch.Generator().manual_seed(int(seed)) if seed is not None else None
        return torch.randn(n, self.z_dim, generator=gen).to(device)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() != 2 or z.shape[1] != self.z_dim:
            raise ValueError(f"Expected noise shaped N x {self.z_dim}, got {tuple(z.shape)}.")
        return self.net(self.fc(z).view(-1, self.ngf * 8, 4, 4))

    @torch.no_grad()
    def sample(self, n: int, seed: int | None = None) -> torch.Tensor:
        """n new roofs. Uses the running batch-norm statistics (eval mode) and restores the previous mode."""
        was_training = self.training
        self.eval()
        try:
            return self(self.noise(n, seed))
        finally:
            self.train(was_training)


class Discriminator(nn.Module):
    """GAN critic: 64 -> 32 -> 16 -> 8 -> 4 pixels per side, then one 4x4 convolution to a single score (a logit)."""

    def __init__(self, ndf: int = 48):
        super().__init__()
        self.ndf = int(ndf)
        n = self.ndf
        self.net = nn.Sequential(
            nn.Conv2d(3, n, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(n, n * 2, 4, 2, 1), nn.BatchNorm2d(n * 2), nn.LeakyReLU(0.2, True),
            nn.Conv2d(n * 2, n * 4, 4, 2, 1), nn.BatchNorm2d(n * 4), nn.LeakyReLU(0.2, True),
            nn.Conv2d(n * 4, n * 8, 4, 2, 1), nn.BatchNorm2d(n * 8), nn.LeakyReLU(0.2, True),
            nn.Conv2d(n * 8, 1, 4, 1, 0), nn.Flatten())

    def settings(self) -> dict:
        return {"ndf": self.ndf}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """N x 1 scores. Higher means "looks real"; sigmoid(score) is the probability."""
        _check_images(x)
        return self.net(x)


MODELS = {cls.__name__: cls for cls in (RoofAE, RoofVAE, Generator, Discriminator)}
