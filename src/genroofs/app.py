"""An optional web page for the three models, built with Gradio and shown inside the notebook.

    from genroofs.app import launch
    launch(ae, vae, generator, test.images, test.manifest)

Guard rails: roofs come only from the pack's test set (there is no upload box, so nobody uploads a photo of a
real home), and no public link is ever created (share=False). Gradio is optional: pip install "genroofs[app]".
"""
import numpy as np
import torch
from PIL import Image

from .data import to_bytes, to_float
from .metrics import reconstruction_errors
from .viz import morph_images

__all__ = ["RoofApp", "build_app", "launch"]

SHOW_PX = 256       # roofs are 64 px; shown larger so they are easy to see


def _require_gradio():
    try:
        import gradio as gr
    except ImportError:
        raise ImportError('The web app needs Gradio, which is optional. Install it with: pip install "genroofs[app]" '
                          "(or pip install gradio)") from None
    return gr


def _big(image: torch.Tensor) -> np.ndarray:
    arr = image.detach().cpu().permute(1, 2, 0).numpy()
    return np.asarray(Image.fromarray(arr).resize((SHOW_PX, SHOW_PX), Image.Resampling.NEAREST))


class RoofApp:
    """What the page does, without Gradio (so it can be tested on its own)."""

    def __init__(self, ae, vae, generator, test_images: torch.Tensor, test_manifest=None, reference_errors=None):
        if test_images.dtype != torch.uint8 or test_images.dim() != 4:
            raise TypeError("test_images must be the uint8 test roofs, for example test.images.")
        self.ae, self.vae, self.generator = ae, vae, generator
        self.images = test_images.cpu()
        self.manifest = test_manifest
        self.errors = (torch.as_tensor(reference_errors, dtype=torch.float64) if reference_errors is not None
                       else reconstruction_errors(ae, self.images).double())
        self.count = len(self.images)

    def _index(self, i) -> int:
        return int(min(max(int(i), 0), self.count - 1))

    @torch.no_grad()
    def inspect(self, i):
        """Original, autoencoder copy and a line of text for test roof i."""
        i = self._index(i)
        self.ae.eval()
        device = next(self.ae.parameters()).device
        copy = to_bytes(self.ae(to_float(self.images[i:i + 1].to(device))))[0].cpu()
        error = float(self.errors[i])
        pct = float((self.errors < error).double().mean() * 100)
        where = f" from {self.manifest[i]['place']}" if self.manifest is not None else ""
        text = (f"**Roof {i}{where}**: unusual score {error:,.0f}, higher than {pct:.1f}% of the test roofs. "
                f"{'An inspector would look at this one.' if pct >= 95 else 'No inspection.'}")
        return _big(self.images[i]), _big(copy), text

    def morph(self, a, b, t):
        """The VAE's picture at fraction t of the way from test roof a to test roof b."""
        a, b, t = self._index(a), self._index(b), float(min(max(t, 0.0), 1.0))
        frames = morph_images(self.vae, self.images[a], self.images[b], steps=11)
        return _big(frames[int(round(t * 10))])

    def new_roof(self, seed):
        """One new roof from the GAN generator; the same seed gives the same roof."""
        return _big(to_bytes(self.generator.sample(1, seed=int(seed)))[0].cpu())


def build_app(ae, vae, generator, test_images, test_manifest=None, reference_errors=None):
    """The Gradio page (not started). launch() starts it."""
    gr = _require_gradio()
    logic = RoofApp(ae, vae, generator, test_images, test_manifest, reference_errors)
    last = logic.count - 1
    with gr.Blocks(title="Roof models") as page:
        gr.Markdown("### Roof models: inspect, morph, generate\nRoofs come from the test set only. "
                    "Nothing you do here leaves this notebook.")
        with gr.Tab("Inspect"):
            pick = gr.Slider(0, last, value=0, step=1, label="Test roof")
            with gr.Row():
                original = gr.Image(label="Original", type="numpy")
                copy = gr.Image(label="Autoencoder copy", type="numpy")
            verdict = gr.Markdown()
            pick.change(logic.inspect, pick, [original, copy, verdict])
        with gr.Tab("Morph"):
            with gr.Row():
                a = gr.Slider(0, last, value=0, step=1, label="Roof A")
                b = gr.Slider(0, last, value=min(1, last), step=1, label="Roof B")
            t = gr.Slider(0.0, 1.0, value=0.5, step=0.1, label="From A (0) to B (1)")
            blend = gr.Image(label="VAE picture", type="numpy")
            for control in (a, b, t):
                control.change(logic.morph, [a, b, t], blend)
        with gr.Tab("New roof"):
            seed = gr.Number(value=1, precision=0, label="Seed")
            button = gr.Button("Draw a new roof")
            new = gr.Image(label="GAN roof", type="numpy")
            button.click(logic.new_roof, seed, new)
        page.load(logic.inspect, pick, [original, copy, verdict])
    return page


def launch(ae, vae, generator, test_images, test_manifest=None, reference_errors=None, share: bool = False,
           **kwargs):
    """Start the page inside the notebook. Public share links are off for this lab."""
    if share:
        raise ValueError("Public share links are off for this lab: the page stays inside the notebook.")
    page = build_app(ae, vae, generator, test_images, test_manifest, reference_errors)
    return page.launch(share=False, **kwargs)
