"""An optional explorer for the three models, drawn inside the notebook as one small web page.

    from genroofs.app import launch
    launch(ae, vae, generator, test.images, test.manifest)

Three tabs: Inspect (a test roof, its autoencoder copy and its unusual score), Morph (slide from roof A to roof B
through the VAE) and New roof (GAN roofs).

When launch() runs, Python prepares every picture once (a few seconds). After that the page needs nothing more from
Python: the controls swap pictures that are already there, so they answer at once, even while another cell is
training. No web server, no proxy, no link and no ipywidgets. It was built this way in G4.8 because a page served
from a Colab session stays blank in Safari's private windows, and an ipywidgets panel stopped responding there.

What the page offers is a fixed set prepared from the test set only (there is no upload box):
  Inspect   24 roofs: the 8 most ordinary, the 8 strangest and 8 at random, ordered from ordinary to strange
  Morph     any two of 6 roofs, in 11 steps
  New roof  24 GAN roofs (seeds 1 to 24)
"""
import base64
import html
import io
import json
import re
import secrets

import numpy as np
import torch
from PIL import Image

from .data import to_bytes, to_float
from .metrics import reconstruction_errors
from .viz import morph_images

__all__ = ["RoofApp", "choose_roofs", "build_page", "launch"]

SHOW_PX = 256       # roofs are 64 px; the page shows them 4 times larger, keeping the pixels square
MORPH_STEPS = 11    # A to B in steps of 0.1


def _array(image: torch.Tensor) -> np.ndarray:
    """A 3 x 64 x 64 uint8 tensor as a 64 x 64 x 3 array."""
    return image.detach().cpu().permute(1, 2, 0).numpy()


def _uri(arr: np.ndarray, exact: bool) -> str:
    """A picture as a data URI: PNG for real roofs (exact), high-quality JPEG for what the models draw (smaller)."""
    buf = io.BytesIO()
    if exact:
        Image.fromarray(arr).save(buf, format="PNG", optimize=True)
        kind = "png"
    else:
        Image.fromarray(arr).save(buf, format="JPEG", quality=95, subsampling=0)
        kind = "jpeg"
    return f"data:image/{kind};base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _html(markdown_bold: str) -> str:
    """Escape text for the page, then turn **x** into bold."""
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", html.escape(markdown_bold, quote=False))


def choose_roofs(errors, n: int = 24, seed: int = 7) -> list:
    """Test-roof positions for the Inspect tab: the n/3 lowest scores, the n/3 highest and the rest at random from
    the middle, ordered from the most ordinary to the strangest. A test set of n roofs or fewer is used whole."""
    errors = torch.as_tensor(errors, dtype=torch.float64)
    count, n = len(errors), int(n)
    order = sorted(range(count), key=lambda i: (float(errors[i]), i))
    if count <= n:
        return order
    k = n // 3
    middle = order[k:count - k]
    gen = torch.Generator().manual_seed(int(seed))
    picked = [middle[j] for j in torch.randperm(len(middle), generator=gen)[:n - 2 * k].tolist()]
    chosen = set(order[:k]) | set(order[count - k:]) | set(picked)
    return sorted(chosen, key=lambda i: (float(errors[i]), i))


class RoofApp:
    """What the explorer shows, without any page (so it can be tested on its own). Pictures are 64 x 64 x 3 arrays."""

    def __init__(self, ae, vae, generator, test_images: torch.Tensor, test_manifest=None, reference_errors=None):
        if test_images.dtype != torch.uint8 or test_images.dim() != 4:
            raise TypeError("test_images must be the uint8 test roofs, for example test.images.")
        self.ae, self.vae, self.generator = ae, vae, generator
        self.images = test_images.cpu()
        self.manifest = test_manifest
        self.errors = (torch.as_tensor(reference_errors, dtype=torch.float64) if reference_errors is not None
                       else reconstruction_errors(ae, self.images).double())
        if len(self.errors) != len(self.images):
            raise ValueError("reference_errors must have one score per test roof.")
        self.count = len(self.images)

    def _index(self, i) -> int:
        return int(min(max(int(i), 0), self.count - 1))

    @torch.no_grad()
    def inspect(self, i):
        """Original, autoencoder copy and a line of text for test roof i."""
        i = self._index(i)
        was_training = self.ae.training
        self.ae.eval()
        try:
            device = next(self.ae.parameters()).device
            copy = to_bytes(self.ae(to_float(self.images[i:i + 1].to(device))))[0].cpu()
        finally:
            self.ae.train(was_training)
        error = float(self.errors[i])
        pct = float((self.errors < error).double().mean() * 100)
        where = f" from {self.manifest[i]['place']}" if self.manifest is not None else ""
        text = (f"**Test roof {i}{where}**: unusual score {error:,.0f}, higher than {pct:.1f}% of the test roofs. "
                f"{'An inspector would look at this one.' if pct >= 95 else 'No inspection.'}")
        return _array(self.images[i]), _array(copy), text

    def morph_frames(self, a, b, steps: int = MORPH_STEPS) -> list:
        """The VAE's pictures on the straight line from test roof a to test roof b, a first."""
        frames = morph_images(self.vae, self.images[self._index(a)], self.images[self._index(b)], steps=steps)
        return [_array(f) for f in frames]

    def morph(self, a, b, t):
        """The VAE's picture at fraction t (0 to 1, in steps of 0.1) of the way from test roof a to test roof b."""
        t = float(min(max(t, 0.0), 1.0))
        return self.morph_frames(a, b)[int(round(t * (MORPH_STEPS - 1)))]

    def new_roof(self, seed):
        """One new roof from the GAN generator; the same seed gives the same roof."""
        return _array(to_bytes(self.generator.sample(1, seed=int(seed)))[0].cpu())


def _page_data(logic: RoofApp, n_inspect: int, n_morph: int, n_new: int, seed: int) -> dict:
    """Every picture and line of text the page needs, prepared once."""
    inspect = []
    for i in choose_roofs(logic.errors, n_inspect, seed):
        original, copy, text = logic.inspect(i)
        inspect.append({"original": _uri(original, True), "copy": _uri(copy, False), "text": _html(text)})

    gen = torch.Generator().manual_seed(int(seed) + 1)
    roofs = torch.randperm(logic.count, generator=gen)[:min(int(n_morph), logic.count)].tolist()
    pairs = {}
    for x in range(len(roofs)):
        for y in range(x, len(roofs)):
            frames = logic.morph_frames(roofs[x], roofs[y])
            pairs[f"{x}-{y}"] = [_uri(f, False) for f in frames]
    morph = {"roofs": [_uri(_array(logic.images[r]), True) for r in roofs], "test_ids": roofs, "pairs": pairs}

    new = [{"seed": s, "picture": _uri(logic.new_roof(s), False)} for s in range(1, int(n_new) + 1)]
    return {"inspect": inspect, "morph": morph, "new": new}


_STYLE = """
#{id} {{ font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 14px;
        color: inherit; max-width: 780px; line-height: 1.4; }}
#{id} .gr-tabs {{ display: flex; gap: 4px; border-bottom: 1px solid rgba(128,128,128,.5); margin: 8px 0 12px; }}
#{id} .gr-tab {{ font: inherit; color: inherit; background: transparent; border: 1px solid rgba(128,128,128,.5);
        border-bottom: none; border-radius: 6px 6px 0 0; padding: 6px 14px; cursor: pointer; opacity: .7; }}
#{id} .gr-tab.gr-on {{ opacity: 1; font-weight: 600; border-color: #3b82f6; box-shadow: inset 0 2px 0 #3b82f6; }}
#{id} .gr-row {{ display: flex; flex-wrap: wrap; gap: 12px; align-items: flex-start; }}
#{id} figure {{ margin: 0; }}
#{id} figcaption {{ font-weight: 600; margin-bottom: 4px; }}
#{id} img.gr-big {{ width: {px}px; height: {px}px; image-rendering: crisp-edges; image-rendering: pixelated;
        display: block; border-radius: 4px; }}
#{id} img.gr-thumb {{ width: 64px; height: 64px; image-rendering: crisp-edges; image-rendering: pixelated;
        display: block; border: 3px solid transparent; border-radius: 4px; }}
#{id} img.gr-thumb.gr-a {{ border-color: #3b82f6; }}
#{id} img.gr-thumb.gr-b {{ border-color: #f59e0b; }}
#{id} .gr-thumbs {{ display: flex; gap: 6px; margin-bottom: 10px; text-align: center; }}
#{id} .gr-control {{ display: flex; align-items: center; gap: 10px; margin: 6px 0; }}
#{id} .gr-control label {{ min-width: 72px; }}
#{id} .gr-control input[type=range] {{ width: 260px; }}
#{id} .gr-button {{ font: inherit; color: inherit; background: transparent; border: 1px solid rgba(128,128,128,.7);
        border-radius: 6px; padding: 6px 14px; cursor: pointer; }}
#{id} .gr-note {{ opacity: .75; font-size: 13px; }}
"""

_SCRIPT = """
(function () {
  const root = document.getElementById(%(id)s);
  if (!root) { return; }
  const D = %(data)s;
  const q = (sel) => root.querySelector(sel);
  const qa = (sel) => Array.prototype.slice.call(root.querySelectorAll(sel));

  qa(".gr-tab").forEach(function (tab) {
    tab.addEventListener("click", function () {
      qa(".gr-tab").forEach(function (t) {
        const on = t === tab;
        t.classList.toggle("gr-on", on);
        t.setAttribute("aria-selected", on ? "true" : "false");
      });
      qa(".gr-page").forEach(function (p) { p.hidden = p.getAttribute("data-page") !== tab.getAttribute("data-tab"); });
    });
  });

  const pick = q(".gr-pick");
  function showInspect() {
    const n = Number(pick.value), r = D.inspect[n];
    q(".gr-pick-n").textContent = (n + 1) + " of " + D.inspect.length;
    q(".gr-original").src = r.original;
    q(".gr-copy").src = r.copy;
    q(".gr-verdict").innerHTML = r.text;
  }
  pick.addEventListener("input", showInspect);

  const sa = q(".gr-a-slider"), sb = q(".gr-b-slider"), st = q(".gr-t-slider");
  function showMorph() {
    const a = Number(sa.value), b = Number(sb.value), t = Number(st.value), last = D.steps - 1;
    const frame = a <= b ? D.morph.pairs[a + "-" + b][t] : D.morph.pairs[b + "-" + a][last - t];
    q(".gr-blend").src = frame;
    q(".gr-a-n").textContent = String(a + 1);
    q(".gr-b-n").textContent = String(b + 1);
    q(".gr-t-n").textContent = (t / last).toFixed(1);
    qa(".gr-thumb").forEach(function (img, k) {
      img.classList.toggle("gr-a", k === a);
      img.classList.toggle("gr-b", k === b && k !== a);
    });
  }
  [sa, sb, st].forEach(function (s) { s.addEventListener("input", showMorph); });

  let k = 0;
  function showNew() {
    const r = D.new[k];
    q(".gr-new").src = r.picture;
    q(".gr-new-n").textContent = "GAN roof " + (k + 1) + " of " + D.new.length + " (seed " + r.seed + ")";
  }
  q(".gr-draw").addEventListener("click", function () { k = (k + 1) %% D.new.length; showNew(); });

  qa(".gr-page").forEach(function (p) { p.hidden = p.getAttribute("data-page") !== "0"; });
  q(".gr-nojs").hidden = true;
  showInspect(); showMorph(); showNew();
})();
"""


def _script_json(value) -> str:
    """JSON that is safe inside a <script> element."""
    return json.dumps(value, separators=(",", ":")).replace("<", "\\u003c").replace(">", "\\u003e")


def build_page(ae, vae, generator, test_images, test_manifest=None, reference_errors=None,
               n_inspect: int = 24, n_morph: int = 6, n_new: int = 24, seed: int = 7) -> str:
    """The explorer as one self-contained HTML fragment (pictures inside it as data URIs; no links)."""
    logic = RoofApp(ae, vae, generator, test_images, test_manifest, reference_errors)
    d = _page_data(logic, n_inspect, n_morph, n_new, seed)
    d["steps"] = MORPH_STEPS
    pid = "genroofs-" + secrets.token_hex(4)
    first, n_ins, n_roofs = d["inspect"][0], len(d["inspect"]), len(d["morph"]["roofs"])
    b = min(1, n_roofs - 1)
    mid = (MORPH_STEPS - 1) // 2
    start_blend = d["morph"]["pairs"][f"0-{b}"][mid]
    thumbs = "".join(
        f'<figure><img class="gr-thumb{" gr-a" if j == 0 else ""}{" gr-b" if j == b and j != 0 else ""}" '
        f'src="{u}" alt="Roof {j + 1}"><figcaption>{j + 1}</figcaption></figure>'
        for j, u in enumerate(d["morph"]["roofs"]))

    def slider(cls, label, lo, hi, value, shown):
        return (f'<div class="gr-control"><label>{label}</label><input class="{cls}-slider" type="range" '
                f'min="{lo}" max="{hi}" step="1" value="{value}" aria-label="{label}"> '
                f'<span class="{cls}-n">{shown}</span></div>')

    page = f"""
<div id="{pid}">
<style>{_STYLE.format(id=pid, px=SHOW_PX)}</style>
<div><b>Roof models: inspect, morph, generate.</b> Roofs come from the test set only. The pictures were prepared
when this cell ran, so the controls answer at once, even while another cell is running.</div>
<div class="gr-tabs" role="tablist">
  <button class="gr-tab gr-on" data-tab="0" role="tab" aria-selected="true">Inspect</button>
  <button class="gr-tab" data-tab="1" role="tab" aria-selected="false">Morph</button>
  <button class="gr-tab" data-tab="2" role="tab" aria-selected="false">New roof</button>
</div>
<div class="gr-page" data-page="0">
  <div class="gr-control"><label>Roof</label><input class="gr-pick" type="range" min="0" max="{n_ins - 1}" step="1"
    value="0" aria-label="Roof"> <span class="gr-pick-n">1 of {n_ins}</span></div>
  <div class="gr-row">
    <figure><figcaption>Original</figcaption><img class="gr-big gr-original" src="{first['original']}" alt="Original"></figure>
    <figure><figcaption>Autoencoder copy</figcaption><img class="gr-big gr-copy" src="{first['copy']}" alt="Copy"></figure>
  </div>
  <p class="gr-verdict">{first['text']}</p>
  <p class="gr-note">Left end of the slider: the most ordinary roofs. Right end: the strangest. In between: roofs
  picked at random.</p>
</div>
<div class="gr-page" data-page="1">
  <div class="gr-thumbs">{thumbs}</div>
  {slider("gr-a", "Roof A", 0, n_roofs - 1, 0, 1)}
  {slider("gr-b", "Roof B", 0, n_roofs - 1, b, b + 1)}
  {slider("gr-t", "A to B", 0, MORPH_STEPS - 1, mid, f"{mid / (MORPH_STEPS - 1):.1f}")}
  <figure><figcaption>VAE picture</figcaption><img class="gr-big gr-blend" src="{start_blend}" alt="VAE picture"></figure>
  <p class="gr-note">Blue frame: roof A. Orange frame: roof B.</p>
</div>
<div class="gr-page" data-page="2">
  <div class="gr-control"><button class="gr-button gr-draw" type="button">Draw a new roof</button>
    <span class="gr-new-n">GAN roof 1 of {len(d['new'])} (seed {d['new'][0]['seed']})</span></div>
  <figure><img class="gr-big gr-new" src="{d['new'][0]['picture']}" alt="GAN roof"></figure>
</div>
<p class="gr-note gr-nojs">The controls need JavaScript, which this viewer does not run; the first picture of each
tab is shown.</p>
<script>{_SCRIPT % {"id": json.dumps(pid), "data": _script_json(d)}}</script>
</div>
"""
    return page.strip()


def launch(ae, vae, generator, test_images, test_manifest=None, reference_errors=None, **options) -> None:
    """Prepare the pictures and show the explorer below the cell. Options: n_inspect, n_morph, n_new, seed."""
    from IPython.display import HTML, display
    display(HTML(build_page(ae, vae, generator, test_images, test_manifest, reference_errors, **options)))
