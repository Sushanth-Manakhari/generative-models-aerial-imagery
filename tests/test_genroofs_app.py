"""Offline tests for genroofs.app: the explorer's logic, and the self-contained page it shows in the notebook."""
import base64
import io
import json
import re
import sys

import numpy as np
import pytest
import torch
from PIL import Image

from conftest import make_squares
from genroofs import app
from genroofs.models import Generator, RoofAE, RoofVAE


@pytest.fixture
def logic():
    x = make_squares(12)
    manifest = [{"place": "Gorham"}] * 12
    return app.RoofApp(RoofAE(latent=8), RoofVAE(), Generator(), x, manifest)


def page_data(page: str) -> dict:
    """The data object the page's script uses."""
    found = re.search(r"const D = (\{.*?\});\n", page)
    assert found, "the page carries its data in the script"
    return json.loads(found.group(1))


def picture(uri: str):
    kind, b64 = re.fullmatch(r"data:image/(png|jpeg);base64,([A-Za-z0-9+/=]+)", uri).groups()
    with Image.open(io.BytesIO(base64.b64decode(b64))) as im:
        return kind, im.size, np.asarray(im.convert("RGB"))


def test_inspect(logic):
    original, copy, text = logic.inspect(3)
    assert original.shape == copy.shape == (64, 64, 3) and original.dtype == np.uint8
    assert np.array_equal(original, logic.images[3].permute(1, 2, 0).numpy())
    assert text.startswith("**Test roof 3 from Gorham**") and "unusual score" in text
    assert logic.inspect(99)[2].startswith("**Test roof 11"), "out-of-range picks are clamped to the test set"
    assert logic.inspect(-5)[2].startswith("**Test roof 0")


def test_morph_and_new_roof(logic):
    frames = logic.morph_frames(0, 5)
    assert len(frames) == 11 and frames[0].shape == (64, 64, 3)
    assert np.array_equal(logic.morph(0, 5, 0.0), frames[0]) and np.array_equal(logic.morph(0, 5, 1.0), frames[-1])
    assert not np.array_equal(frames[0], frames[-1])
    assert np.array_equal(logic.new_roof(4), logic.new_roof(4))
    assert not np.array_equal(logic.new_roof(4), logic.new_roof(5))


def test_only_test_roofs_are_accepted():
    with pytest.raises(TypeError, match="uint8 test roofs"):
        app.RoofApp(RoofAE(), RoofVAE(), Generator(), torch.zeros(3, 3, 64, 64))
    with pytest.raises(ValueError, match="one score per test roof"):
        app.RoofApp(RoofAE(), RoofVAE(), Generator(), make_squares(4), reference_errors=[1.0, 2.0])


def test_text_is_escaped_then_bold():
    assert app._html("**Roof <3>** & co") == "<b>Roof &lt;3&gt;</b> &amp; co"


def test_choose_roofs():
    errors = torch.randperm(100, generator=torch.Generator().manual_seed(1)).double()
    chosen = app.choose_roofs(errors, n=24, seed=7)
    scores = [float(errors[i]) for i in chosen]
    assert len(chosen) == len(set(chosen)) == 24
    assert scores == sorted(scores), "ordered from the most ordinary to the strangest"
    assert scores[:8] == list(range(8)) and scores[-8:] == list(range(92, 100)), "8 lowest and 8 highest"
    assert all(8 <= s < 92 for s in scores[8:16]), "8 more from the middle"
    assert chosen == app.choose_roofs(errors, n=24, seed=7)
    assert app.choose_roofs([3.0, 1.0, 2.0], n=24) == [1, 2, 0], "a small test set is used whole"


def test_page_is_self_contained_and_complete():
    x = make_squares(30)
    page = app.build_page(RoofAE(latent=8), RoofVAE(), Generator(), x, [{"place": "York"}] * 30)
    assert page.startswith('<div id="genroofs-') and page.count("<script>") == 1
    assert "http://" not in page and "https://" not in page and "<script src" not in page, "no links of any kind"
    assert [t in page for t in (">Inspect<", ">Morph<", ">New roof<")] == [True] * 3
    assert len(re.findall(r'data-page="\d"', page)) == 3
    assert 'class="gr-pick" type="range" min="0" max="23"' in page
    assert 'class="gr-a-slider" type="range" min="0" max="5"' in page
    assert 'class="gr-t-slider" type="range" min="0" max="10"' in page
    d = page_data(page)
    assert d["steps"] == 11 and len(d["inspect"]) == 24 and len(d["new"]) == 24
    assert [r["seed"] for r in d["new"]] == list(range(1, 25))
    assert len(d["morph"]["roofs"]) == 6 and len(d["morph"]["pairs"]) == 21
    assert all(len(f) == 11 for f in d["morph"]["pairs"].values())
    assert all(r["text"].startswith("<b>Test roof ") and " from York</b>" in r["text"] for r in d["inspect"])
    kind, size, original = picture(d["inspect"][0]["original"])
    assert kind == "png" and size == (64, 64), "real roofs are kept exact"
    assert any(np.array_equal(original, x[i].permute(1, 2, 0).numpy()) for i in range(30))
    assert picture(d["inspect"][0]["copy"])[:2] == ("jpeg", (64, 64))
    assert picture(d["new"][0]["picture"])[:2] == ("jpeg", (64, 64))
    assert d["inspect"][0]["original"] in page.split("<script>")[0], "the first pictures show even without JavaScript"


def test_page_is_the_same_each_time_apart_from_its_name():
    x = make_squares(10)
    models = (RoofAE(latent=8), RoofVAE(), Generator())
    first, second = app.build_page(*models, x), app.build_page(*models, x)
    assert page_data(first) == page_data(second)
    assert first.split('"', 2)[1] != second.split('"', 2)[1], "each page has its own name, so two pages never clash"


def test_page_with_few_test_roofs():
    d = page_data(app.build_page(RoofAE(latent=8), RoofVAE(), Generator(), make_squares(3)))
    assert len(d["inspect"]) == 3 and len(d["morph"]["roofs"]) == 3 and len(d["morph"]["pairs"]) == 6


def test_place_names_cannot_inject_markup():
    page = app.build_page(RoofAE(latent=8), RoofVAE(), Generator(), make_squares(4),
                          [{"place": "<script>alert(1)</script>"}] * 4, n_inspect=4)
    assert "<script>alert" not in page and page.count("<script>") == 1
    assert "&lt;script&gt;" in page.split("<script>")[0]


def test_no_ipywidgets_needed(monkeypatch):
    monkeypatch.setitem(sys.modules, "ipywidgets", None)
    assert "genroofs-" in app.build_page(RoofAE(latent=8), RoofVAE(), Generator(), make_squares(4))


def test_launch_shows_the_page(monkeypatch):
    shown = []
    import IPython.display
    monkeypatch.setattr(IPython.display, "display", lambda obj: shown.append(obj))
    assert app.launch(RoofAE(latent=8), RoofVAE(), Generator(), make_squares(4), n_new=3) is None
    assert len(shown) == 1 and isinstance(shown[0], IPython.display.HTML)
    assert len(page_data(shown[0].data)["new"]) == 3
