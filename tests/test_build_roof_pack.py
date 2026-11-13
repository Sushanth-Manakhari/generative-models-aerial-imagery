"""Offline tests for scripts/build_roof_pack.py: geometry, filters and a full fake run (no network)."""
import csv
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("brp", ROOT / "scripts" / "build_roof_pack.py")
brp = importlib.util.module_from_spec(spec)
sys.modules["brp"] = brp
spec.loader.exec_module(brp)

RES = 0.1875  # metres per pixel used by the pilot


def rect_ring(cx, cy, long_m, short_m, angle_deg):
    """Closed ring (map metres, y up) of a rectangle whose long side points angle_deg counter-clockwise from east."""
    t = math.radians(angle_deg)
    u = np.array([math.cos(t), math.sin(t)])
    w = np.array([-u[1], u[0]])
    c = np.array([cx, cy])
    pts = [c + a * u * long_m / 2 + b * w * short_m / 2 for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    return np.asarray(pts + [pts[0]])


def scene(ring_m, origin, size_px, res=RES, roof=(200, 40, 40), ground=(90, 120, 80)):
    """Synthetic block image: green ground with one red roof drawn from ring_m."""
    img = Image.new("RGB", (size_px, size_px), ground)
    x0, y0 = origin
    px = [((x - x0) / res, (y0 - y) / res) for x, y in ring_m]
    ImageDraw.Draw(img).polygon(px, fill=roof)
    return img


def red_box(img):
    a = np.asarray(img).astype(int)
    red = (a[..., 0] > 150) & (a[..., 1] < 100)
    ys, xs = np.nonzero(red)
    return red, xs.min(), xs.max(), ys.min(), ys.max()


def settings(tmp_path, **kw):
    return brp.Settings(out=tmp_path / "pack", pause_s=0, **kw)


# ---------------------------------------------------------------- geometry
def test_area_centroid_of_rectangle():
    ring = rect_ring(100, 50, 16, 10, 0)
    area, c = brp.polygon_area_centroid(ring)
    assert area == pytest.approx(160, rel=1e-6)
    assert c == pytest.approx([100, 50], abs=1e-6)


@pytest.mark.parametrize("angle", [0, 17, 30, 45, 63, 90, 120, 170, -35])
def test_min_area_rect_finds_long_side(angle):
    ring = rect_ring(0, 0, 16, 10, angle)
    px = np.stack([ring[:, 0], -ring[:, 1]], axis=1)          # to y-down pixel axes
    u, lo, sh = brp.min_area_rect(px)
    assert lo == pytest.approx(16, abs=1e-4) and sh == pytest.approx(10, abs=1e-4)
    # turning by the returned angle makes the long side horizontal
    turned = brp.rotate_points(px, brp.turn_angle_deg(u), (0, 0))
    _, l2, s2 = brp.min_area_rect(turned)
    width = turned[:, 0].max() - turned[:, 0].min()
    height = turned[:, 1].max() - turned[:, 1].min()
    assert width == pytest.approx(16, abs=1e-4) and height == pytest.approx(10, abs=1e-4)


def test_rotate_points_matches_pillow():
    img = Image.new("L", (200, 200), 0)
    img.putpixel((150, 100), 255)
    out = img.rotate(90, resample=Image.NEAREST, center=(100, 100))
    ys, xs = np.nonzero(np.asarray(out))
    got = brp.rotate_points(np.array([[150.0, 100.0]]), 90, (100, 100))[0]
    assert abs(xs.mean() - got[0]) <= 1 and abs(ys.mean() - got[1]) <= 1


# ---------------------------------------------------------------- cutting one roof
@pytest.mark.parametrize("angle", [0, 25, 60, 135])
def test_cut_chip_turns_and_masks(tmp_path, angle):
    s = settings(tmp_path)
    origin = (1000.0, 2000.0)
    ring = rect_ring(1060, 1940, 14, 9, angle)
    img = scene(ring, origin, 640)
    chip = brp.cut_chip(img, origin, RES, ring, s, variants=("A", "B", "C"))
    assert chip.reason == ""
    assert chip.area_m2 == pytest.approx(126, rel=0.01)
    assert chip.long_m == pytest.approx(14, abs=0.05) and chip.short_m == pytest.approx(9, abs=0.05)
    for v in "ABC":
        assert chip.images[v].size == (128, 128)

    # B: long side horizontal, roof centred, red rectangle fills its own bounding box
    red, x0, x1, y0, y1 = red_box(chip.images["B"])
    w, h = (x1 - x0 + 1) * RES, (y1 - y0 + 1) * RES
    assert w == pytest.approx(14, abs=0.6) and h == pytest.approx(9, abs=0.6)
    assert red[y0:y1 + 1, x0:x1 + 1].mean() > 0.95
    assert abs((x0 + x1) / 2 - 63.5) <= 2 and abs((y0 + y1) / 2 - 63.5) <= 2

    # C: everything well outside the roof is black, the roof itself is kept
    a = np.asarray(chip.images["C"]).astype(int)
    assert a[:6].max() <= 3 and a[-6:].max() <= 3
    assert red_box(chip.images["C"])[0].sum() > 0.9 * red.sum()


def test_north_up_keeps_original_angle(tmp_path):
    s = settings(tmp_path)
    origin = (0.0, 120.0)
    ring = rect_ring(60, 60, 14, 9, 45)
    chip = brp.cut_chip(scene(ring, origin, 640), origin, RES, ring, s, variants=("A",))
    red, x0, x1, y0, y1 = red_box(chip.images["A"])
    assert red[y0:y1 + 1, x0:x1 + 1].mean() < 0.75        # a 45-degree rectangle fills its box poorly


@pytest.mark.parametrize("long_m,short_m,reason", [(5, 4, "area"), (30, 12, "area"), (22, 8, "shape"), (20, 3, "shape")])
def test_size_and_shape_filters(tmp_path, long_m, short_m, reason):
    s = settings(tmp_path)
    origin = (0.0, 120.0)
    ring = rect_ring(60, 60, long_m, short_m, 10)
    chip = brp.cut_chip(scene(ring, origin, 640), origin, RES, ring, s)
    assert chip.reason == reason


def test_nodata_and_edge_rejected(tmp_path):
    s = settings(tmp_path)
    origin = (0.0, 120.0)
    ring = rect_ring(60, 60, 14, 9, 0)
    img = scene(ring, origin, 640)
    a = np.asarray(img).copy()
    a[:, :300] = 0                                         # left half of the block has no imagery
    assert brp.cut_chip(Image.fromarray(a), origin, RES, ring, s).reason == "nodata"
    near_edge = rect_ring(4, 60, 14, 9, 0)
    assert brp.cut_chip(scene(near_edge, origin, 640), origin, RES, near_edge, s).reason == "edge"


def test_block_grid_covers_extent_without_overlap():
    blocks = brp.block_grid((0, 0, 1300, 700), 600, 37.5)
    assert len(blocks) == 6
    assert all(b[2] - b[0] <= 600 and b[3] - b[1] <= 600 for b in blocks)


# ---------------------------------------------------------------- full fake run (no network)
class FakeImages:
    def __init__(self, homes, url, wkid, pause_s=0, naip=False):
        self.homes, self.naip, self.calls = homes, naip, 0

    def info(self):
        return {"bandCount": 4, "pixelSizeX": 0.075}

    def export(self, bbox, size_px):
        self.calls += 1
        xmin, ymin, xmax, ymax = bbox
        res = (xmax - xmin) / size_px
        img = Image.new("RGB", (size_px, size_px), (90, 120, 80))
        d = ImageDraw.Draw(img)
        for ring in self.homes:
            d.polygon([((x - xmin) / res, (ymax - y) / res) for x, y in ring], fill=(200, 40, 40))
        return img


class FakeFootprints:
    def __init__(self, homes, *a, **k):
        self.homes, self.where = homes, "OCC_CLS = 'Residential'"
        self.fields = brp.FOOTPRINTS["fields"]

    def _inside(self, env):
        xmin, ymin, xmax, ymax = env
        out = []
        for r in self.homes:
            if r[:, 0].max() >= xmin and r[:, 0].min() <= xmax and r[:, 1].max() >= ymin and r[:, 1].min() <= ymax:
                out.append(r)
        return out

    def count(self, env, where=None):
        return len(self._inside(env))

    def query(self, env):
        return [brp.Footprint(r, "Residential", "Single Family Dwelling") for r in self._inside(env)]


def test_full_fake_run_writes_pack_without_identifying_fields(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    extent = (0.0, 0.0, 1275.0, 675.0)                     # two blocks wide, one high
    homes = [rect_ring(x, y, rng.uniform(10, 16), rng.uniform(7, 10), rng.uniform(0, 180))
             for x in np.arange(60, 1220, 30) for y in np.arange(60, 620, 30)]
    monkeypatch.setattr(brp, "ImageService", lambda url, wkid, pause_s=0, **kw: FakeImages(homes, url, wkid))
    monkeypatch.setattr(brp, "FootprintService", lambda *a, **k: FakeFootprints(homes))
    src = dict(brp.SOUTH_PORTLAND_2022, extent=extent)
    s = settings(tmp_path, target=120, variants=("A", "B", "C"), naip_compare_blocks=1, zip_output=True,
                 max_chips_per_block=80, min_homes_per_block=1)
    report = brp.build(s, src)
    assert report["chips"] == 120 and report["blocks_used"] == 2
    for d in ("A_north_up", "B_turned", "C_turned_masked"):
        assert len(list((s.out / d).glob("*.jpg"))) == 120
    assert len(list((s.out / "N_naip_turned_masked").glob("*.jpg"))) > 0
    rows = list(csv.DictReader(open(s.out / "manifest.csv")))
    assert len(rows) == 120
    banned = {"address", "addr", "lat", "lon", "longitude", "latitude", "x", "y", "build_id", "block", "zip", "city"}
    assert not {c.lower() for c in rows[0]} & banned
    rep = json.loads((s.out / "build_report.json").read_text())
    assert rep["chips"] == 120 and "ATTRIBUTION.txt" in [p.name for p in s.out.iterdir()]
    assert Path(report["zip"]).exists()
    assert "PROP_ADDR" not in brp.FOOTPRINTS["fields"] and all("ADDR" not in f for f in brp.FOOTPRINTS["fields"])
