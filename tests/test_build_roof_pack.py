"""Offline tests for scripts/build_roof_pack.py (no network): geometry, map coordinates, tile stitching,
outline alignment, filters, full fake builds of both source kinds, resume, coverage skip and privacy."""
import csv
import hashlib
import importlib.util
import json
import math
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("brp", ROOT / "scripts" / "build_roof_pack.py")
brp = importlib.util.module_from_spec(spec)
sys.modules["brp"] = brp
spec.loader.exec_module(brp)

RES = 0.1875  # metres per pixel used for the packs
GROUND, ROOF = (90, 120, 80), (200, 40, 40)


def rect_ring(cx, cy, long_m, short_m, angle_deg):
    """Closed ring (map units, y up) of a rectangle whose long side points angle_deg counter-clockwise from east."""
    t = math.radians(angle_deg)
    u = np.array([math.cos(t), math.sin(t)])
    w = np.array([-u[1], u[0]])
    c = np.array([cx, cy])
    pts = [c + a * u * long_m / 2 + b * w * short_m / 2 for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    return np.asarray(pts + [pts[0]])


def scene(rings, origin, size_px, res=RES):
    """Synthetic block image: green ground with red roofs."""
    img = Image.new("RGB", (size_px, size_px), GROUND)
    d = ImageDraw.Draw(img)
    x0, y0 = origin
    for ring in rings:
        d.polygon([((x - x0) / res, (y0 - y) / res) for x, y in ring], fill=ROOF)
    return img


def red_box(img):
    a = np.asarray(img).astype(int)
    red = (a[..., 0] > 150) & (a[..., 1] < 100)
    ys, xs = np.nonzero(red)
    return red, xs.min(), xs.max(), ys.min(), ys.max()


def settings(tmp_path, **kw):
    kw.setdefault("pause_s", 0)
    kw.setdefault("tile_pause_s", 0)
    return brp.Settings(out=tmp_path / "pack", **kw)


# ---------------------------------------------------------------- geometry
def test_area_centroid_of_rectangle():
    area, c = brp.polygon_area_centroid(rect_ring(100, 50, 16, 10, 0))
    assert area == pytest.approx(160, rel=1e-6)
    assert c == pytest.approx([100, 50], abs=1e-6)


@pytest.mark.parametrize("angle", [0, 17, 30, 45, 63, 90, 120, 170, -35])
def test_min_area_rect_finds_long_side(angle):
    ring = rect_ring(0, 0, 16, 10, angle)
    px = np.stack([ring[:, 0], -ring[:, 1]], axis=1)          # to y-down pixel axes
    u, lo, sh = brp.min_area_rect(px)
    assert lo == pytest.approx(16, abs=1e-4) and sh == pytest.approx(10, abs=1e-4)
    turned = brp.rotate_points(px, brp.turn_angle_deg(u), (0, 0))
    assert np.ptp(turned[:, 0]) == pytest.approx(16, abs=1e-4)
    assert np.ptp(turned[:, 1]) == pytest.approx(10, abs=1e-4)


def test_rotate_points_matches_pillow():
    img = Image.new("L", (200, 200), 0)
    img.putpixel((150, 100), 255)
    out = img.rotate(90, resample=Image.NEAREST, center=(100, 100))
    ys, xs = np.nonzero(np.asarray(out))
    got = brp.rotate_points(np.array([[150.0, 100.0]]), 90, (100, 100))[0]
    assert abs(xs.mean() - got[0]) <= 1 and abs(ys.mean() - got[1]) <= 1


# ---------------------------------------------------------------- map coordinates (reference values from pyproj)
@pytest.mark.parametrize("lon,lat,zone,x,y", [
    (-70.25, 43.64, 19, 399177.995, 4832648.758),
    (-72.735, 41.77, 18, 688261.491, 4626719.602),
    (-85.075, 41.12, 16, 661605.633, 4553864.043),
    (-97.675, 30.51, 14, 627140.223, 3376046.118),
    (-69.0, 47.0, 19, 500000.000, 5205164.110),
])
def test_lonlat_to_utm_matches_reference(lon, lat, zone, x, y):
    gx, gy = brp.lonlat_to_utm(lon, lat, zone)
    assert gx == pytest.approx(x, abs=0.05) and gy == pytest.approx(y, abs=0.05)


def test_web_mercator_round_trip_and_scale():
    x, y = brp.lonlat_to_merc(-111.925, 33.36)
    assert x == pytest.approx(-12459434.007, abs=0.01) and y == pytest.approx(3943186.012, abs=0.01)
    assert brp.merc_y_to_lat(y) == pytest.approx(33.36, abs=1e-9)
    env = brp.lonlat_box_to([-111.95, 33.34, -111.90, 33.38], 3857)
    assert env[0] < env[2] and env[1] < env[3]


def test_every_configured_source_is_complete():
    cfg = brp.load_config()
    assert set(cfg["packs"]) == {"train", "variety"}
    for pack in cfg["packs"].values():
        assert pack["sources"] and pack["target"] > 0
        for sid in pack["sources"]:
            src = cfg["sources"][sid]
            for key in ("kind", "region", "place", "year", "url", "licence", "attribution"):
                assert src.get(key), f"{sid} lacks {key}"
            assert src["url"].startswith("https://")
            if src["kind"] == "image":
                assert src.get("area") == "service" or len(src["area_lonlat"]) == 4
                brp.utm_zone_of(int(src["wkid"]))
            else:
                assert src["kind"] == "tiles" and isinstance(src["level"], int) and len(src["area_lonlat"]) == 4
            if "area_lonlat" in src:                         # neighbourhood-sized boxes, never a single home
                lon0, lat0, lon1, lat1 = src["area_lonlat"]
                assert 0.02 <= lon1 - lon0 <= 0.2 and 0.02 <= lat1 - lat0 <= 0.2
    fields = cfg["footprints"]["fields"]
    assert set(fields) == {"OCC_CLS", "PRIM_OCC", "SQMETERS"} and not any("ADDR" in f for f in fields)
    train = cfg["packs"]["train"]["sources"]
    assert len(train) == 8 and "me_wells_2022" in train          # Wells included at the instructor's decision
    assert all(cfg["sources"][sid]["region"] == "Maine" for sid in train)


# ---------------------------------------------------------------- tile stitching
def fake_tile_world(level_res, T=256, origin=(-1000.0, 1000.0)):
    """A pretend tile server: the world is green with one red square from x 100..140, y 100..140."""
    def fetch(url):
        level, r, c = map(int, url.rsplit("/tile/", 1)[1].split("/"))
        res = level_res[level]
        x0, y0 = origin[0] + c * res * T, origin[1] - r * res * T
        if x0 > 900:                                         # tiles past x = 900 do not exist (404)
            return None
        img = Image.new("RGB", (T, T), GROUND)
        px = lambda x, y: ((x - x0) / res, (y0 - y) / res)
        ImageDraw.Draw(img).polygon([px(100, 100), px(140, 100), px(140, 140), px(100, 140)], fill=ROOF)
        return img
    return fetch


def test_tile_service_stitches_the_requested_box():
    level_res = {0: 4.0, 1: 2.0, 2: 0.5}
    ti = {"origin": {"x": -1000.0, "y": 1000.0}, "cols": 256, "rows": 256,
          "lods": [{"level": l, "resolution": r} for l, r in level_res.items()]}
    svc = brp.TileService("https://example.test/MapServer", level=2, pause_s=0, tile_pause_s=0,
                          fetch=fake_tile_world(level_res))
    svc.set_tile_info(ti)
    img = svc.export((80, 80, 160, 160), 160)                 # 0.5 units per pixel: level 2
    red, x0, x1, y0, y1 = red_box(img)
    assert abs(x0 - 40) <= 1 and abs(x1 - 119) <= 1 and abs(y0 - 40) <= 1 and abs(y1 - 119) <= 1
    assert svc._level_for(3.0)[0] == 1 and svc._level_for(100)[0] == 0 and svc._level_for(0.1)[0] == 2
    missing = svc.export((880, 80, 960, 160), 64)              # half of this box has no tiles
    assert 0.3 < brp.nodata_fraction(missing) < 0.8
    with pytest.raises(brp.ServiceError):
        brp.TileService("https://example.test/x", level=9, fetch=lambda u: None).set_tile_info(ti)


# ---------------------------------------------------------------- outline alignment
def test_block_shift_recovers_known_offset():
    rng = np.random.default_rng(1)
    origin, size = (0.0, 600.0), 3200
    true = (2.6, -1.9)                                        # roofs sit 2.6 m east and 1.9 m south of outlines
    rings = [rect_ring(x + rng.uniform(-3, 3), y + rng.uniform(-3, 3), rng.uniform(10, 16), rng.uniform(7, 10),
                       rng.uniform(0, 180)) for x in range(40, 560, 45) for y in range(40, 560, 45)]
    roofs = [r + np.array(true) for r in rings]
    img = scene(roofs, origin, size)
    a = np.asarray(img).astype(np.int16)
    a = np.clip(a + rng.normal(0, 12, a.shape), 0, 255).astype(np.uint8)
    d = ImageDraw.Draw(img := Image.fromarray(a))
    for x in range(20, 580, 60):                              # distractors: driveways that are not in the outlines
        d.rectangle([x / RES, 300 / RES, (x + 4) / RES, 340 / RES], fill=(150, 150, 150))
    est = brp.estimate_block_shift(img, [brp.to_block_px(r, origin, RES) for r in rings], RES)
    assert est is not None and est[2] >= 6
    east, north = est[0] * RES, -est[1] * RES
    assert east == pytest.approx(true[0], abs=0.3) and north == pytest.approx(true[1], abs=0.3)


def test_block_shift_refuses_to_guess_on_plain_noise():
    rng = np.random.default_rng(2)
    img = Image.fromarray(rng.integers(0, 255, (1200, 1200, 3), dtype=np.uint8))
    rings = [brp.to_block_px(rect_ring(x, y, 12, 8, 30), (0.0, 225.0), RES)
             for x in range(30, 200, 30) for y in range(30, 200, 30)]
    assert brp.estimate_block_shift(img, rings, RES) is None


# ---------------------------------------------------------------- cutting one roof
@pytest.mark.parametrize("angle", [0, 25, 60, 135])
def test_cut_chip_turns_and_centres(tmp_path, angle):
    s = settings(tmp_path)
    origin = (1000.0, 2000.0)
    ring = rect_ring(1060, 1940, 14, 9, angle)
    chip = brp.cut_chip(scene([ring], origin, 640), origin, RES, ring, s)
    assert chip.reason == "" and chip.image.size == (128, 128)
    assert chip.area_m2 == pytest.approx(126, rel=0.01)
    assert chip.long_m == pytest.approx(14, abs=0.05) and chip.short_m == pytest.approx(9, abs=0.05)
    red, x0, x1, y0, y1 = red_box(chip.image)
    assert (x1 - x0 + 1) * RES == pytest.approx(14, abs=0.6) and (y1 - y0 + 1) * RES == pytest.approx(9, abs=0.6)
    assert red[y0:y1 + 1, x0:x1 + 1].mean() > 0.95
    assert abs((x0 + x1) / 2 - 63.5) <= 2 and abs((y0 + y1) / 2 - 63.5) <= 2
    outline = chip.to_chip(ring)                              # the outline lands on the roof in chip pixels
    assert abs(outline[:, 0].min() - x0) <= 2 and abs(outline[:, 1].max() - y1) <= 2


def test_cut_chip_in_web_mercator_units_measures_metres(tmp_path):
    s = settings(tmp_path)
    k = 1.2                                                   # Web Mercator units per metre at about 33.6 N
    origin = (0.0, 120.0 * k)
    ring = rect_ring(60 * k, 60 * k, 14 * k, 9 * k, 20)       # a 14 m x 9 m roof drawn in Mercator units
    chip = brp.cut_chip(scene([ring], origin, 768, res=RES * k), origin, RES * k, ring, s, k=k)
    assert chip.reason == ""
    assert chip.area_m2 == pytest.approx(126, rel=0.01) and chip.long_m == pytest.approx(14, abs=0.05)
    red, x0, x1, y0, y1 = red_box(chip.image)
    assert (x1 - x0 + 1) * RES == pytest.approx(14, abs=0.6)


@pytest.mark.parametrize("long_m,short_m,reason", [(5, 4, "area"), (30, 12, "area"), (22, 8, "shape"),
                                                   (20, 3, "shape")])
def test_size_and_shape_filters(tmp_path, long_m, short_m, reason):
    s = settings(tmp_path)
    origin = (0.0, 120.0)
    ring = rect_ring(60, 60, long_m, short_m, 10)
    assert brp.cut_chip(scene([ring], origin, 640), origin, RES, ring, s).reason == reason


def test_nodata_and_edge_rejected(tmp_path):
    s = settings(tmp_path)
    origin = (0.0, 120.0)
    ring = rect_ring(60, 60, 14, 9, 0)
    a = np.asarray(scene([ring], origin, 640)).copy()
    a[:, :300] = 0                                            # left half of the block has no photo
    assert brp.cut_chip(Image.fromarray(a), origin, RES, ring, s).reason == "nodata"
    near_edge = rect_ring(4, 60, 14, 9, 0)
    assert brp.cut_chip(scene([near_edge], origin, 640), origin, RES, near_edge, s).reason == "edge"


def test_no_photo_test_ignores_shadows_and_white_roofs():
    rng = np.random.default_rng(3)
    a = np.full((128, 128, 3), (90, 120, 80), np.uint8)
    a[20:60, 20:70] = rng.integers(0, 25, (40, 50, 3))           # deep, noisy shadow (near black, not flat)
    a[70:120, 20:80] = 255                                        # bright flat white roof
    assert brp.nodata_fraction(Image.fromarray(a)) < 0.05
    a[:, 112:] = 0                                                # a real strip with no photo
    assert brp.nodata_fraction(Image.fromarray(a)) == pytest.approx(0.125, abs=0.01)


def test_texas_checkerboard_counts_as_no_photo():
    rng = np.random.default_rng(4)
    a = np.clip(rng.normal(120, 25, (128, 128, 3)), 0, 255).astype(np.uint8)
    for i in range(4):                                            # a 4 x 4 checkerboard of 8-pixel squares
        for j in range(4):
            a[96 + 8 * i:104 + 8 * i, 96 + 8 * j:104 + 8 * j] = 0 if (i + j) % 2 else 255
    img = Image.fromarray(a)
    assert brp.has_checkerboard(img) and brp.nodata_fraction(img) == pytest.approx(16 / 256)
    b = np.clip(rng.normal(120, 25, (128, 128, 3)), 0, 255).astype(np.uint8)
    b[10:60, 10:90] = 255                                         # a white roof alone is not a checkerboard
    assert not brp.has_checkerboard(Image.fromarray(b)) and brp.nodata_fraction(Image.fromarray(b)) == 0


def test_a_sliver_of_checkerboard_in_a_corner_is_caught(tmp_path):
    s = settings(tmp_path)
    origin, size = (0.0, 120.0), 640
    near = rect_ring(66, 47, 12, 8, 0)               # centre at block pixel (352, 389); window reaches x 416, y 453
    far = rect_ring(30, 90, 12, 8, 0)
    a = np.asarray(noisy(scene([near, far], origin, size), seed=5)).copy()
    for i in range(0, 230, 32):                      # checkerboard from pixel 410 on: a 6-pixel sliver in the window
        for j in range(0, 230, 32):
            a[410 + i:442 + i, 410 + j:442 + j] = 0 if (i + j) // 32 % 2 else 255
    img = Image.fromarray(a)
    mask = brp.checkerboard_mask(img)
    assert mask is not None
    plain = brp.cut_chip(img, origin, RES, near, s)
    assert plain.reason == ""                        # the check on the finished chip alone misses the sliver
    assert brp.cut_chip(img, origin, RES, near, s, missing=mask).reason == "nodata"
    assert brp.cut_chip(img, origin, RES, far, s, missing=mask).reason == ""
    assert brp.checkerboard_mask(noisy(scene([far], origin, size))) is None


def test_only_option_needs_trial_and_known_sources():
    with pytest.raises(SystemExit):
        brp.main(["--pack", "variety", "--only", "tx_round_rock_2021"])
    with pytest.raises(SystemExit):
        brp.main(["--pack", "variety", "--trial", "--only", "nowhere_2020"])


def noisy(img, sd=10, seed=0):
    a = np.asarray(img).astype(np.int16)
    return Image.fromarray(np.clip(a + np.random.default_rng(seed).normal(0, sd, a.shape), 0, 255).astype(np.uint8))


@pytest.mark.parametrize("angle", [0, 30, 75, 140])
def test_house_nudge_centres_the_roof(tmp_path, angle):
    s = settings(tmp_path)
    origin = (0.0, 120.0)
    ring = rect_ring(60, 60, 13, 8, angle)
    img = noisy(scene([ring + np.array([1.3, -0.9])], origin, 640))     # roof 1.3 m east, 0.9 m south of outline
    chip = brp.cut_chip(img, origin, RES, ring, s)
    dx, dy, clarity, fit = brp.chip_fit(chip, ring, int(round(s.snap_m / s.fetch_res)))
    assert clarity >= s.snap_clarity and fit > 1.0
    moved = ring + brp.chip_nudge_to_work(chip, dx, dy, RES, s, 1.0)
    assert moved[0] - ring[0] == pytest.approx([1.3, -0.9], abs=0.25)
    chip2 = brp.cut_chip(img, origin, RES, moved, s)
    red, x0, x1, y0, y1 = red_box(chip2.image)
    assert abs((x0 + x1) / 2 - 63.5) <= 1.5 and abs((y0 + y1) / 2 - 63.5) <= 1.5


def test_outline_on_an_empty_lot_scores_low(tmp_path):
    s = settings(tmp_path)
    origin = (0.0, 120.0)
    ring = rect_ring(60, 60, 13, 8, 20)
    empty = noisy(scene([], origin, 640))
    chip = brp.cut_chip(empty, origin, RES, ring, s)
    assert brp.chip_fit(chip, ring, 13)[3] < s.fit_min
    full = noisy(scene([ring], origin, 640))
    assert brp.chip_fit(brp.cut_chip(full, origin, RES, ring, s), ring, 13)[3] > 3 * s.fit_min


def test_block_grid_covers_extent_without_overlap():
    blocks = brp.block_grid((0, 0, 1300, 700), 600, 37.5)
    assert len(blocks) == 6 and all(b[2] - b[0] <= 600 and b[3] - b[1] <= 600 for b in blocks)


# ---------------------------------------------------------------- full fake builds (no network)
def make_homes(extent, step=30.0, seed=0):
    rng = np.random.default_rng(seed)
    x0, y0, x1, y1 = extent
    return [rect_ring(x, y, rng.uniform(10, 16), rng.uniform(7, 10), rng.uniform(0, 180))
            for x in np.arange(x0 + 50, x1 - 50, step) for y in np.arange(y0 + 50, y1 - 50, step)]


class FakeImages:
    """Pretend image service drawing every home 2 m east of its outline; optional dead zone with no photos."""
    wkid = 26919

    def __init__(self, homes, extent, no_photo_x=None, fail_after=None, shift=(2.0, 0.0)):
        self.homes, self.extent, self.no_photo_x = homes, extent, no_photo_x
        self.fail_after, self.calls, self.big_calls, self.shift = fail_after, 0, 0, np.array(shift)

    def info(self):
        x0, y0, x1, y1 = self.extent
        return {"extent": {"xmin": x0, "ymin": y0, "xmax": x1, "ymax": y1, "spatialReference": {"wkid": 6348}}}

    def export(self, bbox, size_px):
        self.calls += 1
        if size_px > 64:
            self.big_calls += 1
            if self.fail_after is not None and self.big_calls > self.fail_after:
                raise brp.ServiceError("pretend network drop")
        xmin, ymin, xmax, ymax = bbox
        res = (xmax - xmin) / size_px
        img = scene([h + self.shift for h in self.homes], (xmin, ymax), size_px, res)
        if self.no_photo_x is not None:
            a = np.asarray(img).copy()
            cut = int(np.clip((self.no_photo_x - xmin) / res, 0, size_px))
            a[:, cut:] = 0
            img = Image.fromarray(a)
        return img


class FakeFootprints:
    def __init__(self, homes):
        self.homes = homes

    def _inside(self, env):
        xmin, ymin, xmax, ymax = env
        return [r for r in self.homes if r[:, 0].max() >= xmin and r[:, 0].min() <= xmax
                and r[:, 1].max() >= ymin and r[:, 1].min() <= ymax]

    def count(self, env, where=None):
        return len(self._inside(env))

    def query(self, env):
        return [brp.Footprint(r, "Single Family Dwelling") for r in self._inside(env)]


def fake_cfg(extents):
    cfg = brp.load_config()
    cfg = {"footprints": cfg["footprints"], "packs": {}, "sources": {}}
    for i, _ in enumerate(extents):
        cfg["sources"][f"town{i}"] = {"kind": "image", "region": "Maine", "place": f"Town {i}", "year": 2022,
                                      "url": f"https://example.test/{i}/ImageServer", "wkid": 26919,
                                      "area": "service", "licence": "test licence", "attribution": f"Imagery: Town {i}"}
    return cfg


def patch_services(monkeypatch, fakes):
    def make(src, cfg, s):
        i = int(src["url"].split("/")[3])
        return fakes[i], FakeFootprints(fakes[i].homes)
    monkeypatch.setattr(brp, "make_services", make)


def test_full_fake_build_writes_a_clean_pack(tmp_path, monkeypatch):
    ext = [(0.0, 0.0, 1275.0, 675.0), (5000.0, 0.0, 6275.0, 675.0)]
    fakes = [FakeImages(make_homes(e, seed=i), e) for i, e in enumerate(ext)]
    patch_services(monkeypatch, fakes)
    s = settings(tmp_path, target=120, max_per_block=40, min_homes_per_block=1, zip_dir=tmp_path / "dist",
                 name="roofs_test", title="Test pack")
    rep = brp.build(s, ["town0", "town1"], fake_cfg(ext), log=lambda *a: None)
    assert rep["chips"] == 120 and rep["privacy_check"] == "passed"
    assert {v["chips"] for v in rep["sources"].values()} == {60}               # balanced across towns
    rows = list(csv.DictReader(open(s.out / "manifest.csv")))
    assert len(rows) == 120 and list(rows[0]) == brp.MANIFEST_FIELDS
    assert len({r["chip_id"] for r in rows}) == 120 and len(list((s.out / "images").glob("*.jpg"))) == 120
    for v in rep["sources"].values():                                           # 2 m east shift was found
        assert v["median_outline_shift_m"] == pytest.approx(2.0, abs=0.4)
    att = (s.out / "ATTRIBUTION.txt").read_text()
    assert "Imagery: Town 0" in att and "Imagery: Town 1" in att and "USA Structures" in att
    for line in (s.out / "SHA256SUMS.txt").read_text().splitlines():           # every checksum is right
        digest, name = line.split("  ", 1)
        assert hashlib.sha256((s.out / name).read_bytes()).hexdigest() == digest
    meta = json.loads((tmp_path / "dist" / "roofs_test.json").read_text())
    z = tmp_path / "dist" / "roofs_test.zip"
    assert meta["sha256"] == hashlib.sha256(z.read_bytes()).hexdigest() and meta["chips"] == 120
    names = zipfile.ZipFile(z).namelist()
    assert not any(brp.STATE_FILE in n or "/review/" in n for n in names)
    assert "roofs_test/manifest.csv" in names and sum(n.endswith(".jpg") for n in names) == 120
    assert brp.privacy_problems(s.out) == []


def test_blocks_without_photos_are_skipped_before_the_big_download(tmp_path, monkeypatch):
    ext = [(0.0, 0.0, 2550.0, 675.0)]
    fake = FakeImages(make_homes(ext[0]), ext[0], no_photo_x=1300.0)          # the east half has no photos
    patch_services(monkeypatch, [fake])
    s = settings(tmp_path, target=10_000, max_per_block=500, min_homes_per_block=1)
    rep = brp.build(s, ["town0"], fake_cfg(ext), log=lambda *a: None)
    st = rep["sources"]["town0"]
    assert st["blocks_no_photos"] >= 1 and fake.big_calls == st["blocks_used"]


def test_blocks_where_outlines_miss_are_skipped(tmp_path, monkeypatch):
    good, empty = (0.0, 0.0, 1275.0, 675.0), (5000.0, 0.0, 6275.0, 675.0)
    fakes = [FakeImages(make_homes(good), good), FakeImages(make_homes(empty, seed=1), empty)]
    fakes[1].export = lambda bbox, size_px: noisy(Image.new("RGB", (size_px, size_px), GROUND), seed=size_px)
    patch_services(monkeypatch, fakes)
    s = settings(tmp_path, target=100, max_per_block=60, min_homes_per_block=1)
    rep = brp.build(s, ["town0", "town1"], fake_cfg([good, empty]), log=lambda *a: None)
    assert rep["sources"]["town1"]["chips"] == 0 and rep["sources"]["town1"]["blocks_outlines_off"] >= 1
    assert rep["sources"]["town0"]["chips"] == rep["chips"] > 0


def test_quota_moves_to_towns_that_still_have_homes(tmp_path, monkeypatch):
    small, large = (0.0, 0.0, 675.0, 675.0), (5000.0, 0.0, 7550.0, 1275.0)
    fakes = [FakeImages(make_homes(small, step=60), small), FakeImages(make_homes(large), large)]
    patch_services(monkeypatch, fakes)
    s = settings(tmp_path, target=300, max_per_block=100, min_homes_per_block=1)
    rep = brp.build(s, ["town0", "town1"], fake_cfg([small, large]), log=lambda *a: None)
    assert rep["chips"] == 300 and rep["sources"]["town0"]["chips"] < 150
    assert rep["sources"]["town0"]["chips"] + rep["sources"]["town1"]["chips"] == 300


def test_resume_after_a_network_drop_gives_the_same_pack(tmp_path, monkeypatch):
    ext = [(0.0, 0.0, 1875.0, 1275.0)]
    homes = make_homes(ext[0])
    cfg = fake_cfg(ext)
    patch_services(monkeypatch, [FakeImages(homes, ext[0])])
    ref = settings(tmp_path / "ref", target=200, max_per_block=40, min_homes_per_block=1)
    brp.build(ref, ["town0"], cfg, log=lambda *a: None)

    patch_services(monkeypatch, [FakeImages(homes, ext[0], fail_after=2)])
    s = settings(tmp_path / "run", target=200, max_per_block=40, min_homes_per_block=1)
    with pytest.raises(brp.ServiceError):
        brp.build(s, ["town0"], cfg, log=lambda *a: None)
    with pytest.raises(SystemExit):                                             # no silent overwrite
        brp.build(s, ["town0"], cfg, log=lambda *a: None)
    patch_services(monkeypatch, [FakeImages(homes, ext[0])])
    rep = brp.build(s, ["town0"], cfg, resume=True, log=lambda *a: None)
    assert rep["chips"] == 200
    a = (ref.out / "manifest.csv").read_text().splitlines()
    b = (s.out / "manifest.csv").read_text().splitlines()
    assert a == b                                                               # identical to an unbroken run
    s2 = settings(tmp_path / "run", target=200, max_per_block=41, min_homes_per_block=1)
    with pytest.raises(SystemExit):                                             # settings changed: refuse
        brp.build(s2, ["town0"], cfg, resume=True, log=lambda *a: None)


def test_trial_saves_review_pictures_and_no_zip(tmp_path, monkeypatch):
    ext = [(0.0, 0.0, 1275.0, 675.0)]
    patch_services(monkeypatch, [FakeImages(make_homes(ext[0]), ext[0])])
    s = settings(tmp_path, target=30, max_per_block=25, min_homes_per_block=1, trial=True, review=True,
                 zip_dir=tmp_path / "dist")
    rep = brp.build(s, ["town0"], fake_cfg(ext), log=lambda *a: None)
    assert rep["chips"] == 30 and "zip" not in rep and not (tmp_path / "dist").exists()
    assert len(list((s.out / "review").glob("*.jpg"))) == 30 and (s.out / "trial_sheet.jpg").exists()


def test_rezip_refreshes_readme_and_zip_without_touching_images(tmp_path, monkeypatch):
    ext = [(0.0, 0.0, 1275.0, 675.0)]
    patch_services(monkeypatch, [FakeImages(make_homes(ext[0]), ext[0])])
    cfg = fake_cfg(ext)
    s = settings(tmp_path, target=40, max_per_block=40, min_homes_per_block=1, zip_dir=tmp_path / "dist",
                 name="roofs_test", pack="train")
    brp.build(s, ["town0"], cfg, log=lambda *a: None)
    z = tmp_path / "dist" / "roofs_test.zip"
    before = {n: zipfile.ZipFile(z).read(n) for n in zipfile.ZipFile(z).namelist() if n.endswith(".jpg")}
    first = json.loads((tmp_path / "dist" / "roofs_test.json").read_text())
    assert "Known limits" not in (s.out / "README.txt").read_text()
    cfg["packs"] = {"train": {"notes": ["About 4% show no house."]}}
    out = brp.rezip(s, cfg, log=lambda *a: None)
    assert "Known limits" in (s.out / "README.txt").read_text() and "About 4% show no house." in \
        zipfile.ZipFile(z).read("roofs_test/README.txt").decode()
    after = {n: zipfile.ZipFile(z).read(n) for n in zipfile.ZipFile(z).namelist() if n.endswith(".jpg")}
    assert before == after and len(after) == 40                                  # every image byte-identical
    meta = json.loads((tmp_path / "dist" / "roofs_test.json").read_text())
    assert meta["sha256"] == hashlib.sha256(z.read_bytes()).hexdigest() == out["zip_sha256"] != first["sha256"]
    assert meta["built"] == first["built"] and "docs_updated" in meta
    for line in (s.out / "SHA256SUMS.txt").read_text().splitlines():
        digest, name = line.split("  ", 1)
        assert hashlib.sha256((s.out / name).read_bytes()).hexdigest() == digest
    (s.out / "build_report.json").unlink()
    with pytest.raises(SystemExit):
        brp.rezip(s, cfg, log=lambda *a: None)


def test_both_packs_carry_known_limits_notes():
    cfg = brp.load_config()
    assert any("4%" in n for n in cfg["packs"]["train"]["notes"])
    assert cfg["packs"]["variety"]["notes"]


def test_privacy_check_catches_leaks(tmp_path, monkeypatch):
    ext = [(0.0, 0.0, 1275.0, 675.0)]
    patch_services(monkeypatch, [FakeImages(make_homes(ext[0]), ext[0])])
    s = settings(tmp_path, target=20, max_per_block=25, min_homes_per_block=1)
    brp.build(s, ["town0"], fake_cfg(ext), log=lambda *a: None)
    rep = json.loads((s.out / "build_report.json").read_text())
    rep["blocks"][0]["bbox"] = [399177.995, 4832648.758]
    (s.out / "build_report.json").write_text(json.dumps(rep))
    (s.out / "homes.geojson").write_text("{}")
    problems = " | ".join(brp.privacy_problems(s.out))
    assert "bbox" in problems and "coordinate-like" in problems and "homes.geojson" in problems
