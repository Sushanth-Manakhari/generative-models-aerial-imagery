#!/usr/bin/env python3
"""Build the roof-image pack for the generative-models lecture from free, public data.

    python scripts/build_roof_pack.py --pilot           # G1 data test: ~2,000 roofs + a NAIP comparison
    python scripts/build_roof_pack.py --count 20000     # full pack (G3)

Sources (details and licences: docs/DATA_SOURCES.md)
  imagery     Maine GeoLibrary orthoimagery, South Portland 2022 (7.5 cm per pixel, leaf-off spring)
  fallback    USDA NAIP through USGS The National Map (public domain, about 0.6 m per pixel)
  footprints  USA Structures (Oak Ridge National Laboratory and FEMA Geospatial Response Office)

How it works (the same steps an insurer's pipeline uses, minus the address lookup)
  1. Split the town into 600 m squares ("blocks") and pick blocks in random order.
  2. Ask the footprint service for the homes in a block (outline + building type only).
  3. Download ONE aerial image for the whole block (far fewer requests than one per house).
  4. For each home: cut a 24 m square around it, turn it so the long side of the roof is
     horizontal, optionally black out everything that is not roof, and save it.

Privacy: the pack stores roof pictures plus roof area, angle and building type. It never requests
or stores addresses, coordinates, building IDs or owner data.

Only the Python standard library plus numpy and Pillow are needed (requirements-data.txt).
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import random
import shutil
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------------------------
# Data sources
# ---------------------------------------------------------------------------------------------
SOUTH_PORTLAND_2022 = {
    "name": "maine_south_portland_2022",
    "label": "Maine GeoLibrary - Maine Orthoimagery, South Portland 2022 (7.5 cm)",
    "url": "https://gis.maine.gov/image/rest/services/Municipal/orthoMunicipalSouthPortland2022/ImageServer",
    "wkid": 26919,                                         # NAD83 / UTM zone 19N, metres
    "extent": (389250.0, 4827750.0, 402375.0, 4834875.0),  # from the service description
}
NAIP_USGS = {
    "name": "naip_usgs",
    "label": "USDA NAIP via USGS The National Map (public domain)",
    "url": "https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPImagery/ImageServer",
}
FOOTPRINTS = {
    "label": "USA Structures - Oak Ridge National Laboratory (ORNL) and FEMA Geospatial Response Office",
    "url": "https://services2.arcgis.com/FiaPA4ga0iQKduv3/arcgis/rest/services/USA_Structures_View/FeatureServer/0",
    # Only non-identifying fields. Address fields exist in the service and are deliberately NOT requested.
    "fields": ["OCC_CLS", "PRIM_OCC", "SQMETERS"],
    "residential": "OCC_CLS = 'Residential'",
}

ATTRIBUTION = """\
Roof image pack - built for an educational guest lecture on generative models.

Aerial imagery
  Maine GeoLibrary, Maine Orthoimagery - South Portland 2022 (7.5 cm, 4-band, leaf-off).
  https://gis.maine.gov/image/rest/services/Municipal/orthoMunicipalSouthPortland2022/ImageServer
  Statewide orthoimagery metadata: "No restrictions apply to this data."

Comparison imagery (pilot only)
  USDA Farm Service Agency, National Agriculture Imagery Program (NAIP), via USGS The National Map.
  Public domain.

Building outlines (used to find, turn and mask each roof; outlines and addresses are NOT redistributed)
  USA Structures - Oak Ridge National Laboratory (ORNL) and FEMA Geospatial Response Office.
  Licensed CC BY 4.0 (also listed on data.gov as a U.S. Government Work).

The pack contains no addresses, coordinates, building IDs or owner information.
"""

USER_AGENT = "roof-pack-builder/0.1 (educational, non-commercial; generative-models guest lecture)"


# ---------------------------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------------------------
@dataclass
class Settings:
    out: Path
    target: int = 2000
    seed: int = 7
    window_m: float = 24.0        # side of the square cut around each roof, in metres
    chip_px: int = 128            # saved image size (the lecture trains at 64 x 64; 128 keeps options open)
    block_m: float = 600.0        # side of one download block
    margin_m: float = 37.5        # extra ring around each block so rotated cuts never fall off the edge
    min_area_m2: float = 50.0     # skip sheds and tiny outbuildings
    max_area_m2: float = 350.0    # skip apartment blocks and large commercial roofs
    max_long_m: float = 21.0      # roof must fit inside the 24 m window with a little room
    min_short_m: float = 4.0
    min_homes_per_block: int = 15
    max_chips_per_block: int = 250  # spread the pack across many neighbourhoods
    variants: tuple = ("C",)      # A north-up, B turned, C turned + masked
    naip_compare_blocks: int = 0  # pilot: also cut the same roofs from NAIP in this many blocks
    pause_s: float = 0.4          # be polite to free public servers
    zip_output: bool = False

    @property
    def fetch_res(self) -> float:  # metres per pixel requested from the server
        return self.window_m / self.chip_px


VARIANT_DIRS = {"A": "A_north_up", "B": "B_turned", "C": "C_turned_masked", "N": "N_naip_turned_masked"}


# ---------------------------------------------------------------------------------------------
# HTTP (standard library only, with retries)
# ---------------------------------------------------------------------------------------------
def _ssl_context():
    try:
        import certifi  # macOS python.org builds lack system certificates; certifi fixes that
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


_SSL = _ssl_context()


class ServiceError(RuntimeError):
    pass


def http_get(url: str, params: dict, want: str = "json", tries: int = 5, timeout: int = 90):
    """GET url?params. want='json' returns a dict; want='image' returns a PIL image."""
    full = url + "?" + urllib.parse.urlencode(params)
    delay = 2.0
    last = None
    for attempt in range(1, tries + 1):
        try:
            req = urllib.request.Request(full, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout, context=_SSL) as resp:
                body = resp.read()
                ctype = resp.headers.get("Content-Type", "")
            if want == "json":
                data = json.loads(body.decode("utf-8"))
                if isinstance(data, dict) and "error" in data:
                    raise ServiceError(f"service error: {data['error']}")
                return data
            if "image" not in ctype and not (body[:3] == b"\xff\xd8\xff" or body[:4] == b"\x89PNG"):
                raise ServiceError(f"expected an image, got {ctype}: {body[:200]!r}")
            img = Image.open(io.BytesIO(body))
            img.load()
            return img.convert("RGB")
        except (urllib.error.URLError, TimeoutError, ConnectionError, ServiceError, json.JSONDecodeError, OSError) as e:
            last = e
            if isinstance(e, urllib.error.URLError) and isinstance(getattr(e, "reason", None), ssl.SSLError):
                raise SystemExit(
                    "❌ HTTPS certificate problem. Run:  python -m pip install --upgrade certifi   and try again."
                ) from e
            if attempt < tries:
                time.sleep(delay)
                delay *= 2
    raise ServiceError(f"failed after {tries} tries: {full[:160]}... ({last})")


# ---------------------------------------------------------------------------------------------
# Services
# ---------------------------------------------------------------------------------------------
class ImageService:
    def __init__(self, url: str, wkid: int, pause_s: float = 0.4, tries: int = 5, timeout: int = 90):
        self.url, self.wkid, self.pause_s, self.tries, self.timeout = url, wkid, pause_s, tries, timeout

    def info(self) -> dict:
        return http_get(self.url, {"f": "json"}, tries=self.tries, timeout=self.timeout)

    def export(self, bbox, size_px: int) -> Image.Image:
        """One square image covering bbox (xmin, ymin, xmax, ymax) in self.wkid, size_px x size_px."""
        xmin, ymin, xmax, ymax = bbox
        params = {
            "bbox": f"{xmin},{ymin},{xmax},{ymax}",
            "bboxSR": self.wkid, "imageSR": self.wkid,
            "size": f"{size_px},{size_px}",
            "format": "jpg", "compressionQuality": 92,
            "bandIds": "0,1,2",                       # red, green, blue (band 4 is near-infrared)
            "interpolation": "RSP_BilinearInterpolation",
            "f": "image",
        }
        img = http_get(self.url + "/exportImage", params, want="image", tries=self.tries, timeout=self.timeout)
        time.sleep(self.pause_s)
        if img.size != (size_px, size_px):
            img = img.resize((size_px, size_px), Image.BILINEAR)
        return img


@dataclass
class Footprint:
    ring: np.ndarray               # (n, 2) outer ring, map metres
    occ_cls: str = ""
    prim_occ: str = ""


class FootprintService:
    def __init__(self, url: str, wkid: int, fields, where: str, pause_s: float = 0.4):
        self.url, self.wkid, self.fields, self.where, self.pause_s = url, wkid, list(fields), where, pause_s

    def _geom(self, env):
        xmin, ymin, xmax, ymax = env
        return {
            "geometry": f"{xmin},{ymin},{xmax},{ymax}",
            "geometryType": "esriGeometryEnvelope",
            "inSR": self.wkid,
            "spatialRel": "esriSpatialRelIntersects",
        }

    def count(self, env, where: str | None = None) -> int:
        params = {"where": where or self.where, "returnCountOnly": "true", "f": "json", **self._geom(env)}
        data = http_get(self.url + "/query", params)
        time.sleep(self.pause_s)
        return int(data.get("count", 0))

    def query(self, env) -> list:
        out, offset = [], 0
        while True:
            params = {
                "where": self.where, "outFields": ",".join(self.fields), "returnGeometry": "true",
                "outSR": self.wkid, "geometryPrecision": 2, "resultOffset": offset,
                "resultRecordCount": 2000, "f": "json", **self._geom(env),
            }
            data = http_get(self.url + "/query", params)
            time.sleep(self.pause_s)
            feats = data.get("features", [])
            for f in feats:
                rings = (f.get("geometry") or {}).get("rings") or []
                if not rings:
                    continue
                ring = np.asarray(max(rings, key=len), dtype=float)   # outer ring = longest
                attrs = f.get("attributes") or {}
                out.append(Footprint(ring, str(attrs.get("OCC_CLS") or ""), str(attrs.get("PRIM_OCC") or "")))
            if not data.get("exceededTransferLimit") or not feats:
                return out
            offset += len(feats)


# ---------------------------------------------------------------------------------------------
# Geometry (pixel coordinates: x to the right, y DOWN)
# ---------------------------------------------------------------------------------------------
def polygon_area_centroid(pts: np.ndarray):
    """Shoelace area (absolute) and centroid of a closed or open ring."""
    p = pts[:-1] if np.allclose(pts[0], pts[-1]) else pts
    x, y = p[:, 0], p[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cross = x * y1 - x1 * y
    a = cross.sum() / 2.0
    if abs(a) < 1e-9:
        return 0.0, p.mean(axis=0)
    cx = ((x + x1) * cross).sum() / (6 * a)
    cy = ((y + y1) * cross).sum() / (6 * a)
    return abs(a), np.array([cx, cy])


def convex_hull(pts: np.ndarray) -> np.ndarray:
    """Andrew's monotone chain."""
    p = sorted(set(map(tuple, np.round(pts, 6))))
    if len(p) <= 2:
        return np.asarray(p, dtype=float)

    def half(seq):
        h = []
        for q in seq:
            while len(h) >= 2 and ((h[-1][0] - h[-2][0]) * (q[1] - h[-2][1]) - (h[-1][1] - h[-2][1]) * (q[0] - h[-2][0])) <= 0:
                h.pop()
            h.append(q)
        return h

    lower, upper = half(p), half(reversed(p))
    return np.asarray(lower[:-1] + upper[:-1], dtype=float)


def min_area_rect(pts: np.ndarray):
    """Smallest rotated rectangle around pts. Returns (unit vector of the long side, long, short)."""
    h = convex_hull(pts)
    best = None
    for i in range(len(h)):
        e = h[(i + 1) % len(h)] - h[i]
        n = np.hypot(*e)
        if n < 1e-9:
            continue
        u = e / n
        w = np.array([-u[1], u[0]])
        su, sw = h @ u, h @ w
        lu, lw = su.max() - su.min(), sw.max() - sw.min()
        if best is None or lu * lw < best[0]:
            best = (lu * lw, u, w, lu, lw)
    if best is None:
        return np.array([1.0, 0.0]), 0.0, 0.0
    _, u, w, lu, lw = best
    return (u, lu, lw) if lu >= lw else (w, lw, lu)


def turn_angle_deg(direction: np.ndarray) -> float:
    """Degrees to pass to PIL's Image.rotate (counter-clockwise) so `direction` ends up horizontal."""
    dx, dy = direction
    alpha = math.degrees(math.atan2(-dy, dx))      # on-screen angle, counter-clockwise from the x-axis
    while alpha > 90:
        alpha -= 180
    while alpha <= -90:
        alpha += 180
    return -alpha


def rotate_points(pts: np.ndarray, angle_deg: float, center) -> np.ndarray:
    """Where pixels at pts land after Image.rotate(angle_deg, center=center)."""
    t = math.radians(angle_deg)
    c, s = math.cos(t), math.sin(t)
    cx, cy = center
    dx, dy = pts[:, 0] - cx, pts[:, 1] - cy
    return np.stack([cx + c * dx + s * dy, cy - s * dx + c * dy], axis=1)


# ---------------------------------------------------------------------------------------------
# Cutting one roof out of a block image
# ---------------------------------------------------------------------------------------------
@dataclass
class Chip:
    images: dict = field(default_factory=dict)   # variant letter -> PIL image
    area_m2: float = 0.0
    long_m: float = 0.0
    short_m: float = 0.0
    angle_deg: float = 0.0
    reason: str = ""                              # filled when the roof was skipped


def nodata_fraction(img: Image.Image) -> float:
    a = np.asarray(img)
    dark = (a <= 3).all(axis=2)
    bright = (a >= 252).all(axis=2)
    return float((dark | bright).mean())


def roof_mask(poly_px: np.ndarray, size: int, grow_px: int = 1) -> Image.Image:
    """Soft mask of the polygon (drawn 4x larger, then shrunk for smooth edges), grown by grow_px."""
    k = 4
    m = Image.new("L", (size * k, size * k), 0)
    ImageDraw.Draw(m).polygon([tuple(p) for p in (poly_px * k)], fill=255)
    m = m.resize((size, size), Image.LANCZOS)
    if grow_px > 0:
        m = m.filter(ImageFilter.MaxFilter(2 * grow_px + 1))
    return m


def cut_chip(block_img: Image.Image, block_origin, res: float, ring_m: np.ndarray, s: Settings,
             variants=("A", "B", "C"), ref_angle: float | None = None) -> Chip:
    """block_origin = (xmin, ymax) of block_img in map metres; res = metres per pixel of block_img."""
    x0, y0 = block_origin
    ring_px = np.stack([(ring_m[:, 0] - x0) / res, (y0 - ring_m[:, 1]) / res], axis=1)
    area_px, cen = polygon_area_centroid(ring_px)
    chip = Chip(area_m2=area_px * res * res)
    direction, long_px, short_px = min_area_rect(ring_px)
    chip.long_m, chip.short_m = long_px * res, short_px * res
    if not (s.min_area_m2 <= chip.area_m2 <= s.max_area_m2):
        chip.reason = "area"
        return chip
    if chip.long_m > s.max_long_m or chip.short_m < s.min_short_m:
        chip.reason = "shape"
        return chip

    chip.angle_deg = turn_angle_deg(direction) if ref_angle is None else ref_angle
    win = s.window_m / res                          # window side in block pixels
    half_big = math.ceil(win * math.sqrt(2) / 2) + 3
    cx, cy = float(cen[0]), float(cen[1])
    bx0, by0 = int(math.floor(cx)) - half_big, int(math.floor(cy)) - half_big
    if bx0 < 0 or by0 < 0 or bx0 + 2 * half_big > block_img.width or by0 + 2 * half_big > block_img.height:
        chip.reason = "edge"
        return chip
    big = block_img.crop((bx0, by0, bx0 + 2 * half_big, by0 + 2 * half_big))
    lc = (cx - bx0, cy - by0)                       # roof centre inside `big`
    poly_big = ring_px - np.array([bx0, by0])

    def final(img: Image.Image, poly: np.ndarray):
        box = (lc[0] - win / 2, lc[1] - win / 2, lc[0] + win / 2, lc[1] + win / 2)
        out = img.transform((s.chip_px, s.chip_px), Image.EXTENT, box, resample=Image.BICUBIC)
        scale = s.chip_px / win
        return out, (poly - np.array([box[0], box[1]])) * scale

    north, _ = final(big, poly_big)
    if nodata_fraction(north) > 0.05:
        chip.reason = "nodata"
        return chip
    if "A" in variants:
        chip.images["A"] = north
    if "B" in variants or "C" in variants or "N" in variants:
        turned_big = big.rotate(chip.angle_deg, resample=Image.BICUBIC, center=lc)
        turned, poly_t = final(turned_big, rotate_points(poly_big, chip.angle_deg, lc))
        if "B" in variants:
            chip.images["B"] = turned
        if "C" in variants or "N" in variants:
            mask = roof_mask(poly_t, s.chip_px, grow_px=max(1, round(s.chip_px / 40)))
            black = Image.new("RGB", turned.size, (0, 0, 0))
            chip.images["N" if "N" in variants else "C"] = Image.composite(turned, black, mask)
    return chip


# ---------------------------------------------------------------------------------------------
# Building the pack
# ---------------------------------------------------------------------------------------------
def block_grid(extent, block_m: float, margin_m: float):
    xmin, ymin, xmax, ymax = extent
    xs = np.arange(xmin + margin_m, xmax - margin_m - 1, block_m)
    ys = np.arange(ymin + margin_m, ymax - margin_m - 1, block_m)
    return [(float(x), float(y), float(min(x + block_m, xmax - margin_m)), float(min(y + block_m, ymax - margin_m)))
            for y in ys for x in xs]


def probe(img_svc: ImageService, fp_svc: FootprintService, extent, naip: ImageService | None):
    print("🔎 Checking the free data services ...")
    info = img_svc.info()
    print(f"   imagery service: OK ({info.get('bandCount', '?')} bands, "
          f"{info.get('pixelSizeX', '?')} m pixels)")
    total = fp_svc.count(extent)
    print(f"   footprint service: OK ({total:,} homes marked residential in this town)")
    if total == 0:
        everything = fp_svc.count(extent, where="1=1")
        if everything == 0:
            raise SystemExit("❌ The footprint service returned no buildings here. Nothing was built.")
        print(f"   ⚠️  no residential labels here; using all {everything:,} buildings with the size filter")
        fp_svc.where = "1=1"
        total = everything
    cx, cy = (extent[0] + extent[2]) / 2, (extent[1] + extent[3]) / 2
    test = img_svc.export((cx - 12, cy - 12, cx + 12, cy + 12), 128)
    print(f"   test image: OK {test.size[0]}x{test.size[1]}")
    if naip is not None:
        try:
            naip.export((cx - 12, cy - 12, cx + 12, cy + 12), 40)
            print("   NAIP comparison service: OK")
        except ServiceError as e:
            print(f"   ⚠️  NAIP comparison skipped ({e})")
            naip = None
    return total, naip


def build(s: Settings, src=SOUTH_PORTLAND_2022) -> dict:
    t_start = time.time()
    rng = random.Random(s.seed)
    img_svc = ImageService(src["url"], src["wkid"], s.pause_s)
    fp_svc = FootprintService(FOOTPRINTS["url"], src["wkid"], FOOTPRINTS["fields"], FOOTPRINTS["residential"], s.pause_s)
    naip = ImageService(NAIP_USGS["url"], src["wkid"], s.pause_s, tries=2, timeout=40) if s.naip_compare_blocks else None
    total_homes, naip = probe(img_svc, fp_svc, src["extent"], naip)

    if s.out.exists():
        shutil.rmtree(s.out)
    wanted = list(s.variants) + (["N"] if naip is not None else [])
    for v in wanted:
        (s.out / VARIANT_DIRS[v]).mkdir(parents=True, exist_ok=True)

    blocks = block_grid(src["extent"], s.block_m, s.margin_m)
    rng.shuffle(blocks)
    big_m = s.block_m + 2 * s.margin_m
    big_px = int(round(big_m / s.fetch_res))
    rows, skipped, block_log = [], {}, []
    naip_left = s.naip_compare_blocks if naip is not None else 0
    print(f"🏠 Cutting up to {s.target:,} roofs from {len(blocks)} candidate blocks "
          f"({s.block_m:.0f} m squares, {big_px}x{big_px} px downloads) ...")

    for bi, core in enumerate(blocks):
        if len(rows) >= s.target:
            break
        n_here = fp_svc.count(core)
        if n_here < s.min_homes_per_block:
            continue
        t0 = time.time()
        feats = fp_svc.query(core)
        cx0, cy0, cx1, cy1 = core
        mine = []
        for f in feats:                          # keep each home once: its centre must be in this block
            _, c = polygon_area_centroid(f.ring)
            if cx0 <= c[0] < cx1 and cy0 <= c[1] < cy1:
                mine.append(f)
        rng.shuffle(mine)
        bbox = (cx0 - s.margin_m, cy0 - s.margin_m, cx0 - s.margin_m + big_m, cy0 - s.margin_m + big_m)
        block_img = img_svc.export(bbox, big_px)
        origin = (bbox[0], bbox[3])
        naip_img = None
        if naip_left > 0:
            naip_px = int(round(big_m / 0.6))
            try:
                naip_img = naip.export(bbox, naip_px)
                naip_left -= 1
            except ServiceError as e:
                print(f"   ⚠️  NAIP comparison turned off for this run: the USGS server is slow ({str(e)[:80]})")
                naip_left = 0
        kept_here = 0
        for f in mine:
            if len(rows) >= s.target or kept_here >= s.max_chips_per_block:
                break
            chip = cut_chip(block_img, origin, s.fetch_res, f.ring, s, variants=s.variants)
            if chip.reason:
                skipped[chip.reason] = skipped.get(chip.reason, 0) + 1
                continue
            chip_id = f"{len(rows) + 1:06d}"
            for v, im in chip.images.items():
                im.save(s.out / VARIANT_DIRS[v] / f"{chip_id}.jpg", quality=95, subsampling=0)
            has_naip = False
            if naip_img is not None:
                n_res = big_m / naip_img.width
                nchip = cut_chip(naip_img, origin, n_res, f.ring, s, variants=("N",), ref_angle=chip.angle_deg)
                if not nchip.reason and "N" in nchip.images:
                    nchip.images["N"].save(s.out / VARIANT_DIRS["N"] / f"{chip_id}.jpg", quality=95, subsampling=0)
                    has_naip = True
            rows.append({
                "chip_id": chip_id, "source": src["name"], "area_m2": round(chip.area_m2, 1),
                "long_m": round(chip.long_m, 1), "short_m": round(chip.short_m, 1),
                "angle_deg": round(chip.angle_deg, 1), "occupancy": f.occ_cls, "use": f.prim_occ,
                "has_naip": int(has_naip),
            })
            kept_here += 1
        dt = time.time() - t0
        block_log.append({"homes": len(mine), "kept": kept_here, "seconds": round(dt, 1)})
        print(f"   block {len(block_log):3d}: {len(mine):4d} homes, kept {kept_here:3d}  "
              f"({len(rows):,}/{s.target:,} total, {dt:.0f} s)")

    with open(s.out / "manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["chip_id"])
        w.writeheader()
        w.writerows(rows)
    (s.out / "ATTRIBUTION.txt").write_text(ATTRIBUTION)
    secs = time.time() - t_start
    per_chip = secs / max(1, len(rows))
    report = {
        "chips": len(rows), "target": s.target, "variants": wanted,
        "homes_marked_residential_in_town": total_homes,
        "blocks_used": len(block_log), "skipped": skipped, "seconds": round(secs, 1),
        "minutes_per_1000_chips": round(per_chip * 1000 / 60, 1),
        "window_m": s.window_m, "chip_px": s.chip_px, "metres_per_pixel": s.fetch_res,
        "imagery": src["label"], "footprints": FOOTPRINTS["label"],
        "python": sys.version.split()[0], "seed": s.seed, "blocks": block_log,
    }
    (s.out / "build_report.json").write_text(json.dumps(report, indent=2))
    if s.zip_output:
        zpath = s.out.with_suffix(".zip")
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as z:
            for p in sorted(s.out.rglob("*")):
                if p.is_file():
                    z.write(p, p.relative_to(s.out.parent))
        report["zip"] = str(zpath)
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pilot", action="store_true", help="G1 data test: ~2,000 roofs, 3 styles + NAIP comparison")
    ap.add_argument("--count", type=int, default=None, help="number of roofs to cut")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args(argv)
    if a.pilot:
        s = Settings(out=a.out or ROOT / "data" / "roof_pack_pilot", target=a.count or 2000, seed=a.seed,
                     variants=("A", "B", "C"), naip_compare_blocks=2, zip_output=True)
    else:
        s = Settings(out=a.out or ROOT / "data" / "roof_pack", target=a.count or 20000, seed=a.seed)
    print(f"📦 Output: {s.out}")
    try:
        r = build(s)
    except ServiceError as e:
        raise SystemExit(f"❌ A data service did not answer properly: {e}\n   Nothing is broken on your Mac; "
                         "send this message to Claude.")
    print(f"\n✅ Done: {r['chips']:,} roofs in {r['seconds'] / 60:.1f} min "
          f"(about {r['minutes_per_1000_chips']} min per 1,000). Skipped: {r['skipped']}")
    if "zip" in r:
        print(f"   For Claude: {r['zip']}")


if __name__ == "__main__":
    main()
