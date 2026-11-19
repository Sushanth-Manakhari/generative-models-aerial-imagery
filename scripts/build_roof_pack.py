#!/usr/bin/env python3
"""Build the roof-image packs for the generative-models lecture from free, public data.

    python scripts/build_roof_pack.py --list                   # show the packs and their sources
    python scripts/build_roof_pack.py --pack all --trial       # small trial: ~50 roofs per source, for review
    python scripts/build_roof_pack.py --pack train             # training pack, Maine 2022 (~22,000 roofs)
    python scripts/build_roof_pack.py --pack variety           # variety pack, five U.S. regions (~3,000 roofs)
    python scripts/build_roof_pack.py --pack storm --trial     # storm pack: Florida roofs days after Hurricane Michael,
                                                               # with FEMA's public damage labels (~2,000 roofs)

    --resume   continue a build that stopped (network drop, laptop asleep): nothing is downloaded twice
    --fresh    delete an unfinished build of that pack and start again
    --only tx_round_rock_2021   (with --trial) re-check just some sources; output goes to a separate folder
    --rezip    refresh README, ATTRIBUTION, checksums and zip of finished packs (no downloads, same images)

Sources, licences and privacy rules: docs/DATA_SOURCES.md and scripts/roof_sources.json.

How it works (the same steps an insurer's imagery pipeline uses, minus any address lookup)
  1. Split each town or neighbourhood into square blocks and visit them in a fixed random order.
  2. Skip a block at once if the town's photos do not cover it (one tiny preview image).
  3. Ask the public footprint service for the homes in the block (outline and building use only).
  4. Download ONE aerial image for the whole block (far fewer requests than one per house).
  5. Line the outlines up with the photo: public outlines are often a few metres off, mostly by the same
     amount across a neighbourhood, so one shift per block (the median of many houses) fixes most of it.
  6. For each home: cut a 24 m square around it, turn it so the long side of the roof is horizontal,
     nudge it up to 2.5 m onto its own roof edges when that is clear, and keep it only if the outline
     really sits on a roof (this drops empty lots, trees and homes built after the photo was taken).
     A block where most outlines miss is skipped altogether.

Privacy: a pack stores roof pictures plus roof size, angle, building use, town and year. It never
requests or stores addresses, coordinates, building IDs or owner data.

Only the Python standard library plus numpy and Pillow are needed (requirements-data.txt).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import io
import json
import math
import os
import random
import re
import shutil
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
SOURCES_FILE = Path(__file__).resolve().with_name("roof_sources.json")
USER_AGENT = "roof-pack-builder/0.2 (educational, non-commercial; generative-models guest lecture)"
MANIFEST_FIELDS = ["chip_id", "file", "source", "region", "place", "year",
                   "area_m2", "long_m", "short_m", "angle_deg", "use"]
LABEL_FIELDS = ["damage", "assessment"]    # storm pack only: FEMA's damage level for the home, and how it was judged
DAMAGE_ORDER = ["none recorded", "No Damage", "Affected", "Minor", "Major", "Destroyed"]
DAMAGED = set(DAMAGE_ORDER[2:])
STATE_FILE = "_build_state.json"          # local bookkeeping for --resume; never packed or published
EARTH_R = 6378137.0                       # Web Mercator sphere radius, metres


def load_config(path: Path = SOURCES_FILE) -> dict:
    return json.loads(Path(path).read_text())


# ---------------------------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------------------------
@dataclass
class Settings:
    out: Path
    pack: str = "train"
    name: str = "roofs"
    title: str = ""
    target: int = 2000
    seed: int = 7
    window_m: float = 24.0          # side of the square cut around each roof, in metres
    chip_px: int = 128              # saved image size (the lecture trains at 64 x 64; 128 keeps options open)
    block_m: float = 600.0          # side of one download block
    margin_m: float = 37.5          # extra ring around each block so turned cuts never fall off the edge
    min_area_m2: float = 50.0       # skip sheds and tiny outbuildings
    max_area_m2: float = 350.0      # skip apartment blocks and large commercial roofs
    max_long_m: float = 21.0        # the roof must fit inside the 24 m window with a little room
    min_short_m: float = 4.0
    min_homes_per_block: int = 15
    max_per_block: int = 250        # spread the pack across many neighbourhoods
    coverage_min: float = 0.5       # skip a block when less than half of it has photos
    align: bool = True              # correct the outline shift once per block
    max_shift_m: float = 6.0
    snap_m: float = 2.5             # then nudge each house up to this far onto its own roof edges ...
    snap_clarity: float = 4.0       # ... but only when the best position is clearly better than the rest
    fit_min: float = 0.30           # skip a house whose outline does not sit on roof edges (empty lot, trees)
    block_fit_min: float = 0.45     # skip a block whose outlines mostly miss (new construction, misregistration)
    pause_s: float = 0.3            # be polite to free public servers
    tile_pause_s: float = 0.02
    trial: bool = False
    review: bool = False            # trial only: also save chips with the outlines drawn on, for checking
    zip_dir: Path | None = None     # where the finished zip goes (None: no zip)
    labels: bool = False            # storm pack: add FEMA damage labels (two extra manifest columns)

    @property
    def fields(self) -> list:        # manifest.csv columns for this pack
        return MANIFEST_FIELDS + LABEL_FIELDS if self.labels else MANIFEST_FIELDS

    @property
    def fetch_res(self) -> float:   # metres per pixel requested from the server
        return self.window_m / self.chip_px

    def signature(self) -> str:      # settings that must not change between a build and its --resume
        keys = ("pack", "name", "target", "seed", "window_m", "chip_px", "block_m", "margin_m", "min_area_m2",
                "max_area_m2", "max_long_m", "min_short_m", "min_homes_per_block", "max_per_block",
                "coverage_min", "align", "max_shift_m", "snap_m", "snap_clarity", "fit_min", "block_fit_min",
                "trial", "labels")
        return json.dumps({k: getattr(self, k) for k in keys}, sort_keys=True)


# ---------------------------------------------------------------------------------------------
# HTTP (standard library only, with retries)
# ---------------------------------------------------------------------------------------------
def _ssl_context():
    try:
        import certifi  # python.org builds on macOS lack system certificates; certifi fixes that
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


_SSL = _ssl_context()


class ServiceError(RuntimeError):
    pass


def _to_rgb(img: Image.Image) -> Image.Image:
    """RGB on a black background (transparent tile edges become 'no photo', which the checks catch)."""
    if img.mode in ("RGBA", "LA", "P", "PA"):
        img = img.convert("RGBA")
        bg = Image.new("RGB", img.size, (0, 0, 0))
        bg.paste(img, mask=img.split()[3])
        return bg
    return img.convert("RGB")


def http_get(url: str, params: dict | None, want: str = "json", tries: int = 5, timeout: int = 90,
             missing_ok: bool = False):
    """GET url?params. want='json' -> dict, want='text' -> str, want='image' -> RGB PIL image.
    missing_ok: HTTP 404 -> None."""
    full = url + ("?" + urllib.parse.urlencode(params) if params else "")
    delay, last = 2.0, None
    for attempt in range(1, tries + 1):
        try:
            req = urllib.request.Request(full, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout, context=_SSL) as resp:
                body = resp.read()
                ctype = resp.headers.get("Content-Type", "")
            if want == "text":
                return body.decode("utf-8", "replace")
            if want == "json":
                data = json.loads(body.decode("utf-8"))
                if isinstance(data, dict) and "error" in data:
                    raise ServiceError(f"service error: {data['error']}")
                return data
            is_img = body[:3] == b"\xff\xd8\xff" or body[:4] == b"\x89PNG"
            if not is_img and "image" not in ctype:
                raise ServiceError(f"expected an image, got {ctype}: {body[:120]!r}")
            img = Image.open(io.BytesIO(body))
            img.load()
            return _to_rgb(img)
        except urllib.error.HTTPError as e:
            if missing_ok and e.code == 404:
                return None
            last = e
        except (urllib.error.URLError, TimeoutError, ConnectionError, ServiceError, json.JSONDecodeError,
                OSError) as e:
            last = e
            if isinstance(e, urllib.error.URLError) and isinstance(getattr(e, "reason", None), ssl.SSLError):
                raise SystemExit("❌ HTTPS certificate problem. Run:  .venv/bin/python -m pip install --upgrade "
                                 "certifi   and try again.") from e
        if attempt < tries:
            time.sleep(delay)
            delay = min(delay * 2, 30)
    raise ServiceError(f"failed after {tries} tries: {full[:160]}... ({last})")


# ---------------------------------------------------------------------------------------------
# Map coordinates
# ---------------------------------------------------------------------------------------------
def lonlat_to_utm(lon: float, lat: float, zone: int):
    """GRS80 / WGS84 Transverse Mercator (NAD83 UTM). Accurate to well under a metre inside the zone."""
    a, f = 6378137.0, 1 / 298.257222101
    k0, e2 = 0.9996, f * (2 - f)
    ep2 = e2 / (1 - e2)
    phi, lam = math.radians(lat), math.radians(lon)
    lam0 = math.radians(-183 + 6 * zone)
    n = a / math.sqrt(1 - e2 * math.sin(phi) ** 2)
    t, c = math.tan(phi) ** 2, ep2 * math.cos(phi) ** 2
    aa = math.cos(phi) * (lam - lam0)
    m = a * ((1 - e2 / 4 - 3 * e2 ** 2 / 64 - 5 * e2 ** 3 / 256) * phi
             - (3 * e2 / 8 + 3 * e2 ** 2 / 32 + 45 * e2 ** 3 / 1024) * math.sin(2 * phi)
             + (15 * e2 ** 2 / 256 + 45 * e2 ** 3 / 1024) * math.sin(4 * phi)
             - (35 * e2 ** 3 / 3072) * math.sin(6 * phi))
    x = k0 * n * (aa + (1 - t + c) * aa ** 3 / 6 + (5 - 18 * t + t * t + 72 * c - 58 * ep2) * aa ** 5 / 120) + 500000.0
    y = k0 * (m + n * math.tan(phi) * (aa * aa / 2 + (5 - t + 9 * c + 4 * c * c) * aa ** 4 / 24
                                       + (61 - 58 * t + t * t + 600 * c - 330 * ep2) * aa ** 6 / 720))
    return x, y


def lonlat_to_merc(lon: float, lat: float):
    return EARTH_R * math.radians(lon), EARTH_R * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))


def merc_to_lonlat(x: float, y: float):
    return math.degrees(x / EARTH_R), merc_y_to_lat(y)


def point_in_polygon(x: float, y: float, ring: np.ndarray) -> bool:
    inside, n = False, len(ring)
    for i in range(n):
        x0, y0 = ring[i]
        x1, y1 = ring[(i + 1) % n]
        if (y0 > y) != (y1 > y) and x < (x1 - x0) * (y - y0) / (y1 - y0 + 1e-12) + x0:
            inside = not inside
    return inside


def merc_y_to_lat(y: float) -> float:
    return math.degrees(math.atan(math.sinh(y / EARTH_R)))


def utm_zone_of(wkid: int) -> int:
    if 26901 <= wkid <= 26923:          # NAD83 / UTM zone N
        return wkid - 26900
    if 32601 <= wkid <= 32660:          # WGS84 / UTM zone N
        return wkid - 32600
    raise ValueError(f"not a UTM north wkid: {wkid}")


def lonlat_box_to(box, wkid: int):
    """(lon0, lat0, lon1, lat1) -> envelope (xmin, ymin, xmax, ymax) in wkid (UTM or 3857)."""
    lon0, lat0, lon1, lat1 = box
    corners = [(lon0, lat0), (lon0, lat1), (lon1, lat0), (lon1, lat1)]
    if wkid in (3857, 102100):
        pts = [lonlat_to_merc(*c) for c in corners]
    else:
        z = utm_zone_of(wkid)
        pts = [lonlat_to_utm(lo, la, z) for lo, la in corners]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


# ---------------------------------------------------------------------------------------------
# Services
# ---------------------------------------------------------------------------------------------
class ImageService:
    """ArcGIS ImageServer: one exportImage call returns any box, re-projected to `wkid` by the server."""

    def __init__(self, url: str, wkid: int, pause_s: float = 0.3, tries: int = 5, timeout: int = 90):
        self.url, self.wkid, self.pause_s, self.tries, self.timeout = url, wkid, pause_s, tries, timeout

    def info(self) -> dict:
        return http_get(self.url, {"f": "json"}, tries=self.tries, timeout=self.timeout)

    def export(self, bbox, size_px: int) -> Image.Image:
        xmin, ymin, xmax, ymax = bbox
        params = {
            "bbox": f"{xmin},{ymin},{xmax},{ymax}", "bboxSR": self.wkid, "imageSR": self.wkid,
            "size": f"{size_px},{size_px}", "format": "jpg", "compressionQuality": 92,
            "bandIds": "0,1,2",                                  # red, green, blue (band 4 is near-infrared)
            "interpolation": "RSP_BilinearInterpolation", "f": "image",
        }
        img = http_get(self.url + "/exportImage", params, want="image", tries=self.tries, timeout=self.timeout)
        time.sleep(self.pause_s)
        if img.size != (size_px, size_px):
            img = img.resize((size_px, size_px), Image.BILINEAR)
        return img


class TileService:
    """ArcGIS cached tile layer in Web Mercator: fetch the tiles over a box and stitch them together."""

    def __init__(self, url: str, level: int, pause_s: float = 0.3, tile_pause_s: float = 0.02,
                 tries: int = 4, timeout: int = 60, fetch=None, workers: int = 4):
        self.url, self.max_level, self.pause_s, self.tile_pause_s = url, level, pause_s, tile_pause_s
        self.workers = workers
        self.tries, self.timeout = tries, timeout
        self.wkid = 3857
        self._fetch = fetch or (lambda u: http_get(u, None, want="image", tries=self.tries,
                                                   timeout=self.timeout, missing_ok=True))
        self.tile_info = None

    def info(self) -> dict:
        data = http_get(self.url, {"f": "json"}, tries=self.tries, timeout=self.timeout)
        self.set_tile_info(data["tileInfo"])
        return data

    def set_tile_info(self, tile_info: dict):
        self.tile_info = tile_info
        levels = {l["level"] for l in tile_info["lods"]}
        if self.max_level not in levels:
            raise ServiceError(f"tile level {self.max_level} not offered by {self.url}")

    def tile_url(self, level: int, row: int, col: int) -> str:
        return f"{self.url}/tile/{level}/{row}/{col}"

    def _level_for(self, res_needed: float):
        """Coarsest level that is still at least as sharp as res_needed (never finer than max_level)."""
        lods = sorted((l for l in self.tile_info["lods"] if l["level"] <= self.max_level),
                      key=lambda l: l["resolution"], reverse=True)
        for l in lods:
            if l["resolution"] <= res_needed:
                return l["level"], l["resolution"]
        return lods[-1]["level"], lods[-1]["resolution"]

    def export(self, bbox, size_px: int) -> Image.Image:
        if self.tile_info is None:
            self.info()
        xmin, ymin, xmax, ymax = bbox
        level, res = self._level_for((xmax - xmin) / size_px)
        ox, oy = self.tile_info["origin"]["x"], self.tile_info["origin"]["y"]
        T = int(self.tile_info["cols"])
        span = res * T
        c0, c1 = int(math.floor((xmin - ox) / span)), int(math.floor((xmax - ox) / span))
        r0, r1 = int(math.floor((oy - ymax) / span)), int(math.floor((oy - ymin) / span))
        mosaic = Image.new("RGB", ((c1 - c0 + 1) * T, (r1 - r0 + 1) * T), (0, 0, 0))

        def get(rc):
            time.sleep(self.tile_pause_s)
            return rc, self._fetch(self.tile_url(level, rc[0], rc[1]))

        cells = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
        with ThreadPoolExecutor(max_workers=self.workers) as pool:     # a few at a time, like a map viewer
            for (r, c), tile in pool.map(get, cells):
                if tile is not None:
                    if tile.size != (T, T):
                        tile = tile.resize((T, T), Image.BILINEAR)
                    mosaic.paste(tile, ((c - c0) * T, (r - r0) * T))
        left = (xmin - (ox + c0 * span)) / res
        top = ((oy - r0 * span) - ymax) / res
        box = (left, top, left + (xmax - xmin) / res, top + (ymax - ymin) / res)
        time.sleep(self.pause_s)
        return mosaic.transform((size_px, size_px), Image.EXTENT, box, resample=Image.BILINEAR)


WEB_MERCATOR_TILES = {"origin": {"x": -20037508.342787, "y": 20037508.342787}, "cols": 256, "rows": 256,
                      "lods": [{"level": z, "resolution": 156543.03392804097 / 2 ** z} for z in range(23)]}


class XYZTileService(TileService):
    """Web Mercator tiles addressed as {z}/{x}/{y} (the usual web-map scheme), e.g. NOAA's storm imagery."""

    def __init__(self, template: str, level: int, **kw):
        super().__init__(template, level, **kw)
        self.set_tile_info(WEB_MERCATOR_TILES)

    def info(self) -> dict:
        return {"tileInfo": self.tile_info}

    def tile_url(self, level: int, row: int, col: int) -> str:
        return self.url.format(z=level, x=col, y=row)


def parse_wmts(xml: str) -> list:
    """Layers of a WMTS GetCapabilities document: [{"id", "box": (lon0, lat0, lon1, lat1), "template"}]."""
    out = []
    for block in re.findall(r"<Layer>(.*?)</Layer>", xml, flags=re.S):
        ident = re.search(r"<ows:Identifier>\s*([^<\s]+)\s*</ows:Identifier>", block)
        lo = re.search(r"<ows:LowerCorner>\s*([-\d.]+)\s+([-\d.]+)\s*</ows:LowerCorner>", block)
        hi = re.search(r"<ows:UpperCorner>\s*([-\d.]+)\s+([-\d.]+)\s*</ows:UpperCorner>", block)
        tpl = re.search(r'<ResourceURL[^>]*template="([^"]+)"', block)
        if not (ident and lo and hi and tpl):
            continue
        tms = re.search(r"<TileMatrixSet>\s*([^<\s]+)\s*</TileMatrixSet>", block)
        template = (tpl.group(1).replace("{TileMatrixSet}", tms.group(1) if tms else "GoogleMapsCompatible")
                    .replace("{Style}", "default").replace("{TileMatrix}", "{z}").replace("{TileCol}", "{x}")
                    .replace("{TileRow}", "{y}"))
        out.append({"id": ident.group(1), "box": (float(lo.group(1)), float(lo.group(2)),
                                                  float(hi.group(1)), float(hi.group(2))), "template": template})
    return out


class NoaaEriService:
    """NOAA Emergency Response Imagery for one storm (public domain, flown in the days after landfall).

    The storm's page links a WMTS capabilities file; each flight is one tile layer in Web Mercator. For every
    box asked for, the flight that really has photos there is used: a layer's outline is a rectangle around
    its flight lines, so several layers can claim a street while only one photographed it. NOAA answers a tile
    it does not have with a 1 x 1 transparent picture, which becomes 'no photo' here."""
    wkid = 3857
    CAPS = re.compile(r"""[^"'\s()<>]*(?:WMTSCapabilities\.xml|tileserver\.php/wmts)""", re.I)

    def __init__(self, page_url: str, level: int, prefixes=(), pause_s: float = 0.3, tile_pause_s: float = 0.02,
                 timeout: int = 60, get_text=None, fetch=None, min_level: int = 17, known=None):
        self.page_url, self.level, self.prefixes = page_url, int(level), tuple(prefixes)
        self.known = list(known or [])    # flights checked by hand: [{"id", "box", "template"}]; no page needed
        self.min_level = min(int(min_level), self.level)   # coarser zooms may be missing: never ask for them
        self.pause_s, self.tile_pause_s, self.timeout = pause_s, tile_pause_s, timeout
        self._get_text = get_text or (lambda u: http_get(u, None, want="text", timeout=timeout))
        self._fetch = fetch or (lambda u: _real_tile(http_get(u, None, want="image", tries=4, timeout=timeout,
                                                              missing_ok=True)))
        self.missions, self.caps_url, self._svc, self._choice = [], None, {}, {}

    def _service(self, mission: dict) -> XYZTileService:
        if mission["id"] not in self._svc:
            self._svc[mission["id"]] = XYZTileService(mission["template"], self.level, pause_s=0,
                                                      tile_pause_s=self.tile_pause_s, timeout=self.timeout,
                                                      fetch=self._fetch, workers=6)
        return self._svc[mission["id"]]

    def _find_caps(self) -> str:
        """The WMTS capabilities address: in the storm page, in its scripts (some pages add the link with
        JavaScript), or at the two places NOAA's storm pages have used."""
        page = self._get_text(self.page_url)
        m = self.CAPS.search(page)
        if m:
            return urllib.parse.urljoin(self.page_url, m.group(0))
        for js in re.findall(r"""<script[^>]+src=["']([^"']+\.js)["']""", page)[:12]:
            url = urllib.parse.urljoin(self.page_url, js)
            if urllib.parse.urlparse(url).netloc != urllib.parse.urlparse(self.page_url).netloc:
                continue
            try:
                m = self.CAPS.search(self._get_text(url))
            except ServiceError:
                continue
            if m:
                return urllib.parse.urljoin(url, m.group(0))
        for guess in ("services/WMTSCapabilities.xml", "../tilesj/services/tileserver.php/wmts"):
            url = urllib.parse.urljoin(self.page_url, guess)
            try:
                if parse_wmts(self._get_text(url)):
                    return url
            except ServiceError:
                continue
        raise ServiceError(f"no WMTS link on {self.page_url} or its scripts (NOAA may have changed the page)")

    def info(self) -> dict:
        if self.known:
            self.missions = sorted(self.known, key=lambda l: l["id"])
            self.caps_url = "flights listed in scripts/roof_sources.json"
            return {"missions": [l["id"] for l in self.missions], "level": self.level, "tile_service": self.caps_url}
        self.caps_url = self._find_caps()
        layers = parse_wmts(self._get_text(self.caps_url))
        self.missions = sorted((l for l in layers if not self.prefixes or l["id"].startswith(self.prefixes)),
                               key=lambda l: l["id"])
        if not self.missions:
            raise ServiceError(f"no flights starting {self.prefixes} in {self.caps_url}")
        return {"missions": [l["id"] for l in self.missions], "level": self.level, "tile_service": self.caps_url}

    def _candidates(self, bbox) -> list:
        lon0, lat0 = merc_to_lonlat(bbox[0], bbox[1])
        lon1, lat1 = merc_to_lonlat(bbox[2], bbox[3])

        def overlap(m):
            a0, b0, a1, b1 = m["box"]
            return max(0.0, min(a1, lon1) - max(a0, lon0)) * max(0.0, min(b1, lat1) - max(b0, lat0))
        return [m for m in sorted(self.missions, key=overlap, reverse=True) if overlap(m) > 0]

    def _export(self, mission: dict, bbox, size_px: int) -> Image.Image:
        """One flight's picture of the box, fetched at zoom min_level or finer, then shrunk to size_px."""
        finest_needed = (bbox[2] - bbox[0]) / WEB_MERCATOR_TILES["lods"][self.min_level]["resolution"]
        fetch_px = max(int(size_px), int(math.ceil(finest_needed)))
        img = self._service(mission).export(bbox, fetch_px)
        return img if fetch_px == size_px else img.resize((size_px, size_px), Image.BOX)

    def flight_for(self, bbox):
        """The flight with the most photo cover over this box (None: no flight photographed it)."""
        key = tuple(round(v, -1) for v in bbox)
        if key not in self._choice:
            best, best_cover = None, 0.0
            for m in self._candidates(bbox)[:5]:
                cover = 1.0 - nodata_fraction(self._export(m, bbox, 64))
                if cover > best_cover:
                    best, best_cover = m, cover
                if cover > 0.95:
                    break
            self._choice[key] = best
        return self._choice[key]

    def export(self, bbox, size_px: int) -> Image.Image:
        if not self.missions:
            self.info()
        best = self.flight_for(bbox)
        time.sleep(self.pause_s)
        if best is None:
            return Image.new("RGB", (size_px, size_px), (0, 0, 0))
        return self._export(best, bbox, size_px)


def _real_tile(img):
    """NOAA's 'no tile here' answer is a tiny transparent picture: treat it as missing."""
    return None if img is None or min(img.size) <= 4 else img


class LabelService:
    """FEMA's public geospatial damage assessments: one point per assessed building with its damage level.
    Fields are found by their descriptive names, and coded values (DES, MAJ, ...) are turned into words
    (Destroyed, Major, ...) using the layer's own code lists."""

    WANT = {"damage": "damage level", "assessment": "assessment type", "event": "event name", "dtype": "damage type"}

    def __init__(self, url: str, wkid: int, event: str, exclude_assessment=(), pause_s: float = 0.3,
                 tries: int = 5, timeout: int = 90, get_json=None):
        self.url, self.wkid, self.event, self.exclude = url, wkid, event, tuple(exclude_assessment)
        self.pause_s = pause_s
        self._get = get_json or (lambda u, p: http_get(u, p, tries=tries, timeout=timeout))
        self.fields, self.codes = None, {}

    def info(self) -> dict:
        layer = self._get(self.url, {"f": "json"})
        found = {}
        for f in layer.get("fields", []):
            alias = str(f.get("alias") or f.get("name", "")).strip().lower()
            for key, want in self.WANT.items():
                if key not in found and alias == want:
                    found[key] = f["name"]
                    dom = f.get("domain") or {}
                    self.codes[key] = {str(c["code"]): str(c["name"]) for c in dom.get("codedValues", [])}
        missing = [k for k in ("damage", "event") if k not in found]
        if missing:
            raise ServiceError(f"damage layer lacks fields {missing}: "
                               f"{[f.get('alias') for f in layer.get('fields', [])]}")
        self.fields = found
        return {"fields": found}

    def word(self, key: str, value) -> str:
        v = "" if value is None else str(value).strip()
        return self.codes.get(key, {}).get(v, v)

    def where(self) -> str:
        w = f"{self.fields['event']} = '{self.event}'"
        if self.exclude and "assessment" in self.fields:
            names = {n.lower() for n in self.exclude}
            codes = [c for c, n in self.codes.get("assessment", {}).items() if n.lower() in names]
            codes += [n for n in self.exclude if n not in codes and not self.codes.get("assessment")]
            if codes:
                col = self.fields["assessment"]
                w += f" AND ({col} IS NULL OR {col} NOT IN ({', '.join(repr(c) for c in codes)}))"
        return w

    def query(self, env) -> list:
        """[(x, y, damage word, 'how assessed / damage type')] for the points inside env (work system)."""
        if self.fields is None:
            self.info()
        names = [self.fields[k] for k in ("damage", "assessment", "dtype") if k in self.fields]
        out, offset = [], 0
        while True:
            params = {"where": self.where(), "outFields": ",".join(names), "returnGeometry": "true",
                      "outSR": self.wkid, "resultOffset": offset, "resultRecordCount": 2000, "f": "json",
                      "geometry": f"{env[0]},{env[1]},{env[2]},{env[3]}", "geometryType": "esriGeometryEnvelope",
                      "inSR": self.wkid, "spatialRel": "esriSpatialRelIntersects"}
            data = self._get(self.url + "/query", params)
            time.sleep(self.pause_s)
            feats = data.get("features", [])
            for f in feats:
                g, a = f.get("geometry") or {}, f.get("attributes") or {}
                if "x" in g and "y" in g:
                    how = " / ".join(w for w in (self.word(k, a.get(self.fields[k]))
                                                 for k in ("assessment", "dtype") if k in self.fields) if w)
                    out.append((float(g["x"]), float(g["y"]), self.word("damage", a.get(self.fields["damage"])), how))
            if not data.get("exceededTransferLimit") or not feats:
                return out
            offset += len(feats)


def damage_for(ring: np.ndarray, points: list, near: float):
    """The most severe damage label among the assessment points inside the outline, or within `near` (work units)
    of it. ('none recorded', '') when no point belongs to this home."""
    best, kind = None, ""
    xmin, ymin = ring[:, 0].min() - near, ring[:, 1].min() - near
    xmax, ymax = ring[:, 0].max() + near, ring[:, 1].max() + near
    _, c = polygon_area_centroid(ring)
    for x, y, level, how in points:
        if not (xmin <= x <= xmax and ymin <= y <= ymax):
            continue
        if not (point_in_polygon(x, y, ring) or math.hypot(x - c[0], y - c[1]) <= near):
            continue
        rank = DAMAGE_ORDER.index(level) if level in DAMAGE_ORDER else -1
        if rank >= 0 and (best is None or rank > DAMAGE_ORDER.index(best)):
            best, kind = level, how
    return (best, kind) if best is not None else ("none recorded", "")


@dataclass
class Footprint:
    ring: np.ndarray               # (n, 2) outer ring in the work coordinate system
    use: str = ""                  # e.g. "Single Family Dwelling"


class FootprintService:
    def __init__(self, url: str, wkid: int, fields, where: str, pause_s: float = 0.3, tries: int = 5,
                 timeout: int = 90):
        self.url, self.wkid, self.fields, self.where = url, wkid, list(fields), where
        self.pause_s, self.tries, self.timeout = pause_s, tries, timeout

    def _geom(self, env):
        xmin, ymin, xmax, ymax = env
        return {"geometry": f"{xmin},{ymin},{xmax},{ymax}", "geometryType": "esriGeometryEnvelope",
                "inSR": self.wkid, "spatialRel": "esriSpatialRelIntersects"}

    def count(self, env, where: str | None = None) -> int:
        params = {"where": where or self.where, "returnCountOnly": "true", "f": "json", **self._geom(env)}
        data = http_get(self.url + "/query", params, tries=self.tries, timeout=self.timeout)
        time.sleep(self.pause_s)
        return int(data.get("count", 0))

    def query(self, env) -> list:
        out, offset = [], 0
        while True:
            params = {"where": self.where, "outFields": ",".join(self.fields), "returnGeometry": "true",
                      "outSR": self.wkid, "geometryPrecision": 2, "resultOffset": offset,
                      "resultRecordCount": 2000, "f": "json", **self._geom(env)}
            data = http_get(self.url + "/query", params, tries=self.tries, timeout=self.timeout)
            time.sleep(self.pause_s)
            feats = data.get("features", [])
            for f in feats:
                rings = (f.get("geometry") or {}).get("rings") or []
                if rings:
                    attrs = f.get("attributes") or {}
                    out.append(Footprint(np.asarray(max(rings, key=len), dtype=float),
                                         str(attrs.get("PRIM_OCC") or "")))
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
    return abs(a), np.array([((x + x1) * cross).sum() / (6 * a), ((y + y1) * cross).sum() / (6 * a)])


def convex_hull(pts: np.ndarray) -> np.ndarray:
    """Andrew's monotone chain."""
    p = sorted(set(map(tuple, np.round(pts, 6))))
    if len(p) <= 2:
        return np.asarray(p, dtype=float)

    def half(seq):
        h = []
        for q in seq:
            while len(h) >= 2 and ((h[-1][0] - h[-2][0]) * (q[1] - h[-2][1])
                                   - (h[-1][1] - h[-2][1]) * (q[0] - h[-2][0])) <= 0:
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
    alpha = math.degrees(math.atan2(-dy, dx))
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


def to_block_px(ring: np.ndarray, origin, res: float) -> np.ndarray:
    x0, y0 = origin
    return np.stack([(ring[:, 0] - x0) / res, (y0 - ring[:, 1]) / res], axis=1)


# ---------------------------------------------------------------------------------------------
# Lining the outlines up with the photo (one shift per block)
# ---------------------------------------------------------------------------------------------
def _gradient(gray: np.ndarray) -> np.ndarray:
    g = gray.astype(np.float32)
    gx = np.zeros_like(g)
    gy = np.zeros_like(g)
    gx[:, 1:-1] = g[:, 2:] - g[:, :-2]
    gy[1:-1, :] = g[2:, :] - g[:-2, :]
    return np.hypot(gx, gy)


def house_shift(gray: np.ndarray, ring_px: np.ndarray, r: int):
    """Best (dx, dy) in pixels that moves the outline onto roof edges, and how clear the best is.

    Correlates a drawing of the outline with the photo's edge strength for every shift up to r pixels
    (computed all at once with the FFT)."""
    x0, y0 = np.floor(ring_px.min(axis=0)).astype(int) - r - 4
    x1, y1 = np.ceil(ring_px.max(axis=0)).astype(int) + r + 4
    if x0 < 0 or y0 < 0 or x1 > gray.shape[1] or y1 > gray.shape[0]:
        return None
    patch = gray[y0:y1, x0:x1]
    grad = _gradient(patch)
    grad = (grad - grad.mean()) / (grad.std() + 1e-6)
    e_img = Image.new("L", (x1 - x0, y1 - y0), 0)
    pts = [tuple(p) for p in (ring_px - np.array([x0, y0]))]
    ImageDraw.Draw(e_img).line(pts + [pts[0]], fill=255, width=2)
    edge = np.asarray(e_img, dtype=np.float32) / 255.0
    if edge.sum() < 10:
        return None
    corr = np.real(np.fft.ifft2(np.conj(np.fft.fft2(edge)) * np.fft.fft2(grad)))
    idx = np.arange(-r, r + 1)
    win = corr[np.ix_(idx % corr.shape[0], idx % corr.shape[1])]       # win[dy + r, dx + r]
    k = int(np.argmax(win))
    dy, dx = divmod(k, win.shape[1])
    dy, dx = dy - r, dx - r
    clarity = float((win.max() - np.median(win)) / (win.std() + 1e-6))
    return dx, dy, clarity, (abs(dx) == r or abs(dy) == r)


def estimate_block_shift(block_img: Image.Image, rings_px: list, res: float, k: float = 1.0,
                         max_shift_m: float = 6.0, sample: int = 40, min_good: int = 6,
                         min_clarity: float = 5.0, max_spread_m: float = 2.0, seed: int = 0):
    """Median outline-to-roof shift for one neighbourhood, in block pixels: (dx, dy, houses used) or None."""
    r = max(2, int(round(max_shift_m * k / res)))
    gray = np.asarray(block_img.convert("L"))
    order = list(range(len(rings_px)))
    random.Random(seed).shuffle(order)
    good = []
    for i in order[:sample]:
        est = house_shift(gray, rings_px[i], r)
        if est is None:
            continue
        dx, dy, clarity, on_border = est
        if clarity >= min_clarity and not on_border:
            good.append((dx, dy))
    if len(good) < min_good:
        return None
    g = np.asarray(good, dtype=float)
    med = np.median(g, axis=0)
    spread = np.median(np.abs(g - med), axis=0)
    if (spread > max_spread_m * k / res).any():                         # houses disagree: do not guess
        return None
    return float(med[0]), float(med[1]), len(good)


# ---------------------------------------------------------------------------------------------
# Cutting one roof out of a block image
# ---------------------------------------------------------------------------------------------
@dataclass
class Chip:
    image: Image.Image | None = None
    area_m2: float = 0.0
    long_m: float = 0.0
    short_m: float = 0.0
    angle_deg: float = 0.0
    reason: str = ""                     # filled when the roof was skipped
    to_chip: object = None               # function: ring in work units -> chip pixel coordinates


def _flat_cells(img: Image.Image, cell: int = 8):
    """8 x 8 cells that are perfectly flat black, and perfectly flat white."""
    a = np.asarray(img.convert("RGB")).astype(np.int16)
    h, w = (a.shape[0] // cell) * cell, (a.shape[1] // cell) * cell
    c = a[:h, :w].reshape(h // cell, cell, w // cell, cell, 3)
    flat = (c.max(axis=(1, 3)).max(axis=-1) - c.min(axis=(1, 3)).min(axis=-1)) <= 6
    mean = c.mean(axis=(1, 3, 4))
    return flat & (mean <= 4), flat & (mean >= 251)


def _touching(a: np.ndarray, b: np.ndarray, d: int = 2) -> np.ndarray:
    """Cells of a within d cells of a cell of b (turned checkerboards have soft edges between squares)."""
    nb = np.zeros_like(b)
    h, w = b.shape
    for dy in range(-d, d + 1):
        for dx in range(-d, d + 1):
            nb[max(dy, 0):h + min(dy, 0), max(dx, 0):w + min(dx, 0)] |= \
                b[max(-dy, 0):h + min(-dy, 0), max(-dx, 0):w + min(-dx, 0)]
    return a & nb


def nodata_fraction(img: Image.Image, cell: int = 8) -> float:
    """Share of the picture with no photo. Services fill missing areas with flat black, and the Texas service
    draws a black-and-white checkerboard. Count flat black cells, plus flat white cells that touch flat black
    ones (the checkerboard). Deep shadows (noisy) and white roofs on their own are not counted."""
    black, white = _flat_cells(img, cell)
    return float((black | _touching(white, black)).mean())


def checkerboard_mask(block_img: Image.Image, cell: int = 8):
    """Whole-block map of the Texas 'no photo' checkerboard (grown by one cell), or None when there is none.
    Checked on the full block, so even a sliver of checkerboard in a roof's corner is caught."""
    black, white = _flat_cells(block_img, cell)
    chk = _touching(white, black) | _touching(black, white)
    if not chk.any():
        return None
    chk = _touching(np.ones_like(chk), chk, d=1)
    full = np.zeros((block_img.height, block_img.width), np.uint8)
    up = np.kron(chk.astype(np.uint8) * 255, np.ones((cell, cell), np.uint8))
    full[:up.shape[0], :up.shape[1]] = up
    return Image.fromarray(full)


def has_checkerboard(img: Image.Image, cell: int = 8) -> bool:
    """Flat white cells next to flat black ones: the Texas 'no photo' pattern. Checked on 650 trial roofs:
    it marks only the two Texas roofs with a checkerboard corner."""
    black, white = _flat_cells(img, cell)
    return int(_touching(white, black).sum()) >= 2


def cut_chip(block_img: Image.Image, block_origin, res: float, ring: np.ndarray, s: Settings,
             k: float = 1.0, missing: Image.Image | None = None) -> Chip:
    """block_origin = (xmin, ymax) of block_img in work units; res = work units per block pixel;
    k = work units per metre (1 for UTM; 1/cos(latitude) for Web Mercator);
    missing = optional block-sized mask of 'no photo' areas (checkerboard_mask)."""
    ring_px = to_block_px(ring, block_origin, res)
    area_px, cen = polygon_area_centroid(ring_px)
    m_per_px = res / k
    chip = Chip(area_m2=area_px * m_per_px ** 2)
    direction, long_px, short_px = min_area_rect(ring_px)
    chip.long_m, chip.short_m = long_px * m_per_px, short_px * m_per_px
    if not (s.min_area_m2 <= chip.area_m2 <= s.max_area_m2):
        chip.reason = "area"
        return chip
    if chip.long_m > s.max_long_m or chip.short_m < s.min_short_m:
        chip.reason = "shape"
        return chip
    chip.angle_deg = turn_angle_deg(direction)
    win = s.window_m / m_per_px                       # window side in block pixels
    half_big = math.ceil(win * math.sqrt(2) / 2) + 3
    cx, cy = float(cen[0]), float(cen[1])
    bx0, by0 = int(math.floor(cx)) - half_big, int(math.floor(cy)) - half_big
    if bx0 < 0 or by0 < 0 or bx0 + 2 * half_big > block_img.width or by0 + 2 * half_big > block_img.height:
        chip.reason = "edge"
        return chip
    big = block_img.crop((bx0, by0, bx0 + 2 * half_big, by0 + 2 * half_big))
    lc = (cx - bx0, cy - by0)                          # roof centre inside `big`
    box = (lc[0] - win / 2, lc[1] - win / 2, lc[0] + win / 2, lc[1] + win / 2)
    turned = big.rotate(chip.angle_deg, resample=Image.BICUBIC, center=lc)
    out = turned.transform((s.chip_px, s.chip_px), Image.EXTENT, box, resample=Image.BICUBIC)
    if nodata_fraction(out) > 0.05 or has_checkerboard(out):
        chip.reason = "nodata"
        return chip
    if missing is not None:
        m_big = missing.crop((bx0, by0, bx0 + 2 * half_big, by0 + 2 * half_big))
        m_out = m_big.rotate(chip.angle_deg, resample=Image.NEAREST, center=lc).transform(
            (s.chip_px, s.chip_px), Image.EXTENT, box, resample=Image.NEAREST)
        if np.asarray(m_out).max() > 0:
            chip.reason = "nodata"
            return chip
    scale = s.chip_px / win

    def to_chip(any_ring):
        p = to_block_px(any_ring, block_origin, res) - np.array([bx0, by0])
        return (rotate_points(p, chip.angle_deg, lc) - np.array([box[0], box[1]])) * scale

    chip.image, chip.to_chip = out, to_chip
    return chip


def chip_fit(chip: Chip, ring: np.ndarray, r: int, pad: int = 16):
    """How well the outline sits on roof edges in the finished chip, and the best nudge (dx, dy) in chip pixels.

    fit = average edge strength (in standard deviations) under the outline at the best nudge;
    clarity = how much that best nudge stands out from all the other nudges."""
    gray = np.asarray(chip.image.convert("L"))
    grad = _gradient(gray)
    grad = (grad - grad.mean()) / (grad.std() + 1e-6)
    e_img = Image.new("L", chip.image.size, 0)
    pts = [tuple(q) for q in chip.to_chip(ring)]
    ImageDraw.Draw(e_img).line(pts + [pts[0]], fill=255, width=1)
    edge = np.asarray(e_img, dtype=np.float32) / 255.0
    n = float(edge.sum())
    if n < 10:
        return 0, 0, 0.0, 0.0
    h, w = grad.shape
    gp = np.zeros((h + 2 * pad, w + 2 * pad), np.float32)
    ep = np.zeros_like(gp)
    gp[pad:pad + h, pad:pad + w], ep[pad:pad + h, pad:pad + w] = grad, edge
    corr = np.real(np.fft.ifft2(np.conj(np.fft.fft2(ep)) * np.fft.fft2(gp)))
    idx = np.arange(-r, r + 1)
    win = corr[np.ix_(idx % corr.shape[0], idx % corr.shape[1])] / n
    k = int(np.argmax(win))
    dy, dx = divmod(k, win.shape[1])
    clarity = float((win.max() - np.median(win)) / (win.std() + 1e-6))
    return dx - r, dy - r, clarity, float(win.max())


def chip_nudge_to_work(chip: Chip, dx: float, dy: float, res: float, s: Settings, k: float) -> np.ndarray:
    """A nudge measured in the turned chip (pixels) as a move in the work coordinate system."""
    win = s.window_m * k / res
    vx, vy = dx * win / s.chip_px, dy * win / s.chip_px           # block pixels, still turned
    t = math.radians(chip.angle_deg)
    c, sn = math.cos(t), math.sin(t)
    bx, by = c * vx - sn * vy, sn * vx + c * vy                   # undo the turn
    return np.array([bx * res, -by * res])


# ---------------------------------------------------------------------------------------------
# Building a pack
# ---------------------------------------------------------------------------------------------
def block_grid(extent, block: float, margin: float):
    xmin, ymin, xmax, ymax = extent
    xs = np.arange(xmin + margin, xmax - margin - 1e-6, block)
    ys = np.arange(ymin + margin, ymax - margin - 1e-6, block)
    return [(float(x), float(y), float(min(x + block, xmax - margin)), float(min(y + block, ymax - margin)))
            for y in ys for x in xs]


def make_services(src: dict, cfg: dict, s: Settings):
    fp = cfg["footprints"]
    timeout = int(src.get("timeout", 90))
    if src["kind"] == "image":
        img = ImageService(src["url"], int(src["wkid"]), s.pause_s, tries=5, timeout=timeout)
    elif src["kind"] == "tiles":
        img = TileService(src["url"], int(src["level"]), s.pause_s, s.tile_pause_s, timeout=timeout)
    elif src["kind"] == "noaa_eri":
        known = [{"id": f["id"], "box": tuple(f["box"]), "template": f["template"]}
                 for f in cfg.get("noaa_flights", {}).get(src["url"], [])]
        img = NoaaEriService(src["url"], int(src["level"]), src.get("flights", ()), s.pause_s,
                             s.tile_pause_s, timeout=timeout, known=known)
    else:
        raise ValueError(f"unknown source kind {src['kind']}")
    fps = FootprintService(fp["url"], img.wkid, fp["fields"], fp["where"], s.pause_s)
    return img, fps


def make_labels(lab_cfg: dict, wkid: int, s: Settings) -> LabelService:
    return LabelService(lab_cfg["url"], wkid, lab_cfg["event"], lab_cfg.get("exclude_assessment", ()), s.pause_s)


def source_area(src: dict, info: dict, wkid: int):
    """Work-system envelope of the area to cut from, and k (work units per metre)."""
    if src.get("area") == "service":
        e = info["extent"]
        sr = e.get("spatialReference", {})
        native = sr.get("latestWkid") or sr.get("wkid")
        if native not in (26919, 6348) or wkid != 26919:   # both are UTM zone 19N in metres (NAD83 variants)
            raise SystemExit(f"❌ {src['url']}: 'area: service' needs a UTM 19N service, got wkid {native}")
        return (e["xmin"], e["ymin"], e["xmax"], e["ymax"]), 1.0
    env = lonlat_box_to(src["area_lonlat"], wkid)
    if wkid in (3857, 102100):
        lat = merc_y_to_lat((env[1] + env[3]) / 2)
        return env, 1.0 / math.cos(math.radians(lat))
    return env, 1.0


@dataclass
class Run:
    """Progress through one source."""
    sid: str
    src: dict
    img: object = None
    fp: object = None
    labels: object = None
    extent: tuple = ()
    k: float = 1.0
    blocks: list = field(default_factory=list)
    next_block: int = 0
    quota: int = 0
    kept: int = 0
    done: bool = False
    shifts: list = field(default_factory=list)       # per-block (dx_m, dy_m) offsets, metres
    stats: dict = field(default_factory=dict)


def _atomic_write(path: Path, text: str):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _home_key(ring: np.ndarray, wkid: int) -> str:
    """One-way key so overlapping towns never give the same house twice. Only the hash is kept."""
    _, c = polygon_area_centroid(ring)
    return hashlib.sha256(f"{wkid}:{round(c[0])}:{round(c[1])}".encode()).hexdigest()[:20]


def _review_image(chip: Chip, ring_raw, ring_used) -> Image.Image:
    im = chip.image.copy()
    d = ImageDraw.Draw(im)
    for ring, colour in ((ring_raw, (255, 60, 60)), (ring_used, (255, 230, 0))):
        p = [tuple(q) for q in chip.to_chip(ring)]
        d.line(p + [p[0]], fill=colour, width=1)
    return im


def _redistribute(runs: list):
    """A source that ran out of blocks hands its unused quota to the sources still going."""
    spare = sum(r.quota - r.kept for r in runs if r.done and r.kept < r.quota)
    if spare <= 0:
        return
    for r in runs:
        if r.done:
            r.quota = r.kept
    active = [r for r in runs if not r.done]
    for i, r in enumerate(active):
        r.quota += spare // len(active) + (1 if i < spare % len(active) else 0)


def build(s: Settings, sources: list, cfg: dict, resume: bool = False, fresh: bool = False,
          log=print) -> dict:
    """Build one pack from the given source ids. Returns the build report."""
    t_start = time.time()
    if s.out.exists():
        if fresh:
            shutil.rmtree(s.out)
        elif not resume:
            raise SystemExit(f"❌ {s.out} already exists. Add --resume to continue it, or --fresh to start over.")
    resuming = resume and (s.out / STATE_FILE).exists()
    if resume and not resuming and s.out.exists():
        raise SystemExit(f"❌ {s.out} has no saved progress to resume. Use --fresh to start over.")
    (s.out / "images").mkdir(parents=True, exist_ok=True)
    if s.review:
        (s.out / "review").mkdir(exist_ok=True)

    log("🔎 Checking the free data services ...")
    runs = []
    lab_cfg = cfg["packs"].get(s.pack, {}).get("labels") if s.labels else None
    for sid in sources:
        src = cfg["sources"][sid]
        img, fp = make_services(src, cfg, s)
        info = img.info()
        extent, k = source_area(src, info, img.wkid)
        total = fp.count(extent)
        log(f"   {src['place']} ({src['region']} {src['year']}): photos OK, {total:,} homes marked residential")
        if "missions" in info:
            log(f"      NOAA flight days: {', '.join(info['missions'])}; tile level {info['level']}")
        labels = None
        if lab_cfg:
            labels = make_labels(lab_cfg, img.wkid, s)
            found = labels.info()["fields"]
            n_lab = len(labels.query(extent)) if s.trial else None
            here = f", {n_lab:,} assessed homes here" if n_lab is not None else ""
            log(f"      damage labels: fields {found}{here}")
        blocks = block_grid(extent, s.block_m * k, s.margin_m * k)
        random.Random(f"{s.seed}-{sid}").shuffle(blocks)
        stats = {"blocks_tried": 0, "blocks_no_photos": 0, "blocks_few_homes": 0,
                 "blocks_outlines_off": 0, "blocks_used": 0, "skipped": {}, "shift_fallbacks": 0,
                 "houses_nudged": 0}
        if labels is not None:
            stats.update({"blocks_unassessed": 0, "damage": {}})
        runs.append(Run(sid, src, img, fp, extent=extent, k=k, blocks=blocks, labels=labels, stats=stats))
    base, extra = divmod(s.target, len(runs))
    for i, r in enumerate(runs):
        r.quota = base + (1 if i < extra else 0)

    seen, n_chips, block_log, elapsed_before = set(), 0, [], 0.0
    manifest = s.out / "manifest.csv"
    if resuming:
        st = json.loads((s.out / STATE_FILE).read_text())
        if st.get("signature") != s.signature():
            raise SystemExit("❌ The settings changed since this build started. Use --fresh to start over.")
        for r in runs:
            rs = st["runs"][r.sid]
            r.next_block, r.quota, r.kept, r.done = rs["next_block"], rs["quota"], rs["kept"], rs["done"]
            r.shifts, r.stats = [tuple(x) for x in rs["shifts"]], rs["stats"]
        seen, n_chips = set(st["seen"]), st["chips"]
        block_log, elapsed_before = st["block_log"], st["elapsed_s"]
        rows = list(csv.DictReader(open(manifest))) if manifest.exists() else []
        if len(rows) < n_chips:
            raise SystemExit("❌ manifest.csv is shorter than the saved progress. Use --fresh to start over.")
        if len(rows) > n_chips:                       # stopped mid-block: drop that block's partial work
            for row in rows[n_chips:]:
                for folder in ("images", "review"):
                    (s.out / folder / row["file"]).unlink(missing_ok=True)
            with open(manifest, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=s.fields)
                w.writeheader()
                w.writerows(rows[:n_chips])
        log(f"↩️  Resuming: {n_chips:,} roofs already done.")
    else:
        with open(manifest, "w", newline="") as fh:
            csv.DictWriter(fh, fieldnames=s.fields).writeheader()

    def save_state():
        state = {"signature": s.signature(), "chips": n_chips, "seen": sorted(seen), "block_log": block_log,
                 "elapsed_s": elapsed_before + (time.time() - t_start),
                 "runs": {r.sid: {"next_block": r.next_block, "quota": r.quota, "kept": r.kept, "done": r.done,
                                  "shifts": r.shifts, "stats": r.stats} for r in runs}}
        _atomic_write(s.out / STATE_FILE, json.dumps(state))

    big_px = int(round((s.block_m + 2 * s.margin_m) / s.fetch_res))
    log(f"🏠 Cutting {s.target:,} roofs from {len(runs)} sources "
        f"({s.block_m:.0f} m blocks, {big_px}x{big_px} px downloads) ...")
    while n_chips < s.target and any(not r.done for r in runs):
        for r in runs:
            if r.done or n_chips >= s.target:
                continue
            if r.kept >= r.quota or r.next_block >= len(r.blocks):
                r.done = True
                _redistribute(runs)
                continue
            bi = r.next_block
            core = tuple(r.blocks[bi])
            r.next_block += 1
            r.stats["blocks_tried"] += 1
            t0 = time.time()
            m, big = s.margin_m * r.k, (s.block_m + 2 * s.margin_m) * r.k
            bbox = (core[0] - m, core[1] - m, core[0] - m + big, core[1] - m + big)   # always square
            if r.fp.count(core) < s.min_homes_per_block:
                r.stats["blocks_few_homes"] += 1
                save_state()
                continue
            if 1.0 - nodata_fraction(r.img.export(bbox, 64)) < s.coverage_min:
                r.stats["blocks_no_photos"] += 1
                save_state()
                continue
            points = []
            if r.labels is not None:
                points = r.labels.query(bbox)
                if lab_cfg.get("assessed_blocks_only", True) and not any(p[2] in DAMAGED for p in points):
                    r.stats["blocks_unassessed"] += 1      # FEMA recorded no damage here: no basis for labels
                    save_state()
                    continue
            mine = []
            for f in r.fp.query(core):                 # each home once: its centre must lie in this block
                _, c = polygon_area_centroid(f.ring)
                if core[0] <= c[0] < core[2] and core[1] <= c[1] < core[3]:
                    key = _home_key(f.ring, r.img.wkid)
                    if key not in seen:
                        mine.append((f, key))
            random.Random(f"{s.seed}-{r.sid}-{bi}").shuffle(mine)
            side = bbox[2] - bbox[0]
            block_img = r.img.export(bbox, big_px)
            res = side / big_px
            origin = (bbox[0], bbox[3])

            shift_px = None
            if s.align and mine:
                est = estimate_block_shift(block_img, [to_block_px(f.ring, origin, res) for f, _ in mine],
                                           res, r.k, s.max_shift_m, seed=bi)
                if est is not None:
                    shift_px = (est[0], est[1])
                    r.shifts.append((round(est[0] * res / r.k, 2), round(-est[1] * res / r.k, 2)))
                elif len(r.shifts) >= 3:                 # fall back to this town's usual shift
                    med = np.median(np.asarray(r.shifts), axis=0)
                    shift_px = (med[0] * r.k / res, -med[1] * r.k / res)
                    r.stats["shift_fallbacks"] += 1
                else:
                    r.stats["shift_fallbacks"] += 1
            move = np.array([shift_px[0] * res, -shift_px[1] * res]) if shift_px else np.zeros(2)

            missing = checkerboard_mask(block_img)
            snap_px = max(1, int(round(s.snap_m / s.fetch_res)))      # chip pixels
            good, fits, skipped_here = [], [], {}
            for f, key in mine:                                     # judge every house in the block first
                ring_used = f.ring + move
                chip = cut_chip(block_img, origin, res, ring_used, s, r.k, missing)
                if chip.reason:
                    skipped_here[chip.reason] = skipped_here.get(chip.reason, 0) + 1
                    continue
                dx, dy, clarity, fit = chip_fit(chip, ring_used, snap_px)
                fits.append(fit)
                if fit < s.fit_min:
                    skipped_here["no_roof"] = skipped_here.get("no_roof", 0) + 1
                    continue
                if clarity >= s.snap_clarity and (dx or dy):
                    nudged = ring_used + chip_nudge_to_work(chip, dx, dy, res, s, r.k)
                    chip2 = cut_chip(block_img, origin, res, nudged, s, r.k, missing)
                    if not chip2.reason:
                        chip, ring_used = chip2, nudged
                        r.stats["houses_nudged"] = r.stats.get("houses_nudged", 0) + 1
                good.append((f, key, chip, ring_used))
            if fits and float(np.median(fits)) < s.block_fit_min:   # outlines mostly miss: skip the block
                r.stats["blocks_outlines_off"] = r.stats.get("blocks_outlines_off", 0) + 1
                good = []
            for reason, n in skipped_here.items():
                r.stats["skipped"][reason] = r.stats["skipped"].get(reason, 0) + n

            new_rows, kept_here = [], 0
            if r.labels is not None:
                near = float(lab_cfg.get("match_m", 6.0)) * r.k
                labelled = [(f, key, chip, ring_used, damage_for(f.ring, points, near))
                            for f, key, chip, ring_used in good]
                hit = [x for x in labelled if x[4][0] in DAMAGED]
                calm = [x for x in labelled if x[4][0] not in DAMAGED]
                ratio = float(lab_cfg.get("undamaged_per_damaged", 1.0))
                calm = calm[:max(int(lab_cfg.get("min_undamaged_per_block", 3)), int(round(ratio * len(hit))))]
                labelled = [x for pair in itertools.zip_longest(hit, calm) for x in pair if x is not None]
            else:
                labelled = [(f, key, chip, ring_used, None) for f, key, chip, ring_used in good]
            for f, key, chip, ring_used, dmg in labelled:
                if kept_here >= s.max_per_block or r.kept + kept_here >= r.quota or n_chips + kept_here >= s.target:
                    break
                if dmg is not None:
                    r.stats["damage"][dmg[0]] = r.stats["damage"].get(dmg[0], 0) + 1
                chip_id = f"{n_chips + kept_here + 1:06d}"
                fname = f"{chip_id}.jpg"
                chip.image.save(s.out / "images" / fname, quality=95, subsampling=0)
                if s.review:
                    _review_image(chip, f.ring, ring_used).save(s.out / "review" / fname, quality=90)
                seen.add(key)
                new_rows.append({"chip_id": chip_id, "file": fname, "source": r.sid,
                                 "region": r.src["region"], "place": r.src["place"], "year": r.src["year"],
                                 "area_m2": round(chip.area_m2, 1), "long_m": round(chip.long_m, 1),
                                 "short_m": round(chip.short_m, 1), "angle_deg": round(chip.angle_deg, 1),
                                 "use": f.use, **({"damage": dmg[0], "assessment": dmg[1]} if dmg else {})})
                kept_here += 1
            with open(manifest, "a", newline="") as fh:
                csv.DictWriter(fh, fieldnames=s.fields).writerows(new_rows)
            n_chips += kept_here
            r.kept += kept_here
            r.stats["blocks_used"] += 1
            dt = time.time() - t0
            block_log.append({"source": r.sid, "homes": len(mine), "kept": kept_here, "seconds": round(dt, 1),
                              "shift_m": r.shifts[-1] if (shift_px and r.shifts) else None,
                              "median_fit": round(float(np.median(fits)), 2) if fits else None})
            save_state()
            log(f"   {r.src['place']:<18} block {r.stats['blocks_used']:3d}: kept {kept_here:3d} "
                f"({n_chips:,}/{s.target:,} total, {dt:.0f} s)")

    return finish(s, runs, cfg, n_chips, block_log, elapsed_before + (time.time() - t_start), log)


# ---------------------------------------------------------------------------------------------
# Finishing: attribution, checksums, report, privacy check, zip
# ---------------------------------------------------------------------------------------------
PRIVACY_NOTE = ("This pack contains no addresses, coordinates, building IDs or owner information. Building "
                "outlines were used only to find, turn and centre each roof and are not included.")


def attribution_text(s: Settings, runs: list, cfg: dict) -> str:
    lines = [f"{s.title or s.name}", "Built for an educational guest lecture on generative models.", "",
             "Aerial imagery"]
    seen = set()
    for r in runs:                        # one entry per distinct source (several towns can share one program)
        entry = (r.src["attribution"], r.src["url"], r.src["licence"])
        if entry not in seen:
            seen.add(entry)
            lines += [f"  {entry[0]}", f"    {entry[1]}", f"    Licence: {entry[2]}"]
    fp = cfg["footprints"]
    lines += ["", "Building outlines (used to find, turn and centre each roof; NOT included in this pack)",
              f"  {fp['label']}", f"    {fp['url']}", f"    Licence: {fp['licence']}"]
    lab = cfg.get("packs", {}).get(s.pack, {}).get("labels") if s.labels else None
    if lab:
        lines += ["", "Damage labels (the 'damage' and 'assessment' columns; the points themselves are NOT included)",
                  f"  {lab['label']}", f"    {lab['url']}", f"    Licence: {lab['licence']}"]
    lines += ["", PRIVACY_NOTE, ""]
    return "\n".join(lines)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


BANNED_WORDS = re.compile(r"addr|address|street|owner|parcel|lat|lon|coord|build_id|bbox|extent|utm|zip", re.I)


def privacy_problems(out: Path) -> list:
    """What would leak location or identity if this folder were published (empty list = fine)."""
    problems = []
    with open(out / "manifest.csv") as fh:
        header = next(csv.reader(fh))
    if header not in (MANIFEST_FIELDS, MANIFEST_FIELDS + LABEL_FIELDS):
        problems.append(f"unexpected manifest columns: {header}")
    report = json.loads((out / "build_report.json").read_text())

    def walk(o, path=""):
        if isinstance(o, dict):
            for key, val in o.items():
                if BANNED_WORDS.search(str(key)):
                    problems.append(f"build_report.json has a location-like field: {path}{key}")
                walk(val, f"{path}{key}.")
        elif isinstance(o, list):
            for v in o:
                walk(v, path)
    walk(report)
    big_number = re.compile(r"\b\d{6,}\.\d+|\b-?\d{1,3}\.\d{5,}\b")       # UTM metres or decimal degrees
    for name in ("manifest.csv", "build_report.json", "ATTRIBUTION.txt"):
        text = (out / name).read_text()
        if name == "ATTRIBUTION.txt":
            text = re.sub(r"https?://\S+", "", text)
        if big_number.search(text):
            problems.append(f"{name} contains a coordinate-like number")
    allowed_top = {"images", "review", "manifest.csv", "ATTRIBUTION.txt", "build_report.json", "SHA256SUMS.txt",
                   "README.txt", STATE_FILE}
    for p in out.iterdir():
        if p.name not in allowed_top:
            problems.append(f"unexpected file in the pack: {p.name}")
    return problems


def readme_text(s: Settings, n_chips: int, notes: list) -> str:
    text = (f"{s.title or s.name}\n\n{n_chips:,} roof images, {s.chip_px}x{s.chip_px} JPEG, each a "
            f"{s.window_m:.0f} m x {s.window_m:.0f} m square turned so the roof's long side is horizontal.\n"
            "manifest.csv: one row per image (town, year, roof area and size in metres, turn angle, building use"
            + ("; FEMA's damage level for the home and how it was judged" if s.labels else "") + ").\n"
            "ATTRIBUTION.txt: sources and licences. SHA256SUMS.txt: checksums of every file.\n\n" + PRIVACY_NOTE + "\n")
    if notes:
        text += "\nKnown limits\n" + "".join(f"  - {n}\n" for n in notes)
    return text


def pack_notes(cfg: dict, s: Settings) -> list:
    return list(cfg.get("packs", {}).get(s.pack, {}).get("notes", []))


def seal(s: Settings, n_chips: int, per_source_chips: dict, built: str | None = None) -> dict:
    """Checksums, privacy check and (for real packs) the zip with its SHA-256. Images are never touched."""
    out = {}
    files = sorted(p for p in s.out.rglob("*") if p.is_file() and p.name != STATE_FILE
                   and p.name != "SHA256SUMS.txt" and p.name != "trial_sheet.jpg"
                   and "review" not in p.relative_to(s.out).parts)
    (s.out / "SHA256SUMS.txt").write_text(
        "".join(f"{sha256_of(p)}  {p.relative_to(s.out).as_posix()}\n" for p in files))
    problems = privacy_problems(s.out)
    if problems:
        raise SystemExit("❌ Privacy check failed, nothing was zipped:\n   " + "\n   ".join(problems))
    out["privacy_check"] = "passed"
    if s.zip_dir is not None and not s.trial:
        s.zip_dir.mkdir(parents=True, exist_ok=True)
        zpath = s.zip_dir / f"{s.name}.zip"
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as z:
            for p in files + [s.out / "SHA256SUMS.txt"]:
                z.write(p, f"{s.name}/{p.relative_to(s.out).as_posix()}")
        meta = {"name": s.name, "file": zpath.name, "bytes": zpath.stat().st_size, "sha256": sha256_of(zpath),
                "chips": n_chips, "chip_px": s.chip_px, "window_m": s.window_m, "per_source": per_source_chips,
                "built": built or time.strftime("%Y-%m-%d")}
        if built:
            meta["docs_updated"] = time.strftime("%Y-%m-%d")
        (s.zip_dir / f"{s.name}.json").write_text(json.dumps(meta, indent=2) + "\n")
        out["zip"], out["zip_sha256"], out["zip_bytes"] = str(zpath), meta["sha256"], meta["bytes"]
    return out


def finish(s: Settings, runs: list, cfg: dict, n_chips: int, block_log: list, seconds: float, log=print) -> dict:
    (s.out / "ATTRIBUTION.txt").write_text(attribution_text(s, runs, cfg))
    (s.out / "README.txt").write_text(readme_text(s, n_chips, pack_notes(cfg, s)))
    per_source = {}
    for r in runs:
        sh = np.asarray(r.shifts) if r.shifts else np.zeros((0, 2))
        per_source[r.sid] = {
            "place": r.src["place"], "region": r.src["region"], "year": r.src["year"], "chips": r.kept,
            **r.stats,
            "median_outline_shift_m": round(float(np.median(np.hypot(sh[:, 0], sh[:, 1]))), 2) if len(sh) else None,
        }
    report = {
        "pack": s.name, "title": s.title, "trial": s.trial, "chips": n_chips, "target": s.target,
        "sources": per_source, "seconds": round(seconds, 1),
        "minutes_per_1000_chips": round(seconds / max(1, n_chips) * 1000 / 60, 1),
        "window_m": s.window_m, "chip_px": s.chip_px, "metres_per_pixel": s.fetch_res, "block_m": s.block_m,
        "python": sys.version.split()[0], "seed": s.seed, "blocks": block_log,
    }
    (s.out / "build_report.json").write_text(json.dumps(report, indent=2))   # final: checksummed below
    report = dict(report)                                                  # extra keys stay out of the file
    report.update(seal(s, n_chips, {k: v["chips"] for k, v in per_source.items()}))
    if s.review:
        make_sheet(s.out, runs, log)
    return report


def rezip(s: Settings, cfg: dict, log=print) -> dict:
    """Refresh README.txt (for example new 'Known limits' notes), ATTRIBUTION.txt, checksums and the zip of a
    finished pack, without downloading or changing a single roof image."""
    report_path, manifest = s.out / "build_report.json", s.out / "manifest.csv"
    if not report_path.exists() or not manifest.exists():
        raise SystemExit(f"❌ {s.out} is not a finished pack (no build_report.json). Build it first.")
    report = json.loads(report_path.read_text())
    n_chips = sum(1 for _ in csv.DictReader(open(manifest)))
    if n_chips != report["chips"]:
        raise SystemExit("❌ manifest.csv and build_report.json disagree. Rebuild this pack with --fresh.")
    old_meta = s.zip_dir / f"{s.name}.json" if s.zip_dir else None
    built = json.loads(old_meta.read_text()).get("built") if old_meta and old_meta.exists() else None
    (s.out / "README.txt").write_text(readme_text(s, n_chips, pack_notes(cfg, s)))
    runs = [Run(sid, cfg["sources"][sid], None, None) for sid in report["sources"] if sid in cfg["sources"]]
    if len(runs) == len(report["sources"]):
        (s.out / "ATTRIBUTION.txt").write_text(attribution_text(s, runs, cfg))
    out = seal(s, n_chips, {k: v["chips"] for k, v in report["sources"].items()}, built=built or report.get("built"))
    log(f"   README and ATTRIBUTION refreshed, checksums rebuilt, privacy check {out['privacy_check']}.")
    return out


def make_sheet(out: Path, runs: list, log=print, per_source: int = 20, cols: int = 10):
    """Trial only: one review picture with up to 20 roofs per source (red = public outline, yellow = shifted)."""
    rows = list(csv.DictReader(open(out / "manifest.csv")))
    damage_of = {row["file"]: row["damage"] for row in rows if row.get("damage")}
    tile, label_w = 130, 170
    bands = []
    for r in runs:
        files = [row["file"] for row in rows if row["source"] == r.sid][:per_source]
        bands.append((f"{r.src['place']}\n{r.src['region']} {r.src['year']}", files))
    height = sum(max(1, math.ceil(len(f) / cols)) * tile for _, f in bands) + 30
    sheet = Image.new("RGB", (label_w + cols * tile, height), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    d.text((6, 8), "Trial review: red = public outline, yellow = after the neighbourhood shift and house nudge",
           fill=(0, 0, 0))
    y = 30
    for label, files in bands:
        d.text((6, y + 6), label, fill=(0, 0, 0))
        for i, fname in enumerate(files):
            im = Image.open(out / "review" / fname).resize((tile - 4, tile - 4))
            dmg = damage_of.get(fname)
            if dmg:
                ImageDraw.Draw(im).rectangle((0, tile - 18, tile, tile), fill=(0, 0, 0))
                ImageDraw.Draw(im).text((3, tile - 17), dmg, fill=(255, 255, 255))
            sheet.paste(im, (label_w + (i % cols) * tile, y + (i // cols) * tile))
        y += max(1, math.ceil(len(files) / cols)) * tile
    sheet.save(out / "trial_sheet.jpg", quality=88)
    log(f"   review picture: {out / 'trial_sheet.jpg'}")


# ---------------------------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------------------------
def settings_for(pack: str, cfg: dict, trial: bool, out: Path | None = None, seed: int = 7) -> tuple:
    p = cfg["packs"][pack]
    n_src = len(p["sources"])
    target = 50 * n_src if trial else int(p["target"])
    folder = (ROOT / "data" / "trial" / p["name"]) if trial else (ROOT / "data" / "roof_packs" / p["name"])
    s = Settings(out=out or folder, pack=pack, name=p["name"], title=p["title"], target=target, seed=seed,
                 block_m=float(p["block_m"]), max_per_block=int(p["max_per_block"]),
                 min_homes_per_block=int(p["min_homes_per_block"]), trial=trial, review=trial,
                 zip_dir=None if trial else ROOT / "dist", labels="labels" in p)
    if trial:
        s.max_per_block = min(s.max_per_block, 25)     # a trial looks at several neighbourhoods per town
    return s, p["sources"]


def print_sources(cfg: dict):
    for pack, p in cfg["packs"].items():
        print(f"\n{pack}: {p['name']}  (target {p['target']:,} roofs)\n   {p['title']}")
        for sid in p["sources"]:
            src = cfg["sources"][sid]
            print(f"   - {src['place']}, {src['region']} {src['year']}  [{src['kind']}]  licence: {src['licence'][:70]}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", choices=["train", "variety", "storm", "all"], default=None,
                    help="'all' builds train and variety; the storm pack is built on its own")
    ap.add_argument("--trial", action="store_true", help="about 50 roofs per source plus a review picture")
    ap.add_argument("--resume", action="store_true", help="continue a build that stopped")
    ap.add_argument("--fresh", action="store_true", help="delete an unfinished build and start again")
    ap.add_argument("--list", action="store_true", help="show the packs and sources, then stop")
    ap.add_argument("--rezip", action="store_true",
                    help="refresh README/checksums/zip of finished packs (no downloads, images unchanged)")
    ap.add_argument("--only", default=None, help="comma-separated source ids, for a quick check of a few sources")
    ap.add_argument("--out", type=Path, default=None, help="output folder (one pack only)")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args(argv)
    cfg = load_config()
    if a.list or not a.pack:
        print_sources(cfg)
        if not a.list:
            print("\nChoose a pack, for example:  python scripts/build_roof_pack.py --pack all --trial")
        return
    packs = ["train", "variety"] if a.pack == "all" else [a.pack]
    if a.rezip:
        if a.trial or a.only or a.resume or a.fresh:
            raise SystemExit("❌ --rezip works on finished packs only (no --trial, --only, --resume or --fresh).")
        for pack in packs:
            s, _ = settings_for(pack, cfg, False, a.out if len(packs) == 1 else None, a.seed)
            print(f"\n📦 {s.name}: refreshing README, ATTRIBUTION and zip")
            r = rezip(s, cfg)
            print(f"   Zip: {r['zip']}\n   SHA-256: {r['zip_sha256']}")
        return
    only = [x.strip() for x in a.only.split(",")] if a.only else None
    if only:
        unknown = [x for x in only if x not in cfg["sources"]]
        if unknown:
            raise SystemExit(f"❌ unknown source id(s): {unknown}. See --list.")
        if not a.trial:
            raise SystemExit("❌ --only is for trial checks; the real packs always use every source.")
    for pack in packs:
        s, sources = settings_for(pack, cfg, a.trial, a.out if len(packs) == 1 else None, a.seed)
        if only:
            sources = [x for x in sources if x in only]
            if not sources:
                continue
            s.target = 50 * len(sources)
            s.out = s.out.with_name(s.out.name + "-only")
        print(f"\n📦 {s.title}\n   Output: {s.out}")
        try:
            r = build(s, sources, cfg, resume=a.resume, fresh=a.fresh)
        except ServiceError as e:
            raise SystemExit(f"❌ A data service did not answer properly: {e}\n   Nothing is broken on your Mac. "
                             f"Run the same command again with --resume to continue.")
        print(f"\n✅ {s.name}: {r['chips']:,} roofs in {r['seconds'] / 60:.1f} min "
              f"(about {r['minutes_per_1000_chips']} min per 1,000). Privacy check: {r['privacy_check']}.")
        for sid, v in r["sources"].items():
            print(f"   {v['place']:<18} {v['chips']:6,} roofs   outline shift ~{v['median_outline_shift_m']} m   "
                  f"skipped {v['skipped']}" + (f"   damage {v['damage']}" if "damage" in v else ""))
        if "zip" in r:
            print(f"   Zip: {r['zip']}\n   SHA-256: {r['zip_sha256']}")


if __name__ == "__main__":
    main()
