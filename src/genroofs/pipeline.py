"""The insurer's pipeline, end to end, on one public building: address -> point -> photo -> roof -> score.

The building is fixed: South Portland City Hall, 25 Cottage Road, South Portland, Maine (a public building).
The functions take no address argument. You can read and change the code, and that is the point of the ethics
discussion: the same few lines would score anyone's home.

Two free geocoders are asked for the same building, and they disagree:
- the U.S. Census Bureau geocoder interpolates along the street's address range, so its point lands on the street;
- OpenStreetMap's Nominatim finds the building by name, so its point lands on the roof.
The demo pictures both, so the class can see how a 36-metre gap decides which "roof" gets scored.

OpenStreetMap's usage policy allows at most one request per second for all users of an application together, so
by default the demo uses the answer Nominatim gave on November 13, 2026 (a public building does not move). The
instructor passes osm="live" to ask again in class; everyone else stays well inside the policy.

Nothing is written to disk. Credits: U.S. Census Bureau Geocoder (public domain); OpenStreetMap Nominatim
(data (c) OpenStreetMap contributors, ODbL); Maine GeoLibrary, Maine Orthoimagery - South Portland 2022.
"""
import io
import json
import math
import urllib.error
import urllib.parse
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from . import data
from .metrics import reconstruction_errors

__all__ = ["BUILDING", "RECORDED_OSM", "Point", "PipelineError", "geocode_census", "geocode_osm", "geocode_public_building",
           "distance_m", "lonlat_to_utm", "fetch_photo", "roof_crop", "score_roof", "run_demo", "CREDITS"]

BUILDING = {"name": "South Portland City Hall", "address": "25 Cottage Road, South Portland, ME 04106",
            "osm_query": "South Portland City Hall, 25 Cottage Road, South Portland, Maine"}
CENSUS_URL = "https://geocoding.geo.census.gov/geocoder/locations/onelineaddress"
OSM_URL = "https://nominatim.openstreetmap.org/search"
PHOTO_URL = "https://gis.maine.gov/image/rest/services/Municipal/orthoMunicipalSouthPortland2022/ImageServer"
PHOTO_WKID, UTM_ZONE = 26919, 19          # NAD83 / UTM zone 19N, as the Maine pack was built
WINDOW_M, PHOTO_PX = 24.0, 128            # the same window and size as every roof in the packs
RECORDED_OSM = {"lon": -70.2518709, "lat": 43.6373371, "name": "South Portland City Hall", "date": "2026-11-13"}
CREDITS = ("Geocoding: U.S. Census Bureau Geocoder (public domain) and OpenStreetMap Nominatim (data (c) "
           "OpenStreetMap contributors, ODbL). Photo: Maine GeoLibrary, Maine Orthoimagery - South Portland 2022.")


class PipelineError(RuntimeError):
    """A free service didn't answer or answered something unexpected. The message names the service."""


@dataclass
class Point:
    lon: float
    lat: float
    source: str
    matched: str

    def __str__(self) -> str:
        return f"{self.lat:.5f}, {self.lon:.5f} ({self.source}: {self.matched})"


def _get(url: str, params: dict, service: str, timeout: float = 30) -> bytes:
    full = f"{url}?{urllib.parse.urlencode(params)}"
    try:
        with data._open_url(full, timeout) as resp:
            return resp.read()
    except (OSError, ValueError) as err:
        reason = f"HTTP {err.code}" if isinstance(err, urllib.error.HTTPError) else str(getattr(err, "reason", err))
        raise PipelineError(f"{service} did not answer ({reason}). Check the internet connection, or skip this "
                            f"demo; the slides have a screenshot.") from None


def geocode_census() -> Point:
    """Ask the U.S. Census Bureau geocoder (free, no key) for the building's address."""
    body = _get(CENSUS_URL, {"address": BUILDING["address"], "benchmark": "Public_AR_Current", "format": "json"},
                "The Census geocoder")
    try:
        match = json.loads(body)["result"]["addressMatches"][0]
        return Point(float(match["coordinates"]["x"]), float(match["coordinates"]["y"]), "Census",
                     match["matchedAddress"])
    except (ValueError, KeyError, IndexError, TypeError):
        raise PipelineError("The Census geocoder found no match for the address.") from None


def geocode_osm(live: bool = True) -> Point:
    """Ask OpenStreetMap's Nominatim (free, no key) for the building by name. With live=False, return the answer
    recorded on RECORDED_OSM["date"] instead of asking (see the usage-policy note at the top of this module)."""
    if not live:
        r = RECORDED_OSM
        return Point(r["lon"], r["lat"], "OpenStreetMap", f"{r['name']} (answer recorded {r['date']})")
    body = _get(OSM_URL, {"q": BUILDING["osm_query"], "format": "jsonv2", "limit": 1}, "OpenStreetMap Nominatim")
    try:
        hit = json.loads(body)[0]
        point = Point(float(hit["lon"]), float(hit["lat"]), "OpenStreetMap", hit.get("name") or hit["display_name"])
    except (ValueError, KeyError, IndexError, TypeError):
        raise PipelineError("OpenStreetMap Nominatim found no match for the building.") from None
    if "city hall" not in point.matched.lower():
        raise PipelineError(f"OpenStreetMap Nominatim answered with {point.matched!r}, not the City Hall.")
    return point


def distance_m(a: Point, b: Point) -> float:
    """Distance on the ground between two points, in metres (haversine)."""
    r = 6371008.8
    p1, p2 = math.radians(a.lat), math.radians(b.lat)
    dp, dl = p2 - p1, math.radians(b.lon - a.lon)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def _check_osm_mode(osm: str) -> bool:
    if osm not in ("recorded", "live"):
        raise ValueError('osm must be "recorded" (the default, for everyone) or "live" (the instructor, once).')
    return osm == "live"


def geocode_public_building(osm: str = "recorded") -> dict:
    """Both geocoders for the same building, and how far apart their points are.
    osm: "recorded" uses OpenStreetMap's answer from November 13, 2026; "live" asks Nominatim again."""
    census, osm = geocode_census(), geocode_osm(live=_check_osm_mode(osm))
    gap = distance_m(census, osm)
    if gap > 500:
        raise PipelineError(f"The two geocoders disagree by {gap:.0f} m, more than expected; check the services.")
    return {"census": census, "osm": osm, "gap_m": gap}


def lonlat_to_utm(lon: float, lat: float, zone: int = UTM_ZONE) -> tuple:
    """Longitude/latitude to NAD83 UTM metres (GRS80 Transverse Mercator; well under a metre inside the zone)."""
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


def fetch_photo(point: Point, window_m: float = WINDOW_M, size_px: int = PHOTO_PX) -> torch.Tensor:
    """A window_m x window_m square of the South Portland 2022 photo centred on the point, north up,
    as a uint8 tensor 3 x size_px x size_px (the same window and size as the pack's roofs)."""
    x, y = lonlat_to_utm(point.lon, point.lat)
    h = window_m / 2
    params = {"bbox": f"{x - h:.2f},{y - h:.2f},{x + h:.2f},{y + h:.2f}", "bboxSR": PHOTO_WKID,
              "imageSR": PHOTO_WKID, "size": f"{size_px},{size_px}", "format": "jpg", "compressionQuality": 92,
              "bandIds": "0,1,2", "interpolation": "RSP_BilinearInterpolation", "f": "image"}
    body = _get(PHOTO_URL + "/exportImage", params, "The Maine image service")
    try:
        with Image.open(io.BytesIO(body)) as im:
            im = im.convert("RGB")
            if im.size != (size_px, size_px):
                im = im.resize((size_px, size_px), Image.Resampling.BILINEAR)
            arr = np.asarray(im, dtype=np.uint8).transpose(2, 0, 1).copy()
    except OSError:
        raise PipelineError("The Maine image service did not send a picture.") from None
    if (arr.max(axis=0) < 8).mean() > 0.5:
        raise PipelineError("The Maine image service has no photo at this point (the picture is mostly black).")
    return torch.from_numpy(arr)


def roof_crop(photo: torch.Tensor, size: int = 64) -> torch.Tensor:
    """Shrink the photo exactly as the pack loader shrinks every roof (area averaging)."""
    im = Image.fromarray(photo.permute(1, 2, 0).numpy())
    if im.size != (size, size):
        im = im.resize((size, size), Image.Resampling.BOX)
    return torch.from_numpy(np.asarray(im, dtype=np.uint8).transpose(2, 0, 1).copy())


def score_roof(model, crop: torch.Tensor, reference_errors) -> dict:
    """The autoencoder's unusual score for one crop, and where it falls among the reference (test) roofs."""
    error = float(reconstruction_errors(model, crop.unsqueeze(0))[0])
    ref = torch.as_tensor(reference_errors, dtype=torch.float64)
    percentile = float((ref < error).double().mean() * 100) if len(ref) else float("nan")
    return {"error": error, "percentile": percentile}


def run_demo(ae, reference_errors, threshold: float = 95.0, osm: str = "recorded", show: bool = True):
    """The whole chain for both geocoders, printed step by step and pictured side by side.

    reference_errors: reconstruction_errors(ae, test_images) for the Maine test roofs.
    threshold: the percentile above which the roof "goes to an inspector".
    osm: "recorded" (default) or "live"; the instructor uses "live" once in class.
    """
    from .exercises import Result, _copies
    from .viz import show_rows
    _check_osm_mode(osm)
    print(f"1. Address: {BUILDING['name']}, {BUILDING['address']} (a public building)")
    points = geocode_public_building(osm)
    census, osm = points["census"], points["osm"]
    print(f"2. Census geocoder point:        {census}")
    print(f"   OpenStreetMap building point: {osm}")
    print(f"   The two points are {points['gap_m']:.0f} m apart.")
    photos = {"Census point": fetch_photo(census), "OpenStreetMap point": fetch_photo(osm)}
    print(f"3. Photo: a {WINDOW_M:g} m square around each point from the South Portland 2022 orthophoto")
    crops = {k: roof_crop(v) for k, v in photos.items()}
    scores = {k: score_roof(ae, v, reference_errors) for k, v in crops.items()}
    print("4. Unusual score from the autoencoder:")
    for k, s in scores.items():
        verdict = "goes to an inspector" if s["percentile"] >= threshold else "no inspection"
        print(f"   {k:<20} score {s['error']:8.0f}, higher than {s['percentile']:5.1f}% of the Maine test roofs "
              f"-> {verdict}")
    print("5. Questions: which square is the building? Who decided which point to trust? The model learned from "
          "houses; is a 19th-century former church 'unusual' or just different?")
    print(f"\nCredits: {CREDITS}")
    if show:
        names = list(photos)
        show_rows({"photo (128 px)": torch.stack([photos[k] for k in names]),
                   "crop (64 px)": torch.stack([crops[k] for k in names]),
                   "AE copy": torch.cat([_copies(ae, crops[k].unsqueeze(0)) for k in names])},
                  labels=names, title="From an address to a score: two geocoders, two different squares")
    return Result(gap_m=points["gap_m"], census_score=scores["Census point"]["error"],
                  census_percentile=scores["Census point"]["percentile"],
                  osm_score=scores["OpenStreetMap point"]["error"],
                  osm_percentile=scores["OpenStreetMap point"]["percentile"], points=points, photos=photos)
