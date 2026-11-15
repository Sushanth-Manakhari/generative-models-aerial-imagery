"""Offline tests for genroofs.pipeline: the three free services are replaced by a fake that answers like them."""
import importlib.util
import io
import json
import sys
import urllib.error
import urllib.parse
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from genroofs import data, pipeline
from genroofs.models import RoofAE

ROOT = Path(__file__).resolve().parents[1]
CENSUS = (-70.251446636979, 43.637431693344)        # what the Census geocoder answered for the address
OSM = (-70.2518709, 43.6373371)                      # what OpenStreetMap answered for the building's name


class Reply:
    def __init__(self, body: bytes):
        self.body = body
        self.headers = {}

    def read(self, n=-1):
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def photo_bytes(dark: bool = False) -> bytes:
    rng = np.random.default_rng(0)
    arr = np.zeros((128, 128, 3), np.uint8) if dark else rng.integers(60, 200, (128, 128, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "JPEG", quality=92)
    return buf.getvalue()


class FakeServices:
    def __init__(self):
        self.calls, self.down, self.census_matches, self.osm_name, self.dark, self.osm_point = [], set(), True, \
            "South Portland City Hall", False, OSM

    def open(self, url, timeout, headers=None):
        self.calls.append(url)
        host = urllib.parse.urlparse(url).netloc
        if host in self.down:
            raise urllib.error.HTTPError(url, 503, "Service Unavailable", {}, None)
        if host == "geocoding.geo.census.gov":
            matches = [{"coordinates": {"x": CENSUS[0], "y": CENSUS[1]},
                        "matchedAddress": "25 COTTAGE RD, SOUTH PORTLAND, ME, 04106"}] if self.census_matches else []
            return Reply(json.dumps({"result": {"addressMatches": matches}}).encode())
        if host == "nominatim.openstreetmap.org":
            lon, lat = self.osm_point
            return Reply(json.dumps([{"lat": str(lat), "lon": str(lon), "name": self.osm_name,
                                      "display_name": f"{self.osm_name}, 25, Cottage Road"}]).encode())
        if host == "gis.maine.gov":
            return Reply(photo_bytes(self.dark))
        raise AssertionError(f"unexpected URL {url}")


@pytest.fixture
def services(monkeypatch):
    fake = FakeServices()
    monkeypatch.setattr(data, "_open_url", fake.open)
    return fake


def query(url: str) -> dict:
    return dict(urllib.parse.parse_qsl(urllib.parse.urlparse(url).query))


def test_both_geocoders(services):
    census, osm = pipeline.geocode_census(), pipeline.geocode_osm()
    assert (census.lon, census.lat, census.source) == (*CENSUS, "Census")
    assert census.matched == "25 COTTAGE RD, SOUTH PORTLAND, ME, 04106"
    assert (osm.lon, osm.lat, osm.matched) == (*OSM, "South Portland City Hall")
    assert query(services.calls[0])["address"] == "25 Cottage Road, South Portland, ME 04106"
    assert query(services.calls[1])["q"].startswith("South Portland City Hall")
    both = pipeline.geocode_public_building(osm="live")
    assert 34 < both["gap_m"] < 38


def test_geocoder_problems(services):
    services.census_matches = False
    with pytest.raises(pipeline.PipelineError, match="no match"):
        pipeline.geocode_census()
    services.osm_name = "Cottage Road Pizza"
    with pytest.raises(pipeline.PipelineError, match="not the City Hall"):
        pipeline.geocode_osm()
    services.census_matches, services.osm_name, services.osm_point = True, "South Portland City Hall", (-70.3, 43.7)
    with pytest.raises(pipeline.PipelineError, match="disagree"):
        pipeline.geocode_public_building(osm="live")


@pytest.mark.parametrize("host, name", [("geocoding.geo.census.gov", "The Census geocoder"),
                                        ("nominatim.openstreetmap.org", "OpenStreetMap Nominatim"),
                                        ("gis.maine.gov", "The Maine image service")])
def test_a_service_that_is_down_is_named(services, host, name):
    services.down.add(host)
    with pytest.raises(pipeline.PipelineError, match=f"{name} did not answer \\(HTTP 503\\)"):
        pipeline.run_demo(RoofAE(latent=8), torch.arange(10.0), osm="live", show=False)


def test_distance():
    a = pipeline.Point(0.0, 0.0, "a", "")
    b = pipeline.Point(0.0, 1.0, "b", "")
    assert pipeline.distance_m(a, b) == pytest.approx(111_195, rel=1e-3)
    assert pipeline.distance_m(a, a) == 0.0


def test_utm_matches_the_pack_builder():
    spec = importlib.util.spec_from_file_location("brp_pipe", ROOT / "scripts" / "build_roof_pack.py")
    brp = importlib.util.module_from_spec(spec)
    sys.modules["brp_pipe"] = brp
    spec.loader.exec_module(brp)
    for lon, lat in (CENSUS, OSM, (-69.0, 45.0), (-71.5, 43.1)):
        assert pipeline.lonlat_to_utm(lon, lat) == pytest.approx(brp.lonlat_to_utm(lon, lat, 19), abs=1e-6)
    x, y = pipeline.lonlat_to_utm(*CENSUS)
    assert (round(x, 1), round(y, 1)) == (399057.0, 4832365.3)


def test_photo_service_is_the_one_the_maine_pack_used():
    cfg = json.loads((ROOT / "scripts" / "roof_sources.json").read_text())["sources"]["me_south_portland_2022"]
    assert pipeline.PHOTO_URL == cfg["url"] and pipeline.PHOTO_WKID == cfg["wkid"]


def test_fetch_photo_asks_for_a_24_metre_square(services):
    photo = pipeline.fetch_photo(pipeline.Point(*OSM, "OpenStreetMap", "City Hall"))
    assert photo.shape == (3, 128, 128) and photo.dtype == torch.uint8
    q = query(services.calls[-1])
    xmin, ymin, xmax, ymax = map(float, q["bbox"].split(","))
    assert xmax - xmin == pytest.approx(24.0) and ymax - ymin == pytest.approx(24.0)
    x, y = pipeline.lonlat_to_utm(*OSM)
    assert (xmin + xmax) / 2 == pytest.approx(x, abs=0.01) and (ymin + ymax) / 2 == pytest.approx(y, abs=0.01)
    assert q["bboxSR"] == q["imageSR"] == "26919" and q["bandIds"] == "0,1,2" and q["size"] == "128,128"


def test_no_photo_and_not_a_picture(services, monkeypatch):
    services.dark = True
    with pytest.raises(pipeline.PipelineError, match="no photo at this point"):
        pipeline.fetch_photo(pipeline.Point(*OSM, "x", ""))
    monkeypatch.setattr(data, "_open_url", lambda url, timeout, headers=None: Reply(b"<html>error</html>"))
    with pytest.raises(pipeline.PipelineError, match="did not send a picture"):
        pipeline.fetch_photo(pipeline.Point(*OSM, "x", ""))


def test_crop_and_score():
    photo = torch.from_numpy(np.random.default_rng(1).integers(0, 255, (3, 128, 128), dtype=np.uint8))
    crop = pipeline.roof_crop(photo)
    want = np.asarray(Image.fromarray(photo.permute(1, 2, 0).numpy()).resize((64, 64), Image.Resampling.BOX))
    assert np.array_equal(crop.permute(1, 2, 0).numpy(), want)
    ae = RoofAE(latent=8)
    s = pipeline.score_roof(ae, crop, torch.tensor([0.0, 1e9]))
    assert s["percentile"] == 50.0 and s["error"] > 0


def test_recorded_openstreetmap_answer_is_used_by_default(services):
    both = pipeline.geocode_public_building()
    assert (both["osm"].lon, both["osm"].lat) == OSM and "recorded 2026-11-13" in both["osm"].matched
    assert not any("nominatim" in url for url in services.calls), "no request to OpenStreetMap by default"
    assert 34 < both["gap_m"] < 38
    with pytest.raises(ValueError, match="recorded"):
        pipeline.geocode_public_building(osm="sometimes")
    with pytest.raises(ValueError, match="recorded"):
        pipeline.run_demo(RoofAE(latent=8), torch.arange(10.0), osm="always", show=False)


def test_run_demo_end_to_end(services, capsys, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    r = pipeline.run_demo(RoofAE(latent=8), torch.linspace(0, 1e5, 100), osm="live")
    out = capsys.readouterr().out
    assert "South Portland City Hall, 25 Cottage Road" in out
    assert "Census geocoder point" in out and "OpenStreetMap building point" in out
    assert "The two points are 36 m apart" in out and "Credits:" in out and "ODbL" in out
    assert 34 < r.gap_m < 38 and set(r.photos) == {"Census point", "OpenStreetMap point"}
    assert 0 <= r.osm_percentile <= 100 and 0 <= r.census_percentile <= 100
    assert list(tmp_path.iterdir()) == [], "the demo writes nothing to disk"
    assert len(services.calls) == 4         # Census, OpenStreetMap, two photos


def test_real_network_stays_blocked_in_tests():
    with pytest.raises(RuntimeError, match="network access is blocked"):
        pipeline.geocode_census()
