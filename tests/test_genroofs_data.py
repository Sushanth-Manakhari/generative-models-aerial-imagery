"""Offline tests for genroofs.data: pack info, cache folder, checked downloads, loading, slices, splits, batches."""
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from conftest import FAKE_NAME, FAKE_SOURCES, MANIFEST_FIELDS, build_zip, marker_value
from genroofs import data

ROOT = Path(__file__).resolve().parents[1]
REAL_OPEN_URL = data._open_url


def leftovers(folder: Path) -> list:
    return sorted(p.name for p in folder.glob("*")) if folder.exists() else []


def marker_of(images: torch.Tensor, k: int) -> float:
    return images[k, :, 0, 0].float().mean().item()


# ----------------------------------------------------------------------------------------------- pack info

def test_bundled_pack_info_matches_packs_folder():
    bundled = json.loads((ROOT / "src" / "genroofs" / "pack_info.json").read_text())
    on_disk = {p.stem: json.loads(p.read_text()) for p in sorted((ROOT / "packs").glob("*.json"))}
    assert bundled == on_disk, "src/genroofs/pack_info.json must be an exact copy of packs/*.json"


def test_available_packs_and_unknown_name():
    assert data.available_packs() == ["roofs_maine_2022", "roofs_us_variety"]
    with pytest.raises(data.PackError, match="Available packs: roofs_maine_2022, roofs_us_variety"):
        data.pack_info("roofs_nowhere")


def test_pack_info_is_a_copy():
    info = data.pack_info("roofs_maine_2022")
    info["sha256"] = "changed"
    info["per_source"]["me_wells_2022"] = 0
    fresh = data.pack_info("roofs_maine_2022")
    assert fresh["sha256"].startswith("d779d1a7")
    assert fresh["per_source"]["me_wells_2022"] == 2760
    assert fresh["chips"] == sum(fresh["per_source"].values()) == 22000


# --------------------------------------------------------------------------------------------- cache folder

def test_cache_dir_env_var_wins(monkeypatch, tmp_path):
    monkeypatch.setenv("GENROOFS_DATA", str(tmp_path / "mine"))
    monkeypatch.setattr(data, "_on_colab", lambda: True)
    assert data.cache_dir() == tmp_path / "mine"
    assert (tmp_path / "mine").is_dir()


def test_cache_dir_on_colab(monkeypatch):
    monkeypatch.delenv("GENROOFS_DATA", raising=False)
    monkeypatch.setattr(data, "_on_colab", lambda: True)
    assert data.cache_dir(create=False) == Path("/content/genroofs_data")


def test_colab_is_detected_from_its_environment(monkeypatch):
    monkeypatch.delitem(sys.modules, "google.colab", raising=False)
    monkeypatch.delenv("COLAB_RELEASE_TAG", raising=False)
    assert not data._on_colab()
    monkeypatch.setenv("COLAB_RELEASE_TAG", "release-colab-test")
    assert data._on_colab()


def test_cache_dir_inside_project_folder(monkeypatch, tmp_path):
    project = tmp_path / "project"
    (project / "packs").mkdir(parents=True)
    (project / "packs" / "roofs_maine_2022.json").write_text("{}")
    (project / "src" / "genroofs").mkdir(parents=True)
    (project / "notebooks").mkdir()
    monkeypatch.delenv("GENROOFS_DATA", raising=False)
    monkeypatch.setattr(data, "_on_colab", lambda: False)
    monkeypatch.chdir(project / "notebooks")
    assert data.cache_dir(create=False) == (project / "data" / "cache").resolve()


def test_cache_dir_falls_back_to_home(monkeypatch, tmp_path):
    monkeypatch.delenv("GENROOFS_DATA", raising=False)
    monkeypatch.setattr(data, "_on_colab", lambda: False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
    monkeypatch.chdir(tmp_path)
    assert data.cache_dir(create=False) == tmp_path / "home" / ".cache" / "genroofs"


# ------------------------------------------------------------------------------------------------ download

def test_download_checks_and_caches(fake_env):
    path = fake_env.data.download_pack(FAKE_NAME, quiet=True)
    assert path == fake_env.zip_path
    assert path.read_bytes() == fake_env.pack.zip_bytes
    assert leftovers(fake_env.cache) == [f"{FAKE_NAME}.zip", f"{FAKE_NAME}.zip.ok"]
    assert len(fake_env.server.calls) == 1
    assert fake_env.server.calls[0].endswith(f"/releases/download/data-v1/{FAKE_NAME}.zip")
    fake_env.data.download_pack(FAKE_NAME, quiet=True)
    assert len(fake_env.server.calls) == 1, "a checked zip in the cache must not be downloaded again"


def test_release_url_points_at_the_public_repo():
    assert data.RELEASE_URL.format(file="x.zip") == (
        "https://github.com/Sushanth-Manakhari/generative-models-aerial-imagery/releases/download/data-v1/x.zip")


def test_ok_note_saves_rehashing(fake_env, monkeypatch):
    fake_env.data.download_pack(FAKE_NAME, quiet=True)
    calls = []
    real = data.sha256_of
    monkeypatch.setattr(data, "sha256_of", lambda p: calls.append(p) or real(p))
    fake_env.data.download_pack(FAKE_NAME, quiet=True)
    assert calls == []
    fake_env.zip_path.with_name(f"{FAKE_NAME}.zip.ok").unlink()
    fake_env.data.download_pack(FAKE_NAME, quiet=True)
    assert len(calls) == 1 and len(fake_env.server.calls) == 1


def test_wrong_checksum_is_deleted_not_used(fake_env):
    fake_env.server.mode = "corrupt"
    with pytest.raises(data.PackError, match="SHA-256 does not match"):
        fake_env.data.download_pack(FAKE_NAME, quiet=True)
    assert leftovers(fake_env.cache) == []


def test_different_file_on_release_says_update(fake_env):
    fake_env.server.mode = "wrong_length"
    with pytest.raises(data.PackError, match="Update genroofs"):
        fake_env.data.download_pack(FAKE_NAME, quiet=True)
    assert leftovers(fake_env.cache) == []
    assert len(fake_env.server.calls) == 1


def test_dropped_connection_is_retried(fake_env):
    fake_env.server.mode = "drop_once"
    path = fake_env.data.download_pack(FAKE_NAME, quiet=True)
    assert path.read_bytes() == fake_env.pack.zip_bytes
    assert len(fake_env.server.calls) == 2
    assert leftovers(fake_env.cache) == [f"{FAKE_NAME}.zip", f"{FAKE_NAME}.zip.ok"]


@pytest.mark.parametrize("mode", ["drop_always", "short"])
def test_gives_up_after_three_tries_and_cleans_up(fake_env, mode):
    fake_env.server.mode = mode
    with pytest.raises(data.PackError, match="Could not download roofs_fake_test.zip"):
        fake_env.data.download_pack(FAKE_NAME, quiet=True)
    assert len(fake_env.server.calls) == 3
    assert leftovers(fake_env.cache) == []


def test_missing_file_is_not_retried(fake_env):
    fake_env.server.mode = "http404"
    with pytest.raises(data.PackError, match="HTTP 404"):
        fake_env.data.download_pack(FAKE_NAME, quiet=True)
    assert len(fake_env.server.calls) == 1


def test_damaged_zip_in_cache_is_replaced(fake_env, capsys):
    fake_env.cache.mkdir(parents=True)
    fake_env.zip_path.write_bytes(b"\0" * fake_env.pack.info["bytes"])
    path = fake_env.data.download_pack(FAKE_NAME)
    assert path.read_bytes() == fake_env.pack.zip_bytes
    assert "does not match its checksum" in capsys.readouterr().out


def test_not_enough_disk_space(fake_env, monkeypatch):
    class Usage:
        free = 1000
    monkeypatch.setattr(data.shutil, "disk_usage", lambda p: Usage)
    with pytest.raises(data.PackError, match="Not enough free disk space"):
        fake_env.data.download_pack(FAKE_NAME, quiet=True)
    assert fake_env.server.calls == []


def test_explicit_cache_folder(fake_env, tmp_path):
    path = fake_env.data.download_pack(FAKE_NAME, cache=tmp_path / "elsewhere", quiet=True)
    assert path == tmp_path / "elsewhere" / f"{FAKE_NAME}.zip"


def test_real_network_is_blocked_in_tests(fake_env, monkeypatch):
    monkeypatch.setattr(data, "_open_url", REAL_OPEN_URL)
    with pytest.raises(RuntimeError, match="network access is blocked"):
        data.download_pack(FAKE_NAME, quiet=True)
    assert leftovers(fake_env.cache) == []


def test_open_url_uses_a_certificate_bundle(monkeypatch):
    import ssl
    import urllib.request
    seen = {}

    def fake_urlopen(req, timeout, context):
        seen.update(req=req, timeout=timeout, context=context)
        return "response"
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    assert REAL_OPEN_URL("https://example.invalid/pack.zip", 7) == "response"
    assert seen["timeout"] == 7
    assert isinstance(seen["context"], ssl.SSLContext)
    assert seen["context"].verify_mode == ssl.CERT_REQUIRED
    assert seen["req"].get_header("User-agent").startswith("genroofs/")


# ------------------------------------------------------------------------------------------------- loading

def test_load_shapes_and_types(fake_env):
    images, manifest = data.load_roofs(FAKE_NAME, quiet=True)
    assert images.shape == (48, 3, 64, 64) and images.dtype == torch.uint8
    assert len(manifest) == 48
    row = manifest[0]
    assert list(row) == list(MANIFEST_FIELDS) == list(data.MANIFEST_FIELDS)
    assert isinstance(row["area_m2"], float) and isinstance(row["year"], int) and isinstance(row["place"], str)


def test_images_follow_manifest_order(fake_env):
    for size in (8, 32, 64, 128):
        images, manifest = data.load_roofs(FAKE_NAME, size=size, quiet=True)
        for k, row in enumerate(manifest):
            assert abs(marker_of(images, k) - marker_value(int(row["chip_id"]) - 1)) <= 3


def test_shrinking_uses_area_averaging(fake_env):
    images, manifest = data.load_roofs(FAKE_NAME, size=64, quiet=True)
    for k in (0, 17, 47):
        with Image.open(io.BytesIO(fake_env.pack.images[manifest[k]["file"]])) as im:
            want = np.asarray(im.convert("RGB").resize((64, 64), Image.Resampling.BOX)).transpose(2, 0, 1)
        assert np.array_equal(images[k].numpy(), want)


def test_size_limits(fake_env):
    for bad in (7, 129):
        with pytest.raises(ValueError, match="between 8 and 128"):
            data.load_roofs(FAKE_NAME, size=bad, quiet=True)


@pytest.mark.parametrize("angle, want", [(-90, 90), (90, 90), (-89.9, -89.9), (0, 0), (135, -45), (-135, 45),
                                         (180, 0), (-270, 90), (45.5, 45.5)])
def test_normalize_angle(angle, want):
    assert data.normalize_angle(angle) == pytest.approx(want)


def test_manifest_angles_are_cleaned(fake_env):
    _, manifest = data.load_roofs(FAKE_NAME, quiet=True)
    angles = [r["angle_deg"] for r in manifest]
    assert all(-90 < a <= 90 for a in angles)
    assert manifest[0]["angle_deg"] == 90.0          # stored as -90 in the pack
    assert -90.0 not in angles


def test_manifest_has_no_location_fields(fake_env):
    _, manifest = data.load_roofs(FAKE_NAME, quiet=True)
    assert all(tuple(r) == data.MANIFEST_FIELDS for r in manifest)
    for banned in ("address", "street", "lat", "latitude", "lon", "longitude", "x", "y", "building_id", "owner"):
        assert banned not in data.MANIFEST_FIELDS


def test_decoded_copy_is_reused(fake_env, monkeypatch):
    first, _ = data.load_roofs(FAKE_NAME, quiet=True)
    npys = list(fake_env.cache.glob("*.npy"))
    assert [p.name for p in npys] == [f"{FAKE_NAME}_64px_{fake_env.pack.info['sha256'][:12]}.npy"]
    monkeypatch.setattr(data, "_decode_all", lambda *a, **k: pytest.fail("decoded again"))
    second, _ = data.load_roofs(FAKE_NAME, quiet=True)
    assert torch.equal(first, second)


def test_damaged_decoded_copy_is_rebuilt(fake_env, capsys):
    first, _ = data.load_roofs(FAKE_NAME, quiet=True)
    npy = next(fake_env.cache.glob("*.npy"))
    npy.write_bytes(b"not a numpy file")
    second, _ = data.load_roofs(FAKE_NAME)
    assert torch.equal(first, second)
    assert "is damaged; decoding again" in capsys.readouterr().out


def test_full_disk_only_skips_the_decoded_copy(fake_env, monkeypatch, capsys):
    data.download_pack(FAKE_NAME, quiet=True)
    real = data._check_space

    def no_room_for_npy(folder, need, what):
        if what.endswith(".npy"):
            raise data.PackError("disk full")
        return real(folder, need, what)
    monkeypatch.setattr(data, "_check_space", no_room_for_npy)
    images, _ = data.load_roofs(FAKE_NAME)
    assert images.shape == (48, 3, 64, 64)
    assert not list(fake_env.cache.glob("*.npy*"))
    assert "could not save the decoded copy" in capsys.readouterr().out


def test_limit_picks_a_seeded_subset(fake_env):
    a = data.load_roofs(FAKE_NAME, limit=10, seed=1, quiet=True)
    b = data.load_roofs(FAKE_NAME, limit=10, seed=1, quiet=True)
    c = data.load_roofs(FAKE_NAME, limit=10, seed=2, quiet=True)
    assert len(a.manifest) == 10 and a.images.shape[0] == 10
    assert [r["chip_id"] for r in a.manifest] == [r["chip_id"] for r in b.manifest]
    assert [r["chip_id"] for r in a.manifest] != [r["chip_id"] for r in c.manifest]
    for k, row in enumerate(a.manifest):
        assert abs(marker_of(a.images, k) - marker_value(int(row["chip_id"]) - 1)) <= 3
    assert len(data.load_roofs(FAKE_NAME, limit=500, quiet=True).manifest) == 48
    with pytest.raises(ValueError):
        data.load_roofs(FAKE_NAME, limit=0, quiet=True)


def test_credits_printed_once_per_session(fake_env, capsys):
    data.load_roofs(FAKE_NAME)
    out = capsys.readouterr().out
    assert "roofs_fake_test: Fake test pack: roofs drawn by the test suite" in out
    assert "Imagery: Test Imagery Office; Second Office." in out
    assert "Building outlines (used only to find each roof, not included): Fake Outlines Project." in out
    assert "ATTRIBUTION.txt" in out and "Loaded 48 roofs at 64x64" in out
    data.load_roofs(FAKE_NAME)
    assert "Imagery:" not in capsys.readouterr().out


def test_quiet_prints_nothing(fake_env, capsys):
    data.load_roofs(FAKE_NAME, quiet=True)
    assert capsys.readouterr().out == ""


def test_full_attribution_text(fake_env):
    text = data.attribution(FAKE_NAME)
    assert text.startswith("Fake test pack") and "Licence: CC BY 4.0" in text


def test_image_missing_from_zip(fake_env):
    pack = fake_env.pack
    fake_env.install(build_zip(pack.rows, pack.images, skip={"000005.jpg"}))
    with pytest.raises(data.PackError, match="000005.jpg is listed in the manifest but missing"):
        data.load_roofs(FAKE_NAME, quiet=True)


def test_manifest_with_wrong_columns(fake_env):
    pack = fake_env.pack
    fake_env.install(build_zip(pack.rows, pack.images, manifest_fields=MANIFEST_FIELDS + ["lat"]))
    with pytest.raises(data.PackError, match="unexpected columns"):
        data.load_roofs(FAKE_NAME, quiet=True)


def test_roof_count_must_match_pack_info(fake_env):
    pack = fake_env.pack
    fake_env.install(pack.zip_bytes, chips=50)
    with pytest.raises(data.PackError, match="lists 48 roofs, expected 50"):
        data.load_roofs(FAKE_NAME, quiet=True)


# ----------------------------------------------------------------------------------------- slices, splits

def per_source(manifest) -> dict:
    counts = {}
    for r in manifest:
        counts[r["source"]] = counts.get(r["source"], 0) + 1
    return counts


def test_class_slice_takes_an_even_share_per_town(fake_env):
    roofs = data.load_roofs(FAKE_NAME, quiet=True)
    part = data.class_slice(roofs, n=20)
    assert per_source(part.manifest) == {s: 5 for s in FAKE_SOURCES}
    assert part.images.shape[0] == 20
    for k, row in enumerate(part.manifest):
        assert abs(marker_of(part.images, k) - marker_value(int(row["chip_id"]) - 1)) <= 3


def test_class_slice_passes_on_a_small_towns_share(fake_env):
    images, manifest = data.load_roofs(FAKE_NAME, quiet=True)
    part = data.class_slice(images, manifest, n=40)
    counts = per_source(part.manifest)
    assert sum(counts.values()) == 40
    assert counts["me_delta_2022"] == 8                       # all it has
    others = [counts[s] for s in ("me_alpha_2022", "me_beta_2022", "me_gamma_2022")]
    assert max(others) - min(others) <= 1


def test_class_slice_is_the_same_for_everyone(fake_env):
    roofs = data.load_roofs(FAKE_NAME, quiet=True)
    ids = lambda r: [x["chip_id"] for x in r.manifest]
    assert ids(data.class_slice(roofs, n=20, seed=7)) == ids(data.class_slice(roofs, n=20, seed=7))
    assert ids(data.class_slice(roofs, n=20, seed=7)) != ids(data.class_slice(roofs, n=20, seed=8))
    assert ids(data.class_slice(roofs, n=20)) == sorted(ids(data.class_slice(roofs, n=20)))


def test_class_slice_bigger_than_the_pack_returns_everything(fake_env):
    roofs = data.load_roofs(FAKE_NAME, quiet=True)
    assert len(data.class_slice(roofs, n=100).manifest) == 48
    with pytest.raises(ValueError):
        data.class_slice(roofs, n=0)
    with pytest.raises(TypeError):
        data.class_slice(roofs.images)


def test_even_quotas_for_the_real_maine_pack():
    sizes = data.pack_info("roofs_maine_2022")["per_source"]
    quotas = data._even_quotas(sizes, data.CLASS_SIZE)
    assert sum(quotas.values()) == 10_000 and set(quotas.values()) == {1250}


def test_train_test_split(fake_env):
    roofs = data.load_roofs(FAKE_NAME, quiet=True)
    train, test = data.train_test_split(roofs, test_frac=0.25)
    tr, te = {r["chip_id"] for r in train.manifest}, {r["chip_id"] for r in test.manifest}
    assert not tr & te and len(tr | te) == 48
    assert per_source(test.manifest) == {"me_alpha_2022": 4, "me_beta_2022": 3, "me_gamma_2022": 3,
                                         "me_delta_2022": 2}
    assert train.images.shape[0] == len(train.manifest) == 36
    again_train, again_test = data.train_test_split(*roofs, test_frac=0.25)
    assert [r["chip_id"] for r in again_test.manifest] == [r["chip_id"] for r in test.manifest]
    for bad in (0, 1, 1.5):
        with pytest.raises(ValueError):
            data.train_test_split(roofs, test_frac=bad)


def test_mismatched_images_and_manifest(fake_env):
    images, manifest = data.load_roofs(FAKE_NAME, quiet=True)
    with pytest.raises(ValueError, match="must match"):
        data.class_slice(images[:10], manifest, n=5)


# ------------------------------------------------------------------------------------------------ tensors

def test_to_float_and_back_is_exact():
    every = torch.arange(256, dtype=torch.uint8).view(1, 1, 16, 16)
    f = data.to_float(every)
    assert f.dtype == torch.float32 and f.min().item() == -1.0 and f.max().item() == 1.0
    assert torch.equal(data.to_bytes(f), every)
    assert torch.equal(data.to_bytes(torch.tensor([-3.0, 0.0, 5.0])), torch.tensor([0, 128, 255], dtype=torch.uint8))
    with pytest.raises(TypeError):
        data.to_float(f)


def numbered(n: int) -> torch.Tensor:
    return torch.arange(n, dtype=torch.uint8).view(n, 1, 1, 1).expand(n, 3, 4, 4).contiguous()


def ids_in(batch: torch.Tensor) -> list:
    return data.to_bytes(batch)[:, 0, 0, 0].tolist()


def test_batches_cover_each_roof_once():
    x = numbered(48)
    got = list(data.batches(x, batch_size=10, seed=3))
    assert len(got) == 4 and all(b.shape == (10, 3, 4, 4) and b.dtype == torch.float32 for b in got)
    seen = [i for b in got for i in ids_in(b)]
    assert len(seen) == len(set(seen)) == 40
    tail = list(data.batches(x, batch_size=10, seed=3, drop_last=False))
    assert [len(b) for b in tail] == [10, 10, 10, 10, 8]
    assert sorted(i for b in tail for i in ids_in(b)) == list(range(48))


def test_batches_order():
    x = numbered(48)
    a = [ids_in(b) for b in data.batches(x, batch_size=8, seed=5)]
    b = [ids_in(b) for b in data.batches(x, batch_size=8, seed=5)]
    c = [ids_in(b) for b in data.batches(x, batch_size=8, seed=6)]
    assert a == b and a != c
    plain = [ids_in(b) for b in data.batches(x, batch_size=8, shuffle=False)]
    assert plain[0] == list(range(8))
    torch.manual_seed(11)
    g1 = [ids_in(b) for b in data.batches(x, batch_size=8)]
    torch.manual_seed(11)
    g2 = [ids_in(b) for b in data.batches(x, batch_size=8)]
    assert g1 == g2


def test_batches_device_and_small_inputs():
    x = numbered(16)
    assert next(data.batches(x, batch_size=4, device="cpu")).device.type == "cpu"
    with pytest.raises(ValueError, match="fewer than one batch"):
        list(data.batches(numbered(5), batch_size=8))
    with pytest.raises(ValueError):
        list(data.batches(x, batch_size=0))


# ------------------------------------------------------------------------------------------------- pandas

def test_to_pandas_without_pandas(fake_env, monkeypatch):
    _, manifest = data.load_roofs(FAKE_NAME, quiet=True)
    monkeypatch.setitem(sys.modules, "pandas", None)
    with pytest.raises(ImportError, match="pip install pandas"):
        data.to_pandas(manifest)


def test_to_pandas_with_pandas(fake_env):
    pytest.importorskip("pandas")
    _, manifest = data.load_roofs(FAKE_NAME, quiet=True)
    df = data.to_pandas(manifest)
    assert list(df.columns) == list(data.MANIFEST_FIELDS) and len(df) == 48
