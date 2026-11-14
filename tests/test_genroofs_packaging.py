"""Offline checks that the packaging, the laptop setup and the docs agree with each other."""
import re
from pathlib import Path

import pytest

import genroofs
from genroofs import pipeline

ROOT = Path(__file__).resolve().parents[1]
tomllib = pytest.importorskip("tomllib")        # Python 3.11+; on 3.10 these few checks are skipped


def pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def name_of(requirement: str) -> str:
    return re.split(r"[<>=!~\[;\s]", requirement.strip(), maxsplit=1)[0].lower()


def requirement_lines() -> list:
    lines = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]


def test_metadata():
    project = pyproject()["project"]
    assert project["name"] == "genroofs"
    assert project["requires-python"] == ">=3.10"
    assert pyproject()["tool"]["setuptools"]["dynamic"]["version"] == {"attr": "genroofs.__version__"}
    assert re.fullmatch(r"\d+\.\d+\.\d+(\.dev\d+)?", genroofs.__version__)
    assert set(project["optional-dependencies"]) == {"app", "pandas", "dev"}


def test_torch_is_not_pinned_anywhere():
    deps = pyproject()["project"]["dependencies"]
    assert "torch" in deps
    assert "torch" in requirement_lines()


def test_requirements_cover_the_package():
    package = {name_of(d) for d in pyproject()["project"]["dependencies"]}
    laptop = {name_of(r) for r in requirement_lines()}
    assert package <= laptop
    assert laptop - package == {"jupyterlab", "pytest"}


def test_data_files_are_packaged():
    assert pyproject()["tool"]["setuptools"]["package-data"]["genroofs"] == ["*.json"]
    for name in ("pack_info.json", "weights_info.json"):
        assert (ROOT / "src" / "genroofs" / name).is_file()


def test_setup_scripts_stay_inside_the_project():
    mac = (ROOT / "scripts" / "setup_mac.sh").read_text(encoding="utf-8")
    win = (ROOT / "scripts" / "setup_windows.ps1").read_text(encoding="utf-8")
    for text in (mac, win):
        code = "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("#"))
        assert "sudo" not in code and "--user" not in code and "-g " not in code
        assert ".venv" in code and "requirements.txt" in code and "genroofs.check" in code
        assert not re.search(r"\b3\.(1[1-9]|[2-9]\d)\b", text), "no Python version is hard-coded"
    assert 'VENV="$PROJECT/.venv"' in mac
    assert 'Join-Path $Project ".venv"' in win
    assert win.isascii(), "Windows PowerShell 5.1 misreads non-ASCII characters in scripts without a BOM"


def test_public_repo_gets_the_setup_files():
    listing = ROOT / ".public-exclude"            # exists only in the instructor's private copy
    excluded = listing.read_text(encoding="utf-8").splitlines() if listing.exists() else []
    for path in ("scripts/setup_mac.sh", "scripts/setup_windows.ps1", "docs/SETUP.md", "pyproject.toml",
                 "requirements.txt"):
        assert path not in excluded
        assert (ROOT / path).is_file()


def test_docs_credit_openstreetmap():
    sources = (ROOT / "docs" / "DATA_SOURCES.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for text in (sources, readme, pipeline.CREDITS):
        assert "OpenStreetMap" in text and "ODbL" in text
    assert "operations.osmfoundation.org/policies/nominatim" in sources
    assert pipeline.RECORDED_OSM["date"] in ("2026-11-13",)
    assert "November 13, 2026" in sources
