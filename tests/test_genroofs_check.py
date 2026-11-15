"""Offline tests for python -m genroofs.check."""
import sys
from pathlib import Path

from conftest import FakeServer
from genroofs import check, data


def test_tidy_hides_the_home_folder(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "someone"))
    assert check.tidy(tmp_path / "someone" / "Projects" / "x") == "~/Projects/x"
    assert check.tidy("/content/genroofs_data") == "/content/genroofs_data"


def test_all_good(monkeypatch, tmp_path, capsys):
    server = FakeServer(b"P")
    monkeypatch.setattr(data, "_open_url", server.open)
    monkeypatch.setenv("GENROOFS_DATA", str(tmp_path / "cache"))
    assert check.main() == 0
    out = capsys.readouterr().out
    assert out.rstrip().endswith("ALL GOOD")
    for line in ("Python ", "torch ", "safetensors", "Device ", "Cache ", "Release     data-v1 answers",
                 "Training    2 steps"):
        assert line in out
    assert server.headers == {"Range": "bytes=0-0"}, "only one byte of the pack is asked for"
    assert server.calls[0].endswith("/releases/download/data-v1/roofs_us_variety.zip")


def test_problems_are_listed(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("GENROOFS_DATA", str(tmp_path / "cache"))

    class Usage:
        free = 2e8
    monkeypatch.setattr(check.shutil, "disk_usage", lambda p: Usage)
    monkeypatch.setitem(sys.modules, "pandas", None)
    assert check.main() == 1                          # the real network is blocked in tests
    out = capsys.readouterr().out
    assert "Release     data-v1 did NOT answer" in out and "Fix these" in out
    assert "1. Only 0.2 GB free" in out and "2. Can't reach the roof packs" in out
    assert "pandas      not installed (optional" in out
    assert "ALL GOOD" not in out


def test_never_prints_the_home_folder(monkeypatch, tmp_path, capsys):
    home = tmp_path / "home-of-someone"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setenv("GENROOFS_DATA", str(home / ".cache" / "genroofs"))
    monkeypatch.setattr(data, "_open_url", FakeServer(b"P").open)
    check.main()
    out = capsys.readouterr().out
    assert str(home) not in out and "~/.cache/genroofs" in out
