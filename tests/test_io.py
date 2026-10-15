import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from beacon.io import make_run_dir, save_figure


def test_make_run_dir_writes_params(tmp_path):
    d = make_run_dir("smoke", {"a": 1}, root=str(tmp_path), date="2026-10-14")
    assert d.name == "2026-10-14_smoke"
    p = json.loads((d / "params.json").read_text())
    assert p["a"] == 1 and len(p["git_commit"]) == 40 and isinstance(p["git_dirty"], bool)


def test_save_figure_writes_pdf_and_png(tmp_path):
    fig = plt.figure()
    save_figure(fig, tmp_path, "f")
    assert (tmp_path / "f.pdf").exists() and (tmp_path / "f.png").exists()


def test_require_full_run_rejects_smoke(tmp_path):
    import pytest
    from beacon.io import require_full_run
    (tmp_path / "params.json").write_text(json.dumps({"smoke": True}))
    with pytest.raises(SystemExit):
        require_full_run(tmp_path)
    require_full_run(tmp_path, allow_smoke=True)  # explicit override passes


def test_require_full_run_accepts_full(tmp_path):
    from beacon.io import require_full_run
    (tmp_path / "params.json").write_text(json.dumps({"smoke": False}))
    require_full_run(tmp_path)


def test_git_dirty_ignores_untracked_files(tmp_path):
    import subprocess
    repo = tmp_path / "repo"
    repo.mkdir()
    def git(*a):
        subprocess.run(["git", *a], cwd=repo, check=True, capture_output=True)
    git("init", "-q")
    (repo / "f.txt").write_text("x")
    git("add", "f.txt")
    git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "init")
    (repo / "untracked.txt").write_text("y")
    d = make_run_dir("a", {}, root=str(tmp_path / "out"), repo=repo)
    assert json.loads((d / "params.json").read_text())["git_dirty"] is False
    (repo / "f.txt").write_text("changed")
    d = make_run_dir("b", {}, root=str(tmp_path / "out"), repo=repo)
    assert json.loads((d / "params.json").read_text())["git_dirty"] is True
