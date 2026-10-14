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
