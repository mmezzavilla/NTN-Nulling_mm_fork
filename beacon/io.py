"""Run directories and figure saving for experiments (results/<date>_<name>/)."""
import datetime
import json
import pathlib
import subprocess

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True,
                          check=True, cwd=REPO_ROOT).stdout.strip()


def make_run_dir(name, params, root="results", date=None):
    """Create <root>/<date>_<name>/ and write params.json with the git state."""
    date = date or datetime.date.today().isoformat()
    run_dir = pathlib.Path(root) / f"{date}_{name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    record = dict(params)
    record["git_commit"] = _git("rev-parse", "HEAD")
    record["git_dirty"] = bool(_git("status", "--porcelain"))
    (run_dir / "params.json").write_text(json.dumps(record, indent=2, default=str))
    return run_dir


def save_figure(fig, run_dir, stem):
    """Save a figure as <stem>.pdf and <stem>.png."""
    run_dir = pathlib.Path(run_dir)
    fig.savefig(run_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(run_dir / f"{stem}.png", dpi=200, bbox_inches="tight")
