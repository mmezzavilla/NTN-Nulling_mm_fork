"""Run directories and figure saving for experiments (results/<date>_<name>/)."""
import datetime
import json
import pathlib
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _git(repo, *args):
    return subprocess.run(["git", *args], capture_output=True, text=True,
                          check=True, cwd=repo).stdout.strip()


def make_run_dir(name, params, root="results", date=None, repo=REPO_ROOT):
    """Create <root>/<date>_<name>/ and write params.json with the git state.

    git_dirty reflects uncommitted changes to tracked files only, so untracked
    outputs such as results/ do not mark every run dirty.
    """
    date = date or datetime.date.today().isoformat()
    run_dir = pathlib.Path(root) / f"{date}_{name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    record = dict(params)
    record["git_commit"] = _git(repo, "rev-parse", "HEAD")
    record["git_dirty"] = bool(_git(repo, "status", "--porcelain", "--untracked-files=no"))
    (run_dir / "params.json").write_text(json.dumps(record, indent=2, default=str))
    return run_dir


def require_full_run(run_dir, allow_smoke=False):
    """Exit if run_dir was produced by a smoke run, unless explicitly allowed."""
    params = json.loads((pathlib.Path(run_dir) / "params.json").read_text())
    if params.get("smoke") and not allow_smoke:
        sys.exit(f"{run_dir} is a smoke run; pass --allow-smoke to use it anyway")


def save_figure(fig, run_dir, stem):
    """Save a figure as <stem>.pdf and <stem>.png."""
    run_dir = pathlib.Path(run_dir)
    fig.savefig(run_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(run_dir / f"{stem}.png", dpi=200, bbox_inches="tight")
