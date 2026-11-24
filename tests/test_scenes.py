"""Every stage on every scene in room-data/, once per intrinsics mode.

Writes pictures and metrics.json to results/<time>/<scene>/<mode>/. Thresholds only catch
breakage; judge quality from the metrics. ROOM_DATA and RR_DENSE_SIZE override the defaults.
"""
import os
import time
import traceback
from pathlib import Path

import numpy as np
import pytest

from rr.benchmark import MODES, find_scenes, run_scene, write_summary
from rr.ply import read_ply
from rr.sources import open_source

pytestmark = pytest.mark.scenes
ROOT = Path(__file__).parents[1]
SCENES = find_scenes(Path(os.environ.get("ROOM_DATA", ROOT / "room-data")))
OUT = Path(os.environ.get("RR_RESULTS", ROOT / "results")) / time.strftime("%Y%m%d-%H%M%S")
CASES = [pytest.param((s, m), id=f"{s.name}-{m}") for s in SCENES for m in MODES] or [
    pytest.param(None, marks=pytest.mark.skip(reason="no scenes; run scripts/download_replica.py"))
]


@pytest.fixture(scope="module", params=CASES)
def run(request):
    scene, mode = request.param
    out = OUT / scene.name / mode
    error, metrics = None, {}
    try:
        metrics = run_scene(scene, mode, out, int(os.environ.get("RR_DENSE_SIZE", 1200)))
    except Exception as e:  # reported by each test, so earlier stages can still pass
        error = e
        out.mkdir(parents=True, exist_ok=True)
        (out / "FAILED.txt").write_text("".join(traceback.format_exception(e)))
    write_summary(OUT)
    return scene, out, metrics, error


def test_recording(run):
    r = open_source(run[0].recording)
    assert len(r) and r.color(0).shape[:2][::-1] == r.color_size
    assert r.depth(0).max() < 20
    poses = [p for i in range(len(r)) if (p := r.color_pose(i)) is not None]
    assert poses and all(np.allclose(p[:3, :3] @ p[:3, :3].T, np.eye(3), atol=1e-2) for p in poses)


def test_keyframes(run):
    assert len(list((run[1] / "images").glob("*.jpg"))) >= 10
    assert (run[1] / "pictures/original/sheet.jpg").exists() and (run[1] / "pictures/keyframes.jpg").exists()


def test_sparse(run):
    _, out, m, error = run
    assert (out / "pictures/inside_sparse/sheet.png").exists(), error
    if error is None:
        assert m["registered_fraction"] > 0.5 and m["trajectory"]["ate_rmse_m"] < 0.5


def test_dense(run):
    _, out, m, error = run
    assert error is None, error
    assert m["clouds"]["fused"]["precision_5cm"] > 0.3
    for name in ("inside", "outside"):
        assert (out / "pictures" / name / "sheet.png").exists()


def test_plane_files(run):
    _, out, _, error = run
    assert error is None, error
    z = np.load(out / "dense/planes.npz")
    fused, fill = read_ply(out / "dense/fused.ply"), read_ply(out / "dense/plane_fill.ply")
    assert len(z["point_plane"]) == len(fused) and len(z["fill_plane"]) == len(fill)
    for i, e in enumerate(z["equations"]):
        assert np.abs(fill.points[z["fill_plane"] == i] @ e[:3] + e[3]).max(initial=0) < 1e-3
