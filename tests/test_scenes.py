"""Every stage on every scene in room-data/, once per intrinsics mode.

Writes pictures and metrics.json to results/<time>/<scene>/<mode>/. Thresholds only catch
breakage; judge quality from the metrics. ROOM_DATA and RR_DENSE_SIZE override the defaults.
Stage 4 needs the .venv-seg environment (or RR_SEG_PYTHON); without it, stages 4-5 fail.
"""
import json
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
        if (out / "metrics.json").exists():
            metrics = json.loads((out / "metrics.json").read_text())
    write_summary(OUT)
    return scene, out, metrics, error


def completed(run, stage):
    _, _, m, error = run
    assert m.get("completed", 0) >= stage, f"stopped after stage {m.get('completed', 0)}: {error!r}"


def test_recording(run):
    r = open_source(run[0].recording)
    assert len(r) and r.color(0).shape[:2][::-1] == r.color_size
    assert r.depth(0).max() < 20
    poses = [p for i in range(len(r)) if (p := r.color_pose(i)) is not None]
    assert poses and all(np.allclose(p[:3, :3] @ p[:3, :3].T, np.eye(3), atol=1e-2) for p in poses)


def test_keyframes(run):
    completed(run, 1)
    pictures = run[1] / "pictures"
    assert len(list((run[1] / "images").glob("*.jpg"))) >= 10
    for name in ("p1_original/sheet.jpg", "p1_keyframes.jpg", "p1_timeline.png"):
        assert (pictures / name).exists(), name


def test_sparse(run):
    completed(run, 2)
    _, out, m, _ = run
    for name in ("p2_sparse_inside/sheet.png", "p2_sparse_outside/sheet.png", "p2_trajectory.png"):
        assert (out / "pictures" / name).exists(), name
    assert m["registered_fraction"] > 0.5 and m["trajectory"]["ate_rmse_m"] < 0.5


def test_dense(run):
    completed(run, 3)
    _, out, m, _ = run
    assert m["clouds"]["fused"]["precision_5cm"] > 0.3
    for name in ("p3_dense_inside/sheet.png", "p3_dense_outside/sheet.png", "p3_scores.png", "all_in_one.jpg"):
        assert (out / "pictures" / name).exists(), name


def test_plane_files(run):
    completed(run, 3)
    out = run[1]
    z = np.load(out / "dense/planes.npz")
    fused, fill = read_ply(out / "dense/fused.ply"), read_ply(out / "dense/plane_fill.ply")
    assert len(z["point_plane"]) == len(fused) and len(z["fill_plane"]) == len(fill)
    for i, e in enumerate(z["equations"]):
        assert np.abs(fill.points[z["fill_plane"] == i] @ e[:3] + e[3]).max(initial=0) < 1e-3


def test_segmentation(run):
    completed(run, 4)
    _, out, m, _ = run
    labels = list((out / "segmentation/labels").glob("*.png"))
    assert len(labels) == len(list((out / "dense/images").iterdir())) == m["segmentation"]["images"]
    # A room: the network should find a good deal of wall, floor or ceiling.
    shares = m["segmentation"]["class_fractions"]
    assert shares["wall"] + shares["floor"] + shares["ceiling"] > 0.2
    for name in ("p4_segmentation/sheet.png", "p4_classes.png"):
        assert (out / "pictures" / name).exists(), name


def test_fusion(run):
    completed(run, 5)
    _, out, m, _ = run
    z = np.load(out / "semantic/labels.npz")
    dense = read_ply(out / "dense/dense.ply")
    assert len(z["classes"]) == len(dense) and int(z["num_fused"]) == len(read_ply(out / "dense/fused.ply"))
    assert (z["classes"] >= 0).all(), "every point gets a label"
    parts = sum(len(read_ply(out / "semantic" / f"{n}.ply")) for n in ("structure", "objects"))
    assert parts == len(dense)
    assert m["fusion"]["labeled_by_votes"] > 0.5 * len(dense)
    for name in ("p5_labels_inside/sheet.png", "p5_labels_outside/sheet.png", "p5_classes.png", "all_in_one.jpg"):
        assert (out / "pictures" / name).exists(), name
    if "labels" in m:  # only with Replica's semantic ground truth
        assert m["labels"]["final (+ plane prior, neighbours)"]["accuracy"] > 0.6
