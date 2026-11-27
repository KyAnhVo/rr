"""Stages 4-5 and the label evaluation, without torch, a GPU or real data."""
import gzip
import http.server
import importlib.util
import io
import json
import tarfile
import threading
from pathlib import Path

import cv2
import numpy as np
import pytest

from rr.charts import class_shares, cloud_scores, keyframe_timeline, label_scores, trajectory
from rr.phase4 import read_labels, read_lookup, segment_images
from rr.phase5 import FROM_NEIGHBOURS, FROM_PLANE, FROM_VOTES, FusionConfig, View, fuse, visible
from rr.semantic_gt import label_scores as score_labels
from rr.semantic_gt import read_semantic_mesh, sample_labeled
from rr.taxonomy import OBJECT, UNKNOWN, ade_class, ade_lookup, is_structure, replica_class
from rr.visualize import segmentation_sheet

ADE = {0: "wall", 3: "floor, flooring", 5: "ceiling", 8: "windowpane, window", 14: "door", 19: "chair"}


def test_taxonomy():
    assert [ade_class(ADE[i]) for i in (0, 3, 5, 8, 14, 19)] == [0, 1, 2, 4, 3, OBJECT]
    table = ade_lookup(ADE)
    assert table[8] == 4 and table[19] == OBJECT and table[255] == UNKNOWN and table[1] == UNKNOWN
    assert replica_class("wall-plug") == 5 and replica_class("sofa") == OBJECT
    assert replica_class("undefined") == UNKNOWN and replica_class("anonymize_text") == UNKNOWN
    assert is_structure(np.array([0, 5, OBJECT, UNKNOWN])).tolist() == [True, True, False, False]


@pytest.fixture
def segmented(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    for i in range(3):
        cv2.imwrite(str(images / f"{i:05d}.jpg"), np.full((24, 32, 3), 80 * i, np.uint8))

    def fake(image):  # left half wall, right half chair
        labels = np.zeros(image.shape[:2], np.uint8)
        labels[:, image.shape[1] // 2:] = 19
        return labels

    result = segment_images(images, tmp_path / "seg", fake, ADE, "fake")
    return images, tmp_path / "seg", result


def test_segment_images(segmented):
    images, seg, result = segmented
    assert result.images == 3 and result.class_fractions["wall"] == pytest.approx(0.5)
    assert result.class_fractions["object"] == pytest.approx(0.5)
    labels = read_labels(seg, "00001.jpg")
    assert labels is not None and labels.shape == (24, 32) and labels[0, -1] == 19
    assert read_lookup(seg)[19] == OBJECT
    with pytest.raises(FileExistsError):
        segment_images(images, seg, lambda i: np.zeros(i.shape[:2], np.uint8), ADE, "fake")


def test_segmentation_sheet(segmented, tmp_path):
    images, seg, _ = segmented
    segmentation_sheet(images, seg, tmp_path / "p4")
    assert (tmp_path / "p4" / "sheet.png").exists()


def grid(lo, hi, step, z):
    xs = np.arange(lo, hi + 1e-9, step)
    x, y = np.meshgrid(xs, xs)
    return np.column_stack([x.ravel(), y.ravel(), np.full(x.size, z)])


def test_fusion_votes_occlusion_plane_and_neighbours():
    """A wall at z=2 with a small object at z=1 in front, seen by one camera at the origin."""
    wall, thing = grid(-1, 1, 0.02, 2.0), grid(-0.1, 0.1, 0.01, 1.0)
    stray = np.array([[0.0, 0.0, -1.0]])  # behind the camera: never seen, on no plane
    points = np.concatenate([wall, thing, stray])
    normals = np.tile([0, 0, -1.0], (len(points), 1))
    plane_of = np.concatenate([np.zeros(len(wall), int), np.full(len(thing) + 1, -1)])
    K = np.array([[50, 0, 32], [0, 50, 32], [0, 0, 1.0]])
    labels = np.zeros((64, 64), np.int8)
    labels[26:39, 26:39] = OBJECT  # where the object projects (pixels 27-37), plus a pixel of slack
    view = View("a", K, np.eye(4)[:3], labels)

    f = fuse(points, normals, [view], plane_of, FusionConfig())
    w, t = slice(0, len(wall)), slice(len(wall), len(wall) + len(thing))
    assert np.all(f.classes[t] == OBJECT) and np.all(f.source[t] == FROM_VOTES)
    assert np.all(f.classes[w] == 0), "wall points hidden behind the object must not take its label"
    hidden = f.source[w] == FROM_PLANE
    assert hidden.any() and np.all(f.votes_only[w][hidden] == UNKNOWN)
    assert f.source[-1] == FROM_NEIGHBOURS and f.votes_only[-1] == UNKNOWN
    assert np.all(f.single_view[t] == OBJECT)


def test_visibility_uses_depth_map():
    K = np.array([[50, 0, 32], [0, 50, 32], [0, 0, 1.0]])
    labels = np.zeros((64, 64), np.int8)
    points = np.array([[0, 0, 1.5], [0, 0, 2.0]])
    depth = np.full((32, 32), 2.0, np.float32)  # half resolution, measured wall at 2 m
    idx, _, _ = visible(points, View("a", K, np.eye(4)[:3], labels, depth), FusionConfig())
    assert idx.tolist() == [1], "a point in front of the measured surface is floating, not visible"
    idx, _, _ = visible(points[:1], View("a", K, np.eye(4)[:3], labels, np.zeros_like(depth)), FusionConfig())
    assert idx.tolist() == [0], "without a measurement, the z-buffer decides"


def write_semantic_mesh(folder: Path):
    """Two unit quads: a floor (object 1) and a sofa seat (object 2)."""
    folder.mkdir(parents=True)
    v = np.zeros(8, [(k, "<f4") for k in "xyz"] + [(k, "u1") for k in ("red", "green", "blue")])
    v["x"] = [0, 1, 1, 0, 0, 1, 1, 0]
    v["y"] = [0, 0, 1, 1, 0, 0, 1, 1]
    v["z"] = [0, 0, 0, 0, 0.5, 0.5, 0.5, 0.5]
    f = np.array([(4, [0, 1, 2, 3], 1), (4, [4, 5, 6, 7], 2)], [("n", "u1"), ("i", "<i4", 4), ("object_id", "<i4")])
    head = "ply\nformat binary_little_endian 1.0\nelement vertex 8\n"
    head += "".join(f"property float {k}\n" for k in "xyz")
    head += "".join(f"property uchar {k}\n" for k in ("red", "green", "blue"))
    head += "element face 2\nproperty list uchar int vertex_indices\nproperty int object_id\nend_header\n"
    (folder / "mesh_semantic.ply").write_bytes(head.encode() + v.tobytes() + f.tobytes())
    (folder / "info_semantic.json").write_text(json.dumps({"objects": [
        {"id": 1, "class_id": 40, "class_name": "floor"}, {"id": 2, "class_id": 76, "class_name": "sofa"}]}))
    return folder / "mesh_semantic.ply", folder / "info_semantic.json"


def test_semantic_mesh_and_scores(tmp_path):
    mesh, info = write_semantic_mesh(tmp_path / "room9_semantic")
    v, faces, ids = read_semantic_mesh(mesh)
    assert len(v) == 8 and faces.shape == (4, 3) and sorted(ids.tolist()) == [1, 1, 2, 2]
    points, normals, classes = sample_labeled(mesh, info, 2000)
    assert set(classes.tolist()) == {1, OBJECT}
    assert np.all(classes[points[:, 2] > 0.25] == OBJECT) and np.allclose(np.abs(normals[:, 2]), 1)
    truth = np.where(points[:, 2] > 0.25, OBJECT, 1).astype(np.int8)
    wrong = truth.copy()
    wrong[:100] = np.where(truth[:100] == OBJECT, 0, OBJECT)
    s = score_labels(points[:500], {"perfect": truth[:500], "wrong": wrong[:500]}, points, classes)
    assert s["perfect"]["accuracy"] == 1 and s["perfect"]["miou"] == 1 and s["perfect"]["points_scored"] == 500
    assert s["wrong"]["accuracy"] == pytest.approx(0.8)
    far = score_labels(points[:10] + 1.0, {"far": truth[:10]}, points, classes)
    assert far["far"]["points_scored"] == 0


def test_charts(tmp_path):
    kf = {"frames_examined": 4, "config": {"min_matches": 40}, "frame_sharpness": [[0, 50, 1], [3, 5, 0], [6, 60, 1], [9, 70, 1]],
          "keyframes": [{"frame_index": 0, "reason": "first", "inliers": 0, "parallax_deg": 0},
                        {"frame_index": 9, "reason": "parallax", "inliers": 300, "parallax_deg": 2.0}]}
    keyframe_timeline(kf, tmp_path / "p1.png")
    truth = np.random.default_rng(0).random((20, 3))
    trajectory(truth + 0.001, truth, tmp_path / "p2.png")
    cloud_scores({"sparse": {"precision_5cm": 0.9, "recall_5cm": 0.3, "fscore_5cm": 0.45}}, tmp_path / "p3.png")
    class_shares({"wall": 0.5, "object": 0.5}, tmp_path / "p4.png", "shares")
    label_scores({"final": {"accuracy": 0.9, "miou": 0.8}}, tmp_path / "p5.png")
    assert all((tmp_path / f"p{i}.png").stat().st_size > 1000 for i in range(1, 6))


def test_semantics_downloader_streams_only_wanted_files(tmp_path):
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as tar:
        for name in ("replica_v1/room_0/mesh.ply", "replica_v1/room_0/habitat/mesh_semantic.ply",
                     "replica_v1/room_0/habitat/info_semantic.json", "replica_v1/room_1/habitat/mesh_semantic.ply"):
            data = (name * 200).encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    blob = gzip.compress(raw.getvalue())
    parts = {"aa": blob[:len(blob) // 2], "ab": blob[len(blob) // 2:]}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = parts.get(self.path[-2:])
            self.send_response(200 if body is not None else 404)
            self.send_header("Content-Length", str(len(body or b"")))
            self.end_headers()
            self.wfile.write(body or b"")

        def log_message(self, *args): ...

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    spec = importlib.util.spec_from_file_location(
        "dls", Path(__file__).parents[1] / "scripts" / "download_replica_semantics.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = tmp_path / "out"
    module.download(f"http://127.0.0.1:{server.server_port}/replica.tar.gz.part", ["room0"], out)
    server.shutdown()
    files = sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file())
    assert files == ["room0_semantic/info_semantic.json", "room0_semantic/mesh_semantic.ply"]
    assert (out / "room0_semantic/mesh_semantic.ply").read_text().startswith("replica_v1/room_0/habitat/mesh_semantic.ply")
