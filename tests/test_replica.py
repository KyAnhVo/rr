import http.server
import importlib.util
import json
import threading
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from rr.benchmark import find_scenes
from rr.evaluation import read_mesh, visible_mask
from rr.replica import ReplicaReader


@pytest.fixture
def scene(tmp_path):
    (tmp_path / "room9/results").mkdir(parents=True)
    (tmp_path / "cam_params.json").write_text(json.dumps(
        {"camera": {"w": 64, "h": 48, "fx": 60, "fy": 60, "cx": 31.5, "cy": 23.5, "scale": 6553.5}}))
    pose = np.eye(4)
    pose[0, 3] = 0.5
    np.savetxt(tmp_path / "room9/traj.txt", [np.eye(4).ravel(), pose.ravel()])
    for i in range(2):
        cv2.imwrite(str(tmp_path / f"room9/results/frame{i:06d}.jpg"), np.full((48, 64, 3), (200, 30, 60), np.uint8))
        cv2.imwrite(str(tmp_path / f"room9/results/depth{i:06d}.png"), np.full((48, 64), 13107, np.uint16))
    (tmp_path / "room9_mesh.ply").touch()
    return tmp_path / "room9"


def test_reader(scene):
    r = ReplicaReader(scene)
    assert len(r) == 2 and r.color_size == (64, 48) and r.color_intrinsics[0, 2] == 31.5
    assert r.color_pose(1)[0, 3] == 0.5
    assert r.depth(0)[0, 0] == pytest.approx(2.0)
    assert visible_mask(np.array([[0, 0, 2.0], [0, 0, 1.0]]), r, [0]).tolist() == [True, False]
    assert [s.name for s in find_scenes(scene.parent)] == ["room9"]
    next((scene / "results").glob("*.png")).unlink()
    with pytest.raises(ValueError, match="incomplete"):
        ReplicaReader(scene)


def test_quad_mesh(tmp_path):
    v = np.zeros(4, [(k, "<f4") for k in "xyz"] + [(k, "u1") for k in ("red", "green", "blue")])
    v["x"], v["y"] = [0, 1, 1, 0], [0, 0, 1, 1]
    f = np.array([(4, [0, 1, 2, 3])], [("n", "u1"), ("i", "<i4", 4)])
    head = "ply\nformat binary_little_endian 1.0\nelement vertex 4\n"
    head += "".join(f"property float {k}\n" for k in "xyz")
    head += "".join(f"property uchar {k}\n" for k in ("red", "green", "blue"))
    head += "element face 1\nproperty list uchar int vertex_indices\nend_header\n"
    (tmp_path / "q.ply").write_bytes(head.encode() + v.tobytes() + f.tobytes())
    assert read_mesh(tmp_path / "q.ply")[1].tolist() == [[0, 1, 2], [0, 2, 3]]


def test_downloader_fetches_only_requested_scene(tmp_path):
    archive = tmp_path / "Replica.zip"
    with zipfile.ZipFile(archive, "w") as z:
        for name in ("cam_params.json", "room0_mesh.ply", "room1_mesh.ply", "room0/traj.txt", "room1/traj.txt"):
            z.writestr("Replica/" + name, name * 50)
    data = archive.read_bytes()

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_HEAD(self):
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()

        def do_GET(self):
            start, end = map(int, self.headers["Range"][6:].split("-"))
            body = data[start:end + 1]
            self.send_response(206)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args): ...

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    spec = importlib.util.spec_from_file_location(
        "dl", Path(__file__).parents[1] / "scripts" / "download_replica.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = tmp_path / "out"
    module.download(f"http://127.0.0.1:{server.server_port}/Replica.zip", ["room0"], out)
    server.shutdown()
    files = sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file())
    assert files == ["cam_params.json", "room0/traj.txt", "room0_mesh.ply"]
    assert (out / "room0/traj.txt").read_text() == "room0/traj.txt" * 50
