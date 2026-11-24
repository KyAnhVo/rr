import struct
import subprocess
import zlib
from pathlib import Path

import cv2
import numpy as np
import pytest

from rr.evaluation import read_mesh, sample_mesh, score, umeyama, visible_mask
from rr.sens import SensReader

W, H, DW, DH = 64, 48, 32, 24


def write_sens(path, poses, colors, depths_mm, K_color, K_depth):
    cw, ch = colors[0].shape[1], colors[0].shape[0]
    dw, dh = depths_mm[0].shape[1], depths_mm[0].shape[0]
    with path.open("wb") as f:
        f.write(struct.pack("<IQ", 4, 4) + b"mock")
        for m in (K_color, np.eye(4), K_depth, np.eye(4)):
            f.write(np.asarray(m, "<f4").tobytes())
        f.write(struct.pack("<iiIIIIfQ", 2, 1, cw, ch, dw, dh, 1000.0, len(poses)))
        for i, (pose, color, depth) in enumerate(zip(poses, colors, depths_mm)):
            jpg = cv2.imencode(".jpg", color, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()
            z = zlib.compress(depth.astype("<u2").tobytes())
            f.write(np.asarray(pose, "<f4").tobytes() + struct.pack("<4Q", i, i, len(jpg), len(z)) + jpg + z)


def intrinsics(f, w, h):
    K = np.eye(4)
    K[0, 0] = K[1, 1] = f
    K[0, 2], K[1, 2] = w / 2, h / 2
    return K


@pytest.fixture
def sens(tmp_path):
    poses = [np.eye(4), np.full((4, 4), -np.inf)]
    colors = [np.full((H, W, 3), (200, 30, 60), np.uint8), np.zeros((H, W, 3), np.uint8)]
    depth = np.full((DH, DW), 2000)
    depth[0, 1] = 0
    write_sens(tmp_path / "s.sens", poses, colors, [depth, depth], intrinsics(60, W, H), intrinsics(30, DW, DH))
    return tmp_path / "s.sens"


def test_sens_reader(sens):
    r = SensReader(sens)
    assert len(r) == 2 and r.color_size == (W, H) and r.sensor_name == "mock"
    assert r.color_intrinsics[0, 0] == 60
    assert r.pose(1) is None and np.array_equal(r.pose(0), np.eye(4))
    assert np.abs(r.color(0).astype(int) - (200, 30, 60)).max() < 6  # BGR order kept
    d = r.depth(0)
    assert d.shape == (DH, DW) and d[0, 0] == 2 and d[0, 1] == 0
    assert [i for i, _ in r.frames(2)] == [0]


def test_truncated_sens_is_rejected(sens):
    sens.write_bytes(sens.read_bytes()[:-5])
    with pytest.raises(ValueError, match="truncated"):
        SensReader(sens)


def test_visibility_uses_depth(sens):
    points = np.array([[0, 0, 2.0], [0, 0, 1.0], [0, 0, 3.0], [9, 0, 2.0]])
    assert visible_mask(points, SensReader(sens), [0, 1]).tolist() == [True, False, False, False]


def test_umeyama_and_score():
    rng = np.random.default_rng(0)
    p = rng.random((100, 3))
    R = cv2.Rodrigues(np.array([0.3, -0.5, 1.0]))[0]
    sim = umeyama(p, 2.5 * p @ R.T + 3)
    assert sim.scale == pytest.approx(2.5) and np.allclose(sim.R, R)
    assert score(p, p, np.ones(100, bool))["chamfer_m"] == 0
    half = score(p[:50], p, np.ones(100, bool), 1e-9)
    assert half["precision_5cm"] == 1 and half["recall_5cm"] == 0.5
    with pytest.raises(ValueError):
        umeyama(np.outer(np.arange(5.0), [1, 2, 3]), np.outer(np.arange(5.0), [1, 2, 3]))


def write_mesh(path, vertices, faces, color=(120, 120, 120)):
    v = np.empty(len(vertices), [(k, "<f4") for k in "xyz"] + [(k, "u1") for k in ("red", "green", "blue", "alpha")])
    for i, k in enumerate("xyz"):
        v[k] = np.asarray(vertices)[:, i]
    v["red"], v["green"], v["blue"], v["alpha"] = *color, 255
    f = np.empty(len(faces), [("n", "u1"), ("i", "<i4", 3)])
    f["n"], f["i"] = 3, faces
    header = "ply\nformat binary_little_endian 1.0\n" + f"element vertex {len(v)}\n"
    header += "".join(f"property float {k}\n" for k in "xyz")
    header += "".join(f"property uchar {k}\n" for k in ("red", "green", "blue", "alpha"))
    header += f"element face {len(f)}\nproperty list uchar int vertex_indices\nend_header\n"
    path.write_bytes(header.encode() + v.tobytes() + f.tobytes())


def test_mesh(tmp_path):
    write_mesh(tmp_path / "m.ply", [[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]])
    assert len(read_mesh(tmp_path / "m.ply")[0]) == 3
    c = sample_mesh(tmp_path / "m.ply", 100)
    assert np.allclose(c.normals, [0, 0, 1]) and (c.points[:, :2].sum(1) <= 1 + 1e-9).all()


SCRIPT = Path(__file__).parents[1] / "scripts" / "download_scannet.sh"


def test_download_script(tmp_path):
    run = lambda *a: subprocess.run(
        [SCRIPT, *a], env={"PATH": "/usr/bin:/bin", "ROOM_DATA": str(tmp_path)}, capture_output=True, text=True
    )
    assert run("../evil").returncode == 2
    assert "scan-net.org" in run().stderr
    (tmp_path / "download-scannet.py").write_text(
        "import sys,os\na=sys.argv;o,s,t=a[2],a[4],a[6]\nd=os.path.join(o,'scans',s);os.makedirs(d,exist_ok=True)\n"
        "t.endswith('.txt') or open(os.path.join(d,s+t),'w').write('x')\n"
    )
    result = run("scene0000_00")
    assert result.returncode == 1 and "FAILED: scene0000_00.txt" in result.stderr
    assert (tmp_path / "scans/scene0000_00/scene0000_00.sens").exists()
    assert "have scene0000_00.sens" in run("scene0000_00").stdout
