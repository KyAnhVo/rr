import cv2
import numpy as np
import pytest

from rr.evaluation import read_mesh, sample_mesh, score, umeyama


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

