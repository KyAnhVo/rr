import cv2
import numpy as np
import pytest

from rr.colmap import read_depth_map
from rr.phase1 import pair_geometry, sharp_mask
from rr.phase3 import seen_through
from rr.planes import PlaneConfig, fill_plane, fit_planes, scene_size
from rr.ply import PointCloud, read_ply, write_ply

K = np.array([[1000.0, 0, 640], [0, 1000, 360], [0, 0, 1]])


def observe(points, rotation_deg=0.0, translation=(0.0, 0.0, 0.0)):
    rng = np.random.default_rng(0)
    R = cv2.Rodrigues(np.radians([0.0, rotation_deg, 0.0]))[0]
    views = []
    for p in (points, points @ R.T + np.array(translation)):
        q = p @ K.T
        views.append(q[:, :2] / q[:, 2:] + rng.normal(0, 0.4, (len(p), 2)))
    return views


def scatter(planar):
    rng = np.random.default_rng(1)
    z = np.full(400, 4.0) if planar else rng.uniform(2.5, 6, 400)
    return np.column_stack([rng.uniform(-1.2, 1.2, 400), rng.uniform(-0.8, 0.8, 400), z])


@pytest.mark.parametrize("planar", [False, True])
def test_pure_rotation_has_no_parallax(planar):
    g = pair_geometry(*observe(scatter(planar), rotation_deg=10), K, 1.5)
    assert g.inliers > 300 and g.parallax_deg < 1.0


@pytest.mark.parametrize("planar", [False, True])
def test_translation_has_parallax_even_on_a_wall(planar):
    g = pair_geometry(*observe(scatter(planar), 3, (0.4, 0, 0)), K, 1.5)
    assert g.inliers > 300 and g.parallax_deg > 1.5


def test_blur_is_judged_locally():
    scores = np.array([100.0] * 20 + [10.0] * 20)  # textured, then a plain wall
    scores[5] = 20
    mask = sharp_mask(scores, 5, 0.6)
    assert not mask[5] and mask.sum() == 39


def test_planes_and_fill_respect_outline():
    rng = np.random.default_rng(2)
    xy = rng.uniform(0, 4, (60000, 2))
    wall = xy[~np.all(np.abs(xy - 2) < 0.15, axis=1)]  # 0.3 x 0.3 hole
    a = np.column_stack([wall, rng.normal(0, 0.003, len(wall))])
    b = np.column_stack([xy[:, 0], rng.normal(0, 0.003, len(xy)), xy[:, 1]])
    cloud = PointCloud(
        np.concatenate([a, b]),
        np.concatenate([np.tile([0, 0, 1.0], (len(a), 1)), np.tile([0, 1.0, 0], (len(b), 1))]),
        np.full((len(a) + len(b), 3), 128, np.uint8),
    )
    planes = fit_planes(cloud, PlaneConfig())
    assert len(planes) == 2
    wall_plane = next(p for p in planes if abs(p.normal[2]) > 0.999)
    fill = fill_plane(cloud, wall_plane, scene_size(cloud.points))
    assert wall_plane.distance(fill.points).max() < 1e-9
    assert np.all(np.abs(fill.points[:, :2] - 2) < 0.2, axis=1).sum() > 50
    assert fill.points[:, :2].min() > -0.1 and fill.points[:, :2].max() < 4.1


def test_seen_through():
    depth = np.full((720, 1280), 5.0, np.float32)
    pose = np.eye(4)[:3]
    points = np.array([[0, 0, 3.0], [0, 0, 5.0], [0, 0, 7.0], [0, 0, -1.0]])
    assert seen_through(points, K, pose, depth, 0.03).tolist() == [True, False, False, False]


def test_file_formats(tmp_path):
    cloud = PointCloud(np.eye(3), np.eye(3)[::-1], np.arange(9, dtype=np.uint8).reshape(3, 3))
    write_ply(tmp_path / "a.ply", cloud)
    back = read_ply(tmp_path / "a.ply")
    assert np.array_equal(back.points, cloud.points) and np.array_equal(back.colors, cloud.colors)
    data = np.arange(6, dtype=np.float32)
    (tmp_path / "d.bin").write_bytes(b"3&2&1&" + data.tobytes())
    assert read_depth_map(tmp_path / "d.bin").tolist() == data.reshape(2, 3).tolist()
