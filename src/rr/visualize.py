"""Headless point-cloud pictures: outside views, and video frame vs. reconstruction."""
from pathlib import Path

import cv2
import numpy as np
from cv2.typing import MatLike
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from rr.evaluation import Similarity
from rr.ply import PointCloud
from rr.sources import FrameSource

BACKGROUND = 255


def render(
    cloud: PointCloud, K: NDArray, cam_from_world: NDArray, size: tuple[int, int], radius: int = 1, cull: bool = False
) -> MatLike:
    """Z-buffered square splats. `cull` hides surfaces seen from behind (needs normals)."""
    w, h = size
    image = np.full((h, w, 3), BACKGROUND, np.uint8)
    cam = cloud.points @ cam_from_world[:, :3].T + cam_from_world[:, 3]
    keep = cam[:, 2] > 1e-6
    if cull:
        keep &= np.sum((cloud.normals @ cam_from_world[:, :3].T) * cam, axis=1) < 0
    cam, color = cam[keep], cloud.colors[keep][:, ::-1]
    proj = cam @ K.T
    xy = np.floor(proj[:, :2] / proj[:, 2:])
    ok = np.all(np.abs(xy) < 1e6, axis=1)
    xy, color, z = xy[ok].astype(int), color[ok], cam[ok, 2]
    xs, ys, zs, src = [], [], [], []
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            x, y = xy[:, 0] + dx, xy[:, 1] + dy
            inside = np.flatnonzero((x >= 0) & (x < w) & (y >= 0) & (y < h))
            xs.append(x[inside])
            ys.append(y[inside])
            zs.append(z[inside])
            src.append(inside)
    x, y, depth, s = (np.concatenate(v) for v in (xs, ys, zs, src))
    order = np.argsort(-depth, kind="stable")  # nearest is written last
    image[y[order], x[order]] = color[s[order]]
    return image


def _look_at(eye: NDArray, target: NDArray, up: NDArray) -> NDArray:
    f = (target - eye) / np.linalg.norm(target - eye)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    R = np.stack([r, np.cross(f, r), f])
    return np.column_stack([R, -R @ eye])


def _label(image: MatLike, text: str) -> MatLike:
    out = cv2.copyMakeBorder(image, 30, 0, 0, 0, cv2.BORDER_CONSTANT, value=(40, 40, 40))
    cv2.putText(out, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _grid(images: list[MatLike], columns: int) -> MatLike:
    h, w = images[0].shape[:2]
    cells = [cv2.resize(i, (w, h)) for i in images]
    while len(cells) % columns:
        cells.append(np.full_like(cells[0], BACKGROUND))
    return np.vstack([np.hstack(cells[i:i + columns]) for i in range(0, len(cells), columns)])


def _save(path: Path, image: MatLike) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"could not write {path}")


def original_frames(source: FrameSource, output: Path, count: int = 12) -> None:
    """Evenly spaced frames of the input, individually and as one sheet."""
    indices = [i for i, _ in source.frames(max(1, round(source.fps)))]
    wanted = {indices[j] for j in np.linspace(0, len(indices) - 1, min(count, len(indices))).round().astype(int)}
    cells = []
    for i, frame in source.frames(max(1, round(source.fps))):
        if i in wanted:
            _save(output / f"frame_{i:06d}.jpg", frame)
            cells.append(_label(cv2.resize(frame, (480, round(480 * frame.shape[0] / frame.shape[1]))), f"frame {i}"))
    _save(output / "sheet.jpg", _grid(cells, 4))


def keyframe_sheet(image_dir: Path, names: list[str], output: Path, count: int = 24) -> None:
    pick = [names[i] for i in np.unique(np.linspace(0, len(names) - 1, min(count, len(names))).round().astype(int))]
    cells = []
    for name in pick:
        image = cv2.imread(str(image_dir / name))
        assert image is not None
        cells.append(_label(cv2.resize(image, (320, round(320 * image.shape[0] / image.shape[1]))), name))
    _save(output, _grid(cells, 6))


def gravity_frame(model) -> Similarity:
    """z-up frame from the average camera 'up', for when there is no ground truth."""
    up = -np.mean([np.array(i.cam_from_world().matrix())[1, :3] for i in model.images.values()], axis=0)
    up /= np.linalg.norm(up)
    x = np.cross(up, np.eye(3)[np.argmin(np.abs(up))])
    x /= np.linalg.norm(x)
    return Similarity(1.0, np.stack([x, np.cross(up, x), up]), np.zeros(3))


def facing_cameras(cloud: PointCloud, model) -> PointCloud:
    """Sparse points have no normals: point them at the cameras, so near walls can be culled."""
    centers = np.array([i.projection_center() for i in model.images.values()])
    nearest = centers[cKDTree(centers).query(cloud.points)[1]]
    n = nearest - cloud.points
    return PointCloud(cloud.points, n / np.linalg.norm(n, axis=1, keepdims=True), cloud.colors)


def outside_views(clouds: dict[str, tuple[PointCloud, int]], output: Path, size: tuple[int, int] = (720, 540)) -> None:
    """Top view plus four corners, one row per view and one column per cloud.

    Clouds are z-up, and all are drawn from the same cameras, framed on the first one.
    """
    reference = next(iter(clouds.values()))[0].points
    lo, hi = np.percentile(reference, [1, 99], axis=0)
    center, extent = (lo + hi) / 2, float(np.linalg.norm(hi - lo))
    margin = 0.15 * (hi - lo)
    K = np.array([[size[1] * 1.2, 0, size[0] / 2], [0, size[1] * 1.2, size[1] / 2], [0, 0, 1]])
    rows = []
    for name, azimuth, elevation in [("top", 0, 89.0)] + [(f"corner_{a}", a, 35.0) for a in (45, 135, 225, 315)]:
        a, e = np.radians(azimuth), np.radians(elevation)
        eye = center + 1.25 * extent * np.array([np.cos(a) * np.cos(e), np.sin(a) * np.cos(e), np.sin(e)])
        cells = []
        for label, (cloud, radius) in clouds.items():
            inside = np.all((cloud.points > lo - margin) & (cloud.points < hi + margin), axis=1)
            pose = _look_at(eye, center, np.array([0, 0, 1.0]))
            image = render(cloud[inside], K, pose, size, radius, bool(np.any(cloud.normals)))
            _save(output / f"{name}_{label.replace(' ', '_')}.png", image)
            cells.append(_label(image, f"{label}: {name}"))
        rows.append(np.hstack(cells))
        _save(output / f"{name}.png", rows[-1])
    _save(output / "sheet.png", np.vstack(rows))


def frame_comparisons(
    model, image_dir: Path, clouds: dict[str, tuple[PointCloud, int]], output: Path, count: int = 6
) -> None:
    """Keyframe next to each cloud seen from the same estimated camera."""
    images = sorted(model.images.values(), key=lambda i: i.name)
    rows = []
    for k in np.unique(np.linspace(0, len(images) - 1, min(count, len(images))).round().astype(int)):
        image = images[k]
        photo = cv2.imread(str(image_dir / image.name))
        assert photo is not None
        camera = model.cameras[image.camera_id]
        K = np.array(camera.calibration_matrix())
        scale = 640 / camera.width
        size = (640, round(camera.height * scale))
        K[:2] *= scale
        # Undo lens distortion so that the photo matches the pinhole renders.
        if camera.model.name == "SIMPLE_RADIAL":
            full = np.array(camera.calibration_matrix())
            photo = cv2.undistort(photo, full, np.array([camera.params[3], 0, 0, 0], dtype=np.float64))
        pose = np.array(image.cam_from_world().matrix())
        cells = [_label(cv2.resize(photo, size), f"video {image.name}")]
        cells += [_label(render(c, K, pose, size, r), n) for n, (c, r) in clouds.items()]
        row = np.hstack(cells)
        _save(output / f"{Path(image.name).stem}.png", row)
        rows.append(row)
    _save(output / "sheet.png", np.vstack(rows))
