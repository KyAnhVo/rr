"""Headless pictures: outside views, video frame vs. reconstruction, segmentation, overview."""
import json
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
from cv2.typing import MatLike
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from rr.evaluation import Similarity
from rr.ply import PointCloud
from rr.sources import FrameSource
from rr.taxonomy import COLORS, UNKNOWN_COLOR, binary_colors, class_colors

# Extra image columns for frame_comparisons: (image name, cell size) -> [(title, BGR image)].
ExtraColumns = Callable[[str, tuple[int, int]], list[tuple[str, MatLike]]]
OUTSIDE = [("top", 0, 89.0)] + [(f"corner_{a}", a, 35.0) for a in (45, 135, 225, 315)]

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


class OutsideCameras:
    """Cameras around a z-up scene, framed on `reference`, shared by every cloud drawn."""

    def __init__(self, reference: NDArray, size: tuple[int, int]) -> None:
        self.lo, self.hi = np.percentile(reference, [1, 99], axis=0)
        self.size = size
        self.K = np.array([[size[1] * 1.2, 0, size[0] / 2], [0, size[1] * 1.2, size[1] / 2], [0, 0, 1]])
        center, extent = (self.lo + self.hi) / 2, float(np.linalg.norm(self.hi - self.lo))
        self.poses = {}
        for name, azimuth, elevation in OUTSIDE:
            a, e = np.radians(azimuth), np.radians(elevation)
            eye = center + 1.25 * extent * np.array([np.cos(a) * np.cos(e), np.sin(a) * np.cos(e), np.sin(e)])
            self.poses[name] = _look_at(eye, center, np.array([0, 0, 1.0]))

    def render(self, view: str, cloud: PointCloud, radius: int) -> MatLike:
        margin = 0.15 * (self.hi - self.lo)
        inside = np.all((cloud.points > self.lo - margin) & (cloud.points < self.hi + margin), axis=1)
        return render(cloud[inside], self.K, self.poses[view], self.size, radius, bool(np.any(cloud.normals)))


def outside_views(clouds: dict[str, tuple[PointCloud, int]], output: Path, size: tuple[int, int] = (720, 540)) -> None:
    """Top view plus four corners, one row per view and one column per cloud.

    Clouds are z-up, and all are drawn from the same cameras, framed on the first one.
    """
    cameras = OutsideCameras(next(iter(clouds.values()))[0].points, size)
    rows = []
    for name in cameras.poses:
        cells = []
        for label, (cloud, radius) in clouds.items():
            image = cameras.render(name, cloud, radius)
            _save(output / f"{name}_{_slug(label)}.png", image)
            cells.append(_label(image, f"{label}: {name}"))
        rows.append(np.hstack(cells))
        _save(output / f"{name}.png", rows[-1])
    _save(output / "sheet.png", np.vstack(rows))


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in text.lower()).strip("_")


def _photo(model, image, image_dir: Path, width: int) -> tuple[MatLike, NDArray, NDArray, tuple[int, int]]:
    """Keyframe undistorted to match pinhole renders, with K, pose and size at `width`."""
    photo = cv2.imread(str(image_dir / image.name))
    if photo is None:
        raise FileNotFoundError(image_dir / image.name)
    camera = model.cameras[image.camera_id]
    K = np.array(camera.calibration_matrix())
    if camera.model.name == "SIMPLE_RADIAL":
        photo = cv2.undistort(photo, K.copy(), np.array([camera.params[3], 0, 0, 0], dtype=np.float64))
    scale = width / camera.width
    size = (width, round(camera.height * scale))
    K[:2] *= scale
    return cv2.resize(photo, size), K, np.array(image.cam_from_world().matrix()), size


def _spread(model, count: int) -> list:
    images = sorted(model.images.values(), key=lambda i: i.name)
    return [images[k] for k in np.unique(np.linspace(0, len(images) - 1, min(count, len(images))).round().astype(int))]


def frame_comparisons(
    model, image_dir: Path, clouds: dict[str, tuple[PointCloud, int]], output: Path, count: int = 6,
    extra: ExtraColumns | None = None,
) -> None:
    """Keyframe next to each cloud seen from the same estimated camera (clouds in model frame)."""
    rows = []
    for image in _spread(model, count):
        photo, K, pose, size = _photo(model, image, image_dir, 640)
        cells = [_label(photo, f"video {image.name}")]
        cells += [_label(cv2.resize(e, size, interpolation=cv2.INTER_NEAREST), n) for n, e in (extra(image.name, size) if extra else [])]
        cells += [_label(render(c, K, pose, size, r), n) for n, (c, r) in clouds.items()]
        row = np.hstack(cells)
        _save(output / f"{Path(image.name).stem}.png", row)
        rows.append(row)
    _save(output / "sheet.png", np.vstack(rows))


def recolored(cloud: PointCloud, colors: NDArray[np.uint8]) -> PointCloud:
    return PointCloud(cloud.points, cloud.normals, colors)


def class_cloud(cloud: PointCloud, classes: NDArray) -> PointCloud:
    return recolored(cloud, class_colors(classes))


def binary_cloud(cloud: PointCloud, classes: NDArray) -> PointCloud:
    return recolored(cloud, binary_colors(classes))


def ade_palette() -> NDArray[np.uint8]:
    """BGR colour per ADE20K id; 255 (unlabelled) is grey."""
    palette = np.random.default_rng(3).integers(40, 256, (256, 3)).astype(np.uint8)
    palette[255] = UNKNOWN_COLOR
    return palette


def coarse_image(coarse: NDArray[np.int8]) -> MatLike:
    """BGR picture of a coarse class map."""
    image = np.empty((*coarse.shape, 3), np.uint8)
    image[:] = UNKNOWN_COLOR[::-1]
    known = coarse >= 0
    image[known] = COLORS[coarse[known]][:, ::-1]
    return image


def structure_overlay(photo: MatLike, coarse: NDArray[np.int8]) -> MatLike:
    """Photo tinted blue on structure and orange on objects."""
    tint = np.ascontiguousarray(binary_colors(coarse.ravel()).reshape(*coarse.shape, 3)[..., ::-1])
    return cv2.addWeighted(photo, 0.5, tint, 0.5, 0)


def segmentation_sheet(image_dir: Path, seg_dir: Path, output: Path, count: int = 6) -> None:
    """Per keyframe: photo | ADE20K classes | coarse classes | structure/object overlay."""
    from rr.phase4 import read_labels, read_lookup

    lookup, palette = read_lookup(seg_dir), ade_palette()
    id2label = json.loads((seg_dir / "classes.json").read_text())["id2label"]
    names = sorted(p.name for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    rows = []
    for name in [names[i] for i in np.unique(np.linspace(0, len(names) - 1, min(count, len(names))).round().astype(int))]:
        photo, ade = cv2.imread(str(image_dir / name)), read_labels(seg_dir, name)
        if photo is None or ade is None:
            continue
        coarse = lookup[ade]
        classes = palette[ade]
        # Name the three largest ADE20K classes in the picture.
        ids, counts = np.unique(ade[ade != 255], return_counts=True)
        top = ", ".join(id2label.get(str(i), "?").split(",")[0] for i in ids[np.argsort(-counts)][:3])
        cells = [_label(photo, f"keyframe {name}"), _label(classes, f"p4 ADE20K: {top}"),
                 _label(coarse_image(coarse), "p4 coarse classes"), _label(structure_overlay(photo, coarse), "p4 structure / object")]
        width = 480
        cells = [cv2.resize(c, (width, round(c.shape[0] * width / c.shape[1])), interpolation=cv2.INTER_NEAREST) for c in cells]
        row = np.hstack(cells)
        _save(output / f"{Path(name).stem}.png", row)
        rows.append(row)
    if rows:
        _save(output / "sheet.png", np.vstack(rows))


def in_world(cam_from_model: NDArray, world: Similarity) -> NDArray:
    """A model-frame camera as a camera on world points (same image, depth scaled by world.scale)."""
    R = cam_from_model[:, :3] @ world.R.T
    return np.column_stack([R, world.scale * cam_from_model[:, 3] - R @ world.t])


def _blank(size: tuple[int, int], text: str) -> MatLike:
    image = np.full((size[1], size[0], 3), 235, np.uint8)
    cv2.putText(image, text, (12, size[1] // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (90, 90, 90), 2, cv2.LINE_AA)
    return image


def overview(
    model, image_dir: Path, world: Similarity, clouds: list[tuple[str, PointCloud, int]], output: Path,
    labels_2d: Callable[[str], MatLike | None] | None = None, cell: int = 420, inside_count: int = 3,
) -> None:
    """All phases on one sheet. Columns: p1 video, [p4 2D labels], then each cloud (world frame).
    Rows: top and two corner views from outside, then a few keyframe cameras from inside."""
    outside = OutsideCameras(clouds[0][1].points, (cell, round(cell * 0.75)))
    titles = ["p1 video"] + (["p4 2D labels"] if labels_2d else []) + [t for t, _, _ in clouds]
    rows = []
    for view in ("top", "corner_45", "corner_225"):
        cells = [_blank(outside.size, f"outside: {view}")] + ([_blank(outside.size, "")] if labels_2d else [])
        cells += [outside.render(view, c, r) for _, c, r in clouds]
        rows.append(cells)
    for image in _spread(model, inside_count + 2)[1:-1]:
        photo, K, pose, size = _photo(model, image, image_dir, cell)
        cells = [photo]
        if labels_2d:
            labels = labels_2d(image.name)
            cells.append(_blank(size, "no labels") if labels is None else cv2.resize(labels, size, interpolation=cv2.INTER_NEAREST))
        cells += [render(c, K, in_world(pose, world), size, r) for _, c, r in clouds]
        rows.append(cells)
    sheet = [np.hstack([_label(c, f"{t}") for c, t in zip(cells, titles)]) for cells in rows]
    _save(output, np.vstack(sheet))
