"""Stage 1: keyframe selection.

A frame becomes a keyframe when it is sharp relative to its neighbours and has
enough *rotation-compensated* parallax w.r.t. the previous keyframe. We avoid a
homography/fundamental score ratio here: a homography explains pure rotation, but
also a translating camera looking at a single wall, which we want to keep.
"""
import json
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from cv2.typing import MatLike
from numpy.typing import NDArray
from scipy.ndimage import median_filter

Reason = Literal["first", "parallax", "overlap", "bridge", "break"]


@dataclass
class KeyframeConfig:
    process_fps: float = 10.0        # frames examined per second of video
    track_size: int = 1280           # long side used for SIFT tracking
    save_size: int = 1920            # long side of the saved keyframes
    blur_ratio: float = 0.6          # sharpness floor, relative to the local median
    blur_window_s: float = 1.0       # half width of that local window
    ratio_test: float = 0.75         # Lowe's ratio test
    max_features: int = 3000
    min_matches: int = 40            # fewer epipolar inliers than this = tracking lost
    min_parallax_deg: float = 1.5    # median parallax after removing rotation
    min_overlap: float = 0.35        # inliers relative to the keyframe's best match count
    focal_factor: float = 0.8        # focal guess = factor * long side (phones ~0.7-0.8)
    ransac_px: float = 1.5


@dataclass
class FrameData:
    index: int
    image: MatLike
    sharpness: float
    keypoints: Sequence[cv2.KeyPoint]
    descriptors: MatLike | None


@dataclass
class PairGeometry:
    inliers: int
    parallax_deg: float


@dataclass
class Keyframe:
    name: str
    frame_index: int
    reason: Reason
    sharpness: float
    inliers: int
    parallax_deg: float


def _resize_long_side(image: MatLike, size: int) -> MatLike:
    scale = size / max(image.shape[:2])
    if scale >= 1:
        return image
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def sharpness(image: MatLike) -> float:
    """Variance of the Laplacian at a fixed resolution."""
    gray = cv2.cvtColor(_resize_long_side(image, 640), cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def sharp_mask(scores: NDArray[np.float64], half_window: int, ratio: float) -> NDArray[np.bool_]:
    """Compare against a local median: plain walls score low even in perfect focus."""
    if len(scores) == 0:
        return np.zeros(0, dtype=bool)
    return scores >= ratio * median_filter(scores, size=2 * half_window + 1, mode="nearest")


def pair_geometry(
    pts_1: NDArray[np.float64], pts_2: NDArray[np.float64], K: NDArray[np.float64], ransac_px: float
) -> PairGeometry:
    """Epipolar inliers, and the parallax that remains once rotation is undone."""
    if len(pts_1) < 8:
        return PairGeometry(0, 0.0)
    E, mask = cv2.findEssentialMat(pts_1, pts_2, K, method=cv2.USAC_MAGSAC, prob=0.999, threshold=ransac_px)
    if E is None or E.shape != (3, 3) or mask is None:
        return PairGeometry(0, 0.0)
    keep = mask.ravel().astype(bool)
    if keep.sum() < 8:
        return PairGeometry(int(keep.sum()), 0.0)

    K_inv = np.linalg.inv(K)

    def rays(pts: NDArray[np.float64]) -> NDArray[np.float64]:
        r = np.column_stack([pts, np.ones(len(pts))]) @ K_inv.T
        return r / np.linalg.norm(r, axis=1, keepdims=True)

    r1, r2 = rays(pts_1[keep]), rays(pts_2[keep])
    # Of the two rotations in E, take the one leaving the least parallax. Unlike a
    # cheirality vote, this stays well defined when translation is ~0.
    parallax = np.inf
    for R in cv2.decomposeEssentialMat(E)[:2]:
        cos = np.clip(np.sum((r1 @ R.T) * r2, axis=1), -1, 1)
        parallax = min(parallax, float(np.degrees(np.median(np.arccos(cos)))))
    return PairGeometry(int(keep.sum()), parallax)


class KeyframeSelector:
    def __init__(self, config: KeyframeConfig | None = None) -> None:
        self.config = config or KeyframeConfig()
        self._sift = cv2.SIFT.create(nfeatures=self.config.max_features)
        self._matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    def select(self, video_path: Path, output_dir: Path) -> list[Keyframe]:
        """Write keyframes to output_dir/images and metadata to output_dir/keyframes.json."""
        cfg = self.config
        image_dir = output_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        if any(image_dir.iterdir()):
            raise FileExistsError(f"{image_dir} is not empty; use a fresh output directory")

        step = self._step(video_path)
        # Two passes: the blur test needs the sharpness of *future* frames too.
        scores = {i: sharpness(f) for i, f in self._frames(video_path, step)}
        if not scores:
            raise ValueError(f"no frames could be read from {video_path}")
        half_window = max(1, round(cfg.blur_window_s * cfg.process_fps))
        mask = sharp_mask(np.array(list(scores.values())), half_window, cfg.blur_ratio)
        sharp = {i for i, ok in zip(scores, mask) if ok}

        keyframes: list[Keyframe] = []
        for frame, reason, geom in self._choose(video_path, step, sharp, scores):
            name = f"{len(keyframes):05d}.jpg"
            ok = cv2.imwrite(
                str(image_dir / name),
                _resize_long_side(frame.image, cfg.save_size),
                [cv2.IMWRITE_JPEG_QUALITY, 97],
            )
            if not ok:
                raise OSError(f"could not write {image_dir / name}")
            keyframes.append(Keyframe(name, frame.index, reason, frame.sharpness, geom.inliers, geom.parallax_deg))

        (output_dir / "keyframes.json").write_text(json.dumps({
            "video": str(video_path),
            "config": asdict(cfg),
            "frames_examined": len(scores),
            "frames_blurry": len(scores) - len(sharp),
            "keyframes": [asdict(k) for k in keyframes],
        }, indent=2))
        return keyframes

    def _choose(
        self, video_path: Path, step: int, sharp: set[int], scores: dict[int, float]
    ) -> Iterator[tuple[FrameData, Reason, PairGeometry]]:
        cfg = self.config
        reference: FrameData | None = None
        candidate: tuple[FrameData, PairGeometry] | None = None  # last frame still well tracked
        best_inliers = 0

        for index, image in self._frames(video_path, step):
            if index not in sharp:
                continue
            frame = self._extract(index, image, scores[index])
            if reference is None:
                if len(frame.keypoints) >= cfg.min_matches:
                    reference = frame
                    yield frame, "first", PairGeometry(0, 0.0)
                continue

            geom = self._compare(reference, frame)
            if geom.inliers < cfg.min_matches and candidate is not None:
                # Tracking was lost: step back to the last good frame, then retry from there.
                reference, candidate_geom = candidate
                yield reference, "bridge", candidate_geom
                candidate, best_inliers = None, 0
                geom = self._compare(reference, frame)

            if geom.inliers < cfg.min_matches:
                if len(frame.keypoints) < cfg.min_matches:
                    continue  # nothing to see here (e.g. blank wall); keep the reference
                # Never hold on to a reference that has left the view.
                reason: Reason | None = "break"
            elif geom.parallax_deg >= cfg.min_parallax_deg:
                reason = "parallax"
            elif geom.inliers < cfg.min_overlap * best_inliers:
                # Mostly rotation: no baseline, but needed to keep the chain connected.
                reason = "overlap"
            else:
                reason = None

            if reason is None:
                candidate, best_inliers = (frame, geom), max(best_inliers, geom.inliers)
            else:
                yield frame, reason, geom
                reference, candidate, best_inliers = frame, None, 0

    def _step(self, video_path: Path) -> int:
        video = cv2.VideoCapture(str(video_path))
        try:
            if not video.isOpened():
                raise FileNotFoundError(f"cannot open video {video_path}")
            fps = video.get(cv2.CAP_PROP_FPS)
        finally:
            video.release()
        return max(1, round(fps / self.config.process_fps)) if fps > 0 else 1

    @staticmethod
    def _frames(video_path: Path, step: int) -> Iterator[tuple[int, MatLike]]:
        video = cv2.VideoCapture(str(video_path))
        try:
            index = 0
            while video.grab():
                if index % step == 0:
                    ok, image = video.retrieve()
                    if ok:
                        yield index, image
                index += 1
        finally:
            video.release()

    def _extract(self, index: int, image: MatLike, score: float) -> FrameData:
        gray = cv2.cvtColor(_resize_long_side(image, self.config.track_size), cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self._sift.detectAndCompute(gray, None)
        return FrameData(index, image, score, keypoints, descriptors)

    def _compare(self, a: FrameData, b: FrameData) -> PairGeometry:
        if a.descriptors is None or b.descriptors is None or len(b.keypoints) < 2:
            return PairGeometry(0, 0.0)
        matches = [
            p[0] for p in self._matcher.knnMatch(a.descriptors, b.descriptors, k=2)
            if len(p) == 2 and p[0].distance < self.config.ratio_test * p[1].distance
        ]
        pts_1 = np.array([a.keypoints[m.queryIdx].pt for m in matches], dtype=np.float64)
        pts_2 = np.array([b.keypoints[m.trainIdx].pt for m in matches], dtype=np.float64)
        h, w = _resize_long_side(a.image, self.config.track_size).shape[:2]
        f = self.config.focal_factor * max(h, w)
        K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float64)
        return pair_geometry(pts_1, pts_2, K, self.config.ransac_px)
