"""Walks the video once for sharpness, then again to pick and write keyframes."""
import json
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
from cv2.typing import MatLike
from numpy.typing import NDArray

from rr.phase1.config import FrameData, Keyframe, KeyframeConfig, PairGeometry, Reason
from rr.phase1.geometry import camera_matrix, pair_geometry
from rr.phase1.sharpness import resize_long_side, sharp_mask, sharpness
from rr.sources import FrameSource


class KeyframeSelector:
    def __init__(self, config: KeyframeConfig | None = None, intrinsics: NDArray[np.float64] | None = None) -> None:
        """`intrinsics` is the 3x3 matrix at the source's resolution, if known."""
        self.config = config or KeyframeConfig()
        self._intrinsics = intrinsics
        self._sift = cv2.SIFT.create(nfeatures=self.config.max_features)
        self._matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    def select(self, source: FrameSource, output_dir: Path) -> list[Keyframe]:
        """Write keyframes to output_dir/images and metadata to output_dir/keyframes.json."""
        cfg = self.config
        image_dir = output_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        if any(image_dir.iterdir()):
            raise FileExistsError(f"{image_dir} is not empty; use a fresh output directory")

        step = self._step(source)
        # Two passes: the blur test needs the sharpness of *future* frames too.
        scores = {i: sharpness(f) for i, f in source.frames(step)}
        if not scores:
            raise ValueError(f"no frames could be read from {source.path}")
        half_window = max(1, round(cfg.blur_window_s * cfg.process_fps))
        mask = sharp_mask(np.array(list(scores.values())), half_window, cfg.blur_ratio)
        sharp = {i for i, ok in zip(scores, mask) if ok}

        keyframes: list[Keyframe] = []
        for frame, reason, geom in self._choose(source, step, sharp, scores):
            name = f"{len(keyframes):05d}.jpg"
            ok = cv2.imwrite(
                str(image_dir / name),
                resize_long_side(frame.image, cfg.save_size),
                [cv2.IMWRITE_JPEG_QUALITY, 97],
            )
            if not ok:
                raise OSError(f"could not write {image_dir / name}")
            keyframes.append(Keyframe(name, frame.index, reason, frame.sharpness, geom.inliers, geom.parallax_deg))

        (output_dir / "keyframes.json").write_text(json.dumps({
            "video": str(source.path),
            "config": asdict(cfg),
            "frames_examined": len(scores),
            "frames_blurry": len(scores) - len(sharp),
            "keyframes": [asdict(k) for k in keyframes],
            # [frame index, sharpness, passed the blur test]
            "frame_sharpness": [[i, round(s, 2), int(i in sharp)] for i, s in scores.items()],
        }, indent=2))
        return keyframes

    def _choose(
        self, source: FrameSource, step: int, sharp: set[int], scores: dict[int, float]
    ) -> Iterator[tuple[FrameData, Reason, PairGeometry]]:
        cfg = self.config
        reference: FrameData | None = None
        candidate: tuple[FrameData, PairGeometry] | None = None  # last frame still well tracked
        best_inliers = 0

        for index, image in source.frames(step):
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

    def _step(self, source: FrameSource) -> int:
        return max(1, round(source.fps / self.config.process_fps)) if source.fps > 0 else 1

    def _extract(self, index: int, image: MatLike, score: float) -> FrameData:
        gray = cv2.cvtColor(resize_long_side(image, self.config.track_size), cv2.COLOR_BGR2GRAY)
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
        h, w = resize_long_side(a.image, self.config.track_size).shape[:2]
        full_h, full_w = a.image.shape[:2]
        K = camera_matrix(w, h, self.config.focal_factor, self._intrinsics, (full_w, full_h))
        return pair_geometry(pts_1, pts_2, K, self.config.ransac_px)
