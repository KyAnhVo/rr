"""Stage 1 settings, and the records passed between its steps."""
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import cv2
from cv2.typing import MatLike

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
