"""Stage 1: keyframe selection.

A frame becomes a keyframe when it is sharp relative to its neighbours and has
enough *rotation-compensated* parallax w.r.t. the previous keyframe. We avoid a
homography/fundamental score ratio here: a homography explains pure rotation, but
also a translating camera looking at a single wall, which we want to keep.

    config     settings and the records passed between the steps
    sharpness  blur test
    geometry   epipolar inliers and parallax between two frames
    selector   reads the video, picks and writes keyframes
"""
from rr.phase1.config import FrameData, Keyframe, KeyframeConfig, PairGeometry, Reason
from rr.phase1.geometry import pair_geometry
from rr.phase1.selector import KeyframeSelector
from rr.phase1.sharpness import sharp_mask, sharpness

__all__ = [
    "FrameData", "Keyframe", "KeyframeConfig", "KeyframeSelector", "PairGeometry", "Reason",
    "pair_geometry", "sharp_mask", "sharpness",
]
