"""Where stage 1 gets its RGB frames."""
from collections.abc import Iterator
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
from cv2.typing import MatLike
from numpy.typing import NDArray

from rr.replica import ReplicaReader


class FrameSource(Protocol):
    path: Path
    fps: float

    def frames(self, step: int) -> Iterator[tuple[int, MatLike]]: ...


class Recording(FrameSource, Protocol):
    """A dataset sequence with ground truth. Poses are camera-to-world, None where unknown."""
    color_size: tuple[int, int]
    depth_size: tuple[int, int]
    color_intrinsics: NDArray[np.float64]
    depth_intrinsics: NDArray[np.float64]

    def __len__(self) -> int: ...
    def color(self, index: int) -> MatLike: ...
    def depth(self, index: int) -> NDArray[np.float32]: ...
    def color_pose(self, index: int) -> NDArray[np.float64] | None: ...
    def depth_pose(self, index: int) -> NDArray[np.float64] | None: ...


class VideoSource:
    def __init__(self, path: Path) -> None:
        self.path = path
        video = cv2.VideoCapture(str(path))
        try:
            if not video.isOpened():
                raise FileNotFoundError(f"cannot open video {path}")
            self.fps = video.get(cv2.CAP_PROP_FPS)
        finally:
            video.release()

    def frames(self, step: int) -> Iterator[tuple[int, MatLike]]:
        video = cv2.VideoCapture(str(self.path))
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


def open_source(path: Path) -> FrameSource:
    if path.is_dir():
        return ReplicaReader(path)
    return VideoSource(path)
