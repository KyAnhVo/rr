"""Where stage 1 gets its RGB frames."""
from collections.abc import Iterator
from pathlib import Path
from typing import Protocol

import cv2
from cv2.typing import MatLike

from rr.sens import SensReader


class FrameSource(Protocol):
    path: Path
    fps: float

    def frames(self, step: int) -> Iterator[tuple[int, MatLike]]: ...


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
    return SensReader(path) if path.suffix == ".sens" else VideoSource(path)
