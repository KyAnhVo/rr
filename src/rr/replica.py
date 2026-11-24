"""Reader for the Replica RGB-D sequences used by iMAP / NICE-SLAM."""
import json
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
from cv2.typing import MatLike
from numpy.typing import NDArray


class ReplicaReader:
    """Also a stage 1 frame source; only RGB is exposed through `frames`."""
    fps = 30.0  # nominal: the renders carry no timestamps

    def __init__(self, path: Path) -> None:
        self.path = path
        camera = json.loads((path.parent / "cam_params.json").read_text())["camera"]
        self.color_size = self.depth_size = (int(camera["w"]), int(camera["h"]))
        self._depth_scale = float(camera["scale"])
        K = np.eye(4)
        K[0, 0], K[1, 1], K[0, 2], K[1, 2] = (float(camera[k]) for k in ("fx", "fy", "cx", "cy"))
        self.color_intrinsics = self.depth_intrinsics = K  # the renders are already registered
        self._poses = np.loadtxt(path / "traj.txt", ndmin=2).reshape(-1, 4, 4)
        self._colors = sorted((path / "results").glob("frame*.jpg"))
        self._depths = sorted((path / "results").glob("depth*.png"))
        if not len(self._poses) == len(self._colors) == len(self._depths) > 0:
            raise ValueError(
                f"{path}: incomplete scene ({len(self._poses)} poses, {len(self._colors)} images, "
                f"{len(self._depths)} depth maps); re-run scripts/download_replica.py"
            )

    def __len__(self) -> int:
        return len(self._poses)

    def color_pose(self, index: int) -> NDArray[np.float64] | None:
        """Camera-to-world, OpenCV axes."""
        pose = self._poses[index]
        return pose if np.isfinite(pose).all() else None

    depth_pose = color_pose

    def color(self, index: int) -> MatLike:
        image = cv2.imread(str(self._colors[index]), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"cannot decode {self._colors[index]}")
        return image

    def depth(self, index: int) -> NDArray[np.float32]:
        """Depth in meters; 0 means no measurement."""
        depth = cv2.imread(str(self._depths[index]), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise ValueError(f"cannot decode {self._depths[index]}")
        return depth.astype(np.float32) / self._depth_scale

    def frames(self, step: int) -> Iterator[tuple[int, MatLike]]:
        for i in range(0, len(self), step):
            yield i, self.color(i)
