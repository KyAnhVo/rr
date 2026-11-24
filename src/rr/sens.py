"""Reader for ScanNet .sens recordings. Frames are read from disk on demand."""
import struct
import zlib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import cv2
import numpy as np
from cv2.typing import MatLike
from numpy.typing import NDArray

_COLOR = {0: "raw", 1: "png", 2: "jpeg"}
_DEPTH = {0: "raw", 1: "zlib", 2: "occi"}
_FRAME = struct.Struct("<16f4Q")


def _read(f: BinaryIO, size: int) -> bytes:
    data = f.read(size)
    if len(data) != size:
        raise ValueError("truncated .sens file")
    return data


@dataclass
class _Entry:
    pose: NDArray[np.float64]
    timestamp_us: int
    offset: int
    color_size: int
    depth_size: int


class SensReader:
    """Also a stage 1 frame source; only RGB is exposed through `frames`."""
    fps = 30.0  # nominal; stored timestamps are not reliable across scans

    def __init__(self, path: Path) -> None:
        self.path = path
        file_size = path.stat().st_size
        with path.open("rb") as f:
            (version, name_length) = struct.unpack("<IQ", _read(f, 12))
            if version != 4:
                raise ValueError(f"{path}: unsupported .sens version {version}")
            if name_length > file_size:
                raise ValueError(f"{path}: corrupt header")
            self.sensor_name = _read(f, name_length).decode(errors="replace")
            m = np.frombuffer(_read(f, 256), "<f4").reshape(4, 4, 4).astype(np.float64)
            self.color_intrinsics, self.color_extrinsics = m[0], m[1]
            self.depth_intrinsics, self.depth_extrinsics = m[2], m[3]
            color, depth = struct.unpack("<ii", _read(f, 8))
            if color not in _COLOR or depth not in _DEPTH:
                raise ValueError(f"{path}: unknown compression {color}/{depth}")
            self.color_compression, self.depth_compression = _COLOR[color], _DEPTH[depth]
            w, h, dw, dh = struct.unpack("<4I", _read(f, 16))
            self.color_size, self.depth_size = (w, h), (dw, dh)
            self.depth_shift, count = struct.unpack("<fQ", _read(f, 12))
            self._entries: list[_Entry] = []
            for _ in range(count):
                values = _FRAME.unpack(_read(f, _FRAME.size))
                entry = _Entry(
                    np.array(values[:16]).reshape(4, 4), values[16], f.tell(), values[18], values[19]
                )
                end = entry.offset + entry.color_size + entry.depth_size
                if end > file_size:
                    raise ValueError(f"{path}: truncated .sens file")
                self._entries.append(entry)
                f.seek(end)

    def __len__(self) -> int:
        return len(self._entries)

    def pose(self, index: int) -> NDArray[np.float64] | None:
        """Camera-to-world, or None where ScanNet lost tracking."""
        pose = self._entries[index].pose
        return pose if np.isfinite(pose).all() else None

    def color(self, index: int) -> MatLike:
        """BGR image."""
        e = self._entries[index]
        with self.path.open("rb") as f:
            f.seek(e.offset)
            data = np.frombuffer(_read(f, e.color_size), np.uint8)
        if self.color_compression == "raw":
            return cv2.cvtColor(data.reshape(self.color_size[1], self.color_size[0], 3), cv2.COLOR_RGB2BGR)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"{self.path}: cannot decode color frame {index}")
        return image

    def depth(self, index: int) -> NDArray[np.float32]:
        """Depth in meters; 0 means no measurement."""
        if self.depth_compression == "occi":
            raise NotImplementedError("occi depth compression is not supported")
        e = self._entries[index]
        with self.path.open("rb") as f:
            f.seek(e.offset + e.color_size)
            data = _read(f, e.depth_size)
        if self.depth_compression == "zlib":
            data = zlib.decompress(data)
        depth = np.frombuffer(data, "<u2").reshape(self.depth_size[1], self.depth_size[0])
        return depth.astype(np.float32) / self.depth_shift

    def frames(self, step: int) -> Iterator[tuple[int, MatLike]]:
        for i in range(0, len(self), step):
            yield i, self.color(i)
