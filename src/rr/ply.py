"""Minimal binary PLY I/O for coloured, oriented point clouds."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

_TYPES = {
    "float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
    "uchar": "u1", "uint8": "u1", "char": "i1", "short": "<i2", "ushort": "<u2",
    "int": "<i4", "int32": "<i4", "uint": "<u4",
}


@dataclass
class PointCloud:
    points: NDArray[np.float64]   # (N, 3)
    normals: NDArray[np.float64]  # (N, 3)
    colors: NDArray[np.uint8]     # (N, 3)

    def __len__(self) -> int:
        return len(self.points)

    def __getitem__(self, index: NDArray) -> "PointCloud":
        return PointCloud(self.points[index], self.normals[index], self.colors[index])


def concatenate(clouds: list[PointCloud]) -> PointCloud:
    return PointCloud(
        np.concatenate([c.points for c in clouds]),
        np.concatenate([c.normals for c in clouds]),
        np.concatenate([c.colors for c in clouds]),
    )


def read_ply(path: Path) -> PointCloud:
    with path.open("rb") as f:
        count, fields, fmt, in_vertex = 0, [], "", False
        if f.readline().strip() != b"ply":
            raise ValueError(f"{path} is not a PLY file")
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: truncated header")
            words = line.decode().split()
            if words[0] == "end_header":
                break
            if words[0] == "format":
                fmt = words[1]
            elif words[0] == "element":
                in_vertex = words[1] == "vertex"
                if in_vertex:
                    count = int(words[2])
            elif words[0] == "property" and in_vertex:
                fields.append((words[2], _TYPES[words[1]]))
        if fmt != "binary_little_endian":
            raise ValueError(f"{path}: unsupported PLY format {fmt}")
        data = np.fromfile(f, dtype=np.dtype(fields), count=count)
    stack = lambda names, t: np.column_stack([data[n] for n in names]).astype(t)
    return PointCloud(
        stack("xyz", np.float64), stack(("nx", "ny", "nz"), np.float64),
        stack(("red", "green", "blue"), np.uint8),
    )


def write_ply(path: Path, cloud: PointCloud) -> None:
    fields = [(n, "<f4") for n in ("x", "y", "z", "nx", "ny", "nz")]
    fields += [(n, "u1") for n in ("red", "green", "blue")]
    data = np.empty(len(cloud), dtype=fields)
    for i, n in enumerate(("x", "y", "z")):
        data[n] = cloud.points[:, i]
        data["n" + n] = cloud.normals[:, i]
    for i, n in enumerate(("red", "green", "blue")):
        data[n] = cloud.colors[:, i]
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {len(cloud)}"]
    header += [f"property {'float' if t == '<f4' else 'uchar'} {n}" for n, t in fields]
    with path.open("wb") as f:
        f.write(("\n".join(header) + "\nend_header\n").encode())
        data.tofile(f)
