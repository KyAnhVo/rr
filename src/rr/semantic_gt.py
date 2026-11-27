"""Ground-truth structure/object labels from Replica's semantic mesh, and label scoring.

The NICE-SLAM archive has no labels. They come from the original Replica release
(scripts/download_replica_semantics.py), saved as room-data/replica/<scene>_semantic/
{mesh_semantic.ply, info_semantic.json}.
"""
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from rr.taxonomy import CLASSES, OBJECT, UNKNOWN, replica_class

_TYPES = {"char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1", "short": "<i2", "int16": "<i2",
          "ushort": "<u2", "uint16": "<u2", "int": "<i4", "int32": "<i4", "uint": "<u4", "uint32": "<u4",
          "float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8"}


def semantic_files(scene_dir: Path) -> tuple[Path, Path] | None:
    folder = scene_dir.with_name(f"{scene_dir.name}_semantic")
    mesh, info = folder / "mesh_semantic.ply", folder / "info_semantic.json"
    return (mesh, info) if mesh.exists() and info.exists() else None


def read_semantic_mesh(path: Path) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    """Vertices, triangles and each triangle's object id, from a binary PLY whose faces have a
    vertex index list followed by an `object_id` property (quads are split in two)."""
    with path.open("rb") as f:
        if f.readline().strip() != b"ply":
            raise ValueError(f"{path} is not a PLY file")
        counts: dict[str, int] = {}
        props: dict[str, list[list[str]]] = {}
        order: list[str] = []
        fmt = current = ""
        while (words := f.readline().decode().split())[:1] != ["end_header"]:
            if not words:
                raise ValueError(f"{path}: bad PLY header")
            if words[0] == "format":
                fmt = words[1]
            elif words[0] == "element":
                current = words[1]
                counts[current] = int(words[2])
                props[current] = []
                order.append(current)
            elif words[0] == "property":
                props[current].append(words[1:])
        if fmt != "binary_little_endian" or order[:2] != ["vertex", "face"]:
            raise ValueError(f"{path}: unsupported layout")
        if any(p[0] == "list" for p in props["vertex"]):
            raise ValueError(f"{path}: list property on vertices")
        v = np.fromfile(f, np.dtype([(p[1], _TYPES[p[0]]) for p in props["vertex"]]), counts["vertex"])
        face = props["face"]
        if face[0][0] != "list" or [p[-1] for p in face[1:]] != ["object_id"]:
            raise ValueError(f"{path}: faces must be an index list then object_id")
        _, n_type, i_type, _ = face[0]
        start = f.tell()
        corners = int(np.fromfile(f, _TYPES[n_type], 1)[0])
        f.seek(start)
        dtype = np.dtype([("n", _TYPES[n_type]), ("i", _TYPES[i_type], corners), ("object_id", _TYPES[face[1][0]])])
        faces = np.fromfile(f, dtype, counts["face"])
    if corners not in (3, 4) or len(faces) != counts["face"] or np.any(faces["n"] != corners):
        raise ValueError(f"{path}: not a pure triangle or quad mesh")
    index, ids = faces["i"].astype(np.int64), faces["object_id"].astype(np.int64)
    if corners == 4:
        index, ids = np.concatenate([index[:, [0, 1, 2]], index[:, [0, 2, 3]]]), np.concatenate([ids, ids])
    return np.column_stack([v[k] for k in "xyz"]).astype(np.float64), index, ids


def object_classes(info: dict) -> dict[int, int]:
    """Object id -> coarse class, from info_semantic.json."""
    if "objects" in info:
        return {int(o["id"]): replica_class(str(o.get("class_name", ""))) for o in info["objects"]}
    names = {int(c["id"]): str(c["name"]) for c in info["classes"]}
    return {i: replica_class(names.get(int(c), "")) for i, c in enumerate(info["id_to_label"])}


def sample_labeled(
    mesh: Path, info: Path, count: int = 2_000_000, seed: int = 0
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int8]]:
    """Points spread uniformly by area, their normals and coarse ground-truth class (-1 = unlabelled)."""
    v, f, ids = read_semantic_mesh(mesh)
    lookup = object_classes(json.loads(info.read_text()))
    unique, inverse = np.unique(ids, return_inverse=True)
    face_class = np.array([lookup.get(int(i), UNKNOWN) for i in unique], np.int8)[inverse]
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    cross = np.cross(b - a, c - a)
    area = np.linalg.norm(cross, axis=1)
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(f), count, p=area / area.sum())
    r1, r2 = np.sqrt(rng.random(count)), rng.random(count)
    w = np.column_stack([1 - r1, r1 * (1 - r2), r1 * r2])
    normals = cross[pick] / np.maximum(area[pick, None], 1e-12)
    return np.einsum("ni,nij->nj", w, v[f[pick]]), normals, face_class[pick]


def frame_offset(labeled: NDArray[np.float64], gt: NDArray[np.float64], count: int = 20_000) -> float:
    """Median distance from semantic-mesh samples to the geometry mesh: ~0 if they share a frame."""
    some = labeled[np.random.default_rng(0).choice(len(labeled), min(count, len(labeled)), replace=False)]
    return float(np.median(cKDTree(gt).query(some, workers=-1)[0]))


def label_scores(
    points: NDArray[np.float64], predictions: dict[str, NDArray[np.int8]],
    gt_points: NDArray[np.float64], gt_classes: NDArray[np.int8], max_distance: float = 0.05,
) -> dict[str, dict[str, float | int]]:
    """Per prediction: point accuracy and IoU, binary (structure/object) and over all classes.

    Each predicted point is compared with the nearest labelled ground-truth sample, if within
    `max_distance`; points farther away, or on unlabelled surfaces, are not scored."""
    labeled = gt_classes >= 0
    distance, nearest = cKDTree(gt_points[labeled]).query(points, workers=-1)
    truth = gt_classes[labeled][nearest]
    near = distance < max_distance
    scores = {}
    for name, predicted in predictions.items():
        ok = near & (predicted >= 0)
        p, t = predicted[ok].astype(np.int64), truth[ok].astype(np.int64)
        result: dict[str, float | int] = {"points_scored": int(ok.sum()), "fraction_scored": float(ok.mean())}
        if ok.any():
            ps, ts = p < OBJECT, t < OBJECT
            iou = {}
            for label, (a, b) in {"structure": (ps, ts), "object": (~ps, ~ts)}.items():
                union = np.sum(a | b)
                iou[label] = float(np.sum(a & b) / union) if union else float("nan")
                result[f"iou_{label}"] = iou[label]
            result["accuracy"] = float(np.mean(ps == ts))
            result["miou"] = float(np.nanmean(list(iou.values())))
            per_class = []
            for c in range(len(CLASSES)):
                union = np.sum((p == c) | (t == c))
                if union:
                    per_class.append(np.sum((p == c) & (t == c)) / union)
            result["class_accuracy"] = float(np.mean(p == t))
            result["class_miou"] = float(np.mean(per_class))
        scores[name] = result
    return scores
