"""Per-phase pictures of a run, for real videos (`rr`) and the benchmark (`rr-benchmark`).

Everything goes to <run>/pictures/, prefixed by phase:

    p1_original/  p1_keyframes.jpg  p1_timeline.png
    p2_sparse_inside/  p2_sparse_outside/
    p3_dense_inside/  p3_dense_outside/
    p4_segmentation/  p4_classes.png
    p5_labels_inside/  p5_labels_outside/  p5_classes.png
    all_in_one.jpg      every phase side by side; rewritten after phases 3 and 5

The benchmark adds ground-truth columns ("target") and charts that need ground truth
(p2_trajectory.png, p3_scores.png, p5_scores.png).

Inside views use the reconstruction's own cameras. Outside views need a z-up frame: the
ground-truth frame when the benchmark has one, otherwise one levelled with the cameras'
average "up" (the phone is held roughly upright), which is all a real video offers.
"""
import json
from pathlib import Path
from typing import Any

import numpy as np

from rr.charts import class_shares, keyframe_timeline
from rr.evaluation import Similarity
from rr.phase4 import read_labels, read_lookup
from rr.ply import PointCloud, concatenate, read_ply
from rr.sources import FrameSource
from rr.taxonomy import CLASSES, OBJECT, is_structure
from rr.visualize import (
    binary_cloud, class_cloud, coarse_image, facing_cameras, frame_comparisons, gravity_frame, keyframe_sheet,
    original_frames, outside_views, overview, recolored, segmentation_sheet,
)

Clouds = dict[str, tuple[PointCloud, int]]  # title -> (cloud, splat radius)
GREY, RED = np.array((170, 170, 170), np.uint8), np.array((230, 20, 20), np.uint8)


class Pictures:
    def __init__(self, run: Path) -> None:
        self.run, self.dir = run, run / "pictures"
        self.world: Similarity | None = None
        self.target_out: Clouds = {}   # ground truth, world frame
        self.target_in: Clouds = {}    # ground truth, model frame

    def set_ground_truth(self, world: Similarity, target: PointCloud) -> None:
        """`world` maps the model frame to the ground-truth frame; `target` is in the latter."""
        self.world = world
        self.target_out = {"target": (target, 1)}
        self.target_in = {"target mesh": (world.inverse().cloud(target), 1)}

    def _frame(self, model: Any) -> Similarity:
        if self.world is None:
            self.world = gravity_frame(model)
        return self.world

    def phase1(self, source: FrameSource) -> None:
        data = json.loads((self.run / "keyframes.json").read_text())
        original_frames(source, self.dir / "p1_original")
        keyframe_sheet(self.run / "images", [k["name"] for k in data["keyframes"]], self.dir / "p1_keyframes.jpg")
        keyframe_timeline(data, self.dir / "p1_timeline.png")

    def phase2(self, model: Any) -> None:
        world = self._frame(model)
        sparse = read_ply(self.run / "sparse" / "sparse.ply")
        shown = world.cloud(facing_cameras(sparse, model))
        outside_views(self.target_out | {"p2 sparse": (shown, 2)}, self.dir / "p2_sparse_outside")
        frame_comparisons(model, self.run / "images", self.target_in | {"p2 sparse": (sparse, 2)}, self.dir / "p2_sparse_inside")

    def phase3(self, model: Any) -> None:
        world = self._frame(model)
        d = self._dense()
        fill_shown = concatenate([recolored(d["fused"], np.tile(GREY, (len(d["fused"]), 1))),
                                  recolored(d["plane_fill"], np.tile(RED, (len(d["plane_fill"]), 1)))])
        clouds = {"p3 fused": d["fused"], "p3 dense": d["dense"], "p3 planes": d["planes"], "p3 fill in red": fill_shown}
        outside_views(self.target_out | {n: (world.cloud(c), 1) for n, c in clouds.items()}, self.dir / "p3_dense_outside")
        frame_comparisons(model, self.run / "images", self.target_in | {n: (c, 1) for n, c in clouds.items()},
                          self.dir / "p3_dense_inside")
        overview(model, self.run / "images", world, self._columns(model, self.target_out), self.dir / "all_in_one.jpg")

    def phase4(self) -> None:
        seg_dir = self.run / "segmentation"
        result = json.loads((seg_dir / "segmentation.json").read_text())
        segmentation_sheet(self.run / "dense" / "images", seg_dir, self.dir / "p4_segmentation")
        class_shares(result["class_fractions"], self.dir / "p4_classes.png", "p4: share of keyframe pixels per class")

    def phase5(self, dense_model: Any, target_labels: PointCloud | None = None) -> None:
        """`dense_model` is the undistorted model in dense/sparse; `target_labels` (world frame,
        coloured by class) replaces the target when the benchmark has semantic ground truth."""
        world = self._frame(dense_model)
        classes = np.load(self.run / "semantic" / "labels.npz")["classes"]
        d = self._dense()
        cloud = concatenate([d["fused"], d["plane_fill"]])
        class_shares({c: float(np.mean(classes == i)) for i, c in enumerate(CLASSES)}, self.dir / "p5_classes.png",
                     "p5: share of dense points per class")
        p5 = {
            "p5 labels": class_cloud(cloud, classes), "p5 structure / object": binary_cloud(cloud, classes),
            "p5 structure": cloud[is_structure(classes)], "p5 objects": cloud[classes == OBJECT],
        }
        if target_labels is not None:
            target_out: Clouds = {"target labels": (target_labels, 1)}
            target_in: Clouds = {"target labels": (world.inverse().cloud(target_labels), 1)}
        else:
            target_out, target_in = self.target_out, self.target_in
        outside_views(target_out | {n: (world.cloud(c), 1) for n, c in p5.items()}, self.dir / "p5_labels_outside")

        seg_dir = self.run / "segmentation"
        lookup = read_lookup(seg_dir)

        def labels_2d(name: str):
            ade = read_labels(seg_dir, name)
            return None if ade is None else coarse_image(lookup[ade])

        images = self.run / "dense" / "images"
        frame_comparisons(dense_model, images, target_in | {n: (c, 1) for n, c in p5.items()}, self.dir / "p5_labels_inside",
                          extra=lambda name, size: [("p4 2D labels", lab)] if (lab := labels_2d(name)) is not None else [])
        columns = self._columns(dense_model, target_out) + [(n, world.cloud(c), 1) for n, c in p5.items()]
        overview(dense_model, images, world, columns, self.dir / "all_in_one.jpg", labels_2d)

    def _dense(self) -> dict[str, PointCloud]:
        return {n: read_ply(self.run / "dense" / f"{n}.ply") for n in ("fused", "plane_fill", "dense", "planes")}

    def _columns(self, model: Any, target: Clouds) -> list[tuple[str, PointCloud, int]]:
        """Target, sparse, dense and planes, in the world frame, for all_in_one.jpg."""
        world = self._frame(model)
        sparse = world.cloud(facing_cameras(read_ply(self.run / "sparse" / "sparse.ply"), model))
        d = {n: read_ply(self.run / "dense" / f"{n}.ply") for n in ("dense", "planes")}
        return [(n, c, r) for n, (c, r) in target.items()] + [
            ("p2 sparse", sparse, 2), ("p3 dense", world.cloud(d["dense"]), 1), ("p3 planes", world.cloud(d["planes"]), 1),
        ]
