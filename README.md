# rr

Room reconstruction from monocular video, separated into structure (walls, floor, ceiling,
doors, windows) and objects. Stages:

1. keyframes: sharp frames with enough rotation-compensated parallax
2. sparse: COLMAP incremental SfM
3. dense: PatchMatch MVS, RANSAC plane segments, gap filling
4. segmentation: Mask2Former (ADE20K) on every keyframe, collapsed to structure vs. object
5. fusion: per-pixel labels voted onto the dense points across views, then a plane prior

Needs an NVIDIA GPU and Python 3.12-3.14 (pycolmap has no 3.15 wheels):

    uv python install 3.12 && uv python pin 3.12 && uv sync
    uv run rr room.mp4 out/                  # all five stages; --through N stops earlier
    uv run pytest

Any room shape or size works: nothing assumes a box, and plane tolerances are measured in
units of the typical camera-to-surface distance (recorded as `unit` in `dense/dense.json`), so
they stay the same in a small room and an open-plan floor. Coplanar pieces, such as one wall
line running through two rooms, become separate plane segments. For long walks through
several rooms, sequential matching adds loop detection above 300 keyframes (`--loops on|off`
to force it; COLMAP fetches its vocabulary tree, or pass `--vocab-tree`).

Filming: walk rather than turn on the spot, keep a few metres from the walls, move slowly, in
good light, landscape, 1-3 minutes per room. `rr` writes per-phase pictures to `out/pictures/`
(see below, without the ground-truth columns and charts).

Torch (stage 4) conflicts with pycolmap over CUDA versions, so stage 4 runs in a second
environment, which the pipeline calls as a subprocess (RR_SEG_PYTHON overrides its location):

    UV_PROJECT_ENVIRONMENT=.venv-seg uv sync --no-group colmap --group segmentation

The Mask2Former weights (about 1 GB) download from Hugging Face on first use. A smaller
model: `--seg-model facebook/mask2former-swin-tiny-ade-semantic`.

Outputs, under `out/`: `dense/` holds `fused.ply` (measured), `plane_fill.ply` (synthetic),
`dense.ply` (both), `planes.ply` and `planes.npz`; `segmentation/labels/` the ADE20K class map of
each keyframe; `semantic/` `labels.ply` (coloured by class), `structure.ply`, `objects.ply` and
`labels.npz`. Units are arbitrary: there is no metric scale.

## Benchmark

    scripts/download_replica.py room0 office0             # non-commercial research/education terms
    scripts/download_replica_semantics.py room0 office0   # optional: labels to score stage 5 (streams tens of GB)
    uv run pytest                    # unit tests + every downloaded scene
    uv run pytest -m "not scenes"    # unit tests only
    uv run rr-benchmark              # same scenes, without pytest

Metrics go to `results/<time>/<scene>/<mode>/metrics.json`, with `summary.csv` (geometry) and
`labels.csv` (structure/object scores) per run. Pictures go to `pictures/`, prefixed by phase
(`p2_trajectory.png`, `p3_scores.png` and `p5_scores.png` need ground truth, so only the benchmark
makes them):

| | |
|---|---|
| `p1_original/`, `p1_keyframes.jpg`, `p1_timeline.png` | input frames, keyframes, sharpness and keyframe reasons over time |
| `p2_sparse_inside/`, `p2_sparse_outside/`, `p2_trajectory.png` | sparse cloud from the keyframe cameras and from outside; camera path vs. ground truth |
| `p3_dense_inside/`, `p3_dense_outside/`, `p3_scores.png` | fused, dense, planes and fill; precision/recall per cloud |
| `p4_segmentation/`, `p4_classes.png` | keyframe, ADE20K classes, coarse classes, structure/object overlay |
| `p5_labels_inside/`, `p5_labels_outside/`, `p5_classes.png`, `p5_scores.png` | 3D labels, structure and objects; label scores |
| `all_in_one.jpg` | every phase side by side, from outside and from keyframe cameras |
