# rr

Room reconstruction from monocular video. Stages 1-3: keyframes, sparse SfM, dense MVS + planes.

Needs an NVIDIA GPU and Python 3.12-3.14 (pycolmap has no 3.15 wheels):

    uv python install 3.12 && uv python pin 3.12 && uv sync
    uv run rr room.mp4 out/
    uv run pytest

`out/dense/` holds `fused.ply` (measured), `plane_fill.ply` (synthetic), `dense.ply` (both),
`planes.ply` (coloured by plane) and `planes.npz`. Units are arbitrary: there is no metric scale.

Torch (stage 4) conflicts with pycolmap over CUDA versions, so it uses a second environment:

    UV_PROJECT_ENVIRONMENT=.venv-seg uv sync --no-group colmap --group segmentation

## Benchmark

    scripts/download_replica.py room0 office0   # non-commercial research/education terms
    uv run pytest                    # unit tests + every downloaded scene
    uv run pytest -m "not scenes"    # unit tests only
    uv run rr-benchmark              # same scenes, without pytest

Metrics and pictures go to `results/<time>/<scene>/<mode>/`. ScanNet also works: request access
at http://www.scan-net.org/, save their script as `room-data/download-scannet.py`, and run
`scripts/download_scannet.sh scene0000_00`.
