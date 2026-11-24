import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(prog="rr", description="Room reconstruction, stages 1-3")
    parser.add_argument("video", type=Path, help="video or ScanNet .sens file")
    parser.add_argument("output", type=Path, help="fresh output directory")
    parser.add_argument("--through", type=int, choices=(1, 2, 3), default=3, help="last stage to run")
    parser.add_argument("--matcher", choices=("sequential", "exhaustive"), default="sequential",
                        help="exhaustive is slower but closes loops")
    parser.add_argument("--focal-factor", type=float, default=0.8, help="focal guess / long image side")
    parser.add_argument("--min-parallax", type=float, default=1.5, help="degrees between keyframes")
    parser.add_argument("--dense-size", type=int, default=1200, help="long image side for MVS")
    args = parser.parse_args()

    from rr.phase1 import KeyframeConfig, KeyframeSelector
    from rr.phase2 import SparseConfig, reconstruct_sparse
    from rr.sources import open_source
    from rr.phase3 import DenseConfig, reconstruct_dense

    out: Path = args.output
    keyframes = KeyframeSelector(
        KeyframeConfig(focal_factor=args.focal_factor, min_parallax_deg=args.min_parallax)
    ).select(open_source(args.video), out)
    print(f"stage 1: {len(keyframes)} keyframes")
    if args.through == 1:
        return
    sparse = reconstruct_sparse(
        out / "images", out / "sparse", SparseConfig(matcher=args.matcher, focal_factor=args.focal_factor)
    )
    print(f"stage 2: {sparse}")
    if sparse.num_models > 1 or sparse.images_registered < sparse.images_total:
        print("warning: some keyframes were not registered in the chosen model")
    if args.through == 2:
        return
    dense = reconstruct_dense(
        out / "images", Path(sparse.model_dir), out / "dense", DenseConfig(max_image_size=args.dense_size)
    )
    print(f"stage 3: {dense}")
