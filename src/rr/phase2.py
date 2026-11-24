"""Stage 2: sparse reconstruction with COLMAP incremental SfM."""
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from rr.colmap import load_pycolmap


@dataclass
class SparseConfig:
    matcher: Literal["sequential", "exhaustive"] = "sequential"
    overlap: int = 15                  # neighbours matched per image (sequential)
    camera_model: str = "SIMPLE_RADIAL"
    focal_factor: float = 0.8          # initial guess only; refined by bundle adjustment
    max_image_size: int = 1920
    seed: int = 0


@dataclass
class SparseResult:
    model_dir: str
    images_total: int
    images_registered: int
    num_models: int                    # >1 means the video broke into disconnected pieces
    num_points: int
    mean_reprojection_error: float


def reconstruct_sparse(image_dir: Path, output_dir: Path, config: SparseConfig | None = None) -> SparseResult:
    """Writes output_dir/{database.db, models/<i>, model, sparse.ply, sparse.json}."""
    cfg = config or SparseConfig()
    pycolmap = load_pycolmap()
    names = sorted(p.name for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if len(names) < 3:
        raise ValueError(f"need at least 3 keyframes, found {len(names)} in {image_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    database = output_dir / "database.db"
    if database.exists():
        raise FileExistsError(f"{database} exists; use a fresh output directory")
    pycolmap.set_random_seed(cfg.seed)

    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = cfg.camera_model
    reader.default_focal_length_factor = cfg.focal_factor
    extraction = pycolmap.FeatureExtractionOptions()
    extraction.max_image_size = cfg.max_image_size
    # One physical camera took every frame, so share a single set of intrinsics.
    pycolmap.extract_features(
        database, image_dir, image_names=names, camera_mode=pycolmap.CameraMode.SINGLE,
        reader_options=reader, extraction_options=extraction,
    )

    if cfg.matcher == "sequential":
        pairing = pycolmap.SequentialPairingOptions()
        pairing.overlap = cfg.overlap
        pycolmap.match_sequential(database, pairing_options=pairing)
    elif cfg.matcher == "exhaustive":
        pycolmap.match_exhaustive(database)
    else:
        raise ValueError(f"unknown matcher {cfg.matcher!r}")

    models_dir = output_dir / "models"
    models_dir.mkdir()
    options = pycolmap.IncrementalPipelineOptions()
    options.random_seed = cfg.seed
    options.min_model_size = 3
    models = pycolmap.incremental_mapping(database, image_dir, models_dir, options=options)
    if not models:
        raise RuntimeError("COLMAP could not reconstruct any model: too little parallax or overlap?")

    best = max(models.values(), key=lambda m: (m.num_reg_images(), m.num_points3D()))
    model_dir = output_dir / "model"
    model_dir.mkdir()
    best.write(model_dir)
    best.export_PLY(output_dir / "sparse.ply")
    result = SparseResult(
        str(model_dir), len(names), best.num_reg_images(), len(models),
        best.num_points3D(), float(best.compute_mean_reprojection_error()),
    )
    (output_dir / "sparse.json").write_text(json.dumps(asdict(result) | {"config": asdict(cfg)}, indent=2))
    return result
