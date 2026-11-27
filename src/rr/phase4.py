"""Stage 4: 2D semantic segmentation of the keyframes.

A Mask2Former pretrained on ADE20K labels every pixel with one of 150 classes; the class map is
saved as a PNG of ADE20K ids, and classes.json says how they collapse to structure vs. object.

torch and transformers live in a second environment, because torch's CUDA conflicts with
pycolmap's (see README). From that environment:

    python -m rr.phase4 <image_dir> <output_dir>

From the pycolmap environment, `segment_in_subprocess` runs the same thing.
"""
import argparse
import json
import os
import subprocess
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from cv2.typing import MatLike
from numpy.typing import NDArray

from rr.taxonomy import CLASSES, ade_lookup

DEFAULT_MODEL = "facebook/mask2former-swin-large-ade-semantic"
UNLABELED = 255
ROOT = Path(__file__).resolve().parents[2]

# BGR image -> ADE20K id per pixel, same height and width.
Segmenter = Callable[[MatLike], NDArray[np.uint8]]


@dataclass
class SegmentationConfig:
    model: str = DEFAULT_MODEL       # any Hugging Face semantic/universal segmentation checkpoint
    device: str = "cuda"
    half: bool = True                # fp16 autocast on the GPU


@dataclass
class SegmentationResult:
    images: int
    model: str
    seconds: float
    class_fractions: dict[str, float]  # share of all pixels per coarse class


def load_segmenter(cfg: SegmentationConfig) -> tuple[Segmenter, dict[int, str]]:
    """The network, and its id -> class name table. Downloads the weights on first use."""
    import torch
    from transformers import AutoImageProcessor, AutoModelForUniversalSegmentation

    if cfg.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("torch sees no CUDA device; pass --device cpu to run (slowly) on the CPU")
    processor = AutoImageProcessor.from_pretrained(cfg.model)
    model = AutoModelForUniversalSegmentation.from_pretrained(cfg.model).to(cfg.device).eval()
    half = cfg.half and cfg.device.startswith("cuda")

    def segment(image: MatLike) -> NDArray[np.uint8]:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        inputs = processor(images=rgb, return_tensors="pt").to(cfg.device)
        with torch.inference_mode(), torch.autocast("cuda", torch.float16, enabled=half):
            outputs = model(**inputs)
        labels = processor.post_process_semantic_segmentation(outputs, target_sizes=[rgb.shape[:2]])[0]
        return labels.to("cpu", torch.uint8).numpy()

    id2label = {int(k): str(v) for k, v in model.config.id2label.items()}
    if max(id2label) >= UNLABELED:
        raise ValueError(f"{cfg.model} has {len(id2label)} classes; labels are stored as 8-bit")
    return segment, id2label


def segment_images(
    image_dir: Path, output_dir: Path, segment: Segmenter, id2label: dict[int, str], model_name: str
) -> SegmentationResult:
    """Writes output_dir/labels/<image stem>.png, classes.json and segmentation.json."""
    names = sorted(p.name for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not names:
        raise ValueError(f"no images in {image_dir}")
    label_dir = output_dir / "labels"
    label_dir.mkdir(parents=True, exist_ok=True)
    if any(label_dir.iterdir()):
        raise FileExistsError(f"{label_dir} is not empty; use a fresh output directory")
    lookup = ade_lookup(id2label)
    (output_dir / "classes.json").write_text(json.dumps({
        "model": model_name,
        "id2label": {str(i): n for i, n in sorted(id2label.items())},
        "coarse": {str(i): int(lookup[i]) for i in sorted(id2label)},
        "classes": CLASSES,
    }, indent=2))

    started = time.monotonic()
    counts = np.zeros(len(CLASSES), np.int64)
    for name in names:
        image = cv2.imread(str(image_dir / name))
        if image is None:
            raise ValueError(f"cannot decode {image_dir / name}")
        labels = segment(image)
        if labels.shape != image.shape[:2]:
            raise ValueError(f"segmenter returned {labels.shape} for a {image.shape[:2]} image")
        coarse = lookup[labels]
        counts += np.bincount(coarse[coarse >= 0], minlength=len(CLASSES))
        if not cv2.imwrite(str(label_dir / f"{Path(name).stem}.png"), labels):
            raise OSError(f"could not write {label_dir / name}")
    result = SegmentationResult(
        len(names), model_name, round(time.monotonic() - started, 1),
        {c: float(n / max(1, counts.sum())) for c, n in zip(CLASSES, counts)},
    )
    (output_dir / "segmentation.json").write_text(json.dumps(asdict(result), indent=2))
    return result


def read_lookup(seg_dir: Path) -> NDArray[np.int8]:
    """ADE20K id -> coarse class, as recorded by stage 4."""
    coarse = json.loads((seg_dir / "classes.json").read_text())["coarse"]
    table = np.full(256, -1, np.int8)
    for i, c in coarse.items():
        table[int(i)] = c
    return table


def read_labels(seg_dir: Path, image_name: str) -> NDArray[np.uint8] | None:
    """ADE20K ids of one keyframe, or None if stage 4 did not label it."""
    path = seg_dir / "labels" / f"{Path(image_name).stem}.png"
    return cv2.imread(str(path), cv2.IMREAD_UNCHANGED) if path.exists() else None


def segmentation_python() -> str:
    return os.environ.get("RR_SEG_PYTHON") or str(ROOT / ".venv-seg" / "bin" / "python")


def segment_in_subprocess(
    image_dir: Path, output_dir: Path, cfg: SegmentationConfig | None = None, python: str | None = None
) -> SegmentationResult:
    """Run stage 4 in the torch environment; RR_SEG_PYTHON overrides its interpreter."""
    cfg = cfg or SegmentationConfig()
    python = python or segmentation_python()
    if not Path(python).exists():
        raise RuntimeError(
            f"stage 4 needs the segmentation environment, but {python} does not exist. Create it with\n"
            "    UV_PROJECT_ENVIRONMENT=.venv-seg uv sync --no-group colmap --group segmentation\n"
            "or point RR_SEG_PYTHON at a Python that has torch and transformers."
        )
    command = [python, "-m", "rr.phase4", str(image_dir), str(output_dir), "--model", cfg.model, "--device", cfg.device]
    if not cfg.half:
        command.append("--no-half")
    subprocess.run(command, check=True)
    return SegmentationResult(**json.loads((output_dir / "segmentation.json").read_text()))


def main() -> None:
    parser = argparse.ArgumentParser(prog="rr-segment", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-half", action="store_true", help="full precision on the GPU")
    args = parser.parse_args()
    cfg = SegmentationConfig(args.model, args.device, not args.no_half)
    segment, id2label = load_segmenter(cfg)
    print(json.dumps(asdict(segment_images(args.image_dir, args.output_dir, segment, id2label, cfg.model)), indent=2))


if __name__ == "__main__":
    main()
