"""Matplotlib charts for the per-phase pictures (headless)."""
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from numpy.typing import NDArray  # noqa: E402

from rr.taxonomy import CLASSES, COLORS  # noqa: E402

REASON_COLORS = {"first": "black", "parallax": "tab:blue", "overlap": "tab:green", "bridge": "tab:orange", "break": "tab:red"}


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def keyframe_timeline(keyframes: dict[str, Any], output: Path) -> None:
    """Sharpness of every examined frame, and why and where keyframes were taken."""
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    frames = np.array(keyframes.get("frame_sharpness", []), dtype=float).reshape(-1, 3)
    if len(frames):
        sharp = frames[:, 2] > 0
        top.semilogy(frames[:, 0], frames[:, 1], color="0.6", lw=0.8, label="sharpness")
        top.scatter(frames[~sharp, 0], frames[~sharp, 1], s=10, color="tab:red", marker="x", label="rejected as blurry")
    kf = keyframes["keyframes"]
    for reason, color in REASON_COLORS.items():
        xs = [k["frame_index"] for k in kf if k["reason"] == reason]
        if xs:
            top.vlines(xs, 0, 1, transform=top.get_xaxis_transform(), color=color, lw=0.6, alpha=0.7,
                       label=f"keyframe: {reason} ({len(xs)})")
    top.set_ylabel("variance of Laplacian")
    top.set_title(f"p1: {len(kf)} keyframes from {keyframes.get('frames_examined', '?')} examined frames")
    top.legend(fontsize=8, loc="upper right", ncols=2)

    x = [k["frame_index"] for k in kf]
    colors = [REASON_COLORS[k["reason"]] for k in kf]
    bottom.scatter(x, [k["inliers"] for k in kf], c=colors, s=12)
    bottom.set_ylabel("inliers to previous keyframe")
    bottom.axhline(keyframes.get("config", {}).get("min_matches", 40), color="tab:red", ls="--", lw=0.8)
    right = bottom.twinx()
    right.plot(x, [k["parallax_deg"] for k in kf], color="tab:purple", lw=0.8, alpha=0.7)
    right.set_ylabel("parallax (deg)", color="tab:purple")
    bottom.set_xlabel("frame")
    _save(fig, output)


def trajectory(estimated: NDArray, truth: NDArray, output: Path) -> None:
    """Aligned estimated camera path against the true one, seen along the least-varying axis."""
    _, _, Vt = np.linalg.svd(truth - truth.mean(0))
    a, b = (truth - truth.mean(0)) @ Vt[:2].T, (estimated - truth.mean(0)) @ Vt[:2].T
    error = np.linalg.norm(estimated - truth, axis=1)
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [1.2, 1]})
    left.plot(*a.T, color="0.3", lw=1.5, label="ground truth")
    left.plot(*b.T, color="tab:blue", lw=1, ls="--", label="estimated (aligned)")
    left.scatter(*a[0], color="tab:green", zorder=3, label="start")
    left.set_aspect("equal")
    left.set_xlabel("m")
    left.set_ylabel("m")
    left.legend(fontsize=8)
    left.set_title("p2: camera path, top-down")
    right.plot(error * 1000, color="tab:red", lw=1)
    right.set_xlabel("keyframe")
    right.set_ylabel("position error (mm)")
    right.set_title(f"RMSE {np.sqrt(np.mean(error**2)) * 1000:.1f} mm, max {error.max() * 1000:.1f} mm")
    _save(fig, output)


def cloud_scores(clouds: dict[str, dict[str, float]], output: Path) -> None:
    """Precision, recall and F-score at 5 cm, per point cloud."""
    keys = [("precision_5cm", "precision"), ("recall_5cm", "recall (whole mesh)"),
            ("visible_recall_5cm", "recall (filmed)"), ("fscore_5cm", "F-score")]
    names = [n for n, c in clouds.items() if "precision_5cm" in c]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    width = 0.8 / len(keys)
    for j, (key, label) in enumerate(keys):
        values = [clouds[n].get(key, np.nan) for n in names]
        bars = ax.bar(np.arange(len(names)) + j * width, values, width, label=label)
        ax.bar_label(bars, fmt="%.2f", fontsize=7)
    ax.set_xticks(np.arange(len(names)) + width * (len(keys) - 1) / 2, names)
    ax.set_ylim(0, 1.05)
    ax.set_title("p2/p3: scores against the ground-truth mesh, 5 cm threshold")
    ax.legend(fontsize=8, ncols=4, loc="lower right")
    _save(fig, output)


def class_shares(shares: dict[str, float], output: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.8))
    values = [shares.get(c, 0.0) for c in CLASSES]
    bars = ax.barh(CLASSES, values, color=COLORS / 255)
    ax.bar_label(bars, labels=[f"{v:.1%}" for v in values], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, max(values + [0.01]) * 1.2)
    ax.set_title(title)
    _save(fig, output)


def label_scores(scores: dict[str, dict[str, float]], output: Path) -> None:
    """Structure/object labelling quality for each fusion variant."""
    keys = [("accuracy", "accuracy"), ("iou_structure", "IoU structure"), ("iou_object", "IoU object"),
            ("miou", "mIoU"), ("class_miou", f"mIoU, {len(CLASSES)} classes")]
    names = list(scores)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    width = 0.8 / len(keys)
    for j, (key, label) in enumerate(keys):
        bars = ax.bar(np.arange(len(names)) + j * width, [scores[n].get(key, np.nan) for n in names], width, label=label)
        ax.bar_label(bars, fmt="%.2f", fontsize=7)
    ax.set_xticks(np.arange(len(names)) + width * (len(keys) - 1) / 2, names)
    ax.set_ylim(0, 1.05)
    ax.set_title("p5: point labels against Replica's semantic mesh")
    ax.legend(fontsize=8, ncols=5, loc="lower right")
    _save(fig, output)
