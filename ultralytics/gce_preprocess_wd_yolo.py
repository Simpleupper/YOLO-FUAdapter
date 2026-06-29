"""Offline GCE preprocessing for WD-YOLO weld X-ray images.

This script implements a practical version of the Gray Value Curve Enhancement (GCE)
step described in WD-YOLO. Use it to preprocess/copy your dataset images before YOLO
training. It is intentionally separate from the model code so the other four algorithms
remain unaffected.

Example:
    python gce_preprocess_wd_yolo.py --src /path/images --dst /path/images_gce
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _remove_impulse_noise(gray: np.ndarray) -> np.ndarray:
    gray = gray.astype(np.float32)
    hist = np.bincount(gray.astype(np.uint8).ravel(), minlength=256)
    # Candidate bad pixels are extremely rare gray levels at both histogram tails.
    rare = np.where(hist <= max(2, gray.size // 200000))[0]
    if rare.size == 0:
        return gray
    mask = np.isin(gray.astype(np.uint8), rare)
    if not mask.any():
        return gray
    padded = np.pad(gray, 1, mode="reflect")
    neigh = (
        padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:] +
        padded[1:-1, :-2] + padded[1:-1, 2:] +
        padded[2:, :-2] + padded[2:, 1:-1] + padded[2:, 2:]
    ) / 8.0
    out = gray.copy()
    out[mask] = neigh[mask]
    return out


def _contrast_score(gray: np.ndarray) -> float:
    gy, gx = np.gradient(gray.astype(np.float32))
    return float(np.mean(np.sqrt(gx * gx + gy * gy)))


def gce_image(img: Image.Image, threshold: float = 70.0, max_iter: int = 2) -> Image.Image:
    """Apply WD-YOLO-style gray value curve enhancement to one image."""
    mode = img.mode
    arr = np.asarray(img.convert("L"), dtype=np.float32)
    for _ in range(max_iter):
        arr = _remove_impulse_noise(arr)
        padded = np.pad(arr, 1, mode="reflect")
        q = (padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:]) / 4.0
        mix = 0.7 * arr + 0.3 * q
        mn, mx = float(mix.min()), float(mix.max())
        if mx - mn < 1e-6:
            break
        arr = (mix - mn) * 255.0 / (mx - mn)
        if _contrast_score(arr) >= threshold:
            break
    out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="L")
    return out.convert(mode) if mode not in {"L", "I;16"} else out


def process_dir(src: Path, dst: Path, threshold: float, max_iter: int) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for path in src.rglob("*"):
        rel = path.relative_to(src)
        out_path = dst / rel
        if path.is_dir():
            out_path.mkdir(parents=True, exist_ok=True)
            continue
        if path.suffix.lower() not in IMG_EXTS:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(path.read_bytes())
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img = Image.open(path)
        gce_image(img, threshold=threshold, max_iter=max_iter).save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, type=Path)
    parser.add_argument("--dst", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=70.0)
    parser.add_argument("--max-iter", type=int, default=2)
    args = parser.parse_args()
    process_dir(args.src, args.dst, args.threshold, args.max_iter)


if __name__ == "__main__":
    main()
