from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
from tqdm import tqdm


def _load_video_frames(path: Path) -> Tuple[List[np.ndarray], float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames: List[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from {path}")
    return frames, fps


def _write_video(path: Path, frames: Sequence[np.ndarray], fps: float) -> None:
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to initialize VideoWriter for {path}")
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()


def _align_frame(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    src_gray = src_gray.astype(np.float32) / 255.0
    ref_gray = ref_gray.astype(np.float32) / 255.0

    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-4)
    try:
        cv2.findTransformECC(ref_gray, src_gray, warp_matrix, cv2.MOTION_AFFINE, criteria)
        aligned = cv2.warpAffine(
            src,
            warp_matrix,
            (ref.shape[1], ref.shape[0]),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        return aligned
    except cv2.error:
        return src


def _gaussian_weights(radius: int, sigma: float) -> Tuple[np.ndarray, np.ndarray]:
    offsets = np.arange(-radius, radius + 1)
    if sigma <= 0:
        sigma = max(radius / 2.0, 1e-3)
    weights = np.exp(-(offsets ** 2) / (2 * sigma ** 2))
    weights /= weights.sum()
    return offsets, weights


def temporal_smooth_video(
    video_path: Path,
    output_path: Path,
    mask_video: Optional[Path],
    window: int,
    sigma: float,
    strength: float,
) -> Path:
    frames, fps = _load_video_frames(video_path)
    mask_frames: Optional[List[np.ndarray]] = None
    if mask_video:
        mask_frames, mask_fps = _load_video_frames(mask_video)
        if len(mask_frames) != len(frames):
            raise ValueError(
                f"Mask frame count ({len(mask_frames)}) does not match video ({len(frames)})"
            )
        if abs(mask_fps - fps) > 1e-2:
            raise ValueError(f"Mask fps {mask_fps} differs from video fps {fps}")

    radius = max(window // 2, 1)
    offsets, weights = _gaussian_weights(radius, sigma)
    center_idx = np.where(offsets == 0)[0]
    if center_idx.size == 0:
        offsets = np.insert(offsets, radius, 0)
        weights = np.insert(weights, radius, 0.0)
        weights /= weights.sum()
    strength = float(np.clip(strength, 0.0, 1.0))

    smoothed_frames: List[np.ndarray] = []

    for idx in tqdm(range(len(frames)), desc="Temporal smoothing"):
        ref = frames[idx].astype(np.float32)
        accum = np.zeros_like(ref)
        total_w = 0.0

        for offset, weight in zip(offsets, weights):
            neighbor_idx = idx + offset
            neighbor_idx = min(max(neighbor_idx, 0), len(frames) - 1)
            neighbor = frames[neighbor_idx]
            aligned = neighbor if offset == 0 else _align_frame(neighbor, frames[idx])
            accum += weight * aligned.astype(np.float32)
            total_w += weight

        temporal_avg = accum / max(total_w, 1e-6)

        if mask_frames is not None:
            mask = mask_frames[idx]
            if mask.ndim == 2:
                mask = mask[:, :, None]
            mask = (mask.astype(np.float32) / 255.0) * strength
            result = ref * (1.0 - mask) + temporal_avg * mask
        else:
            result = ref * (1.0 - strength) + temporal_avg * strength

        smoothed_frames.append(np.clip(result, 0, 255).astype(np.uint8))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_video(output_path, smoothed_frames, fps)
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reduce ghosting by aligning neighboring frames and blending them temporally."
    )
    parser.add_argument("--video", type=Path, required=True, help="Input video to smooth.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Destination path (default: <video_stem>_smooth.mp4).",
    )
    parser.add_argument(
        "--mask",
        type=Path,
        help="Optional binary mask video; smoothing is limited to masked pixels.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=5,
        help="Temporal window size (odd number).",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.0,
        help="Gaussian sigma controlling weights across the window.",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=1.0,
        help="Blending strength (0=no smoothing, 1=full smoothing).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    window = args.window if args.window % 2 == 1 else args.window + 1
    output = args.output or args.video.with_name(f"{args.video.stem}_smooth.mp4")
    temporal_smooth_video(
        video_path=args.video,
        output_path=output,
        mask_video=args.mask,
        window=window,
        sigma=args.sigma,
        strength=args.strength,
    )
    print(f"[ok] Smoothed video written to {output}")


if __name__ == "__main__":
    main()

