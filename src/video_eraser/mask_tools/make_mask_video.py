#!/usr/bin/env python3
import argparse
import os
import re
from typing import List, Tuple

import cv2
import numpy as np


def natural_key(s: str) -> List:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def list_images(input_dir: str, pattern_exts: Tuple[str, ...]) -> List[str]:
    entries = []
    for name in os.listdir(input_dir):
        lower = name.lower()
        if any(lower.endswith(ext) for ext in pattern_exts):
            entries.append(os.path.join(input_dir, name))
    entries.sort(key=lambda p: natural_key(os.path.basename(p)))
    return entries


def to_binary_mask(img: np.ndarray, threshold: int, invert: bool) -> np.ndarray:
    # Accept grayscale, RGB, or RGBA; convert to grayscale first
    if img.ndim == 3:
        # If has alpha, drop it
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    # Any nonzero becomes foreground by default; if threshold provided, use it
    _, bin_img = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    if invert:
        bin_img = cv2.bitwise_not(bin_img)
    return bin_img


def pick_fourcc(output_path: str) -> int:
    ext = os.path.splitext(output_path)[1].lower()
    # Default to mp4v for .mp4, else XVID
    if ext == ".mp4":
        return cv2.VideoWriter_fourcc(*"mp4v")
    if ext in {".avi"}:
        return cv2.VideoWriter_fourcc(*"XVID")
    # Fallback
    return cv2.VideoWriter_fourcc(*"mp4v")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert segmentation frames to a binary mask video.")
    parser.add_argument("input_dir", type=str, help="Directory containing mask frames.")
    parser.add_argument("output", type=str, help="Output video path, e.g., mask.mp4 or mask.avi")
    parser.add_argument("--fps", type=float, default=30.0, help="Frames per second for output video.")
    parser.add_argument("--threshold", type=int, default=1, help="Threshold (0-255) to binarize masks. Default 1 makes any nonzero foreground.")
    parser.add_argument("--invert", action="store_true", help="Invert mask (swap foreground/background).")
    parser.add_argument("--exts", type=str, default=".png,.jpg,.jpeg", help="Comma-separated image extensions to load.")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit on number of frames (0=all).")
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    if not os.path.isdir(input_dir):
        raise SystemExit(f"Input directory not found: {input_dir}")

    exts = tuple(e.strip().lower() for e in args.exts.split(",") if e.strip())
    frames = list_images(input_dir, exts)
    if not frames:
        raise SystemExit(f"No frames found in {input_dir} with extensions {exts}")
    if args.limit > 0:
        frames = frames[: args.limit]

    first = cv2.imread(frames[0], cv2.IMREAD_UNCHANGED)
    if first is None:
        raise SystemExit(f"Failed to read first frame: {frames[0]}")
    bin_first = to_binary_mask(first, args.threshold, args.invert)
    height, width = bin_first.shape[:2]
    # Ensure even dimensions for MP4/H.264 compatibility
    even_width = width - (width % 2)
    even_height = height - (height % 2)
    if (even_width, even_height) != (width, height):
        bin_first = cv2.resize(bin_first, (even_width, even_height), interpolation=cv2.INTER_NEAREST)
        width, height = even_width, even_height

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    fourcc = pick_fourcc(args.output)
    # Write 3-channel frames (grayscale replicated) for broader MP4 compatibility
    writer = cv2.VideoWriter(args.output, fourcc, args.fps, (width, height), isColor=True)
    if not writer.isOpened():
        raise SystemExit(f"Failed to open VideoWriter for: {args.output}")

    try:
        for idx, path in enumerate(frames):
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise SystemExit(f"Failed to read frame: {path}")
            bin_img = to_binary_mask(img, args.threshold, args.invert)
            if bin_img.shape[:2] != (height, width):
                bin_img = cv2.resize(bin_img, (width, height), interpolation=cv2.INTER_NEAREST)
            # replicate grayscale to 3 channels
            bgr = cv2.cvtColor(bin_img, cv2.COLOR_GRAY2BGR)
            writer.write(bgr)
    finally:
        writer.release()

    print(f"Wrote mask video: {os.path.abspath(args.output)} ({len(frames)} frames @ {args.fps} fps)")


if __name__ == "__main__":
    main()


