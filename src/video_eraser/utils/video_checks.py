from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass
class VideoStats:
    path: Path
    frame_count: int
    fps: float
    width: int
    height: int


def probe_video(path: Path) -> VideoStats:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    stats = VideoStats(
        path=path.resolve(),
        frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        fps=float(cap.get(cv2.CAP_PROP_FPS)),
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    cap.release()
    return stats


def assert_same_video_geometry(video_a: Path, video_b: Path, fps_tolerance: float = 1e-2) -> None:
    meta_a = probe_video(video_a)
    meta_b = probe_video(video_b)

    mismatches = []
    if meta_a.frame_count != meta_b.frame_count:
        mismatches.append(f"frame count {meta_a.frame_count} vs {meta_b.frame_count}")
    if abs(meta_a.fps - meta_b.fps) > fps_tolerance:
        mismatches.append(f"fps {meta_a.fps:.3f} vs {meta_b.fps:.3f}")
    if meta_a.width != meta_b.width or meta_a.height != meta_b.height:
        mismatches.append(
            f"resolution {meta_a.width}x{meta_a.height} vs {meta_b.width}x{meta_b.height}"
        )
    if mismatches:
        raise ValueError(
            f"Video mismatch between {video_a} and {video_b}: {', '.join(mismatches)}"
        )

