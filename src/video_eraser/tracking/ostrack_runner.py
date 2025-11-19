import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

DEFAULT_TRACKER_NAME = "ostrack"
DEFAULT_TRACKER_PARAM = "vitb_384_mae_ce_32x4_ep300"
DEFAULT_KEYBOX_FRAME = 1


@dataclass
class TrackRunResult:
    """Container for paths produced by an OSTrack inference run."""

    raw_result_file: Path
    exported_track_file: Path
    exported_track_csv: Path
    boxes_xywh: np.ndarray


def _ensure_local_env_file(ostrack_root: Path) -> None:
    """Create lib/test/evaluation/local.py with sane defaults if absent."""

    env_dir = ostrack_root / "lib" / "test" / "evaluation"
    env_dir.mkdir(parents=True, exist_ok=True)
    env_file = env_dir / "local.py"
    if env_file.exists():
        return

    results_dir = ostrack_root / "tracking_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    content = f"""
import os
from lib.test.evaluation.environment import EnvSettings


def local_env_settings():
    settings = EnvSettings()
    settings.prj_dir = r"{ostrack_root.resolve()}"
    settings.save_dir = r"{ostrack_root.resolve()}"
    settings.results_path = r"{results_dir.resolve()}"
    settings.result_plot_path = os.path.join(settings.results_path, "plots")
    settings.segmentation_path = os.path.join(settings.results_path, "segmentation")
    settings.network_path = os.path.join(settings.save_dir, "checkpoints")
    return settings
"""
    env_file.write_text(content.strip() + "\n", encoding="utf-8")


def load_box_from_csv(
    csv_path: Path, frame_id: Optional[int] = None
) -> Sequence[float]:
    """Return [x1, y1, x2, y2] from a key-frame CSV."""

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"No rows found in {csv_path}")
    if frame_id is None:
        row = df.iloc[0]
    else:
        matches = df[df["frame_id"] == frame_id]
        if matches.empty:
            raise ValueError(
                f"frame_id={frame_id} not found in {csv_path}. "
                f"Available ids: {df['frame_id'].tolist()}"
            )
        row = matches.iloc[0]
    return [row["x1"], row["y1"], row["x2"], row["y2"]]


def _xyxy_to_xywh(box_xyxy: Sequence[float]) -> List[float]:
    x1, y1, x2, y2 = box_xyxy
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


def _maybe_extend_pythonpath(env: dict, ostrack_root: Path) -> None:
    current = env.get("PYTHONPATH")
    root_str = str(ostrack_root.resolve())
    if current:
        if root_str not in current.split(os.pathsep):
            env["PYTHONPATH"] = os.pathsep.join([root_str, current])
    else:
        env["PYTHONPATH"] = root_str


def run_ostrack_tracking(
    video_path: Path,
    output_dir: Path,
    init_box_xyxy: Sequence[float],
    tracker_name: str = DEFAULT_TRACKER_NAME,
    tracker_param: str = DEFAULT_TRACKER_PARAM,
    ostrack_root: Path = Path("third_party") / "OSTrack",
    python_bin: str = sys.executable,
    overwrite: bool = False,
) -> TrackRunResult:
    """Execute OSTrack's video demo with a fixed initialization box."""

    video_path = video_path.resolve()
    output_dir = output_dir.resolve()
    ostrack_root = ostrack_root.resolve()
    if not ostrack_root.exists():
        raise FileNotFoundError(
            f"OSTrack repo not found at {ostrack_root}. "
            "Clone https://github.com/botaoye/OSTrack into third_party/OSTrack."
        )

    _ensure_local_env_file(ostrack_root)

    track_dir = output_dir
    track_dir.mkdir(parents=True, exist_ok=True)

    video_demo = ostrack_root / "tracking" / "video_demo.py"
    if not video_demo.exists():
        raise FileNotFoundError(f"video_demo.py missing at {video_demo}")

    optional_box_xywh = _xyxy_to_xywh(init_box_xyxy)

    env = os.environ.copy()
    _maybe_extend_pythonpath(env, ostrack_root)

    cmd: List[str] = [
        python_bin,
        str(video_demo),
        tracker_name,
        tracker_param,
        str(video_path),
        "--save_results",
    ]
    cmd.extend(["--optional_box", *[f"{v:.2f}" for v in optional_box_xywh]])

    subprocess.run(cmd, check=True, cwd=ostrack_root, env=env)

    results_root = ostrack_root / "tracking_results" / tracker_name / tracker_param
    video_stem = video_path.stem
    raw_result = results_root / f"video_{video_stem}.txt"
    if not raw_result.exists():
        raise FileNotFoundError(
            f"Expected tracking output {raw_result} was not created. "
            "Check OSTrack logs for errors."
        )

    exported_txt = track_dir / f"{video_stem}_ostrack.txt"
    if raw_result.resolve() != exported_txt:
        if exported_txt.exists() and overwrite:
            exported_txt.unlink()
        shutil.copy2(raw_result, exported_txt)

    data = np.loadtxt(exported_txt, delimiter="\t", dtype=float)
    if data.ndim == 1:
        data = data[None, :]

    csv_path = track_dir / f"{video_stem}_ostrack.csv"
    if csv_path.exists() and overwrite:
        csv_path.unlink()

    with csv_path.open("w", encoding="utf-8") as handle:
        handle.write("frame_id,x,y,w,h\n")
        for idx, (x, y, w, h) in enumerate(data):
            handle.write(f"{idx},{x:.2f},{y:.2f},{w:.2f},{h:.2f}\n")

    return TrackRunResult(
        raw_result_file=raw_result,
        exported_track_file=exported_txt,
        exported_track_csv=csv_path,
        boxes_xywh=data,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Utility wrapper around OSTrack's video_demo.py to export bounding boxes."
    )
    parser.add_argument("--video", type=Path, required=True, help="Input video file.")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/interim/tracks"),
        help="Directory for exported tracks.",
    )
    parser.add_argument(
        "--ostrack_root",
        type=Path,
        default=Path("third_party/OSTrack"),
        help="Path to the cloned OSTrack repository.",
    )
    parser.add_argument(
        "--tracker_name",
        type=str,
        default=DEFAULT_TRACKER_NAME,
        help="Tracker name passed to video_demo.py.",
    )
    parser.add_argument(
        "--tracker_param",
        type=str,
        default=DEFAULT_TRACKER_PARAM,
        help="Parameter YAML stem located under experiments/ostrack/.",
    )
    parser.add_argument(
        "--init_box",
        type=float,
        nargs=4,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Initial bounding box in XYXY format (pixels). Overrides --keybox_csv.",
    )
    parser.add_argument(
        "--keybox_csv",
        type=Path,
        default=Path("data/annotations/keyboxes.csv"),
        help="CSV with columns frame_id,x1,y1,x2,y2 to source init box from.",
    )
    parser.add_argument(
        "--keybox_frame",
        type=int,
        default=DEFAULT_KEYBOX_FRAME,
        help="frame_id to use from --keybox_csv when --init_box is omitted.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite exported txt/csv if they already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.init_box is not None:
        init_box = args.init_box
    else:
        init_box = load_box_from_csv(args.keybox_csv, args.keybox_frame)

    result = run_ostrack_tracking(
        video_path=args.video,
        output_dir=args.output_dir,
        init_box_xyxy=init_box,
        tracker_name=args.tracker_name,
        tracker_param=args.tracker_param,
        ostrack_root=args.ostrack_root,
        overwrite=args.overwrite,
    )
    print(f"[ok] exported track: {result.exported_track_file}")
    print(f"[ok] csv summary:   {result.exported_track_csv}")


if __name__ == "__main__":
    main()

