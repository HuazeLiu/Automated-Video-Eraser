import argparse
from pathlib import Path
from typing import Optional, Sequence

from video_eraser.diffueraser.runner import run_diffueraser
from video_eraser.segmentation.sam2_mask_generator import generate_mask_video
from video_eraser.tracking.ostrack_runner import (
    DEFAULT_KEYBOX_FRAME,
    run_ostrack_tracking,
    load_box_from_csv,
)
from video_eraser.utils.video_checks import assert_same_video_geometry


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-end pipeline: OSTrack -> SAM2 -> DiffuEraser."
    )
    parser.add_argument("--video", type=Path, required=True, help="Input mp4.")
    parser.add_argument(
        "--keybox_csv",
        type=Path,
        default=Path("data/annotations/keyboxes.csv"),
        help="CSV with manual bounding boxes (frame_id,x1,y1,x2,y2).",
    )
    parser.add_argument(
        "--keybox_frame",
        type=int,
        default=DEFAULT_KEYBOX_FRAME,
        help="frame_id inside keybox_csv used to seed OSTrack.",
    )
    parser.add_argument(
        "--init_box",
        type=float,
        nargs=4,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Manual bounding box override in XYXY format.",
    )
    parser.add_argument(
        "--tracker_param",
        type=str,
        default="vitb_384_mae_ce_32x4_ep300",
        help="Name of OSTrack param yaml (without extension).",
    )
    parser.add_argument(
        "--ostrack_root",
        type=Path,
        default=Path("third_party/OSTrack"),
        help="Path to OSTrack checkout.",
    )
    parser.add_argument(
        "--tracks_dir",
        type=Path,
        default=Path("data/interim/tracks"),
        help="Where to store OSTrack outputs.",
    )
    parser.add_argument(
        "--mask_video",
        type=Path,
        default=Path("data/processed/masks/pipeline_mask.mp4"),
        help="Destination for the binary mask video.",
    )
    parser.add_argument(
        "--mask_frames_dir",
        type=Path,
        default=Path("data/interim/masks"),
        help="Directory for mask PNGs (optional).",
    )
    parser.add_argument(
        "--sam2_model_id",
        type=str,
        default="facebook/sam2-hiera-small",
        help="HuggingFace id for SAM2 weights.",
    )
    parser.add_argument(
        "--diffueraser_root",
        type=Path,
        default=Path("third_party/DiffuEraser"),
        help="Folder containing the DiffuEraser repo.",
    )
    parser.add_argument(
        "--diffueraser_output",
        type=Path,
        default=Path("outputs/videos"),
        help="Directory where DiffuEraser will write diffueraser_result.mp4",
    )
    parser.add_argument(
        "--video_length",
        type=int,
        help="Optional override for DiffuEraser's --video_length argument.",
    )
    parser.add_argument("--max_img_size", type=int)
    parser.add_argument("--mask_dilation_iter", type=int)
    parser.add_argument("--base_model_path", type=Path)
    parser.add_argument("--vae_path", type=Path)
    parser.add_argument("--diffueraser_path", type=Path)
    parser.add_argument("--propainter_model_dir", type=Path)
    parser.add_argument("--ref_stride", type=int)
    parser.add_argument("--neighbor_length", type=int)
    parser.add_argument("--subvideo_length", type=int)
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Run tracking + masking only, skip DiffuEraser.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.init_box is not None:
        init_box = args.init_box
    else:
        init_box = load_box_from_csv(args.keybox_csv, args.keybox_frame)

    track_result = run_ostrack_tracking(
        video_path=args.video,
        output_dir=args.tracks_dir,
        init_box_xyxy=init_box,
        tracker_param=args.tracker_param,
        ostrack_root=args.ostrack_root,
        overwrite=True,
    )

    mask_video = generate_mask_video(
        video_path=args.video,
        track_file=track_result.exported_track_file,
        output_mask_video=args.mask_video,
        frames_dir=args.mask_frames_dir,
        model_id=args.sam2_model_id,
    )

    if args.dry_run:
        print("[info] dry-run enabled: skip DiffuEraser stage.")
        return

    assert_same_video_geometry(args.video, mask_video)

    final_video = run_diffueraser(
        diffueraser_root=args.diffueraser_root,
        input_video=args.video,
        input_mask=mask_video,
        output_dir=args.diffueraser_output,
        video_length=args.video_length,
        mask_dilation_iter=args.mask_dilation_iter,
        max_img_size=args.max_img_size,
        base_model_path=args.base_model_path,
        vae_path=args.vae_path,
        diffueraser_weights=args.diffueraser_path,
        propainter_weights=args.propainter_model_dir,
        ref_stride=args.ref_stride,
        neighbor_length=args.neighbor_length,
        subvideo_length=args.subvideo_length,
    )
    print(f"[ok] DiffuEraser video ready at {final_video}")


if __name__ == "__main__":
    main()

