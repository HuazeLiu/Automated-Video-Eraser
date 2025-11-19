import argparse
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from tqdm import tqdm


def _build_predictor(
    model_id: Optional[str],
    config_file: Optional[str],
    checkpoint: Optional[str],
    device: str,
) -> SAM2ImagePredictor:
    if model_id:
        return SAM2ImagePredictor.from_pretrained(model_id, device=device)
    if not (config_file and checkpoint):
        raise ValueError(
            "Either `model_id` or (`config_file`, `checkpoint`) must be provided."
        )
    sam_model = build_sam2(config_file=config_file, ckpt_path=checkpoint, device=device)
    return SAM2ImagePredictor(sam_model)


def _ensure_numpy_boxes(box_file: Path) -> np.ndarray:
    arr = np.loadtxt(box_file, delimiter="\t", dtype=float)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[1] != 4:
        raise ValueError(f"{box_file} should contain 4 columns (x, y, w, h)")
    return arr


def _prepare_writer(output_path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output_path), fourcc, fps, (width, height), isColor=True
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create VideoWriter for {output_path}")
    return writer


def _select_mask(masks: np.ndarray, scores: np.ndarray, threshold: float) -> np.ndarray:
    idx = int(np.argmax(scores))
    chosen = masks[idx]
    if chosen.dtype != np.bool_:
        chosen = chosen > threshold
    return chosen.astype(np.uint8) * 255


def generate_mask_video(
    video_path: Path,
    track_file: Path,
    output_mask_video: Path,
    frames_dir: Optional[Path] = None,
    model_id: str = "facebook/sam2-hiera-small",
    config_file: Optional[str] = None,
    checkpoint: Optional[str] = None,
    device: Optional[str] = None,
    threshold: float = 0.0,
) -> Path:
    """Create a binary mask video guided by per-frame bounding boxes."""

    video_path = video_path.resolve()
    track_file = track_file.resolve()
    output_mask_video = output_mask_video.resolve()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    boxes_xywh = _ensure_numpy_boxes(track_file)
    if len(boxes_xywh) != frame_count:
        raise ValueError(
            f"Tracking file {track_file} has {len(boxes_xywh)} boxes but "
            f"video {video_path} has {frame_count} frames."
        )

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    predictor = _build_predictor(model_id, config_file, checkpoint, device=device)
    predictor.model.to(device)

    writer = _prepare_writer(output_mask_video, fps, width, height)
    if frames_dir:
        frames_dir = frames_dir.resolve()
        frames_dir.mkdir(parents=True, exist_ok=True)

    try:
        for idx in tqdm(range(frame_count), desc="Generating SAM2 masks"):
            ret, frame_bgr = cap.read()
            if not ret:
                raise RuntimeError(f"Failed to read frame {idx} from {video_path}")
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            predictor.set_image(frame_rgb)

            x, y, w, h = boxes_xywh[idx]
            box_xyxy = np.array([x, y, x + w, y + h], dtype=np.float32)
            masks, scores, _ = predictor.predict(
                box=box_xyxy, multimask_output=False, normalize_coords=True
            )
            mask_img = _select_mask(masks, scores, threshold)

            mask_bgr = cv2.cvtColor(mask_img, cv2.COLOR_GRAY2BGR)
            writer.write(mask_bgr)

            if frames_dir:
                frame_out = frames_dir / f"{idx:06d}.png"
                cv2.imwrite(str(frame_out), mask_img)
    finally:
        writer.release()
        cap.release()

    return output_mask_video


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use SAM2 to turn OSTrack bounding boxes into a binary mask video."
    )
    parser.add_argument("--video", type=Path, required=True, help="Input video.")
    parser.add_argument(
        "--track_file",
        type=Path,
        required=True,
        help="Tab-delimited x,y,w,h file exported by run_ostrack_tracking.",
    )
    parser.add_argument(
        "--output_mask",
        type=Path,
        default=Path("data/processed/masks/sam2_mask.mp4"),
        help="Path for the binary mask video (.mp4).",
    )
    parser.add_argument(
        "--frames_dir",
        type=Path,
        help="Optional directory to also export per-frame PNG masks.",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="facebook/sam2-hiera-small",
        help="HuggingFace model id. Ignored when --config_file/--checkpoint are set.",
    )
    parser.add_argument("--config_file", type=str, help="Local SAM2 YAML config.")
    parser.add_argument("--checkpoint", type=str, help="Local SAM2 checkpoint path.")
    parser.add_argument(
        "--device",
        type=str,
        help="torch device string (default: cuda if available else cpu).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Mask threshold (applied only if predictor outputs logits).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    generate_mask_video(
        video_path=args.video,
        track_file=args.track_file,
        output_mask_video=args.output_mask,
        frames_dir=args.frames_dir,
        model_id=args.model_id,
        config_file=args.config_file,
        checkpoint=args.checkpoint,
        device=args.device,
        threshold=args.threshold,
    )
    print(f"[ok] wrote mask video to {args.output_mask}")


if __name__ == "__main__":
    main()

