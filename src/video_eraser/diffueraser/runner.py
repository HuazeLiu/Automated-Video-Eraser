import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional


def run_diffueraser(
    diffueraser_root: Path,
    input_video: Path,
    input_mask: Path,
    output_dir: Path,
    python_bin: str = sys.executable,
    video_length: Optional[int] = None,
    mask_dilation_iter: Optional[int] = None,
    max_img_size: Optional[int] = None,
    base_model_path: Optional[Path] = None,
    vae_path: Optional[Path] = None,
    diffueraser_weights: Optional[Path] = None,
    propainter_weights: Optional[Path] = None,
    ref_stride: Optional[int] = None,
    neighbor_length: Optional[int] = None,
    subvideo_length: Optional[int] = None,
) -> Path:
    """Invoke DiffuEraser with the provided mask/video."""

    diffueraser_root = diffueraser_root.resolve()
    script = diffueraser_root / "run_diffueraser.py"
    if not script.exists():
        raise FileNotFoundError(
            f"DiffuEraser entry script missing at {script}. "
            "Clone https://github.com/lixiaowen-xw/DiffuEraser.git into third_party/DiffuEraser."
        )

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        python_bin,
        str(script),
        "--input_video",
        str(input_video.resolve()),
        "--input_mask",
        str(input_mask.resolve()),
        "--save_path",
        str(output_dir),
    ]

    def _append(flag: str, value: Optional[object]):
        if value is None:
            return
        cmd.extend([flag, str(value)])

    _append("--video_length", video_length)
    _append("--mask_dilation_iter", mask_dilation_iter)
    _append("--max_img_size", max_img_size)
    _append("--base_model_path", base_model_path)
    _append("--vae_path", vae_path)
    _append("--diffueraser_path", diffueraser_weights)
    _append("--propainter_model_dir", propainter_weights)
    _append("--ref_stride", ref_stride)
    _append("--neighbor_length", neighbor_length)
    _append("--subvideo_length", subvideo_length)

    subprocess.run(cmd, check=True, cwd=diffueraser_root)

    final_video = output_dir / "diffueraser_result.mp4"
    if not final_video.exists():
        raise FileNotFoundError(
            f"DiffuEraser finished but did not create {final_video}. "
            "Check stdout for errors."
        )
    return final_video


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Thin wrapper around DiffuEraser's run_diffueraser.py"
    )
    parser.add_argument("--diffueraser_root", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument(
        "--save_path",
        type=Path,
        default=Path("outputs/videos"),
        help="Where DiffuEraser will drop its result files.",
    )
    parser.add_argument("--video_length", type=int)
    parser.add_argument("--mask_dilation_iter", type=int)
    parser.add_argument("--max_img_size", type=int)
    parser.add_argument("--base_model_path", type=Path)
    parser.add_argument("--vae_path", type=Path)
    parser.add_argument("--diffueraser_path", type=Path)
    parser.add_argument("--propainter_model_dir", type=Path)
    parser.add_argument("--ref_stride", type=int)
    parser.add_argument("--neighbor_length", type=int)
    parser.add_argument("--subvideo_length", type=int)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    final_path = run_diffueraser(
        diffueraser_root=args.diffueraser_root,
        input_video=args.video,
        input_mask=args.mask,
        output_dir=args.save_path,
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
    print(f"[ok] DiffuEraser output ready at {final_path}")


if __name__ == "__main__":
    main()

