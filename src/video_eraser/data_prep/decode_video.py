import os
import subprocess
import argparse

def decode_video(video_path, output_dir, fps=None, scale=None):
    """
    Decode a video into frames using FFmpeg.

    Args:
        video_path (str): Path to input video file.
        output_dir (str): Directory to save extracted frames.
        fps (int, optional): Target frames per second (default = keep original fps).
        scale (tuple, optional): Target resolution as (width, height). Example: (1280, 720).
    """
    os.makedirs(output_dir, exist_ok=True)

    cmd = ["ffmpeg", "-i", video_path]

    # filters
    vf_filters = []
    if fps:
        vf_filters.append(f"fps={fps}")
    if scale:
        w, h = scale
        vf_filters.append(f"scale={w}:{h}")
    if vf_filters:
        cmd.extend(["-vf", ",".join(vf_filters)])

    # Output pattern
    output_pattern = os.path.join(output_dir, "%06d.png")
    cmd.append(output_pattern)

    # Run ffmpeg
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decode video into frames using FFmpeg")
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument("outdir", help="Directory to save frames")
    parser.add_argument("--fps", type=int, default=None, help="Target FPS (default: keep original)")
    parser.add_argument("--scale", nargs=2, type=int, default=None, help="Resize frames: width height")

    args = parser.parse_args()
    scale = tuple(args.scale) if args.scale else None

    decode_video(args.video, args.outdir, fps=args.fps, scale=scale)


# Keep original FPS/resolution
# python decode_video.py input.mp4 frames/

# Force 10 fps
# python decode_video.py input.mp4 frames/ --fps 10

# Resize to 1280x720
# python decode_video.py input.mp4 frames/ --scale 1280 720

# Combine FPS + resize constraints
# python decode_video.py input.mp4 frames/ --fps 10 --scale 1280 720
