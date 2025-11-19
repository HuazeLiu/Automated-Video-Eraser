# Video Eraser Pipeline

This repo bundles a reproducible workflow for removing a target object from video:

1. Sample key frames / bounding boxes (YOLO or manual csv).
2. Run OSTrack to produce per-frame bounding boxes.
3. Prompt SAM2 with those boxes to obtain binary mask videos.
4. Feed the original video + binary mask to [DiffuEraser](https://github.com/lixiaowen-xw/DiffuEraser) for high-quality diffusion-based inpainting.

## Repository Layout

- `src/video_eraser/`
  - `data_prep/`: FFmpeg wrappers and YOLO-to-keybox utilities.
  - `tracking/`: `ostrack_runner.py` ensures the tracker exports `x,y,w,h` files.
  - `segmentation/`: `sam2_mask_generator.py` wraps the SAM2 image predictor.
  - `diffueraser/`: command-line wrapper around the upstream inference script.
  - `pipelines/`: `full_pipeline.py` stitches everything together.
  - `mask_tools/`: legacy utilities for assembling binary masks from PNGs.
- `data/`: opinionated storage layout (`raw/`, `interim/frames+labels+tracks+masks`, `processed/`, `annotations/`, `external/`).
- `docs/media/`: tiny demo clips referenced below (kept under 20 MB).
- `third_party/`: clone-only directories such as OSTrack and DiffuEraser (ignored by Git).

## Environment & Dependencies

1. **Python + FFmpeg**
   ```bash
   # Python 3.10+ recommended
   ffmpeg -version   # ensure available on PATH
   ```
2. **PyTorch** – install the wheel that matches your CUDA stack *before* the rest:
   ```bash
   # Example for CUDA 12.1 (adjust per https://pytorch.org)
   pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
   ```
3. **Project requirements** – SAM2, OpenCV, pandas, etc.
   ```bash
   pip install -r requirements.txt
   ```
4. **Expose the `src/` layout** – either install the package (`pip install -e .`) or set `PYTHONPATH=src` before running the CLI modules:
   ```powershell
   # PowerShell
   $env:PYTHONPATH="src"
   # bash/zsh
   export PYTHONPATH=src
   ```

### Third-Party Components

| Component | Setup |
| --- | --- |
| **OSTrack** | `git clone https://github.com/botaoye/OSTrack third_party/OSTrack` and download the checkpoint referenced by `experiments/ostrack/<param>.yaml` into `third_party/OSTrack/checkpoints/train/ostrack/<param>/OSTrack_epXXXX.pth.tar`. The wrapper auto-generates `lib/test/evaluation/local.py`. |
| **SAM2** | Already installed via `pip install sam2`. Hugging Face weights (e.g., `facebook/sam2-hiera-small`) are pulled on first use; run `huggingface-cli login` if needed. |
| **DiffuEraser** | `git clone https://github.com/lixiaowen-xw/DiffuEraser third_party/DiffuEraser`. Follow their README to download Stable-Diffusion 1.5, VAE, DiffuEraser weights, PCM/ProPainter checkpoints, etc. Our wrapper simply calls `run_diffueraser.py` with proper arguments. |

## Stage-by-Stage CLI

Below commands assume a source video at `data/raw/videos/moose.mp4`.

1. **Decode video (optional)**  
   ```bash
   python src/video_eraser/data_prep/decode_video.py \
       data/raw/videos/moose.mp4 data/interim/frames/moose --fps 10
   ```

2. **Summarize YOLO labels into key boxes**  
   ```bash
   python src/video_eraser/data_prep/yolo_txts_to_keyboxes.py \
       --frames data/interim/frames/moose \
       --labels data/interim/labels/moose \
       --out_csv data/annotations/keyboxes.csv
   ```

3. **Run OSTrack with a known key frame**  
   ```bash
   python -m video_eraser.tracking.ostrack_runner \
       --video data/raw/videos/moose.mp4 \
       --keybox_csv data/annotations/keyboxes.csv \
       --keybox_frame 1 \
       --ostrack_root third_party/OSTrack \
       --tracker_param vitb_384_mae_ce_32x4_ep300 \
       --output_dir data/interim/tracks
   ```
   Produces `data/interim/tracks/moose_ostrack.txt` (tab-delimited `x y w h` per frame) and a CSV summary.

4. **Generate SAM2 masks from tracking results**  
   ```bash
   python -m video_eraser.segmentation.sam2_mask_generator \
       --video data/raw/videos/moose.mp4 \
       --track_file data/interim/tracks/moose_ostrack.txt \
       --output_mask data/processed/masks/moose_sam2.mp4 \
       --frames_dir data/interim/masks/moose \
       --model_id facebook/sam2-hiera-small
   ```
   The output video is single-channel (expanded to RGB) and matches fps/resolution of the original clip.

5. **Invoke DiffuEraser**  
   ```bash
   python -m video_eraser.diffueraser.runner \
       --diffueraser_root third_party/DiffuEraser \
       --video data/raw/videos/moose.mp4 \
       --mask data/processed/masks/moose_sam2.mp4 \
       --save_path outputs/videos \
       --video_length 10 \
       --max_img_size 960
   ```
   See the upstream README for additional arguments (e.g., base model path, Propainter weights).

6. **Full pipeline orchestration**  
   ```bash
   python -m video_eraser.pipelines.full_pipeline \
       --video data/raw/videos/moose.mp4 \
       --keybox_csv data/annotations/keyboxes.csv \
       --keybox_frame 1 \
       --ostrack_root third_party/OSTrack \
       --diffueraser_root third_party/DiffuEraser \
       --mask_video data/processed/masks/moose_sam2.mp4 \
       --diffueraser_output outputs/videos \
       --video_length 10
   ```
   Use `--dry_run` if you only want tracking + mask generation.

### Post-processing (Ghosting Suppression)

Residual flicker can be reduced via the temporal smoother, which registers neighboring frames (ECC) and blends them with Gaussian weights. You can restrict blending to the mask region or apply it globally:

```bash
python -m video_eraser.postprocess.temporal_smoothing \
    --video outputs/videos/diffueraser_result.mp4 \
    --mask data/processed/masks/moose_sam2.mp4 \
    --window 5 --sigma 1.2 --strength 0.8
```

The Moose demo below already uses `window=5`, `sigma=1.2`, `strength=0.8` without a mask to tone down the most obvious ghosting.

### Alignment Reminder

DiffuEraser expects the binary mask video to have the **exact** same frame count, fps, and resolution as the input video. The `sam2_mask_generator` enforces this, and `video_eraser.utils.video_checks.assert_same_video_geometry` is called right before launching DiffuEraser.

## Moose Removal Demo

The following demo video is from Prof. Rachel Mayeri and Prof. Calden Wloka.

| Before (original Moose) | After (Moose removed) |
| --- | --- |
| <img src="docs/media/moose_before.gif?raw=1" width="420" alt="Moose before" /> <br> [download MP4](docs/media/moose_before.mp4?raw=1) | <img src="docs/media/moose_after.gif?raw=1" width="420" alt="Moose after" /> <br> [download MP4](docs/media/moose_after.mp4?raw=1) |

## Pushing Checklist

1. Confirm only lightweight assets are tracked: `git status`.
2. Commit & push:
   ```bash
   git add .
   git commit -m "feat: add full video eraser pipeline"
   git push origin main
   ```
3. Keep large datasets, checkpoints, and videos under `data/` or `outputs/` (ignored by Git). Only tiny demo clips live under `docs/media/`.
