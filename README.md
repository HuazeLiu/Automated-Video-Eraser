# Video Eraser Research Pipeline

End-to-end toolkit for removing foreground objects from video by combining
traditional preprocessing (frame decoding, YOLO box extraction) with mask
generation utilities and optional OSTrack-based tracking experiments.

## Repository Layout

- `src/video_eraser/`: Python modules
  - `data_prep/`: frame decoding and YOLO label conversion scripts
  - `mask_tools/`: utilities for turning segmentation frames into mask videos
- `data/`: standardized storage for raw/interim/processed datasets (ignored by Git)
- `outputs/`: local experiments and rendered results (ignored by Git)
- `docs/media/`: tiny demo clips referenced by this README (`moose_before.mp4`,
  `moose_after.mp4`)
- `third_party/OSTrack/`: optional vendor tracker code (ignored by Git)

## Prerequisites

- Python 3.9+
- FFmpeg available on `PATH` (for `decode_video.py`)
- `pip install -r requirements.txt`
- (Optional) Install OSTrack dependencies inside `third_party/OSTrack/` per the
  upstream README if you plan to run the tracker.

## Typical Workflow

1. **Decode source video**
   ```bash
   python src/video_eraser/data_prep/decode_video.py data/raw/videos/video_2.mp4 data/interim/frames --fps 10
   ```
2. **Generate key boxes from YOLO detections**
   ```bash
   python src/video_eraser/data_prep/yolo_txts_to_keyboxes.py \
       --frames data/interim/frames \
       --labels data/interim/labels \
       --out_csv data/annotations/keyboxes.csv \
       --every 30
   ```
3. **Convert segmentation frames into a mask video**
   ```bash
   python src/video_eraser/mask_tools/make_mask_video.py \
       data/interim/masks \
       outputs/videos/mask.mp4 \
       --fps 24 --threshold 1
   ```
4. **(Optional) Track objects with OSTrack**
   - Place OSTrack under `third_party/OSTrack/`
   - Follow their `install.sh` or conda environment instructions
   - Use `keyboxes.csv` to seed tracker initialization per your experiment.

## Moose Removal Demo

Add lightweight clips to `docs/media/` (see `docs/media/README.md`). GitHub only
renders the `<video>` tag when the file is accessed via `?raw=1`, so keep the
filenames below and make sure they remain small enough for fast loading:

<table>
  <tr>
    <th>Before (original Moose)</th>
    <th>After (Moose removed)</th>
  </tr>
  <tr>
    <td>
      <video src="docs/media/moose_before.mp4?raw=1" controls width="320">
        Your browser does not support embedded videos.
      </video>
    </td>
    <td>
      <video src="docs/media/moose_after.mp4?raw=1" controls width="320">
        Your browser does not support embedded videos.
      </video>
    </td>
  </tr>
</table>

## GitHub Publishing Checklist

1. Run `git status` to confirm only code/docs are tracked (no heavy data).
2. Commit your changes:
   ```bash
   git add .
   git commit -m "Initial research pipeline structure"
   ```
3. Create a new GitHub repository (via the UI) and copy its HTTPS/SSH URL.
4. Connect and push:
   ```bash
   git branch -M main
   git remote add origin <your-repo-url>
   git push -u origin main
   ```
5. For future experiments, keep raw/interim/processed data under `data/` and
   `outputs/`, and only place tiny demo clips under `docs/media/` when you need
   to showcase results publicly.

## License

Add your preferred license text here before publishing (MIT, Apache-2.0, etc.).

