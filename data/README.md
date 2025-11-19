## Data Directory Guide

- `raw/`: Original inputs (e.g., source videos straight from capture). Create subfolders like `videos/` as needed. These files stay local and are ignored by Git.
- `interim/`: Working artifacts generated during preprocessing, such as decoded `frames/` and detection `labels/`.
- `processed/`: Reusable outputs including refined masks or inpainted clips (e.g., binary mask videos under `masks/`).
- `annotations/`: Aggregated supervision files such as `keyboxes.csv`, evaluation metrics, or manual notes.
- `external/`: Snapshots of public datasets or vendor exports (e.g., Ultralytics jobs, Moose segmentation set) that you reference but do not redistribute.

> Each directory ships with a `.gitkeep` placeholder so the structure is versioned while the actual data stays on your machine.

