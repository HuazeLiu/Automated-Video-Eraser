import os, glob, argparse, csv
import cv2
import numpy as np

def parse_indices(arg, n_frames):
    if not arg:
        return []
    idx = []
    for tok in arg.split(","):
        tok = tok.strip()
        if "-" in tok:
            a,b = tok.split("-")
            idx.extend(range(int(a), int(b)+1))
        else:
            idx.append(int(tok))
    idx = [i for i in idx if 1 <= i <= n_frames]
    return sorted(set(idx))

def sample_every(n_frames, step):
    if step and step > 0:
        return list(range(1, n_frames+1, step))
    return []

def read_yolo_txt(txt_path):
    """
    Returns list of (cls, cx, cy, w, h) in normalized coords, or [] if none.
    """
    if not os.path.isfile(txt_path):
        return []
    boxes = []
    with open(txt_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            cx = float(parts[1]); cy = float(parts[2])
            w  = float(parts[3]); h  = float(parts[4])
            boxes.append((cls, cx, cy, w, h))
    return boxes

def yolo_to_xyxy_norm(cx, cy, w, h):
    x1 = cx - w/2.0
    y1 = cy - h/2.0
    x2 = cx + w/2.0
    y2 = cy + h/2.0
    return x1, y1, x2, y2

def norm_to_pixels(x1n, y1n, x2n, y2n, W, H):
    x1 = int(round(x1n * W))
    y1 = int(round(y1n * H))
    x2 = int(round(x2n * W))
    y2 = int(round(y2n * H))
    # clamp and ensure valid
    x1 = max(0, min(x1, W-1)); y1 = max(0, min(y1, H-1))
    x2 = max(0, min(x2, W-1)); y2 = max(0, min(y2, H-1))
    if x2 <= x1: x2 = min(W-1, x1+1)
    if y2 <= y1: y2 = min(H-1, y1+1)
    return x1, y1, x2, y2

def main():
    ap = argparse.ArgumentParser(description="Convert YOLO txt labels to keyboxes.csv for selected key frames.")
    ap.add_argument("--frames", required=True, help="Frames dir (e.g., frames/ with %06d.png)")
    ap.add_argument("--labels", required=True, help="Labels dir with YOLO txt files matching frame names")
    ap.add_argument("--out_csv", default="keyboxes.csv", help="Output CSV file")
    ap.add_argument("--class_id", type=int, default=None, help="If set, only keep this class id")
    ap.add_argument("--indices", default=None, help="1-based frames list/ranges, e.g. '1,60,120-180'")
    ap.add_argument("--every", type=int, default=0, help="Take every Nth frame as keyframe")
    args = ap.parse_args()

    # collect frames (supports .png/.jpg)
    frame_paths = sorted(glob.glob(os.path.join(args.frames, "*.png")) +
                         glob.glob(os.path.join(args.frames, "*.jpg")) +
                         glob.glob(os.path.join(args.frames, "*.jpeg")))
    if not frame_paths:
        raise FileNotFoundError(f"No frames found in {args.frames}")
    n_frames = len(frame_paths)

    # build key-frame set
    key_ids = set(parse_indices(args.indices, n_frames))
    key_ids.update(sample_every(n_frames, args.every))
    if not key_ids:
        # default: first, middle, last
        key_ids.update([1, (n_frames+1)//2, n_frames])
    key_ids = sorted(key_ids)

    # assume labels follow same numbering pattern (e.g., 000001.txt)
    def label_path_for(fid):
        base = os.path.basename(frame_paths[fid-1])
        stem, _ = os.path.splitext(base)
        return os.path.join(args.labels, stem + ".txt")

    # write CSV
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["frame_id", "x1", "y1", "x2", "y2"])

        for fid in key_ids:
            img = cv2.imread(frame_paths[fid-1])
            if img is None:
                print(f"[warn] cannot read frame {fid}, skip")
                continue
            H, W = img.shape[:2]

            txt = label_path_for(fid)
            boxes = read_yolo_txt(txt)
            if not boxes:
                print(f"[warn] no labels in {txt}, skip")
                continue

            # filter by class if requested
            if args.class_id is not None:
                boxes = [b for b in boxes if b[0] == args.class_id]
                if not boxes:
                    print(f"[warn] frame {fid} has no class {args.class_id}, skip")
                    continue

            # choose one box: prefer largest area
            best = None; best_area = -1
            for (cls, cx, cy, w, h) in boxes:
                x1n, y1n, x2n, y2n = yolo_to_xyxy_norm(cx, cy, w, h)
                x1, y1, x2, y2 = norm_to_pixels(x1n, y1n, x2n, y2n, W, H)
                area = (x2 - x1) * (y2 - y1)
                if area > best_area:
                    best_area = area
                    best = (x1, y1, x2, y2)

            if best is None:
                print(f"[warn] frame {fid} has no usable box")
                continue

            wcsv.writerow([fid, best[0], best[1], best[2], best[3]])
            print(f"[ok] frame {fid}: {best}")

    print("Saved key boxes to", args.out_csv)

if __name__ == "__main__":
    main()
