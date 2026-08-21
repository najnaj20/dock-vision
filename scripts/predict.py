#!/usr/bin/env python3
"""Run inference on an image, video, or collection, with counting overlay.

Output: annotated image/video with bounding boxes and a per-frame pallet/box
count overlay. If --csv is set, also writes frame counts to CSV.
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
PALLET_RGB = (200, 90, 50)
BOX_RGB = (90, 150, 210)


def draw_counts(img, pallets, boxes):
    """Overlay translucent colored boxes + count banner + legend."""
    overlay = np.zeros_like(img)
    for x1, y1, x2, y2 in pallets:
        cv2.rectangle(overlay, (x1, y1), (x2, y2), PALLET_RGB, -1)
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(overlay, (x1, y1), (x2, y2), BOX_RGB, -1)
    cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)

    line = f"pallets: {len(pallets)}  boxes: {len(boxes)}"
    (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_DUPLEX, 0.7, 2)
    cv2.rectangle(img, (8, 8), (8 + tw + 12, 8 + th + 12), (20, 20, 20), -1)
    cv2.putText(img, line, (14, 8 + th + 6), cv2.FONT_HERSHEY_DUPLEX, 0.7, (240, 240, 240), 2)

    leg_y = 8 + th + 22
    for i, (color, label) in enumerate(((PALLET_RGB, "pallet"), (BOX_RGB, "box"))):
        y0 = leg_y + i * 20
        cv2.rectangle(img, (14, y0), (34, y0 + 14), color, -1)
        cv2.putText(img, label, (40, y0 + 12), cv2.FONT_HERSHEY_DUPLEX, 0.5, (200, 200, 200), 1)
    return img


def run_inference(model, img):
    """Run model on an image, return (annotated_img, pallet_bboxes, box_bboxes)."""
    results = model(img, verbose=False)[0]
    pallets, boxes = [], []
    for det in results.boxes.data.cpu().numpy():
        x1, y1, x2, y2, conf, cls = det
        cls = int(cls)
        if conf < 0.25:
            continue
        rect = (int(x1), int(y1), int(x2), int(y2))
        if cls == 0:
            pallets.append(rect)
        else:
            boxes.append(rect)
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        cv2.putText(img, f"{conf:.2f}", (int(x1), max(4, int(y1) - 4)),
                    cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 255, 0), 1)
    draw_counts(img, pallets, boxes)
    return img, pallets, boxes


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", required=True, type=Path)
    ap.add_argument("--source", required=True, help="image, video, or folder")
    ap.add_argument("--output", default=None, help="output file or folder")
    ap.add_argument("--csv", default=None, help="write frame counts to CSV")
    ap.add_argument("--device", default="")
    args = ap.parse_args()

    model = YOLO(str(args.weights))
    source = Path(args.source)
    csv_fh = csv_writer = None
    if args.csv:
        csv_fh = open(args.csv, "w", newline="")
        csv_writer = csv.writer(csv_fh)
        csv_writer.writerow(["file", "pallets", "boxes"])

    if source.suffix.lower() in (".jpg", ".jpeg", ".png"):
        out = Path(args.output) if args.output else ROOT / "runs" / "predict" / source.name
        out.parent.mkdir(parents=True, exist_ok=True)
        img = cv2.imread(str(source))
        if img is None:
            raise SystemExit(f"cannot read {source}")
        _, n_pal, n_box = run_inference(model, img)
        cv2.imwrite(str(out), img)
        if csv_writer:
            csv_writer.writerow([source.name, n_pal, n_box])
        print(f"{source.name}: {n_pal} pallets, {n_box} boxes -> {out}")

    elif source.is_dir():
        out_dir = Path(args.output) if args.output else ROOT / "runs" / "predict"
        out_dir.mkdir(parents=True, exist_ok=True)
        for img_path in sorted(source.glob("*")):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            _, n_pal, n_box = run_inference(model, img)
            out_path = out_dir / f"{img_path.stem}_pred{img_path.suffix}"
            cv2.imwrite(str(out_path), img)
            if csv_writer:
                csv_writer.writerow([img_path.name, n_pal, n_box])
            print(f"{img_path.name}: {n_pal} pallets, {n_box} boxes")

    else:  # video
        out = Path(args.output) if args.output else ROOT / "runs" / "predict" / f"{source.stem}_pred.mp4"
        out.parent.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            raise SystemExit(f"cannot open {source}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame, n_pal, n_box = run_inference(model, frame)
            writer.write(frame)
            if csv_writer:
                csv_writer.writerow([f"frame_{frame_idx:06d}", n_pal, n_box])
            frame_idx += 1
            if frame_idx % 30 == 0:
                print(f"  processed {frame_idx} frames")
        cap.release()
        writer.release()
        print(f"{source.name}: {frame_idx} frames -> {out}")

    if csv_fh:
        csv_fh.close()
        print(f"counts -> {args.csv}")


if __name__ == "__main__":
    main()