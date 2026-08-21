#!/usr/bin/env python3
"""Evaluate a trained model on the validation split or a custom dataset.

Prints mAP50, mAP50-95, precision, recall, and writes per-class metrics.
"""

import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("weights", type=Path, help="trained model weights (.pt)")
    ap.add_argument("--data", default=None, help="data.yaml (default: merged/data.yaml)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="")
    ap.add_argument("--output", default=None, help="save predictions + confusion matrix")
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    data_path = args.data or str(ROOT / cfg["data"]["merged_dir"] / "data.yaml")
    out = ROOT / "runs" / "eval" if args.output is None else Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(args.weights))
    metrics = model.val(
        data=data_path,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device or None,
        project=str(out),
        name="val",
        exist_ok=True,
    )
    print(f"\nmAP50:   {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")

    if metrics.box.ap_class_index is not None:
        classes = cfg["data"]["classes"]
        print("\nPer-class (may be in training-id order):")
        # metrics.box.ap is a list of per-class APs
        for i, ap50 in enumerate(getattr(metrics.box, "ap50", []) or []):
            cls_idx = int(metrics.box.ap_class_index[i]) if hasattr(metrics.box, 'ap_class_index') and metrics.box.ap_class_index is not None else i
            label = classes[cls_idx] if cls_idx < len(classes) else f"class_{cls_idx}"
            print(f"  {label}: mAP50={ap50:.4f}")


if __name__ == "__main__":
    main()