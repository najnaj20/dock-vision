#!/usr/bin/env python3
"""Train YOLO on the merged dataset.

Usage:
  python scripts/train.py --model yolo11n.pt --epochs 100 --imgsz 640

For GPU training, specify --device 0  (or use the Colab notebook).
"""

import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=None, help="pretrained model (default: config)")
    ap.add_argument("--epochs", type=int, default=None, help="default from config")
    ap.add_argument("--imgsz", type=int, default=None, help="default from config")
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--data", default=None, help="data.yaml path")
    ap.add_argument("--device", default="", help="e.g. 0 for first GPU, or leave empty for CPU")
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    t_cfg = cfg["training"]
    model_path = args.model or t_cfg["model"]
    data_path = args.data or str(ROOT / cfg["data"]["merged_dir"] / "data.yaml")

    model = YOLO(model_path)
    results = model.train(
        data=data_path,
        epochs=args.epochs or t_cfg["epochs"],
        imgsz=args.imgsz or t_cfg["imgsz"],
        batch=args.batch or t_cfg["batch"],
        device=args.device or None,
        project=str(ROOT / "runs"),
        name="train",
        exist_ok=True,
    )
    print(f"Training complete. Best weights at runs/train/weights/best.pt")


if __name__ == "__main__":
    main()