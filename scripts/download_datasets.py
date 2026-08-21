#!/usr/bin/env python3
"""Ingest downloaded detection datasets into data/raw/<name>/.

Two entry points:
  python scripts/download_datasets.py --zip path/to/pallet_dataset.zip [--name foo]
  python scripts/download_datasets.py --list

Roboflow "download as YOLOv8" zips share the layout
  <project>-<version>/{train,valid,test}/{images,labels}/ + data.yaml
so we normalize any of them to data/raw/<name>/{images,labels} in YOLO format.

Class remapping: each dataset numbers its classes in its own order, so label
files are rewritten against the canonical order in config.yaml (pallet=0,
box=1). Classes outside the canonical list are dropped with a warning; that
keeps the merged dataset coherent across sources.
"""

import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

RECOMMENDED = [
    "https://universe.roboflow.com/projects-fsdqm/pallet-detection-and-count",
    "https://universe.roboflow.com/roboflow-0aslh/pallet-detection-1zlvn",
    "https://universe.roboflow.com/industrialdesign/boxes_and_pallet-r73ja",
]


def _split_class_names(data_yaml: dict):
    names = data_yaml.get("names", {})
    if isinstance(names, dict):
        return {int(k): v for k, v in names.items()}
    return {i: v for i, v in enumerate(names)}


def ingest_zip(zip_path: Path, name: str, canonical: list[str]):
    out_dir = ROOT / "data" / "raw" / name
    img_dir, lbl_dir = out_dir / "images", out_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp)
        root = Path(tmp)
        # the zip may wrap everything in one project folder
        subdirs = [p for p in root.iterdir() if p.is_dir()]
        project = subdirs[0] if len(subdirs) == 1 else root
        data_yaml = yaml.safe_load((project / "data.yaml").read_text())
        src_names = _split_class_names(data_yaml)
        canon_idx = {c: i for i, c in enumerate(canonical)}
        remap = {}
        for idx, cls in src_names.items():
            if cls in canon_idx:
                remap[idx] = canon_idx[cls]
            else:
                print(f"  dropping class '{cls}' (not in canonical {canonical})")

        n_imgs = 0
        for split in ("train", "valid", "test"):
            split_dir = project / split
            if not split_dir.exists():
                continue
            for img_path in sorted((split_dir / "images").glob("*")):
                if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                    continue
                lbl_path = split_dir / "labels" / f"{img_path.stem}.txt"
                if not lbl_path.exists():
                    continue
                dst_img = img_dir / f"{name}_{img_path.name}"
                shutil.copy2(img_path, dst_img)
                lines = []
                for line in lbl_path.read_text().splitlines():
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    cls = int(parts[0])
                    if cls not in remap:
                        continue
                    lines.append(" ".join([str(remap[cls]), *parts[1:]]))
                (lbl_dir / f"{name}_{img_path.stem}.txt").write_text("\n".join(lines) + "\n")
                n_imgs += 1

    print(f"Ingested {n_imgs} images -> {out_dir}")
    print(f"  source classes: {src_names}")
    print(f"  class remap: {remap}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zip", type=Path, help="downloaded dataset zip (Roboflow YOLOv8 format)")
    ap.add_argument("--name", default=None, help="output folder name under data/raw/")
    ap.add_argument("--list", action="store_true", help="print recommended datasets to download")
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    canonical = cfg["data"]["classes"]

    if args.list:
        print("Download these from your browser (select 'YOLOv8' format), then run:")
        for url in RECOMMENDED:
            print(f"  {url}")
        print("\n  python scripts/download_datasets.py --zip ~/Downloads/<file>.zip")
        return

    if not args.zip or not args.zip.exists():
        ap.error("--zip must point to an existing file (or use --list)")

    name = args.name or args.zip.stem.lower().replace(" ", "_")
    ingest_zip(args.zip, name, canonical)


if __name__ == "__main__":
    main()
