#!/usr/bin/env python3
"""Merge all sources under data/raw and data/synthetic into data/merged.

Output layout (YOLO-ready for ultralytics):
  data/merged/{train,val}/{images,labels}/ + data.yaml

Split is seeded and class-stratified per source so every source contributes
to both train and val. Filenames are prefixed with the source name to avoid
collisions.
"""

import argparse
import random
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _load_source(name: str, src_dir: Path):
    images, labels = src_dir / "images", src_dir / "labels"
    if not images.exists() or not labels.exists():
        return []
    pairs = []
    for img in sorted(images.glob("*")):
        if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        lbl = labels / f"{img.stem}.txt"
        if lbl.exists():
            pairs.append((img, lbl, f"{name}_{img.stem}{img.suffix}"))
    return pairs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    classes = cfg["data"]["classes"]
    merged = ROOT / cfg["data"]["merged_dir"]

    sources = []
    raw_dir = ROOT / cfg["data"]["raw_dir"]
    if raw_dir.exists():
        for child in sorted(raw_dir.iterdir()):
            if child.is_dir():
                sources.extend(_load_source(child.name, child))
    syn_dir = ROOT / cfg["data"]["synthetic_dir"]
    if syn_dir.exists():
        sources.extend(_load_source("syn", syn_dir))

    if not sources:
        raise SystemExit("No datasets found. Run the synthetic generator and/or ingest downloads first.")

    rng = random.Random(args.seed)
    # stratify by class composition so val is not skewed toward one source
    def key_fn(pair):
        lbl = pair[1].read_text().splitlines()
        return tuple(sorted(Counter(int(l.split()[0]) for l in lbl if l.strip()).items()))

    buckets = {}
    for pair in sources:
        buckets.setdefault(key_fn(pair), []).append(pair)

    train, val = [], []
    for bucket in buckets.values():
        rng.shuffle(bucket)
        cut = max(1, int(len(bucket) * (1 - args.val_frac)))
        train.extend(bucket[:cut])
        val.extend(bucket[cut:])
    rng.shuffle(train)

    for split_name, pairs in (("train", train), ("val", val)):
        img_dir = merged / split_name / "images"
        lbl_dir = merged / split_name / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for img, lbl, out_name in pairs:
            (img_dir / out_name).write_bytes(img.read_bytes())
            (lbl_dir / f"{Path(out_name).stem}.txt").write_text(lbl.read_text())

    class_counts = Counter()
    for split_name in ("train", "val"):
        for lbl in (merged / split_name / "labels").glob("*.txt"):
            for line in lbl.read_text().splitlines():
                if line.strip():
                    class_counts[int(line.split()[0])] += 1

    (merged / "data.yaml").write_text(
        yaml.safe_dump({
            "path": str(merged),
            "train": "train/images",
            "val": "val/images",
            "nc": len(classes),
            "names": {i: c for i, c in enumerate(classes)},
        })
    )

    print(f"train: {len(train)}  val: {len(val)}  -> {merged}")
    print("class distribution:", {classes[k]: v for k, v in sorted(class_counts.items())})


if __name__ == "__main__":
    main()
