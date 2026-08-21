# dock-vision

Detect pallets and boxes in dock and rack warehouse scenes, count them per
frame, and log occupancy over time. The perception side is YOLO fine-tuned on
a mix of public and synthetic data; the counting side produces a time series
that feeds a dashboard and (next step) occupancy forecasting.

## Why synthetic data

Public labeled pallet datasets are small and narrow. Warehouse camera angles,
lighting, and rack layouts vary a lot, so training on one dataset rarely
transfers. This repo generates its own labeled scenes with domain
randomization (object scale, rotation, occlusion, light, blur, noise), the
same approach industrial CV teams use when labeled footage is scarce. The
generator writes ground-truth occupancy CSVs alongside the images, so the
counting stage can be validated against known counts, not just model output.

## Pipeline

```
download (Roboflow zips)         generate (synthetic scenes)
        |                                |
        +------------> merge -------------+
                         |
                      train (YOLO11)
                         |
                 evaluate (val split)
                         |
               predict + count (images/video)
                         |
                  dashboard (next)
```

## Repo layout

```
config.yaml                  classes, paths, training hyperparams
scripts/
  generate_synthetic.py      procedural scenes: dock floor and rack wall
  download_datasets.py       normalize downloaded zips into data/raw
  merge_datasets.py          stratified train/val split into data/merged
  train.py                   YOLO training
  evaluate.py                val metrics, per-class AP
  predict.py                 inference with counting overlay (image/video/folder)
notebooks/
  train_colab.ipynb          GPU training on Colab (free tier is enough)
data/                        generated locally, not committed
```

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. synthetic baseline (300 scenes, ~2 min on CPU)
python scripts/generate_synthetic.py --num 300

# 2. (optional) add public data. Download from your browser, YOLOv8 format:
#    https://universe.roboflow.com/projects-fsdqm/pallet-detection-and-count
#    https://universe.roboflow.com/roboflow-0aslh/pallet-detection-1zlvn
#    https://universe.roboflow.com/industrialdesign/boxes_and_pallet-r73ja
python scripts/download_datasets.py --zip ~/Downloads/pallet-detection-1.zip

# 3. merge + train
python scripts/merge_datasets.py
python scripts/train.py --model yolo11n.pt --epochs 100   # CPU: slow, use Colab

# 4. evaluate and run inference
python scripts/evaluate.py runs/train/weights/best.pt
python scripts/predict.py --weights runs/train/weights/best.pt --source data/eval
```

For GPU training, open `notebooks/train_colab.ipynb` in Colab, zip
`data/merged` and upload it. The notebook trains YOLO11n/s and downloads the
best weights.

## Data sources

- Roboflow Universe pallet detection datasets (links above)
- `aleksantari/so101_dualcams_palletstack_v1` (Hugging Face, real video for
  evaluation)
- NVIDIA PhysicalAI SimReady Warehouse (OpenUSD assets, CC-BY-4.0; an
  upgrade path for the synthetic pipeline, needs IsaacSim)

## Status

- Synthetic generator: working (dock + rack scenes, YOLO labels, occupancy GT)
- Data pipeline: working
- Training: pending first run
- Dashboard + forecasting: next step

## Roadmap

1. Train baseline on synthetic-only, measure val mAP
2. Add public Roboflow data, retrain, compare
3. Evaluate on real footage (so101 video + self-recorded frames)
4. Streamlit dashboard: counts over time, slot occupancy heatmap
5. Optional: occupancy forecasting (the quant-adjacent part)
