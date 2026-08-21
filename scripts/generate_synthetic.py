#!/usr/bin/env python3
"""Generate synthetic dock/warehouse scenes with YOLO-format labels.

Two scene types:
- dock: open concrete floor, pallets and boxes scattered, sometimes a back wall
- rack: steel rack wall with bays and levels, some slots loaded, some empty

Writes JPEG + YOLO label files (class cx cy w h, normalized) plus a
ground-truth occupancy CSV, so the counting/analytics stage can be validated
against known counts instead of model predictions.

Real labeled pallet data is scarce, so the model is trained mostly on this
synthetic output. The variety below (lighting, blur, noise, occlusion,
object scale) exists to keep the synthetic distribution wide enough that a
fine-tuned YOLO transfers to real footage.
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]

# Wood tones in BGR, jittered per pallet so a batch reads as real inventory
# instead of one repeated material.
WOOD_BASES = [
    (66, 52, 40),     # dark walnut
    (94, 74, 56),
    (122, 100, 78),   # mid oak
    (152, 128, 102),
    (178, 156, 128),  # light pine
]

CARD_BASES = [
    (120, 98, 74),    # kraft
    (140, 116, 88),
    (164, 140, 108),  # light kraft
    (150, 132, 110),
    (190, 178, 158),  # near-white mailer
    (128, 128, 128),  # gray
]


def _bgr(c):
    return tuple(int(v) for v in c)


def make_pallet_sprite(width, height, rng):
    """Pallet as a dock camera sees it: plank deck over a front stringer.

    The stringer band is the counting cue; a deck alone reads as a floor.
    """
    base = np.array(WOOD_BASES[rng.integers(0, len(WOOD_BASES))], np.uint8)
    sprite = np.zeros((height, width, 4), np.uint8)
    deck_h = int(height * rng.uniform(0.7, 0.8))
    plank_h = max(4, deck_h // rng.integers(4, 6))
    y = 0
    while y < deck_h:
        shade = np.clip(base.astype(int) + rng.integers(-20, 20, 3), 0, 255).astype(np.uint8)
        end = min(y + plank_h, deck_h)
        cv2.rectangle(sprite, (0, y), (width, end), _bgr(shade), -1)
        if end < deck_h:
            # plank gap reads as texture; without it the deck is a flat slab
            cv2.rectangle(sprite, (0, end), (width, min(end + 2, deck_h)), (18, 14, 9), -1)
        y = end + rng.integers(2, 5)

    strut = np.clip(base.astype(int) - 35, 0, 255).astype(np.uint8)
    dark = np.clip(base.astype(int) - 60, 0, 255).astype(np.uint8)
    cv2.rectangle(sprite, (0, deck_h), (width, height), _bgr(strut), -1)
    foot_w = max(5, width // 7)
    for cx in (foot_w // 2, width // 2, width - foot_w // 2):
        x0 = cx - foot_w // 2
        cv2.rectangle(sprite, (x0, deck_h), (x0 + foot_w, height), _bgr(dark), -1)

    # per-channel grain, subtle enough not to smear plank edges
    grain = rng.normal(0, 7, (height, width, 1)).astype(np.int16)
    sprite[:, :, :3] = np.clip(sprite[:, :, :3].astype(np.int16) + grain, 0, 255).astype(np.uint8)
    sprite[:, :, 3] = 255
    return sprite


def make_box_sprite(width, height, rng):
    """Cardboard box as a pseudo-3D cuboid: front, top, side faces.

    Face shading contrast is what separates a box from a pallet in a low-res
    frame, so it is kept wide on purpose.
    """
    base = np.array(CARD_BASES[rng.integers(0, len(CARD_BASES))], np.uint8)
    sprite = np.zeros((height, width, 4), np.uint8)
    top_h = int(height * rng.uniform(0.3, 0.42))
    side_w = int(width * rng.uniform(0.18, 0.3))

    front = np.clip(base.astype(int) + rng.integers(-12, 12, 3), 0, 255)
    top = np.clip(front + 28, 0, 255)
    side = np.clip(front - 30, 0, 255)

    cv2.rectangle(sprite, (0, top_h), (width, height), _bgr(front), -1)
    cv2.fillConvexPoly(
        sprite,
        np.array([[0, top_h], [width, top_h], [width - side_w, 0], [side_w, 0]], np.int32),
        _bgr(top),
    )
    cv2.fillConvexPoly(
        sprite,
        np.array([[width, top_h], [width - side_w, 0], [width - side_w - 7, 7], [width - 7, top_h + 7]], np.int32),
        _bgr(side),
    )

    # packing tape: near-universal on real boxes
    tape_y = top_h + int((height - top_h) * rng.uniform(0.2, 0.5))
    tape = np.clip(front + 22, 0, 255)
    cv2.rectangle(sprite, (0, tape_y), (width, tape_y + max(3, height // 28)), _bgr(tape), -1)

    # barcode block on a minority of boxes; skipped when the box is too small
    if rng.random() < 0.4:
        region_w = max(6, min(width // 6, width // 2))
        bx = int(rng.integers(int(width * 0.06), max(int(width * 0.06) + 1, width - region_w)))
        by_lo, by_hi = top_h + 4, height - 10
        if by_hi > by_lo:
            by = int(rng.integers(by_lo, by_hi))
            bar_h = max(6, min(by_hi - by, region_w))
            x = bx
            while x < bx + region_w:
                cv2.rectangle(sprite, (x, by), (x + 2, by + bar_h), (15, 12, 10), -1)
                x += int(rng.integers(3, 7))

    sprite[:, :, 3] = 255
    return sprite


def make_floor(w, h, rng):
    """Concrete floor with faint joints; plain enough that objects pop."""
    base = int(rng.integers(95, 135))
    img = np.full((h, w, 3), base, np.uint8)
    noise = rng.normal(0, 8, (h, w, 1))
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    joint = int(base * 0.82)
    for _ in range(int(rng.integers(1, 3))):
        y = int(rng.integers(h // 3, h - 20))
        cv2.line(img, (0, y), (w, y), (joint, joint, joint), 1)
    for _ in range(int(rng.integers(1, 3))):
        x = int(rng.integers(w // 4, w - 50))
        cv2.line(img, (x, 0), (x, h), (joint, joint, joint), 1)
    # distant light falls off toward the top of the frame
    grad = np.linspace(0.92, 1.0, h)[:, None, None]
    img = np.clip(img.astype(np.float32) * grad, 0, 255).astype(np.uint8)
    return img


def make_dock_background(w, h, rng):
    """Open loading area; returns the floor line so objects stay in front."""
    img = make_floor(w, h, rng)
    wall_y = None
    if rng.random() < 0.6:
        wall_y = int(h * rng.uniform(0.35, 0.5))
        wall = np.full((wall_y, w, 3), int(rng.integers(120, 160)), np.uint8)
        wall = np.clip(wall.astype(np.int16) + rng.normal(0, 6, (wall_y, w, 1)), 0, 255).astype(np.uint8)
        if rng.random() < 0.5:
            # dock door cutout: dark opening reads as depth
            dw = int(w * rng.uniform(0.25, 0.45))
            dx = int(rng.integers(0, w - dw))
            dh = int(wall_y * 0.75)
            cv2.rectangle(wall, (dx, wall_y - dh), (dx + dw, wall_y), (60, 62, 66), -1)
        img[:wall_y] = wall
        cv2.line(img, (0, wall_y), (w, wall_y), (40, 42, 45), 3)
    return img, wall_y


def make_rack_background(w, h, rng):
    """Steel rack wall with bays and levels; returns slot rects + occupancy."""
    img = make_floor(w, h, rng)
    wall_y = int(h * rng.uniform(0.4, 0.55))
    wall = np.full((wall_y, w, 3), int(rng.integers(75, 95)), np.uint8)
    wall = np.clip(wall.astype(np.int16) + rng.normal(0, 5, (wall_y, w, 1)), 0, 255).astype(np.uint8)
    img[:wall_y] = wall
    cv2.line(img, (0, wall_y), (w, wall_y), (30, 32, 35), 2)

    steel = (int(rng.integers(95, 115)), int(rng.integers(100, 120)), int(rng.integers(105, 125)))
    levels = int(rng.integers(2, 3))
    bays = int(rng.integers(3, 5))
    margin_x = int(w * 0.04)
    top_y = int(h * 0.06)
    slot_w = (w - 2 * margin_x) // bays
    slot_h = (wall_y - top_y) // levels

    slots, occupied = [], []
    for li in range(levels):
        for bi in range(bays):
            x1 = margin_x + bi * slot_w + 4
            y1 = top_y + li * slot_h + 4
            x2 = margin_x + (bi + 1) * slot_w - 4
            y2 = top_y + (li + 1) * slot_h - 4
            cv2.rectangle(img, (x1 - 5, y1 - 4), (x1 - 1, y2 + 4), steel, -1)
            cv2.rectangle(img, (x1 - 6, y2 - 3), (x2 + 6, y2 + 3), steel, -1)
            slots.append((x1, y1, x2, y2))
            occupied.append(rng.random() < rng.uniform(0.45, 0.7))
    for bi in range(bays + 1):
        x = margin_x + bi * slot_w
        cv2.line(img, (x, top_y - 6), (x, wall_y), steel, 2)
    return img, slots, occupied


def place_sprite(img, sprite, cx, cy, angle, rng):
    """Rotate a sprite, paste it at (cx, cy), return its axis-aligned bbox.

    The sprite is padded before rotation so corner pixels are never clipped.
    Returns None if the visible part is too small to be a useful label.
    """
    h, w = sprite.shape[:2]
    pad = int(max(w, h) * 0.35)
    canvas = np.zeros((h + 2 * pad, w + 2 * pad, 4), np.uint8)
    canvas[pad:pad + h, pad:pad + w] = sprite
    M = cv2.getRotationMatrix2D((canvas.shape[1] / 2, canvas.shape[0] / 2), angle, 1.0)
    rot = cv2.warpAffine(
        canvas, M, (canvas.shape[1], canvas.shape[0]),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0),
    )
    ys, xs = np.where(rot[:, :, 3] > 10)
    if len(xs) == 0:
        return None
    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    bw, bh = x2 - x1 + 1, y2 - y1 + 1
    x = max(0, min(int(cx - bw / 2), img.shape[1] - 1))
    y = max(0, min(int(cy - bh / 2), img.shape[0] - 1))
    x2c = min(img.shape[1], x + bw)
    y2c = min(img.shape[0], y + bh)
    if x2c - x < 24 or y2c - y < 24:
        return None

    crop = rot[y1:y2 + 1, x1:x2 + 1]
    region = img[y:y2c, x:x2c]
    a = crop[:region.shape[0], :region.shape[1], 3:4].astype(np.float32) / 255.0
    rgb = crop[:region.shape[0], :region.shape[1], :3].astype(np.float32)
    img[y:y2c, x:x2c] = np.clip(rgb * a + region.astype(np.float32) * (1 - a), 0, 255).astype(np.uint8)
    return (x, y, x2c, y2c)


def draw_shadow(img, bbox, rng):
    """Soft contact shadow under an object, before the object is pasted."""
    x1, y1, x2, y2 = bbox
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.ellipse(
        mask,
        (int((x1 + x2) / 2), y2 - 2),
        (int((x2 - x1) * 0.45), max(3, int((y2 - y1) * 0.12))),
        0, 0, 360, 255, -1,
    )
    mask = cv2.GaussianBlur(mask, (0, 0), 4)
    alpha = mask.astype(np.float32)[..., None] / 255.0 * rng.uniform(0.25, 0.45)
    shadow = np.full(img.shape, (25, 25, 30), np.uint8)
    img[:] = np.clip(
        img.astype(np.float32) * (1 - alpha) + shadow.astype(np.float32) * alpha, 0, 255
    ).astype(np.uint8)


def postprocess(img, rng):
    """Global domain randomization: light, blur, noise, vignette."""
    img = np.clip(img.astype(np.float32) * rng.uniform(0.85, 1.18) + rng.integers(-18, 18), 0, 255)
    if rng.random() < 0.2:
        k = int(rng.choice([3, 5]))
        img = cv2.GaussianBlur(img, (k, k), 0)
    if rng.random() < 0.6:
        noise = rng.normal(0, rng.uniform(1.5, 4.5), img.shape[:2])[..., None]
        img = np.clip(img + noise, 0, 255)
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    cx, cy = img.shape[1] / 2, img.shape[0] / 2
    d = np.sqrt(((xx - cx) / (cx * 1.15)) ** 2 + ((yy - cy) / (cy * 1.15)) ** 2)
    v = np.clip(1 - 0.35 * np.clip(d - 0.55, 0, 1), 0.65, 1.0)
    img = img * v[..., None]
    return np.clip(img, 0, 255).astype(np.uint8)


def _bbox_center(bb):
    return ((bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2)


def place_box_on_pallet(img, pal_bb, rng):
    """Drop a box onto the visible deck of a pallet (partially overlapping)."""
    x1, y1, x2, y2 = pal_bb
    bw = int(rng.integers(int((x2 - x1) * 0.25), int((x2 - x1) * 0.6)))
    bh = int(bw * rng.uniform(0.55, 0.8))
    sprite = make_box_sprite(bw, bh, rng)
    cx = int(rng.integers(x1 + bw // 2, max(x1 + bw // 2 + 1, x2 - bw // 2)))
    cy = int(rng.integers(y1 + bh // 2, max(y1 + bh // 2 + 1, y2 - int((y2 - y1) * 0.15))))
    return place_sprite(img, sprite, cx, cy, rng.uniform(-10, 10), rng)


def render_scene(rng, cfg):
    W = cfg["synthetic"]["width"]
    H = cfg["synthetic"]["height"]
    scene_type = cfg["synthetic"]["scene_types"][rng.integers(0, len(cfg["synthetic"]["scene_types"]))]
    bboxes = []          # (x1, y1, x2, y2, class_id)
    slot_records = []    # (slot_id, occupied) for rack scenes
    pallet_count = 0
    box_count = 0

    def add_pallet(cx, cy, size_hint, angle):
        nonlocal pallet_count
        pw = int(rng.integers(size_hint[0], size_hint[1]))
        ph = int(pw * rng.uniform(0.42, 0.55))
        sprite = make_pallet_sprite(pw, ph, rng)
        bb = place_sprite(img, sprite, cx, cy, angle, rng)
        if bb is None:
            return None
        draw_shadow(img, bb, rng)
        # re-paste after shadow so the object sits on top of its own shadow
        place_sprite(img, sprite, cx, cy, angle, rng)
        bboxes.append((*bb, 0))
        pallet_count += 1
        return bb

    if scene_type == "rack":
        img, slots, occupied = make_rack_background(W, H, rng)
        for sid, (slot, occ) in enumerate(zip(slots, occupied)):
            slot_records.append((sid, int(occ)))
            if not occ:
                continue
            x1, y1, x2, y2 = slot
            sw, sh = x2 - x1, y2 - y1
            cx = int(rng.integers(x1 + sw // 4, x2 - sw // 4))
            cy = y2 - int(sh * 0.28)
            bb = add_pallet(cx, cy, (int(sw * 0.5), int(sw * 0.72)), rng.uniform(-8, 8))
            if bb:
                for _ in range(int(rng.integers(0, 3))):
                    if rng.random() < 0.7:
                        b = place_box_on_pallet(img, bb, rng)
                        if b:
                            bboxes.append((*b, 1))
                            box_count += 1
        # loose stock in front of the rack
        if rng.random() < 0.3:
            cx = int(rng.integers(int(W * 0.2), int(W * 0.8)))
            cy = int(rng.integers(int(H * 0.6), int(H * 0.85)))
            add_pallet(cx, cy, (150, 260), rng.uniform(-15, 15))
    else:
        img, wall_y = make_dock_background(W, H, rng)
        y_lo = int((wall_y if wall_y else H * 0.22) + H * 0.06)
        n_pal = int(rng.integers(cfg["synthetic"]["min_objects"], cfg["synthetic"]["max_objects"] + 1))
        for _ in range(n_pal):
            cx = int(rng.integers(int(W * 0.08), int(W * 0.92)))
            cy = int(rng.integers(y_lo, int(H * 0.9)))
            bb = add_pallet(cx, cy, (160, 320), rng.uniform(-15, 15))
            if bb:
                for _ in range(int(rng.integers(0, 3))):
                    if rng.random() < 0.7:
                        b = place_box_on_pallet(img, bb, rng)
                        if b:
                            bboxes.append((*b, 1))
                            box_count += 1
        # loose boxes on the floor
        if rng.random() < 0.35:
            for _ in range(int(rng.integers(0, 3))):
                bw = int(rng.integers(60, 130))
                bh = int(bw * rng.uniform(0.6, 0.9))
                sprite = make_box_sprite(bw, bh, rng)
                cx = int(rng.integers(int(W * 0.08), int(W * 0.92)))
                cy = int(rng.integers(y_lo + 20, int(H * 0.9)))
                bb = place_sprite(img, sprite, cx, cy, rng.uniform(-20, 20), rng)
                if bb:
                    draw_shadow(img, bb, rng)
                    place_sprite(img, sprite, cx, cy, rng.uniform(-20, 20), rng)
                    bboxes.append((*bb, 1))
                    box_count += 1

    return img, bboxes, scene_type, slot_records, pallet_count, box_count


def write_label(path, bboxes, W, H):
    lines = []
    for x1, y1, x2, y2, cls in bboxes:
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        if x2 - x1 < 16 or y2 - y1 < 16:
            continue
        lines.append(f"{cls} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
    path.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--num", type=int, default=None, help="scenes to generate (default: config)")
    ap.add_argument("--out", default=None, help="output dir (default: config synthetic_dir)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    num = args.num or cfg["synthetic"]["num_scenes"]
    out = Path(args.out or ROOT / cfg["data"]["synthetic_dir"])
    img_dir, lbl_dir = out / "images", out / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    classes = cfg["data"]["classes"]
    (out / "classes.yaml").write_text(
        yaml.safe_dump({"nc": len(classes), "names": {i: c for i, c in enumerate(classes)}})
    )

    rng = np.random.default_rng(args.seed)
    frames, occ_rows = [], []
    for i in range(num):
        img, bboxes, scene_type, slot_records, n_pal, n_box = render_scene(rng, cfg)
        img = postprocess(img, rng)
        name = f"scene_{i:05d}"
        cv2.imwrite(str(img_dir / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        write_label(lbl_dir / f"{name}.txt", bboxes, img.shape[1], img.shape[0])
        frames.append((name, scene_type, n_pal, n_box))
        occ_rows.extend((name, sid, occ) for sid, occ in slot_records)

    (out / "frames_summary.csv").write_text(
        "frame,scene_type,pallet_count,box_count\n"
        + "".join(f"{n},{t},{p},{b}\n" for n, t, p, b in frames)
    )
    (out / "occupancy_gt.csv").write_text(
        "frame,slot_id,occupied\n"
        + "".join(f"{n},{s},{o}\n" for n, s, o in occ_rows)
    )
    if not args.quiet:
        n_dock = sum(1 for _, t, _, _ in frames if t == "dock")
        n_rack = num - n_dock
        avg_pal = sum(f[2] for f in frames) / max(1, num)
        avg_box = sum(f[3] for f in frames) / max(1, num)
        print(f"Generated {num} scenes -> {out}")
        print(f"  dock: {n_dock}, rack: {n_rack}")
        print(f"  avg pallets/scene: {avg_pal:.1f}, avg boxes/scene: {avg_box:.1f}")
        print(f"  labels: {lbl_dir}, frames summary: {out / 'frames_summary.csv'}")


if __name__ == "__main__":
    main()
