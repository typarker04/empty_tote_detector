"""
Run the trained tote classifier on a folder of images.

Usage:
    # Print predictions CSV to stdout
    python run_classifier.py --model tote_classifier_best.pth --images_dir /path/to/images

    # Drop new images into images/input/, then sort into empty/ and not_empty/unknown/
    python run_classifier.py --model tote_classifier_best.pth --images_dir images/input/ \
        --sort_output

    # Only flag images with confidence above a threshold
    python run_classifier.py ... --threshold 0.9
"""

import argparse
import csv
import shutil
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def load_model(checkpoint_path: str, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 2),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    img_size = ckpt.get("img_size", 224)
    label_map = ckpt.get("label_map", {"empty": 0, "not_empty": 1})
    return model, img_size, label_map


def make_transform(img_size: int):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


@torch.no_grad()
def predict_batch(model, paths, transform, device, batch_size=64):
    results = []
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i : i + batch_size]
        imgs = []
        for p in batch_paths:
            try:
                imgs.append(transform(Image.open(p).convert("RGB")))
            except Exception as e:
                print(f"Warning: could not read {p}: {e}", file=sys.stderr)
                imgs.append(None)

        valid = [(j, img) for j, img in enumerate(imgs) if img is not None]
        if not valid:
            results.extend([None] * len(batch_paths))
            continue

        indices, tensors = zip(*valid)
        batch = torch.stack(tensors).to(device)
        logits = model(batch)
        probs = F.softmax(logits, dim=1).cpu()

        result_map = {}
        for k, (j, _) in enumerate(valid):
            result_map[j] = probs[k]

        for j in range(len(batch_paths)):
            results.append(result_map.get(j))

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to .pth checkpoint")
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--output_csv", default=None, help="Save predictions CSV here")
    parser.add_argument("--move_empty", default=None,
                        help="If set, move predicted-empty images to this directory")
    parser.add_argument("--sort_output", action="store_true",
                        help="Move images into empty/ and not_empty/unknown/ subdirs of --output_base")
    parser.add_argument("--output_base", default="images",
                        help="Base dir for --sort_output (default: images/)")
    parser.add_argument("--threshold", type=float, default=0.4,
                        help="Confidence threshold for 'empty' prediction")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, img_size, label_map = load_model(args.model, device)
    transform = make_transform(img_size)

    # Invert label_map to get index -> name
    idx_to_label = {v: k for k, v in label_map.items()}
    empty_idx = label_map["empty"]

    images_dir = Path(args.images_dir)
    paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not paths:
        print(f"No images found in {images_dir}")
        sys.exit(1)

    print(f"Running on {len(paths)} images...", file=sys.stderr)
    probs_list = predict_batch(model, paths, transform, device, args.batch_size)

    rows = []
    for path, probs in zip(paths, probs_list):
        if probs is None:
            rows.append({"filename": path.name, "prediction": "error", "empty_prob": ""})
            continue
        empty_prob = probs[empty_idx].item()
        predicted = "empty" if empty_prob >= args.threshold else "not_empty"
        rows.append({
            "filename": path.name,
            "prediction": predicted,
            "empty_prob": f"{empty_prob:.4f}",
        })

    csv_out = args.output_csv or "predictions.csv"
    with open(csv_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "prediction", "empty_prob"])
        writer.writeheader()
        writer.writerows(rows)

    n_empty = sum(1 for r in rows if r["prediction"] == "empty")
    print(f"Results: {n_empty}/{len(rows)} predicted empty  (threshold={args.threshold})")
    print(f"Saved to {csv_out}")

    if args.move_empty:
        dest = Path(args.move_empty)
        dest.mkdir(parents=True, exist_ok=True)
        moved = 0
        for row in rows:
            if row["prediction"] == "empty":
                rgb_name = row["filename"]
                depth_name = rgb_name.replace("_rgb.png", "_depth.png")
                shutil.move(str(images_dir / rgb_name), dest / rgb_name)
                depth_src = images_dir / depth_name
                if depth_src.exists():
                    shutil.move(str(depth_src), dest / depth_name)
                moved += 1
        print(f"Moved {moved} empty-tote image pairs to {dest}")

    if args.sort_output:
        base = Path(args.output_base)
        empty_dir = base / "empty"
        unknown_dir = base / "not_empty" / "unknown"
        empty_dir.mkdir(parents=True, exist_ok=True)
        unknown_dir.mkdir(parents=True, exist_ok=True)
        counts = {"empty": 0, "not_empty": 0}
        for row in rows:
            if row["prediction"] not in ("empty", "not_empty"):
                continue
            dest = empty_dir if row["prediction"] == "empty" else unknown_dir
            rgb_name = row["filename"]
            shutil.move(str(images_dir / rgb_name), dest / rgb_name)
            depth_name = rgb_name.replace("_rgb.png", "_depth.png")
            if (images_dir / depth_name).exists():
                shutil.move(str(images_dir / depth_name), dest / depth_name)
            counts[row["prediction"]] += 1
        print(f"Sorted: {counts['empty']} → {empty_dir}/  {counts['not_empty']} → {unknown_dir}/")


if __name__ == "__main__":
    main()
