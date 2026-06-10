"""
Run the trained SKU homogeneity classifier on a folder of images.

Usage:
    # Print predictions CSV to stdout
    python run_sku_classifier.py --model sku_classifier_best.pth --images_dir /path/to/images

    # Save CSV to a file
    python run_sku_classifier.py --model sku_classifier_best.pth \
        --images_dir /path/to/images --output_csv predictions.csv

    # Sort images from not_empty/unknown/ into not_empty/homogeneous/ and not_empty/not_homogeneous/
    python run_sku_classifier.py --model sku_classifier_best.pth \
        --images_dir images/not_empty/unknown/ --sort_output

    # Only flag images with confidence above a threshold
    python run_sku_classifier.py ... --threshold 0.9
"""

import argparse
import csv
import shutil
import sys
from pathlib import Path

import torch

from classifier_core import load_model, predict_batch, make_transform, IMAGE_EXTS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to .pth checkpoint")
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--output_csv", default=None, help="Save predictions CSV here")
    parser.add_argument("--sort_output", action="store_true",
                        help="Move images into homogeneous/ and not_homogeneous/ siblings of images_dir")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Confidence threshold for 'homogeneous' prediction")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, img_size, label_map, task = load_model(args.model, device)

    if task and task != "sku_homogeneity":
        print(f"Warning: checkpoint task is '{task}', expected 'sku_homogeneity'", file=sys.stderr)

    transform = make_transform(img_size, augment=False)
    idx_to_label = {v: k for k, v in label_map.items()}
    homo_idx = label_map["homogeneous"]

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
            rows.append({"filename": path.name, "prediction": "error", "homogeneous_prob": ""})
            continue
        homo_prob = probs[homo_idx].item()
        predicted = "homogeneous" if homo_prob >= args.threshold else "heterogeneous"
        rows.append({
            "filename": path.name,
            "prediction": predicted,
            "homogeneous_prob": f"{homo_prob:.4f}",
        })

    csv_out = args.output_csv or "sku_predictions.csv"
    with open(csv_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "prediction", "homogeneous_prob"])
        writer.writeheader()
        writer.writerows(rows)

    n_homo = sum(1 for r in rows if r["prediction"] == "homogeneous")
    print(f"Results: {n_homo}/{len(rows)} predicted homogeneous  (threshold={args.threshold})")
    print(f"Saved to {csv_out}")

    if args.sort_output:
        base = images_dir.parent
        homo_dir = base / "homogeneous"
        not_homo_dir = base / "not_homogeneous"
        homo_dir.mkdir(exist_ok=True)
        not_homo_dir.mkdir(exist_ok=True)
        counts = {"homogeneous": 0, "not_homogeneous": 0}
        for row in rows:
            if row["prediction"] not in ("homogeneous", "heterogeneous"):
                continue
            dest = homo_dir if row["prediction"] == "homogeneous" else not_homo_dir
            key = row["prediction"] if row["prediction"] == "homogeneous" else "not_homogeneous"
            rgb_name = row["filename"]
            shutil.move(str(images_dir / rgb_name), dest / rgb_name)
            depth_name = rgb_name.replace("_rgb.png", "_depth.png")
            if (images_dir / depth_name).exists():
                shutil.move(str(images_dir / depth_name), dest / depth_name)
            counts[key] += 1
        print(f"Sorted: {counts['homogeneous']} → homogeneous/  {counts['not_homogeneous']} → not_homogeneous/")


if __name__ == "__main__":
    main()
