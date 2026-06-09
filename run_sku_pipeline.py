"""
Two-stage tote classification pipeline: empty/not-empty → homogeneous/heterogeneous.

Usage:
    python run_sku_pipeline.py \
        --empty_model models/tote_classifier_best.pth \
        --sku_model models/sku_classifier_best.pth \
        --images_dir images/rgb/ \
        --output_csv data/pipeline_predictions.csv

Output CSV columns:
    filename, stage1_prediction, stage2_prediction, empty_prob, homogeneous_prob

    stage2_prediction is 'n_a' for images predicted empty in stage 1.
"""

import argparse
import csv
import sys
from pathlib import Path

import torch

from classifier_core import load_model, predict_batch, make_transform, IMAGE_EXTS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--empty_model", required=True,
                        help="Checkpoint for empty/not-empty classifier")
    parser.add_argument("--sku_model", required=True,
                        help="Checkpoint for SKU homogeneity classifier")
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--output_csv", default="pipeline_predictions.csv")
    parser.add_argument("--empty_threshold", type=float, default=0.5,
                        help="Confidence threshold for 'empty' in stage 1")
    parser.add_argument("--sku_threshold", type=float, default=0.5,
                        help="Confidence threshold for 'homogeneous' in stage 2")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load both models
    empty_model, empty_img_size, empty_label_map, _ = load_model(args.empty_model, device)
    sku_model, sku_img_size, sku_label_map, sku_task = load_model(args.sku_model, device)

    if sku_task and sku_task != "sku_homogeneity":
        print(f"Warning: sku_model task is '{sku_task}', expected 'sku_homogeneity'",
              file=sys.stderr)

    empty_tf = make_transform(empty_img_size, augment=False)
    sku_tf = make_transform(sku_img_size, augment=False)

    empty_idx = empty_label_map["empty"]
    homo_idx = sku_label_map["homogeneous"]

    images_dir = Path(args.images_dir)
    all_paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not all_paths:
        print(f"No images found in {images_dir}")
        sys.exit(1)

    # Stage 1: empty / not-empty
    print(f"Stage 1: classifying {len(all_paths)} images (empty/not-empty)...", file=sys.stderr)
    stage1_probs = predict_batch(empty_model, all_paths, empty_tf, device, args.batch_size)

    stage1_results = {}
    not_empty_paths = []
    for path, probs in zip(all_paths, stage1_probs):
        if probs is None:
            stage1_results[path.name] = ("error", "")
            continue
        empty_prob = probs[empty_idx].item()
        prediction = "empty" if empty_prob >= args.empty_threshold else "not_empty"
        stage1_results[path.name] = (prediction, f"{empty_prob:.4f}")
        if prediction == "not_empty":
            not_empty_paths.append(path)

    # Stage 2: homogeneous / heterogeneous (not_empty images only)
    print(f"Stage 2: classifying {len(not_empty_paths)} not-empty images (SKU homogeneity)...",
          file=sys.stderr)
    stage2_results = {}
    if not_empty_paths:
        stage2_probs = predict_batch(sku_model, not_empty_paths, sku_tf, device, args.batch_size)
        for path, probs in zip(not_empty_paths, stage2_probs):
            if probs is None:
                stage2_results[path.name] = ("error", "")
                continue
            homo_prob = probs[homo_idx].item()
            prediction = "homogeneous" if homo_prob >= args.sku_threshold else "heterogeneous"
            stage2_results[path.name] = (prediction, f"{homo_prob:.4f}")

    # Merge and write output
    rows = []
    for path in all_paths:
        s1_pred, s1_empty_prob = stage1_results.get(path.name, ("error", ""))
        if s1_pred == "not_empty" and path.name in stage2_results:
            s2_pred, s2_homo_prob = stage2_results[path.name]
        else:
            s2_pred = "n_a" if s1_pred != "error" else "error"
            s2_homo_prob = ""
        rows.append({
            "filename": path.name,
            "stage1_prediction": s1_pred,
            "stage2_prediction": s2_pred,
            "empty_prob": s1_empty_prob,
            "homogeneous_prob": s2_homo_prob,
        })

    fieldnames = ["filename", "stage1_prediction", "stage2_prediction",
                  "empty_prob", "homogeneous_prob"]
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    n_empty = sum(1 for r in rows if r["stage1_prediction"] == "empty")
    n_homo = sum(1 for r in rows if r["stage2_prediction"] == "homogeneous")
    n_hetero = sum(1 for r in rows if r["stage2_prediction"] == "heterogeneous")
    print(f"\nResults ({len(rows)} images):")
    print(f"  empty:         {n_empty}")
    print(f"  homogeneous:   {n_homo}")
    print(f"  heterogeneous: {n_hetero}")
    print(f"Saved to {args.output_csv}")


if __name__ == "__main__":
    main()
