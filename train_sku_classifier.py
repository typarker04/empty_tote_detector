"""
Fine-tune EfficientNet-B0 for SKU homogeneity classification.

Usage:
    python train_sku_classifier.py --images_dir /path/to/images --labels_csv sku_labels.csv

Outputs:
    sku_classifier_best.pth  — best checkpoint by val accuracy
"""

import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from classifier_core import (
    ToteDataset,
    load_samples,
    make_weighted_sampler,
    build_model,
    train_epoch,
    eval_epoch,
    make_transform,
)

LABEL_MAP = {"homogeneous": 0, "heterogeneous": 1}
IMG_SIZE = 224


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--labels_csv", default="sku_labels.csv")
    parser.add_argument("--output", default="sku_classifier_best.pth")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--freeze_backbone", action="store_true",
                        help="Freeze backbone, train head only (faster, less data needed)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    samples = load_samples(Path(args.images_dir), Path(args.labels_csv), LABEL_MAP)
    if not samples:
        raise ValueError("No labeled samples found. Run label_sku.py first.")

    n_val = max(1, int(len(samples) * args.val_split))
    random.seed(args.seed)
    random.shuffle(samples)
    val_samples = samples[:n_val]
    train_samples = samples[n_val:]

    counts = [sum(1 for _, l in train_samples if l == c) for c in [0, 1]]
    print(f"Train: {len(train_samples)} ({counts[0]} homogeneous, {counts[1]} heterogeneous)")
    print(f"Val:   {len(val_samples)}")

    train_loader = DataLoader(
        ToteDataset(train_samples, make_transform(IMG_SIZE, augment=True)),
        batch_size=args.batch_size,
        sampler=make_weighted_sampler(train_samples),
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        ToteDataset(val_samples, make_transform(IMG_SIZE, augment=False)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    model = build_model(freeze_backbone=args.freeze_backbone).to(device)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        marker = " *" if val_acc > best_val_acc else ""
        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"train loss={train_loss:.4f} acc={train_acc:.3f}  "
              f"val loss={val_loss:.4f} acc={val_acc:.3f}{marker}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "label_map": LABEL_MAP,
                "img_size": IMG_SIZE,
                "task": "sku_homogeneity",
            }, args.output)

    print(f"\nBest val acc: {best_val_acc:.3f} — saved to {args.output}")


if __name__ == "__main__":
    main()
