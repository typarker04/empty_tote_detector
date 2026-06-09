"""
Fine-tune EfficientNet-B0 for empty-tote binary classification.

Usage:
    python train_classifier.py --images_dir /path/to/images --labels_csv labels.csv

Outputs:
    tote_classifier_best.pth  — best checkpoint by val accuracy
"""

import argparse
import csv
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from PIL import Image

LABEL_MAP = {"empty": 0, "not_empty": 1}
IMG_SIZE = 224


class ToteDataset(Dataset):
    def __init__(self, samples: list, transform):
        self.samples = samples  # list of (path, label_int)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def load_samples(images_dir: Path, csv_path: Path):
    samples = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row["label"] not in LABEL_MAP:
                continue
            img_path = images_dir / row["filename"]
            if img_path.exists():
                samples.append((img_path, LABEL_MAP[row["label"]]))
    return samples


def make_weighted_sampler(samples):
    counts = [0, 0]
    for _, label in samples:
        counts[label] += 1
    weights_per_class = [1.0 / c if c > 0 else 0.0 for c in counts]
    sample_weights = [weights_per_class[label] for _, label in samples]
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def build_model(freeze_backbone: bool = False):
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 2),
    )
    return model


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        correct += (logits.argmax(1) == labels).sum().item()
        total += len(labels)
    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)
        total_loss += loss.item() * len(labels)
        correct += (logits.argmax(1) == labels).sum().item()
        total += len(labels)
    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--labels_csv", default="labels.csv")
    parser.add_argument("--output", default="tote_classifier_best.pth")
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

    samples = load_samples(Path(args.images_dir), Path(args.labels_csv))
    if not samples:
        raise ValueError("No labeled samples found. Run label_images.py first.")

    # deterministic train/val split
    n_val = max(1, int(len(samples) * args.val_split))
    import random
    random.seed(args.seed)
    random.shuffle(samples)
    val_samples = samples[:n_val]
    train_samples = samples[n_val:]

    counts = [sum(1 for _, l in train_samples if l == c) for c in [0, 1]]
    print(f"Train: {len(train_samples)} ({counts[0]} empty, {counts[1]} not_empty)")
    print(f"Val:   {len(val_samples)}")

    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_loader = DataLoader(
        ToteDataset(train_samples, train_tf),
        batch_size=args.batch_size,
        sampler=make_weighted_sampler(train_samples),
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        ToteDataset(val_samples, val_tf),
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
            }, args.output)

    print(f"\nBest val acc: {best_val_acc:.3f} — saved to {args.output}")


if __name__ == "__main__":
    main()
