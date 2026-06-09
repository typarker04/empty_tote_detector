"""
Shared utilities for tote binary classifiers (training + inference).
"""

import csv
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import models, transforms
from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


class ToteDataset(Dataset):
    def __init__(self, samples: list, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def load_samples(images_dir: Path, csv_path: Path, label_map: dict):
    samples = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row["label"] not in label_map:
                continue
            img_path = images_dir / row["filename"]
            if img_path.exists():
                samples.append((img_path, label_map[row["label"]]))
    return samples


def make_weighted_sampler(samples):
    counts = [0, 0]
    for _, label in samples:
        counts[label] += 1
    weights_per_class = [1.0 / c if c > 0 else 0.0 for c in counts]
    sample_weights = [weights_per_class[label] for _, label in samples]
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def build_model(num_classes: int = 2, freeze_backbone: bool = False):
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
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


def make_transform(img_size: int, augment: bool = False):
    """Return a torchvision transform. augment=False for val/inference, True for SKU training."""
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    if not augment:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ])
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        normalize,
    ])


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
    task = ckpt.get("task")
    return model, img_size, label_map, task


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
