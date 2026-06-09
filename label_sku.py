"""
Interactive image labeler for SKU homogeneity classification.

Usage:
    python label_sku.py --images_dir /path/to/images [--labels_csv sku_labels.csv]

    # Skip already-classified empty totes using an existing predictions CSV
    python label_sku.py --images_dir /path/to/images \
        --filter_csv data/empty_predictions.csv \
        --labels_csv data/sku_labels.csv

Keys:
    h  — homogeneous (all items same SKU)
    x  — heterogeneous (multiple SKU types)
    s  — skip
    u  — undo last label
    q  — quit and save
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def load_existing_labels(csv_path: Path) -> dict:
    labels = {}
    if csv_path.exists():
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                labels[row["filename"]] = row["label"]
    return labels


def save_labels(csv_path: Path, labels: dict):
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "label"])
        writer.writeheader()
        for filename, label in labels.items():
            writer.writerow({"filename": filename, "label": label})


def load_not_empty_set(filter_csv: Path) -> set:
    """Return filenames predicted as not_empty from a run_classifier.py output CSV."""
    keep = set()
    with open(filter_csv) as f:
        for row in csv.DictReader(f):
            if row.get("prediction") == "not_empty":
                keep.add(row["filename"])
    return keep


def load_empty_probs(filter_csv: Path) -> dict:
    probs = {}
    with open(filter_csv) as f:
        for row in csv.DictReader(f):
            probs[row["filename"]] = row.get("empty_prob", "")
    return probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--labels_csv", default="sku_labels.csv")
    parser.add_argument(
        "--filter_csv",
        default=None,
        help="Predictions CSV from run_classifier.py — restricts queue to not_empty images",
    )
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    csv_path = Path(args.labels_csv)

    all_images = sorted(
        p for p in images_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTS and not p.name.endswith("_depth.png")
    )
    if not all_images:
        print(f"No images found in {images_dir}")
        sys.exit(1)

    # Optionally restrict to not_empty images from a prior classifier run
    empty_probs = {}
    if args.filter_csv:
        filter_path = Path(args.filter_csv)
        if not filter_path.exists():
            print(f"filter_csv not found: {filter_path}")
            sys.exit(1)
        not_empty_set = load_not_empty_set(filter_path)
        empty_probs = load_empty_probs(filter_path)
        all_images = [p for p in all_images if p.name in not_empty_set]
        print(f"Filtered to {len(all_images)} not_empty images via {args.filter_csv}")

    labels = load_existing_labels(csv_path)
    unlabeled = [p for p in all_images if p.name not in labels]

    print(f"Total images in queue: {len(all_images)}")
    print(f"Already labeled: {len(labels)}")
    print(f"Remaining: {len(unlabeled)}")

    if not unlabeled:
        print("All images labeled.")
        return

    history = []
    idx = 0

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.canvas.manager.set_window_title("SKU Homogeneity Labeler")
    plt.tight_layout()

    def show_image(i):
        if i >= len(unlabeled):
            ax.clear()
            ax.text(0.5, 0.5, "Done! Press q to quit.", ha="center", va="center",
                    transform=ax.transAxes, fontsize=16)
            fig.canvas.draw()
            return
        img_path = unlabeled[i]
        img = mpimg.imread(str(img_path))
        ax.clear()
        ax.imshow(img)
        ax.axis("off")
        progress = f"{i + 1}/{len(unlabeled)} remaining  |  total labeled: {len(labels)}"
        prob_info = ""
        if img_path.name in empty_probs:
            prob_info = f"  |  empty_prob={empty_probs[img_path.name]}"
        ax.set_title(
            f"{img_path.name}\n{progress}{prob_info}\n"
            "[h]=homogeneous  [x]=heterogeneous  [s]=skip  [u]=undo  [q]=quit",
            fontsize=10,
        )
        fig.canvas.draw()

    def on_key(event):
        nonlocal idx
        if event.key == "q":
            save_labels(csv_path, labels)
            print(f"\nSaved {len(labels)} labels to {csv_path}")
            plt.close()
            return

        if idx >= len(unlabeled):
            return

        filename = unlabeled[idx].name

        def apply_label(label):
            nonlocal idx
            labels[filename] = label
            depth_name = filename.replace("_rgb.png", "_depth.png")
            if (images_dir / depth_name).exists():
                labels[depth_name] = label
            history.append(filename)
            idx += 1
            show_image(idx)

        if event.key == "h":
            apply_label("homogeneous")
        elif event.key == "x":
            apply_label("heterogeneous")
        elif event.key == "s":
            idx += 1
            show_image(idx)
        elif event.key == "u" and history:
            removed_name = history.pop()
            del labels[removed_name]
            depth_name = removed_name.replace("_rgb.png", "_depth.png")
            labels.pop(depth_name, None)
            for j, p in enumerate(unlabeled):
                if p.name == removed_name:
                    idx = j
                    break
            show_image(idx)

        if len(labels) % 20 == 0 and len(labels) > 0:
            save_labels(csv_path, labels)

    fig.canvas.mpl_connect("key_press_event", on_key)
    show_image(idx)
    plt.show()

    save_labels(csv_path, labels)
    print(f"\nSaved {len(labels)} labels to {csv_path}")

    counts = {}
    for v in labels.values():
        counts[v] = counts.get(v, 0) + 1
    print("Label counts:", counts)


if __name__ == "__main__":
    main()
