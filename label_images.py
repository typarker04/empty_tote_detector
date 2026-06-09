"""
Interactive image labeler for tote classifier.

Usage:
    python label_images.py --images_dir /path/to/images [--labels_csv labels.csv]

Keys:
    e  — empty tote
    n  — not empty (occupied)
    s  — skip
    u  — undo last label
    q  — quit and save
"""

import argparse
import csv
import os
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--labels_csv", default="labels.csv")
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

    labels = load_existing_labels(csv_path)
    unlabeled = [p for p in all_images if p.name not in labels]

    print(f"Total images: {len(all_images)}")
    print(f"Already labeled: {len(labels)}")
    print(f"Remaining: {len(unlabeled)}")

    if not unlabeled:
        print("All images labeled.")
        return

    history = []  # list of (filename, label) for undo
    idx = 0

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.canvas.manager.set_window_title("Tote Labeler")
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
        ax.set_title(f"{img_path.name}\n{progress}\n[e]=empty  [n]=not empty  [s]=skip  [u]=undo  [q]=quit",
                     fontsize=10)
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

        if event.key == "e":
            apply_label("empty")
        elif event.key == "n":
            apply_label("not_empty")
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

        # autosave every 20 labels
        if len(labels) % 20 == 0:
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
