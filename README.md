# Tote Classifier

Binary image classifier (empty vs. not-empty tote) based on fine-tuned EfficientNet-B0.

## Workflow

1. **Label** images with `label_images.py`
2. **Train** a model with `train_classifier.py`
3. **Run** inference with `run_classifier.py`

## Labeling

```bash
python label_images.py --images_dir /path/to/images --labels_csv data/labels.csv
```

A matplotlib window opens showing each image. Keyboard controls:

| Key | Action |
|-----|--------|
| `e` | empty tote |
| `n` | not empty |
| `s` | skip |
| `u` | undo last label |
| `q` | quit and save |

Labels autosave every 20 images. Already-labeled images are skipped on subsequent runs.

![Tote Labeler](Tote_Labeler.png)

## Training

```bash
python train_classifier.py \
    --images_dir /path/to/images \
    --labels_csv data/labels.csv \
    --output models/tote_classifier_best.pth
```

The best checkpoint (by validation accuracy) is saved automatically. Class imbalance is handled via weighted random sampling.

Key options:

| Flag | Default | Description |
|------|---------|-------------|
| `--epochs` | 20 | Training epochs |
| `--batch_size` | 32 | Batch size |
| `--lr` | 1e-4 | Learning rate (AdamW + cosine decay) |
| `--val_split` | 0.2 | Fraction held out for validation |
| `--freeze_backbone` | off | Train head only — faster, needs less data |

## Inference

```bash
# Print predictions CSV to stdout
python run_classifier.py \
    --model models/tote_classifier_best.pth \
    --images_dir /path/to/images

# Save CSV to a file
python run_classifier.py --model models/tote_classifier_best.pth \
    --images_dir /path/to/images --output_csv predictions.csv

# Move predicted-empty images (and paired *_depth.png files) to another folder
python run_classifier.py --model models/tote_classifier_best.pth \
    --images_dir /path/to/images --move_empty /path/to/empty_output

# Raise the confidence threshold (default 0.5)
python run_classifier.py ... --threshold 0.9
```

Output CSV columns: `filename`, `prediction` (`empty`/`not_empty`), `empty_prob`.

The `--move_empty` flag moves both `*_rgb.png` and the paired `*_depth.png` when present, making it easy to cull empty-tote frames from RGB-D datasets.

## Dependencies

```bash
pip install -r requirements.txt
```

Requires PyTorch with CUDA (`torch==2.6.0+cu124`, `torchvision==0.21.0+cu124`).