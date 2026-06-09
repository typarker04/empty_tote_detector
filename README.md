# Tote Classifier

Two EfficientNet-B0 classifiers for tote inspection:

- **Empty/not-empty** — is the tote occupied?
- **SKU homogeneity** — does the tote contain one SKU type (homogeneous) or multiple (heterogeneous)?

## Image Folder Structure

```
images/
├── empty/      # labeled empty RGB/depth pairs
├── not_empty/  # labeled not-empty RGB/depth pairs
└── input/      # drop new data here to be labeled
```

Depth images (`*_depth.png`) are never shown in the labeler — they are automatically paired with their RGB counterpart and given the same label.

Training scripts search `images/` recursively, so point `--images_dir` at the `images/` parent regardless of which subfolder images live in.

---

## Empty / Not-Empty Classifier

### 1. Score unseen data

```bash
python run_classifier.py \
    --model models/tote_classifier_best.pth \
    --images_dir images/input/ \
    --output_csv data/input_predictions.csv
```

Output CSV columns: `filename`, `prediction` (`empty`/`not_empty`), `empty_prob`.

### 2. Label

```bash
# Basic — shows each image one by one
python label_images.py \
    --images_dir images/input/ \
    --labels_csv data/labels.csv

# With model predictions shown in the title
python label_images.py \
    --images_dir images/input/ \
    --filter_csv data/input_predictions.csv \
    --labels_csv data/labels.csv

# Auto-label high-confidence predictions, review only uncertain ones
python label_images.py \
    --images_dir images/input/ \
    --filter_csv data/input_predictions.csv \
    --skip_threshold 0.95 \
    --labels_csv data/labels.csv
```

Keyboard controls:

| Key | Action |
|-----|--------|
| `e` | empty tote |
| `n` | not empty |
| `s` | skip |
| `u` | undo last label |
| `q` | quit and save |

Labels autosave every 20 images. Already-labeled images are skipped on subsequent runs.

![Tote Labeler](Tote_Labeler.png)

### 3. Train

```bash
python train_classifier.py \
    --images_dir images/ \
    --labels_csv data/labels.csv \
    --output models/tote_classifier_best.pth
```

| Flag | Default | Description |
|------|---------|-------------|
| `--epochs` | 20 | Training epochs |
| `--batch_size` | 32 | Batch size |
| `--lr` | 1e-4 | Learning rate (AdamW + cosine decay) |
| `--val_split` | 0.2 | Fraction held out for validation |
| `--freeze_backbone` | off | Train head only — faster, needs less data |

---

## SKU Homogeneity Classifier

Classifies non-empty totes as **homogeneous** (all items same SKU) or **heterogeneous** (multiple SKU types).

### 1. Pre-filter with the empty/not-empty classifier

```bash
python run_classifier.py \
    --model models/tote_classifier_best.pth \
    --images_dir images/input/ \
    --output_csv data/empty_predictions.csv
```

### 2. Label (only non-empty images are shown)

```bash
python label_sku.py \
    --images_dir images/input/ \
    --filter_csv data/empty_predictions.csv \
    --labels_csv data/sku_labels.csv
```

Keyboard controls:

| Key | Action |
|-----|--------|
| `h` | homogeneous |
| `x` | heterogeneous |
| `s` | skip |
| `u` | undo last label |
| `q` | quit and save |

### 3. Train

```bash
# Quick baseline — frozen backbone
python train_sku_classifier.py \
    --images_dir images/ \
    --labels_csv data/sku_labels.csv \
    --freeze_backbone

# Full fine-tune
python train_sku_classifier.py \
    --images_dir images/ \
    --labels_csv data/sku_labels.csv \
    --output models/sku_classifier_best.pth \
    --epochs 30
```

### 4. Inference

```bash
python run_sku_classifier.py \
    --model models/sku_classifier_best.pth \
    --images_dir images/input/ \
    --output_csv data/sku_predictions.csv
```

Output CSV columns: `filename`, `prediction` (`homogeneous`/`heterogeneous`), `homogeneous_prob`.

---

## Two-Stage Pipeline

Runs both classifiers in sequence and outputs a single merged CSV.

```bash
python run_sku_pipeline.py \
    --empty_model models/tote_classifier_best.pth \
    --sku_model models/sku_classifier_best.pth \
    --images_dir images/input/ \
    --output_csv data/pipeline_predictions.csv
```

Output CSV columns: `filename`, `stage1_prediction`, `stage2_prediction` (`homogeneous`/`heterogeneous`/`n_a`), `empty_prob`, `homogeneous_prob`. Empty totes get `stage2_prediction=n_a`.

---

## Dependencies

```bash
pip install -r requirements.txt
```

Requires PyTorch with CUDA (`torch==2.6.0+cu124`, `torchvision==0.21.0+cu124`).
