# UNet Variants for Medical Tumor Segmentation: A Comparative Study

## 📌 Project Overview

Medical image segmentation is a critical step in many clinical pipelines. In renal cancer analysis, **nephrometry scoring** depends heavily on accurate tumor segmentation, as parameters such as tumor size, depth, and spatial location are derived directly from segmented regions.

The objective of this project is to **systematically evaluate and compare UNet-based segmentation architectures** and identify a robust backbone that can later be extended to **renal tumor (RCC) segmentation on CT scans**.

To build and validate this backbone, we conducted experiments on the **Breast Ultrasound Images Dataset (BUSI)**.

---

## 🧠 Motivation

- Manual tumor segmentation is **time-consuming and subjective**.
- Medical AI requires **high recall without sacrificing precision**.
- Segmentation quality directly affects downstream clinical scores.
- UNet variants are widely used, but their **practical differences are often unclear**.

This project focuses on **understanding, not just benchmarking**, UNet variants—ideal for final-year projects and viva preparation.

---

## 📂 Dataset Description

### Breast Ultrasound Images Dataset (BUSI)

- Grayscale ultrasound images.
- Three classes: `benign`, `malignant`, `normal`.
- Pixel-level **binary segmentation masks**.

### Important Dataset Property

Some images contain **multiple tumor masks** for the same scan (e.g. `*_mask.png`, `*_mask_1.png`). These must be merged into a single ground-truth mask before training (see [Mask merging](#-mask-merging) below).

---

## 🔄 Data Preprocessing Pipeline

### Image Preprocessing

- Converted all images to **grayscale**.
- Normalized pixel values to `[0, 1]`.
- Resized images to a fixed resolution of **256 × 256**.

### Mask Preprocessing

- Converted masks to binary format (0/1).
- **Merged multiple masks into a single binary mask** using logical OR.
- Ensured strict image–mask alignment (same size and indexing).

### Mask Merging

Segmentation models expect **one ground-truth mask per image**. In BUSI, some cases have multiple mask files (e.g. two lesions). We merge them so that any pixel marked in any mask is considered foreground:

```python
# Pseudocode: merge all masks for one image
final_mask = zeros_like(first_mask)
for mask_path in mask_paths:
    mask = load_and_binarize(mask_path)
    final_mask = logical_or(final_mask, mask)
```

**Why this is required:** All tumor regions must be represented in a single binary mask so that the loss and metrics are well-defined. Using logical OR preserves every annotated lesion without double-counting.

---

## 🧩 Models Implemented

We implemented and evaluated three UNet-based architectures.

### 1️⃣ Baseline UNet

| Aspect | Description |
|--------|-------------|
| **Architecture** | Classic encoder–decoder UNet with symmetric skip connections. |
| **Encoder** | Custom CNN (no pretraining); 4 levels, single-channel input. |
| **Training** | Trained entirely from scratch. |
| **Observations** | Learns coarse tumor regions; tends to under-segment; lower confidence near boundaries. |

### 2️⃣ ResNet-UNet (Pretrained Encoder)

| Aspect | Description |
|--------|-------------|
| **Architecture** | Encoder replaced with **ResNet-18** pretrained on ImageNet; UNet-style decoder with skip connections. |
| **Input** | Grayscale images repeated to 3 channels for ResNet compatibility. |
| **Why it helps** | Pretrained encoders provide better low-level features, faster convergence, and more stable segmentation. |
| **Observations** | Cleaner boundaries, higher confidence predictions, best balance between precision and recall. |

### 3️⃣ Attention ResNet-UNet

| Aspect | Description |
|--------|-------------|
| **Architecture** | Builds on ResNet-UNet; adds **attention gates** on skip connections so the decoder focuses on relevant encoder features. |
| **Observations** | Slight improvement in recall in some settings; increased complexity; no consistent gain over ResNet-UNet on this dataset. |

### Summary: UNet Variants and Differences

| Model | Encoder | Skip connections | Pretrained | Main characteristic |
|-------|---------|------------------|------------|---------------------|
| **Baseline UNet** | Custom 4-level CNN | Standard concat | No | Simple, from-scratch baseline. |
| **ResNet-UNet** | ResNet-18 | Standard concat | Yes (ImageNet) | Strong features, best overall. |
| **Attention ResNet-UNet** | ResNet-18 | Attention-gated | Yes (ImageNet) | Focus on salient regions; more parameters. |

---

## ⚙️ Training Details

| Setting | Value |
|---------|--------|
| **Loss** | Dice Loss (optionally Dice + BCE) |
| **Optimizer** | Adam |
| **Learning rate** | 1e-4 |
| **Input size** | 256 × 256 |
| **Prediction threshold** | 0.5 |
| **Strategy** | Same pipeline for all models for fair comparison |

---

## 📊 Evaluation Metrics

Accuracy alone is not suitable for segmentation (class imbalance). We used:

- **Dice Similarity Coefficient (DSC)** — overlap between prediction and ground truth.
- **Intersection over Union (IoU / Jaccard)** — standard segmentation metric.
- **Precision** — fraction of predicted positives that are true positives.
- **Recall (Sensitivity)** — fraction of true positives that are predicted.
- **Specificity** — fraction of true negatives correctly predicted.

---

## 📈 Quantitative Results

| Model | Dice | IoU | Precision | Recall | Specificity |
|-------|------|-----|-----------|--------|-------------|
| Baseline UNet | ~0.68 | ~0.58 | ~0.84 | ~0.66 | ~0.99 |
| ResNet-UNet | **~0.75** | **~0.69** | ~0.96 | ~0.72 | ~0.998 |
| Attention ResNet-UNet | ~0.70 | ~0.64 | ~0.90 | ~0.67 | ~0.995 |

**Best performing model:** ResNet-UNet (highest Dice and IoU with strong precision and recall).

---

## 🚧 Challenges Faced and Solutions

| Challenge | Problem | Solution |
|-----------|---------|----------|
| **Multiple masks per image** | Some BUSI scans have more than one mask file. | Merged all masks using logical OR into one binary mask per image. |
| **Threshold sensitivity** | Predictions visible only at low thresholds initially. | Stabilized training (Dice loss, same pipeline) and fixed inference threshold at 0.5. |
| **Architecture–weight mismatch** | Saved models failed to load after refactoring. | Ensured **exact** architectural consistency between training and inference (same class names and layer order). *PyTorch weights are tied to the architecture that created them.* |
| **Paths and structure** | Path mismatches and module name conflicts across machines. | Clean separation of data path, architectures, and scripts; use relative paths or config where possible. |

---

## 🏆 Final Model Selection

**Selected model: ResNet-UNet**

- Highest Dice and IoU.
- Strong precision without sacrificing recall.
- Stable and reproducible.
- Suitable for extension to CT-based kidney tumor segmentation and downstream tasks (e.g. nephrometry).

---

## 🔮 Future Work

Planned extensions (aligned with nephrometry and clinical use):

1. **Extend segmentation to renal CT images** — same UNet backbone, different modality and preprocessing.
2. **CT-specific preprocessing** — e.g. HU windowing for soft tissue and contrast phases.
3. **Joint kidney and tumor segmentation** — multi-class or two-stage pipeline.
4. **Automated nephrometry score extraction** from segmented tumors:
   - Tumor size (radius or volume)
   - Depth (relationship to kidney surface)
   - Location (polarity, exophytic vs endophytic)
   - Proximity to collecting system

This links the current BUSI-based comparison directly to renal applications and viva discussions.

---

## 📁 Repository Structure

```
├── README.md
├── requirements.txt
├── basic_unet_segmentation.ipynb      # Baseline UNet
├── Res_Unet_segmentation.ipynb        # ResNet-UNet
├── Attemtion_Unet_segmentation.ipynb  # Attention ResNet-UNet
├── dataset/                           # (optional) Dataset loader utilities
├── weights/                           # (optional) Trained model weights
├── train_*.py                         # (optional) Training scripts
├── evaluate.py                        # (optional) Evaluation script
└── inference.py                       # (optional) Inference script
```

The core deliverables are the three Jupyter notebooks; optional Python modules can be added for reproducibility and scripting.

### Additional recommended files (for a clean, reproducible repo)

| File / folder | Purpose |
|---------------|---------|
| `dataset/` | Reusable BUSI dataset loader and mask-merging logic so you don’t duplicate code across notebooks. |
| `dataset/busi_loader.py` | `build_data_dict()` and `BUSIDataset` (same logic as in the notebooks). |
| `evaluate.py` | Script to load saved weights, run the model on a validation/test set, and compute Dice, IoU, precision, recall, specificity. |
| `inference.py` | Script to load a trained model and generate a segmentation mask for a single image (or a folder of images). |

Skeleton versions of these are provided. To use `evaluate.py` and `inference.py`, implement `get_model(arch)` so it returns the same architecture as in the notebooks (or move model definitions into a shared `architectures.py` and import from there).

---

## 📋 Requirements

- **Python:** 3.8+ (3.9 or 3.10 recommended).
- **GPU:** Optional but recommended for training (CUDA-compatible PyTorch).

Install dependencies:

```bash
pip install -r requirements.txt
```

See `requirements.txt` for pinned versions of PyTorch, torchvision, kagglehub, NumPy, Pillow, matplotlib, and Jupyter.

---

## ▶️ How to Run

1. **Install dependencies:** `pip install -r requirements.txt`
2. **Download the dataset** (e.g. using `kagglehub` in the notebook):
   ```python
   import kagglehub
   path = kagglehub.dataset_download("aryashah2k/breast-ultrasound-images-dataset")
   ```
3. **Update the data path** in each notebook to point to the extracted BUSI folder (e.g. `.../Dataset_BUSI_with_GT`).
4. **Run the desired notebook** (Baseline UNet, ResNet-UNet, or Attention ResNet-UNet) from top to bottom to train and evaluate.
5. **(Optional)** Use the evaluation or inference scripts to load saved weights and compute metrics or visualizations.

---

## ⚠️ Disclaimer

This project is intended for **educational and research purposes** only. It is not approved for clinical use.
