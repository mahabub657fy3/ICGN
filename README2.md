<div align="center">

<h1>
  <img src="figures/Overall_pipeline.png" width="100%" alt="ICGN Pipeline"><br><br>
  Many Men, Many Minds
</h1>

<h3>An Instance-Adaptive Conditioning Approach for Data-Efficient Transferable Targeted Attacks</h3>

<br>

[![Paper](https://img.shields.io/badge/Paper-PDF-red?style=for-the-badge&logo=adobeacrobatreader)](https://github.com/mahabubur657fy3/ICGN)
[![GitHub Stars](https://img.shields.io/github/stars/mahabubur657fy3/ICGN?style=for-the-badge&logo=github)](https://github.com/mahabubur657fy3/ICGN)
[![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.1-orange?style=for-the-badge&logo=pytorch)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

<br>

**Md Mahabubur Rahman · Hui Zhang · Biwei Chen · Anjie Peng · Hui Zeng**

*Southwest University of Science and Technology · Beijing Normal University*

</div>

---

## 📌 Overview

**Instance-Conditioned Generative Network (ICGN)** is a novel generative framework for **transfer-based targeted adversarial attacks**. It addresses two fundamental limitations of existing multi-target generative attacks:

1. **Class-static conditioning** — prior methods apply the same target representation to *every* source image, ignoring instance-level visual context (background, pose, texture, illumination).
2. **Single-point injection** — conditioning signals are injected only at one shallow layer, limiting how far target semantics propagate through the generator hierarchy.

ICGN resolves both failures through two coupled innovations:

| Component | Description |
|-----------|-------------|
| **Instance-Adaptive Prompt Learning (IPL)** | A lightweight meta-network predicts a per-image bias over a frozen CLIP model, producing a distinct target embedding for each source image that captures both target-class semantics and instance-level visual context. |
| **Multi-Depth Feature Modulation (MFM)** | The conditioning signal is injected at **17 points** across encoder, residual bottleneck, and decoder via channel-wise affine modulation (CFM), ensuring consistent target guidance throughout the generation hierarchy. |

> **Key result:** ICGN is the *only* method to maintain meaningful attack success rate (>25%) under severely limited training data (10k–20k samples), while consistently outperforming all baselines at every training scale.

---

## 🔥 Highlights

- ✅ **+22.09% minimum transferability gain** over the strongest baseline (ResNet-50 → DenseNet-121, 325k training images)
- ✅ **20.80% average ASR at just 10k training images**, while competing methods converge near 0%
- ✅ **Robust to defenses**: outperforms baselines under JPEG compression and adversarially-trained models
- ✅ **Open-world transferability**: 31/50 on Google Cloud Vision API, 26/50 on GPT-4o mini
- ✅ **More efficient than CGNC**: GPU memory reduced from 16.6 GB → 11.3 GB; training time from 7.05 h → 4.91 h per epoch

---

## 🏗️ Repository Structure

```
ICGN/
├── train.py                   # Unified training script (CIFAR-10 + ImageNet)
├── eval.py                    # Unified evaluation / adversarial example generation
├── inference.py               # Attack success rate measurement across victim models
├── generator.py               # MDMGenerator — Multi-Depth Modulation architecture
├── prompt_learner.py          # CLIP-based Instance-Adaptive Prompt Learner (IPL)
├── utils.py                   # Models, normalization, data loading, class indices
├── imagenet_class_index.json  # ImageNet class label mapping
├── cifar10_class_index.json   # CIFAR-10 class label mapping
└── figures/
    └── Overall_pipeline.png   # Architecture overview figure
```

---

## ⚙️ Installation

**Requirements:** NVIDIA GPU with CUDA 11.8+, Conda

```bash
# Step 1 — Create and activate the Conda environment
conda env create -f environment.yml
conda activate ICGN

# Step 2 — Install CLIP from OpenAI
pip install git+https://github.com/openai/CLIP.git
```

<details>
<summary><b>Pinned dependency versions</b></summary>

| Package | Version |
|---------|---------|
| PyTorch | 2.2.1 |
| torchvision | 0.17.1 |
| timm | 1.0.22 |
| numpy | 1.26.4 |
| pandas | 2.3.3 |

</details>

---

## 📦 Data Preparation

### ImageNet (NeurIPS 2017 Validation Set)

Download the [NeurIPS 2017 adversarial attack dataset](https://www.kaggle.com/c/nips-2017-non-targeted-adversarial-attack) and organize as follows:

```
neurips2017_dev/
├── images/
│   ├── ILSVRC2012_val_00000001.png
│   └── ...
└── images.csv          # columns: ImageId, TrueLabel, TargetClass
```

For training, point `--train_dir` to your local ILSVRC 2012 training set.

### CIFAR-10

Convert the binary CIFAR-10 dataset to PNG folders organized by class:

```
cifar10_png/
├── train/
│   ├── airplane/
│   ├── automobile/
│   └── ...
├── test/
│   ├── airplane/
│   └── ...
└── Cifar_10_val.txt    # format: <filename>,<label> per line
```

---

## 🚀 Quick Start

### Training

**ImageNet — ResNet-50 surrogate (N8 protocol, 8 target classes)**
```bash
python train.py \
  --dataset imagenet \
  --train_dir /data/ImageNet/ILSVRC2012_img_train \
  --model_type res50 \
  --label_flag N8 \
  --epochs 10
```

**ImageNet — Inception-v3 surrogate (C50 protocol, 50 target classes)**
```bash
python train.py \
  --dataset imagenet \
  --train_dir /data/ImageNet/ILSVRC2012_img_train \
  --model_type incv3 \
  --label_flag C50 \
  --epochs 10
```

**CIFAR-10 — ResNet-56 surrogate**
```bash
python train.py \
  --dataset cifar10 \
  --train_dir /data/cifar10/train \
  --model_type cifar10_resnet56 \
  --label_flag ALL \
  --epochs 10
```

Checkpoints are saved to:
```
checkpoints_{dataset}/{model_type}/
├── model-{epoch}.pth    # Generator weights
└── prompt-{epoch}.pth   # Conditioner (IPL) weights
```

---

### Generating Adversarial Examples

**ImageNet — using a trained Inception-v3 checkpoint**
```bash
python eval.py \
  --dataset imagenet \
  --data_dir /data/neurips2017_dev \
  --model_type incv3 \
  --label_flag N8 \
  --load_g_path checkpoints_imagenet/incv3/model-9.pth \
  --load_cond_path checkpoints_imagenet/incv3/prompt-9.pth
```

**CIFAR-10 — using a trained VGG-19_bn checkpoint**
```bash
python eval.py \
  --dataset cifar10 \
  --data_dir /data/cifar10_png/test \
  --model_type cifar10_vgg19_bn \
  --label_flag ALL \
  --load_g_path model-9.pth \
  --load_cond_path prompt-9.pth \
  --val_txt Cifar_10_val.txt
```

Generated adversarial examples are saved to:
```
results_{dataset}/gan_{label_flag}/{model_type}_t{class_id}/images/
```

---

### Measuring Attack Success Rate

**Against standard ImageNet models**
```bash
python inference.py \
  --dataset imagenet \
  --test_dir results_imagenet/gan_n8/res50 \
  --model_t normal \
  --label_flag N8
```

**Against CIFAR-10 classifiers**
```bash
python inference.py \
  --dataset cifar10 \
  --test_dir results_cifar10/gan_all/cifar10_resnet56 \
  --model_t cifar \
  --label_flag ALL
```

---

## 🔧 Key Arguments

### `train.py` / `eval.py`

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `imagenet` | Dataset: `cifar10` or `imagenet` |
| `--model_type` | `res50` | Surrogate model identifier (see table below) |
| `--label_flag` | `N8` | Target class subset (see table below) |
| `--eps` | `16` | L∞ perturbation budget (pixels ÷ 255) |
| `--nz` | `16` | Noise dimension |
| `--k` | auto | Gaussian low-pass kernel size (`2` for CIFAR-10, `4` for ImageNet) |
| `--clip_backbone` | `ViT-B/16` | CLIP vision encoder backbone |
| `--n_ctx` | `16` | Number of learnable context tokens in IPL |
| `--batch_size` | auto | `128` for CIFAR-10, `8` for ImageNet |

### Supported Surrogate Models

| Dataset | `--model_type` | Architecture |
|---------|---------------|--------------|
| ImageNet | `res50` | ResNet-50 |
| ImageNet | `incv3` | Inception-v3 |
| CIFAR-10 | `cifar10_resnet56` | ResNet-56 |
| CIFAR-10 | `cifar10_vgg19_bn` | VGG-19 with BN |

### Class Subset Protocols (`--label_flag`)

| Dataset | Flag | # Target Classes |
|---------|------|-----------------|
| CIFAR-10 | `ALL` | 10 |
| ImageNet | `N8` | 8 (standard benchmark) |
| ImageNet | `C50` | 50 (large-scale evaluation) |

---

## 📊 Main Results

### ImageNet — Standard Black-Box Models (325k training samples)

| Source | Method | Inc-v3 | Res-152 | DN-121 | GoogLeNet | VGG-16 | Res-50 |
|--------|--------|--------|---------|--------|-----------|--------|--------|
| **Inc-v3** | C-GSP | 7.88* | 1.03 | 3.41 | 0.66 | 0.99 | 2.85 |
| | LFAA | 36.41* | 12.94 | 14.78 | 18.75 | 10.91 | 12.70 |
| | CGNC | 22.88* | 2.73 | 6.20 | 3.73 | 3.23 | 5.65 |
| | **ICGN (Ours)** | **65.47*** | **24.59** | **34.31** | **26.27** | **32.55** | **27.80** |
| **Res-50** | C-GSP | 16.81 | 35.21 | 47.24 | 18.22 | 30.46 | 74.10* |
| | LFAA | 9.75 | 43.14 | 58.65 | 24.00 | 35.92 | 89.20* |
| | CGNC | 19.14 | 41.78 | 51.33 | 22.12 | 36.76 | 73.05* |
| | **ICGN (Ours)** | **22.62** | **68.85** | **80.74** | **46.06** | **61.26** | **95.65*** |

*\* denotes white-box (surrogate = victim) result.*

### Data-Efficient Training (Res-50 surrogate, N8 protocol, Avg ASR over 5 victims)

| Training Size | C-GSP | LFAA | GAKer | CGNC | **ICGN (Ours)** |
|--------------|-------|------|-------|------|----------------|
| 10k | 0.06 | 0.22 | 0.09 | 0.05 | **20.80** |
| 20k | 0.05 | 1.53 | 0.11 | 0.07 | **27.93** |
| 50k | 0.95 | 1.97 | 6.86 | 15.22 | **36.02** |
| 100k | 12.06 | 18.99 | 28.14 | 18.86 | **40.67** |
| 325k | 29.59 | 34.29 | 34.54 | 34.23 | **55.91** |

### CIFAR-10 Results

| Source | Method | VGG-19 | VGG-16 | VGG-13 | Res-56 | Res-44 | Res-20 |
|--------|--------|--------|--------|--------|--------|--------|--------|
| **VGG-19** | CGNC | 68.33* | 49.52 | 56.03 | 50.78 | 48.47 | 46.18 |
| | **ICGN** | **76.96*** | **60.77** | **66.38** | **54.89** | **53.65** | **49.10** |
| **Res-56** | CGNC | 37.72 | 33.76 | 41.19 | 73.59* | 53.73 | 46.24 |
| | **ICGN** | **43.89** | **53.82** | **50.33** | **86.50*** | **61.35** | **57.37** |

### Open-World Recognition Systems (50 adversarial examples per method)

| Method | Google Cloud Vision API | GPT-4o mini |
|--------|------------------------|-------------|
| C-GSP | 7 | 0 |
| LFAA | 13 | 5 |
| GAKer | 9 | 8 |
| CGNC | 14 | 4 |
| **ICGN (Ours)** | **31** | **26** |

---

## 🔬 Ablation Study

| Configuration | VGG-16 | GoogLeNet | Inc-v3 | Res-152 | DN-121 | **Avg** |
|--------------|--------|-----------|--------|---------|--------|---------|
| **Full ICGN** | **61.26** | **46.06** | **22.63** | **68.85** | **80.73** | **55.91** |
| w/o Multi-Depth | 54.90 | 40.50 | 19.32 | 61.87 | 77.82 | 50.88 (−5.03) |
| w/o Instance Conditioning | 49.75 | 36.77 | 15.33 | 55.57 | 74.06 | 46.30 (−9.61) |
| w/o Both | 16.21 | 11.71 | 6.32 | 13.47 | 27.33 | 15.01 (−40.90) |

Both components are essential and **synergistic** — removing either causes substantial degradation; removing both collapses performance to near-baseline levels.

---


## 📈 Computational Efficiency

| Metric | CGNC | **ICGN (Ours)** |
|--------|------|----------------|
| GPU Memory | 16.6 GB | **11.3 GB** |
| Training Time / Epoch | 7.05 h | **4.91 h** |

ICGN achieves superior performance with lower computational cost than CGNC.

---
