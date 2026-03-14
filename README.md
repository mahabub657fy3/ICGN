# ICGN: Instance-Conditioned Generative Network

A unified framework for generating targeted adversarial examples using a CLIP-conditioned generator, supporting both **CIFAR-10** and **ImageNet**.

## Overview

![ICGN Framework Overview](figures/Overall_pipeline.png)

Transfer-based targeted adversarial attacks pose significant challenges as they must elicit target-consistent misclassification on unfamiliar victim architectures without any query access. Despite advancements in multi-target generative frameworks that enhance inference speed, they frequently depend on extensive training datasets and utilize very weak conditioning, generally including class labels or precomputed CLIP text embeddings by straightforward concatenation into intermediate feature maps. This class-static conditioning offers merely a rudimentary control signal and does not leverage the more nuanced target semantics that may be adaptively aligned with each source instance. To mitigate these limitations, we propose an Instance-Conditioned Generative Network (ICGN) that generates an input-dependent conditioning representation by deriving an instance-specific semantic token from CLIP through a streamlined network, integrating it with learnable context vectors, and incorporating it into the generator through feature-wise linear modulation at various scales. This instance-adaptive conditioning approach facilitates enhanced semantic alignment between each source image and the target class, unlike the class-static prompts utilized by previous systems, and produces effective adversarial transferability even with limited training data. Experiments on ImageNet and CIFAR-10 reveal that ICGN significantly enhances the targeted success rate (TSR) across all data conditions: in the 325k ImageNet training set, ICGN achieves an average TSR of 62.53\%, representing an increase of 19.09 percentage points over the most robust baseline, while in the severely low-data condition of the 10k training set, ICGN reaches a TSR of 29.66\%, whereas all previous methods falter to below 0.30\%. Ablation investigations confirm the supplementary roles of each suggested component.

## Repository Structure

```
├── train.py                  # Unified training script (CIFAR-10 + ImageNet)
├── eval.py                   # Unified evaluation / generation script
├── inference.py              # Attack success rate measurement across models
├── generator.py              # MDMGenerator
├── prompt_learner.py         # CLIP-based Conditioner
├── utils.py                  # Models, normalization, data loading, class indices
├── imagenet_class_index.json
└── cifar10_class_index.json
```

---

## Installation

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/ICGN.git
cd ICGN

# 2. Create and activate the conda environment (Python 3.10, CUDA 11.8, PyTorch 2.2.1)
conda env create -f environment.yml
conda activate LP-LFGA

# 3. Install CLIP from OpenAI
pip install git+https://github.com/openai/CLIP.git
```
Requirements: NVIDIA GPU with CUDA 11.8+. The environment.yml pins all exact package versions used in our experiments (PyTorch 2.2.1, torchvision 0.17.1, timm 1.0.22, numpy 1.26.4, pandas 2.3.3).

## Quick Start

### Training

**CIFAR-10**
```bash
python train.py \
  --dataset cifar10 \
  --train_dir /data/cifar10_png/train \
  --model_type cifar10_resnet56 \
  --label_flag ALL \
  --epochs 10
```

**ImageNet — ResNet-50**
```bash
python train.py \
  --dataset imagenet \
  --train_dir /data/ImageNet/ILSVRC2012_img_train \
  --model_type res50 \
  --label_flag C50 \
  --epochs 10
```

**ImageNet — Inception-v3**
```bash
python train.py \
  --dataset imagenet \
  --train_dir /data/ImageNet/ILSVRC2012_img_train \
  --model_type incv3 \
  --label_flag N8 \
  --epochs 10
```

Checkpoints are saved to `checkpoints_{dataset}/{model_type}/model-{epoch}.pth` and `prompt-{epoch}.pth`.

### Generating Adversarial Examples

**CIFAR-10**
```bash
python eval.py \
  --dataset cifar10 \
  --data_dir /data/cifar10_png/test \
  --model_type cifar10_vgg19_bn \
  --label_flag ALL \
  --load_g_path checkpoints_cifar10/cifar10_resnet56/model-9.pth \
  --load_cond_path checkpoints_cifar10/cifar10_resnet56/prompt-9.pth \
  --val_txt Cifar_10_val.txt
```

**ImageNet**
```bash
python eval.py \
  --dataset imagenet \
  --data_dir /data/neurips2017_dev \
  --model_type incv3 \
  --label_flag N8 \
  --load_g_path checkpoints_imagenet/incv3/model-9.pth \
  --load_cond_path checkpoints_imagenet/incv3/prompt-9.pth
```

Generated images are saved to `results_{dataset}/gan_{label_flag}/{model_type}_t{class_id}/images/`.

---

### Measuring Attack Success Rate

```bash
# Against a single model
python inference.py \
  --dataset imagenet \
  --test_dir results_imagenet/gan_n8/incv3 \
  --model_t res50 \
  --label_flag N8

# Against all standard models
python inference.py \
  --dataset imagenet \
  --test_dir results_imagenet/gan_n8/incv3 \
  --model_t all \
  --label_flag N8

# Against CIFAR-10 classifiers
python inference.py \
  --dataset cifar10 \
  --test_dir results_cifar10/gan_all/cifar10_resnet56 \
  --model_t cifar \
  --label_flag ALL
```

---

## Key Arguments

### `train.py` / `eval.py`

| Argument | Default | Description |
|---|---|---|
| `--dataset` | `imagenet` | `cifar10` or `imagenet` |
| `--model_type` | `incv3` | Surrogate model (see table below) |
| `--label_flag` | `N8` | Class subset (see table below) |
| `--eps` | `16` | L∞ perturbation budget (pixels, divided by 255) |
| `--nz` | `16` | Noise dimension |
| `--k` | auto | Gaussian lowpass kernel size (2 for CIFAR-10, 4 for ImageNet) |
| `--clip_backbone` | `ViT-B/16` | CLIP backbone |
| `--n_ctx` | `16` | Number of learnable context tokens |
| `--batch_size` | auto | 128 for CIFAR-10, 8 for ImageNet |

### Supported Models

| Dataset | `--model_type` |
|---|---|
| CIFAR-10 | `cifar10_resnet56`, `cifar10_resnet44`, `cifar10_resnet32`, `cifar10_resnet20`, `cifar10_vgg19_bn`, `cifar10_vgg16_bn`, `cifar10_vgg13_bn` |
| ImageNet | `res50`, `res152`, `incv3`, `incv4`, `vgg16`, `googlenet`, `dense121`, `inc_res_v2`, `adv_incv3`, `ens_inc_res_v2` |

### Class Subsets (`--label_flag`)

| Dataset | Flag | # Classes |
|---|---|---|
| CIFAR-10 | `ALL` | 10 |
| CIFAR-10 | `C8`, `C5`, `C3` | 8 / 5 / 3 |
| ImageNet | `N8` | 8 |
| ImageNet | `C20`, `C50`, `C100`, `C200` | 20 / 50 / 100 / 200 |

---

## Data Preparation

**CIFAR-10** — Convert the binary CIFAR-10 dataset to PNG folders organized by class:
```
cifar10_png/
  train/
    airplane/  automobile/  bird/  ...
  test/
    airplane/  automobile/  bird/  ...
  Cifar_10_val.txt   # <filename>,<label> per line
```

**ImageNet (NeurIPS 2017 dev set)** — Organize as:
```
neurips2017_dev/
  images/
    ILSVRC2012_val_00000001.png  ...
  images.csv    # columns: ImageId, TrueLabel, TargetClass
```
