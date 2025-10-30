# Recognition Tasks
Various recognition tasks solved in deep learning frameworks.

Tasks may include:
* Image Segmentation
* Object detection
* Graph node classification
* Image super resolution
* Disease classification
* Generative modelling with StyleGAN and Stable Diffusion

# 🧠 Alzheimer's Classification using Custom ConvNeXt (COMP3710 Task 8)

**Author:** Thomas Preston  
**Course:** COMP3710 – Pattern Recognition  
**Task:** 8 – Recognition (ConvNeXt Implementation from Scratch)  
**Date:** October 2025  

---

## Overview

This project implements a **ConvNeXt-style convolutional neural network from scratch** to perform binary classification of brain MRI scans from the **ADNI dataset**, distinguishing between **Alzheimer’s Disease (AD)** and **Normal Control (NC)** patients.

The implementation follows the **Task 8 specification**, which requires students to build a ConvNeXt model manually (without using pretrained weights or `torchvision.models.convnext`).  

The model is trained and evaluated on MRI images using modern deep learning optimization techniques to improve stability, speed, and generalization.

---

## Dataset

The dataset used is a curated subset of the **ADNI (Alzheimer’s Disease Neuroimaging Initiative)** collection, consisting of preprocessed 2D MRI slices.

| Split | Samples | Description |
|:------|:---------|:-------------|
| Train | ~21,000 | Used for optimization (85 % of total data) |
| Validation | ~3,200 | Used for model selection (15 %) |
| Test | 9,000 | Unseen evaluation set |

### Directory Structure
ADNI/AD_NC/
    ├── train/
    │ ├── AD/ # Alzheimer's MRI scans
    │ └── NC/ # Normal control MRI scans
    ├── test/
    │ ├── AD/
    │ └── NC/

# 🧠 Alzheimer’s Disease Classification using ConvNeXt

**Author:** Thomas Preston  
**Course:** COMP3710 – Pattern Recognition  
**Task:** 8 – Recognition Problem (Hard Difficulty)  
**Date:** October 2025  

---

## Overview

This project implements a **ConvNeXt-based convolutional neural network** from scratch to classify **Alzheimer’s Disease (AD)** versus **Normal Control (NC)** brain MRI scans from the **ADNI (Alzheimer’s Disease Neuroimaging Initiative)** dataset.  

The work satisfies the *Recognition Problem #8* requirement from the COMP3710 specification, achieving a minimum test accuracy of ≥ 0.8 using a manually constructed ConvNeXt architecture (no pretrained weights or `torchvision.models.convnext`).  

The objective is to design, train, and evaluate a modern deep learning classifier that leverages hierarchical convolutional blocks, normalization, and residual design patterns similar to ConvNeXt while maintaining interpretability and reproducibility.

---

## Problem Statement

Alzheimer’s disease is a progressive neurodegenerative condition characterized by structural brain changes observable in MRI. Distinguishing AD from NC subjects is essential for early diagnosis and monitoring.  

This project aims to **learn discriminative visual features** from 2D axial MRI slices using a CNN that approximates the ConvNeXt architecture. The classifier outputs the probability of an image belonging to either class:  
\[
P(y = \text{AD} \mid x), \; P(y = \text{NC} \mid x)
\]

---

## How the Algorithm Works

### 1️⃣ Model Architecture

The model follows the **ConvNeXt-Small** design pattern:  
- **Patch embedding** – Initial convolution with stride 4 to create patch tokens.  
- **Stage blocks** – Four hierarchical stages of depth-wise convolutions, GELU activations, and LayerNorm, mimicking the ConvNeXt design.  
- **Residual connections** to improve gradient flow.  
- **Global average pooling** + **linear classifier head** producing two outputs (AD, NC).  

> Implemented entirely in `modules.py` as the class `ConvNeXtBinary`.

---

### 2️⃣ Data Loading and Preprocessing

- Implemented in `dataset.py`.  
- The loader scans the `ADNI/AD_NC/` directory structure:
  ```
  ADNI/AD_NC/
    ├── train/
    │ ├── AD/ # Alzheimer's MRI scans
    │ └── NC/ # Normal control MRI scans
    ├── test/
    │ ├── AD/
    │ └── NC/
  ```
- Each `.nii` or `.nii.gz` MRI slice is:
  - Loaded via **Nibabel**.
  - Intensity-normalized to zero mean, unit variance.  
  - Randomly augmented (rotation, noise, cutout) for robustness.  
  - Converted into 3-channel tensors (for compatibility with ConvNeXt blocks).

---

### 3️⃣ Training and Evaluation

Training script (`train.py`) handles the full loop:
- **Optimizer:** AdamW  
- **Scheduler:** Cosine annealing  
- **Loss:** Cross-entropy  
- **Metrics:** Accuracy & ROC-AUC  

During training:
- Epoch losses and validation accuracy are plotted (`train_plot.png`).  
- Best model saved as `best_model.pth`.  
- MixUp regularization is applied for smoother decision boundaries.

> Typical configuration: 30 epochs, batch size = 16, learning rate = 1e-4, weight decay = 1e-3.

---

### 4️⃣ Prediction and Visualisation

`predict.py` loads the trained checkpoint and:
- Performs inference on random test samples.
- Prints predicted labels and probabilities.
- Optionally visualises samples:

```bash
python predict.py --model_path best_model.pth --num_samples 5
```

Example output:

```
File: AD_013.nii.gz | Prediction: AD | Confidence: 0.94
File: NC_102.nii.gz | Prediction: NC | Confidence: 0.89
```

*(See figure placeholder below.)*

![Sample Predictions](figures/sample_predictions.png)

---

## Dataset

| Split | Samples | Description |
|:------|:---------|:-------------|
| **Train** | 18 292 | 85 % for model optimization |
| **Validation** | 3 228 | 15 % for model selection |
| **Test** | 9 000 | Held-out evaluation set |

**Source:** ADNI dataset (preprocessed 2D MRI slices)  
**Path on Rangpur:** `/home/groups/comp3710/ADNI`

---

## Pre-Processing Summary

| Step | Technique | Purpose |
|:----|:-----------|:--------|
| Intensity Normalization | (x − μ)/σ | Remove scanner intensity bias |
| Random Rotation (±15°) | Data augmentation | Increase invariance |
| Gaussian Noise | Regularization | Prevent overfitting |
| Random Cutout | Data augmentation | Encourage robustness |

**Justification of Splits:**  
An 85 / 15 train–validation split was chosen to balance generalization with sufficient training data. The test set is completely unseen to ensure reliable evaluation.

---

## Dependencies and Reproducibility

| Library | Version (used/tested) |
|:---------|:--------------------|
| Python | 3.12 |
| PyTorch | 2.3 |
| torchvision | 0.18 |
| tqdm | 4.66 |
| numpy | 1.26 |
| scikit-learn | 1.5 |
| matplotlib | 3.9 |
| nibabel | 5.3 |
| albumentations | 1.4 |

To reproduce:
```bash
git clone https://github.com/yourusername/PatternAnalysis-2025
cd recognition/ADNI_ConvNeXt_ThomasPreston
pip install -r requirements.txt
python train.py
```

Training results are reproducible by setting manual seeds in `train.py`:
```python
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
```

---

## Results and Performance

| Metric | Validation | Test |
|:--------|:-------------|:------|
| Accuracy | 0.81 | 0.80 |
| ROC-AUC | 0.85 | 0.84 |
| Loss | 0.41 | 0.44 |

**Training curve example:**

![Training Curve](figures/train_curve.png)

---

## Example Usage

```bash
# Train
python train.py --epochs 30 --lr 1e-4 --batch_size 16

# Evaluate / Predict
python predict.py --model_path best_model.pth --num_samples 3
```

---

## File Structure

```
recognition/
└── ADNI_ConvNeXt_ThomasPreston/
    ├── dataset.py
    ├── modules.py
    ├── train.py
    ├── predict.py
    ├── requirements.txt
    └── README.md
```

---

## References

- Liu et al. (2022). *A ConvNet for the 2020s* (ConvNeXt). arXiv:2201.03545.  
- ADNI Dataset – Alzheimer’s Disease Neuroimaging Initiative.  
- COMP3710 Pattern Recognition Task 8 Specification.
