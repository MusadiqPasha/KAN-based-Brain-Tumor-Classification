# KAN-based Brain Tumor Classification

## Table of Contents
- [Overview](#overview)
- [Motivation](#motivation)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Code Structure](#code-structure)
- [Results](#results)
- [Future Work](#future-work)
- [References](#references)

---

## Overview

This project explores the use of **Kolmogorov-Arnold Networks (KANs)** as an alternative to Convolutional Neural Networks (CNNs) for classifying brain tumors from MRI images.

KANs use functional approximations via basis functions (like splines) and leverage additive modeling, making them potentially more powerful for datasets where traditional CNNs may overfit or underperform due to limited data.

---

## Motivation

- Traditional CNNs work well with large datasets but may struggle with small, complex datasets like medical images.
- KANs are better suited for learning fine-grained details and offer flexibility in function approximation.
- Our goal is to evaluate the effectiveness of KANs in detecting brain tumors with limited labeled data.

---

## Dataset

- **Source**: [Kaggle - Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- **Structure**: MRI images of human brains categorized into:
  - **No Tumor**
  - **Glioma**
  - **Meningioma**
  - **Pituitary Tumor**

The images are grayscale and preprocessed to standard size (e.g., 128x128 or 224x224). The dataset is split into training and testing sets.

---

## Methodology

### 1. **Data Preprocessing**
- Converted images to grayscale and normalized pixel values.
- Resized all images to a uniform dimension.
- Encoded class labels and split the data into training and test sets.

### 2. **Model Architectures Compared**
- **CNN (Baseline)**:
  - Used 2–3 convolutional layers followed by max pooling, dense layers.
  - Trained using Adam optimizer and categorical cross-entropy loss.
  
- **KAN Model**:
  - Implemented using spline interpolation layers.
  - Replaces linear transformations with spline-based units.
  - Trained with similar hyperparameters for a fair comparison.
  - Used techniques from recent KAN literature to structure layers.

### 3. **Training and Evaluation**
- Evaluated both models on:
  - Accuracy
  - Loss over epochs
  - Confusion matrix
- KAN demonstrated superior edge-detection and performance stability on smaller subsets.

---

## Code Structure

```
├── Matrixx_078_079_095_107_code.ipynb # Jupyter notebook with full training & evaluation
├── Matrixx_078_079_095_107_Paper.pdf # Detailed project report
├── Matrixx_078_079_095_107_PPT.pdf   # Project PPT
├── README.md                         # Project documentation
```

### Key Files Explained:

- `main.py`:
  - CLI-compatible script to preprocess data, train model, and save results.
  - Supports switching between CNN and KAN via flags.

- `Matrixx_078_079_095_107_code.ipynb`:
  - Full model training pipeline with code blocks for:
    - Data loading
    - Visualization
    - Model definitions (CNN and KAN)
    - Accuracy/loss plots
    - Confusion matrix visualization
  - Also includes early stopping and validation accuracy tracking.

---

## Results

- KAN outperformed CNN in terms of:
  - Generalization on small datasets
  - Pixel-level boundary detection
- Accuracy scores improved by ~5–10% in some test runs.
- KAN showed more stable learning curves with reduced overfitting.

---

## Future Work

- Apply **U-KAN** for segmentation of tumor boundaries.
- Experiment with hybrid CNN+KAN architectures.
- Evaluate performance on other medical imaging datasets (e.g., CT, X-ray).
- Explore transfer learning using pretrained KAN backbones.

---

## References

1. [KAN: Kolmogorov-Arnold Networks](https://arxiv.org/abs/2404.19756)
2. [U-KAN: Medical Image Segmentation](https://arxiv.org/abs/2406.02918)
3. [Suitability of KANs for Vision](https://arxiv.org/html/2406.09087v1)
4. [Convolutional Kolmogorov–Arnold Networks](https://arxiv.org/abs/2406.13155)

