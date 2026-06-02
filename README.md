# apple-cut-classification

A multi-modal machine learning pipeline for automated apple quality classification,
combining tabular physical attributes and image-based visual inspection. Submitted for
INF2008: Machine Learning.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mlemxy/apple-cut-classification/blob/main/main.ipynb)

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-337AB7?style=for-the-badge&logo=xgboost&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

---

## Overview

The pipeline is split into two complementary tracks:

**Tabular track** classifies apples as good or bad quality using physical attributes
(size, weight, sweetness, crunchiness, juiciness, ripeness, acidity) from the
[Apple Quality dataset](https://www.kaggle.com/datasets/nelgiriyewithana/apple-quality).
XGBoost, Logistic Regression, and Random Forest were trained, tuned via GridSearchCV,
and combined into a soft-voting ensemble classifier.

**Image track** classifies apples as healthy or rotten using the
[Fruit and Vegetable Disease dataset](https://www.kaggle.com/datasets/muhammad0subhan/fruit-and-vegetable-disease-healthy-vs-rotten/data).
A custom CNN was trained from scratch, followed by transfer learning with a pre-trained
ResNet50 (ImageNet weights, top layers replaced and fine-tuned on apple data).

---

## Pipeline

| Stage | Tabular Track | Image Track |
|---|---|---|
| Preprocessing | Imputation, StandardScaler, OHE, binary target encoding | Resize to 128x128, normalization, categorical encoding |
| Models | XGBoost, Logistic Regression, Random Forest | Custom CNN, ResNet50 (transfer learning) |
| Tuning | GridSearchCV (3-fold CV, scoring=accuracy) | 10 epochs (CNN), 5 epochs (ResNet50) |
| Ensemble | Soft-voting VotingClassifier | Not applicable |
| Evaluation | Accuracy, F1, ROC-AUC, confusion matrix | Accuracy, loss curves, validation accuracy |
| Export | AppleQualityClassifier.pkl | Not applicable |

---

## Datasets

| # | Dataset | Format | Source |
|---|---|---|---|
| 1 | Apple Quality | CSV | [Kaggle](https://www.kaggle.com/datasets/nelgiriyewithana/apple-quality) |
| 2 | Fruit and Vegetable Disease (Healthy vs Rotten) | Images | [Kaggle](https://www.kaggle.com/datasets/muhammad0subhan/fruit-and-vegetable-disease-healthy-vs-rotten/data) |

---

## Getting Started

**Prerequisites:** Python 3.x, Jupyter Notebook or Google Colab

```bash
# Clone the repository
git clone https://github.com/mlemxy/apple-cut-classification

# Install dependencies
pip install -r requirements.txt

# Launch the notebook
jupyter notebook main.ipynb
```

Place `apple_quality.csv` in the project root. For image data, extract the Kaggle zip
into the same directory before running the image track cells.

---

## Future Work

- Multi-modal fusion: combine tabular and image predictions into a unified classifier
- Real-time deployment via web or mobile interface
- Dataset expansion to cover additional fruit types and ripeness gradations
