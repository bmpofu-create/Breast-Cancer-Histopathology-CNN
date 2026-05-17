# Breast Cancer Histopathology Classification Using CNN

This project investigates binary breast cancer classification using the BreaKHis dataset using Convolutional Neural Networks (CNNs).

## Objective

Develop and evaluate CNN-based methods for classifying breast histopathology images into:

- Benign
- Malignant

The study explores:

- Patient-wise splitting
- Class balancing
- Threshold adjustment
- Precision–recall tradeoffs
- Hyperparameter tuning

## Dataset

BreaKHis Dataset

- 7,909 images
- 81 patient IDs extracted
- Magnifications:
  - 40X
  - 100X
  - 200X
  - 400X

Patient-wise splitting was used to prevent data leakage.

## Baseline CNN Results

| Metric | Score |
|----------|--------|
| Accuracy | 85.29% |
| Precision | 93.30% |
| Recall | 86.39% |
| Weighted F1 | 85.72% |

## Threshold Experiment Results

| Threshold | Accuracy | Precision | Recall | Weighted F1 |
|------------|-----------|-----------|----------|-------------|
| 0.3 | 81.87% | 82.91% | 95.20% | 80.04% |
| 0.4 | 78.45% | 84.59% | 86.79% | 78.17% |
| 0.5 | 73.18% | 87.71% | 74.27% | 74.49% |
| 0.6 | 69.99% | 94.07% | 63.56% | 71.86% |
| 0.7 | 65.38% | 94.64% | 56.56% | 67.37% |

## Project Structure

```text
src/
models/
outputs/
```

## Run

Train:

```bash
python src/train_baseline.py
```

Evaluate:

```bash
python src/evaluate_baseline.py
```

Threshold experiment:

```bash
python src/threshold_experiment.py
```