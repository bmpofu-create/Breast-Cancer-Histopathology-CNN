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

## Hyperparameter Tuning Experiments

### Learning Rate Results

| Learning Rate | Best Validation Accuracy |
|---------------|--------------------------|
| 0.001 | 81.16% |
| 0.0005 | 86.75% |
| 0.0001 | 85.45% |

Best learning rate:

```text
0.0005
```

---

### L2 Regularization (Weight Decay)

| Weight Decay | Best Validation Accuracy |
|--------------|--------------------------|
| 0 | 85.73% |
| 1e-5 | 85.07% |
| 1e-4 | 85.82% |
| 1e-3 | 84.89% |

Best:

```text
1e-4
```

---

### Dropout Experiments

| Dropout | Best Validation Accuracy |
|----------|--------------------------|
| 0.3 | 86.75% |
| 0.5 | 81.72% |
| 0.7 | 84.70% |

Best:

```text
0.3
```

---

### Final Tuned Model

Parameters:

```text
Learning Rate = 0.0005
Weight Decay = 1e-4
Dropout = 0.3
```

Test Results:

| Metric | Value |
|----------|--------|
| Accuracy | 80.53% |
| Precision | 87.34% |
| Recall | 86.29% |
| Weighted F1 | 80.64% |

Observation:

Hyperparameter tuning and regularization did not outperform the original baseline CNN and reduced benign classification performance.

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