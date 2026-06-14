Here is a complete, professional `README.md` file tailored specifically to your script's architecture. It outlines the technology stack (PLIP foundation model), data structure expectations, and how to execute the evaluation and interactive testing pipelines.

---

# Breast Cancer Histopathology Classification using PLIP

This repository contains a framework for evaluating and classifying breast cancer histopathology images from the **BreaKHis dataset**. Instead of training a convolutional neural network (CNN) from scratch, this pipeline leverages **PLIP (Pathology Language-Image Pre-training)**—a specialized medical foundation vision model—to extract high-quality latent features.

By utilizing L2-normalized image embeddings, the framework evaluates classification accuracy using matrix dot-product cosine similarity across different magnification tiers (**40X, 100X, 200X, and 400X**). It handles both **Overarching Binary Diagnostics** (Benign vs. Malignant) and **Fine-Grained Tissue Subtypes** (e.g., Adenosis, Fibroadenoma, Carcinoma variants).

---

##  Key Features

* **Foundation Model Backbone**: Uses `vinid/plip` via Hugging Face Transformers to generate vision-language aligned embeddings specifically optimized for pathology tissue.
* **Multi-Tier Magnification Evaluation**: Stratifies datasets dynamically and evaluates model capabilities separately across 40X, 100X, 200X, and 400X zoom scales.
* **Dual-Level Diagnostics**:
* **Binary Classification**: Benign vs. Malignant.
* **Multi-Class Subtype Classification**: Identifies specific histopathological tissue varieties.


* **Zero-Shot & Reference-Based Inference**: Computes exact matrix dot-product calculations to perform swift evaluation without resource-heavy classifier training loops.
* **Interactive Inference Terminal**: Includes a command-line interface to upload and test new local images on the fly.

---

## 📁 Dataset Tree Expectation

The script parses directory paths recursively to extract labels based on folder hierarchies. To match the setup, structure your BreaKHis folder path like this:

```text
BreaKHis_v1/
└── benign/
    └── SOB/
        └── adenosis/
            └── 40X/
                └── sob_b_a_14-22549-40x-001.png
└── malignant/
    └── SOB/
        └── ductal_carcinoma/
            └── 100X/
                └── sob_m_dc_14-5654-100x-001.png

```

---

## 🛠️ Installation & Requirements

Ensure you have Python 3.8+ and a CUDA-compatible environment (optional but recommended for faster feature extraction).

1. Clone this repository to your local workspace.
2. Install the required dependencies:
```bash
pip install torch numpy Pillow tqdm scikit-learn transformers

```


3. Open the script and modify the local directory variable to point to your data:
```python
DATASET_ROOT = "C:/Your/Path/To/BreaKHis_v1"

```



---

##  How to Use

### 1. Run Evaluation Pipeline

Execute the Python script in your terminal window. The framework will find the image files, load the PLIP backbone, extract embeddings, partition data via stratified splits, and generate performance matrix sheets:

```bash
python Final-Model.py

```

### 2. Output Artifacts

Once the evaluation run finishes, two summary sheets are generated in your local repository folder:

* `breakhis_plip_subtypes_results.txt`: A clean text report markdown matrix.
* `breakhis_plip_subtypes_results.csv`: A spreadsheet matrix containing raw calculation points.

### 3. Interactive Inference Loop

After printing global project statistics, the execution converts into a continuous testing shell script loop. You can pass raw path strings to custom images to see real-time predictions:

```text
══════════════════════════════════════════════════
   INTERACTIVE PLIP EXPERIMENT TESTER
══════════════════════════════════════════════════

Enter the path to a histopathology image to classify (or type 'exit' to quit): C:/test_sample.png
Available magnification tiers: ['40X', '100X', '200X', '400X']
Enter the magnification tier for this image (e.g., 40X, 100X, 200X, 400X): 200X

Running inference...
────────────────────────────────────────
   IMAGE:   test_sample.png
   TIER:    200X
   CLASS:   Malignant
   SUBTYPE: DUCTAL_CARCINOMA
────────────────────────────────────────

```

---

## Model Architecture & Methodology

The pipeline relies on the following operational pipeline:

1. **Feature Extraction**: Images pass through PLIP’s Image Transformer model. Spatial visual dimensions are mean-pooled down into a compact context array.
2. **L2 Normalization**: Embedding structures undergo $L_2$ standardization calculations:

$$\text{Normalized Features} = \frac{f}{\|f\|_2}$$


3. **Similarity Vector Indexing**: The testing vectors run matrix dot-product computations against the reference array. This maps an exact Cosine Similarity footprint:

$$\text{Similarity} = X_{\text{test}} \cdot X_{\text{train}}^T$$


4. **Argmax Isolation**: The system locates the highest vector score match and borrows its corresponding metadata parameters to label the query sample.
