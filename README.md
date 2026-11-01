# Breast histopathology classification

Patient-separated experiments on BreaKHis and BRACS, plus a small app that classifies one region into the seven BRACS subtypes.

The datasets and trained weights are not in this repository. Download them separately and keep them in the folders named below.

## Layout

```text
app.py                          Streamlit classifier
src/multiclass_classify.py      7-class PLIP adapter used by the app
experiments/
  01_cnn_baseline/              BreaKHis benign/malignant CNN and tuning runs
  02_breakhis_plip_binary/      PLIP binary adapter, cosine 1-NN, and an early image-level run
  03_breakhis_plip_subtypes/    PLIP subtype runs, including patient-level 1-NN
  04_bracs_3class/              BRACS benign / atypical / malignant
  05_bracs_7class/              BRACS seven-subtype adapter and 1-NN
  06_retrieval/                 Retrieval experiments that call an external API
```

Each experiment folder has a `results/` directory with the report from that test.

On the seven-subtype task, the PLIP adapter separated these three classes best:

| Subtype | Patient-isolated F1 |
|---|---:|
| Invasive carcinoma | 0.90 |
| Flat epithelial atypia | 0.74 |
| Ductal carcinoma in situ | 0.73 |

That run is `experiments/05_bracs_7class`. Its patient-isolated accuracy was 65.19%, ahead of cosine nearest-neighbor at 55.31%.

## Data

Place the extracted datasets here:

```text
BreaKHis_v1/BreaKHis_v1/
BRACS_ROI/
```

BreaKHis images used for these runs: https://drive.google.com/drive/folders/1RjDOnK1qugoRBA66FzTfDt8gPpUGQoFz?usp=sharing

BRACS ROI is the public BRACS region-of-interest release. The metadata tables in `experiments/01_cnn_baseline/results/` store image paths relative to `BreaKHis_v1/BreaKHis_v1/`.

## Setup

```bash
pip install -r requirements.txt
```

The retrieval scripts also need:

```bash
pip install -r requirements-retrieval.txt
```

Copy `.env.example` to `.env` and add an API key before running `experiments/06_retrieval`. The real `.env` is ignored by git.

## Run

CNN baseline, from the repository root:

```bash
python experiments/01_cnn_baseline/train_baseline.py
```

BRACS 7-class experiment:

```bash
python experiments/05_bracs_7class/BRACS_ROI_7_CLASS.py
```

Train the app model, then open the classifier:

```bash
python src/multiclass_classify.py --train
streamlit run app.py
```

Classify one image from the command line:

```bash
python src/multiclass_classify.py --image path\to\image.png
```

The app is a research demo. It is not a diagnostic device.
