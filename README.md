# Heart-disease prediction — EPFL CS-433 Project 1

This project predicts the BRFSS 2015 survey label `_MICHD` (self-reported coronary heart disease or heart attack). The six required methods are in `implementations.py`; the prediction pipeline uses NumPy without external ML libraries.

## Setup and data

Install [uv](https://docs.astral.sh/uv/) and run `uv sync --frozen` from the repository root. Download the course-provided dataset and place these three files, with their original names, directly in `data/raw/`:

```text
data/raw/
├── x_train.csv
├── y_train.csv
└── x_test.csv
```

Create `data/raw/` if it is missing. If the download is a ZIP with a `dataset/` directory, extract the three CSVs from that directory. The dataset is not in this repository; `data/raw/*.csv` is ignored by git. `sample_submission.csv` is not needed.

Run the current prediction recipe with:

```bash
uv run python run.py --jobs 1
```

This rebuilds the processed data, trains on `x_train.csv` and `y_train.csv`, and writes `outputs/submissions/best.csv` for the rows and IDs in `x_test.csv`. A larger `--jobs` value fits members in parallel and uses more memory. The test labels are unavailable locally, so `run.py` does not print a test F1.

## Data preparation

`p1/cleaning.py` applies fixed BRFSS codebook rules to both train and test: it removes 30 survey, duplicate, or mixed-unit columns; converts numeric special codes and units; and fills numeric answers implied by survey skips. This leaves 291 predictors: 56 numeric and 235 categorical. Categorical unknown/refused codes remain distinct. The rules and exploratory checks are documented in `docs/eda_findings.md`.

For linear models, `p1/transforms.py` imputes numeric medians, adds missingness indicators, one-hot encodes categorical levels seen at least 30 times, and standardizes the columns. Each fitted step uses only the training rows of its cross-validation fold.

## Models

We tested the six required methods in `implementations.py`: MSE gradient descent, MSE stochastic gradient descent (batch size 1), least squares, ridge, logistic gradient descent, and regularized logistic gradient descent. Their settings are in `configs/step2/`.

We selected gradient-boosted decision trees (GBDT) because they improved validation F1 over these methods. GBDT builds trees sequentially, with each tree correcting errors left by the earlier ones. Our histogram-based implementation handles missing values and uses only NumPy and the Python standard library; it is in `p1/gbdt.py`. More information on the report.

`run.py` reads `configs/best.toml`: three depth-5 GBDTs from `configs/gbdt_tuned.toml`, each with 550 trees, learning rate 0.05, and model seeds 0–2. It averages their logits with equal weights and classifies at the frozen threshold 0.2166.

## Layout

```text
ML_p1/
├── run.py                # trains the selected model and writes best.csv
├── implementations.py    # six required ML methods
├── helpers.py            # course-provided CSV helpers
├── p1/                   # cleaning, transforms, models, CV and ensembles
├── configs/              # model presets; step2/ has the six methods
├── data/raw/             # locally downloaded CSVs
├── docs/                 # project description, EDA and experiment records
├── notebooks/            # exploratory analysis and cleaning
├── grading_tests/        # course public tests
├── slurm/                # cluster job scripts
├── outputs/              # generated CV results and submission CSVs
├── pyproject.toml        # Python dependencies
└── uv.lock               # locked dependency versions
```

## Evaluation and results

All results below come from local validation on the labeled training set. The labels for `x_test.csv` are hidden locally; AIcrowd computes its score after `best.csv` is submitted.

The six required methods and raw baseline were compared on the same five shuffled folds (seed 42; all 328,135 training rows). Preprocessing is refitted within each training fold. Each model chooses its F1 threshold on its training scores, then predicts the held-out fold. The table reports positive-class F1 as mean ± fold standard deviation and mean ROC AUC. Run `uv run p1 cv configs/step2/*.toml baseline` to repeat this comparison; results and fold scores are written under `outputs/`.

| Method | Validation F1 | Validation AUC |
|---|---:|---:|
| Raw least squares baseline | 0.4095 ± 0.0019 | 0.845 |
| MSE / GD | 0.4289 ± 0.0022 | 0.858 |
| MSE / SGD | 0.4255 ± 0.0033 | 0.855 |
| Least squares | 0.4301 ± 0.0018 | 0.857 |
| Ridge | 0.4297 ± 0.0031 | 0.858 |
| Logistic / GD | 0.4390 ± 0.0027 | 0.864 |
| Regularized logistic / GD | 0.4388 ± 0.0022 | 0.864 |

The later model comparison used three split seeds (42–44), five folds per seed, and a threshold chosen from the other folds' out-of-fold predictions. These results use a different protocol from the table above:

| Model | Validation F1 | Validation AUC |
|---|---:|---:|
| Logistic regression (reference) | 0.4386 | 0.8643 |
| **Three-seed GBDT (current `run.py` configuration)** | **0.4462** | **0.8703** |

The GBDT improves on logistic regression by 0.0077 F1 in the paired comparison (corrected 95% interval [0.0038, 0.0115]).

The GBDT validation score comes from averaging fold models (`fc_bag`); `run.py` refits the same three model seeds on all training rows. Model selection also used the CV data, so the table is not fully nested model-selection CV. See `docs/part3_results.md` and `report/report.pdf` for the comparisons and their limitations.

To repeat the three-seed GBDT comparison, run five-fold CV for each model seed and split seed, then average the out-of-fold logits:

```bash
for m in 0 1 2; do
  for s in 42 43 44; do
    uv run p1 cv gbdt_tuned --set name=fc_m$m --set seed=$s --set model.seed=$m
  done
done
uv run p1 blend fc_m0 fc_m1 fc_m2 --name fc_bag --weights 1 1 1
```
