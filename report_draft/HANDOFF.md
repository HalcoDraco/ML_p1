# Handoff for the report authors

The paper is written as a finished manuscript, with the numerical methods and
neural-network experiments prominent. It is intentionally selective: a short
report is more convincing when each comparison answers a specific question.
This file is working documentation, not an appendix to submit.

## What the paper argues

1. Questionnaire semantics determine the representation: special codes and
   structured missingness cannot be treated as ordinary measurements.
2. The six required implementations provide a controlled comparison of loss
   functions, optimization procedures, and L2 regularization.
3. Neural networks test whether learned nonlinear interactions improve an
   already strong logistic baseline; recorded gains are limited.
4. A NumPy histogram GBDT produces the larger observed improvement. Later
   feature engineering and ensembles provide smaller, uncertain increments.
5. Validation results, competition feedback, and clinical interpretation answer
   different questions and are kept separate.

## Author details

Names supplied by Roger and included in `metadata.tex`:

- Jacobo Romero
- Pablo González
- Roger Baiges

Candidate addresses supplied by Roger, **not yet confirmed**:

- `jacobo.romerodiaz@epfl.ch`
- `pablo.gonzalezmonfort@epfl.ch`
- `roger.baigestrilla@epfl.ch`

The PDF uses the names and EPFL affiliation, without unconfirmed addresses.
Confirm all official addresses when entering the submission form. The team
name and AIcrowd submission identifier have not been supplied.

## Resolve before final submission

### Link the 0.456 score to the exact predictions

The score is confirmed by Roger. Neither a CSV nor an identifier mapping it
to a configuration is present in this checkout. Avoid guessing the model.

- `configs/best.toml` currently selects three `gbdt_tuned` members, model
  seeds 0, 1, 2, and threshold 0.2166. `run.py` refits these on all training rows.
- `docs/round3_summary.md` recommends `fe2_bag5`, `blend_fc_m5`, and `bag_cfg7`.
  The document describes their CSVs as combinations of CV-fold model predictions.
- Refitting on all rows is a different computation from averaging fold models.
  Matching the broad configuration is insufficient to guarantee the same CSV.

Retrieve the chosen CSV, submission identifier, and exact generation procedure
from the cluster session. Verify equality against the selected submission, and
only then update the final recipe and the manuscript's final paragraph. The
report preparation did not modify any model or submission recipe.

### Recover neural-network code if these experiments must be reproducible

The initial MLP architecture is documented (one hidden ReLU layer, widths
64/256, Adam, early stopping, dropout 0.3 for the larger network), but its
source, learning rate, batch size, and stopping details are not archived.
The later `m5` repeated-CV experiment is documented separately; do not assign
the initial architecture to it without checking. The paper explicitly states
the artifact limitation rather than inventing these settings.

The Slurm launcher saves frozen code under `outputs/code/<timestamp>/` on the
cluster. These snapshots, along with scratch scripts and `outputs/oof/`, are
possible places to recover the missing provenance. No full artifact transfer
is needed just to identify the model: a configuration, generation command,
small metric summary, and file checksum may resolve the correspondence.

## Numerical evidence map

| Manuscript component | Source | Interpretation |
|---|---|---|
| Sample counts, prevalence, missingness | `docs/eda_findings.md`, executed `notebooks/01_eda.ipynb` | Course dataset, not the full public BRFSS release |
| Cleaning rules, 291 retained predictors | `p1/cleaning.py`, executed `notebooks/02_cleaning.ipynb` | Fixed codebook rules |
| Table I: six methods and raw baseline | `README.md`, section “Results” | Five-fold seed-42 CV; own-training thresholds; mean ± fold SD |
| Figure 1: encodings A/C/D/E | `docs/eda_findings.md`, scratch-study table | Exploratory holdout with validation-optimized thresholds; optimistic |
| Numerical hyperparameters | `configs/step2/*.toml` | Executable presets |
| Eigenvalue and SGD stability observations | `PROGRESS.md`, `README.md` | Archived numerical diagnostics, not newly rerun |
| Initial MLP experiments | `docs/eda_findings.md`, “Other results on encoding E” | Holdout, widths 64 and 256; provisional scripts not preserved |
| Figure 2: logistic reference | `docs/part3_results.md`, main-results table | Repeated CV, three split seeds × five folds |
| Figure 2: later MLP, GBDT and refinements | `docs/round3_summary.md` | Repeated CV, three split seeds × five folds |
| Corrected paired interval | `docs/part3_results.md` | Reported interval, conditional on selected configurations |
| Public F1 0.456 | Roger's confirmation on 2026-10-29 | Separate competition observation; exact model unassigned |

`data/results.json` retains the numeric values, protocol caveats, and sources.
Figures and Table I are generated from this file. The original per-fold
artifacts are not available here, so the plots show recorded means only.

## Interpretations that must remain precise

- GD, SGD, and direct least squares optimize the same MSE hypothesis class.
  A finite-budget SGD result does not establish that SGD has a worse optimum.
- The logistic-loss expression is evaluated stably with `logaddexp` in code.
  Both regularized numerical presets use lambda 0.001.
- Rare categories can create large row norms after standardization; a full
  gradient's stability bound does not by itself guarantee SGD stability.
- The apparent large “cleaning” gain in the exploratory sequence includes
  recoding categorical unknowns. Numeric cleaning alone changes F1 by only
  -0.0003 after one-hot encoding in a separate paired comparison.
- A 95% paired interval describes the difference between two models. It is
  not a per-model error bar and does not correct for the entire search.
- The code's “nested F1” excludes the scored fold from direct threshold
  fitting; model configurations and blend weights are still selected using
  the same CV data. The manuscript does not call this fully nested CV.
- The late feature ensemble's reported improvement is approximately 0.0011
  (corrected t 1.51). Rounded means can differ by 0.0012; retain the distinction
  between the archived paired comparison and subtracting rounded table values.
- `HAREHAB1` is a post-diagnosis proxy. These results are not prospective
  cardiovascular-risk validation, and a finite model search proves no dataset
  performance ceiling.
- External-library scratch benchmarks and other teams' results are not used
  as evidence in the manuscript. The bibliography cites the actual primary
  methods and CDC documentation, not AI research summaries.

## Figure and layout choices

Two single-column vector figures preserve readability in the supplied IEEE
layout. Fonts are embedded, colors remain distinguishable in print, and axes
are explicitly labelled as positive-class F1. Figure 1 labels the exploratory
protocol; Figure 2 labels repeated CV. Neither figure invents error bars,
training histories, or fold-level measurements.

The original 10-point body text and IEEE page geometry are retained. The build
rejects overflow boxes, unresolved citations, a body longer than two pages,
or references outside the intended third page. Rendered previews are written
to the ignored `build/` directory when `--render` is supplied.

After any content or author edit, rerun the build and inspect the page previews.
Before uploading, use the public GitHub URL pinned to the final commit and
the AIcrowd submission ID corresponding exactly to the reproducible CSV.
