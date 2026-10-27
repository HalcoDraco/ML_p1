# Progress

Last updated: 2026-10-27. Deadline: Thursday 2026-10-29, 16:00.

## Status

| Part | State | Notes |
|---|---|---|
| Repo structure | done | skeleton from `STRUCTURE.md`; most function bodies in `p1/` are still `pass` |
| Step 2: `implementations.py` | buggy | 6 of 25 public tests fail, see `STRUCTURE.md` §7 |
| Part 1: EDA | **done** | `notebooks/01_eda.ipynb` (executed), details in `docs/eda_findings.md` |
| Part 2: cleaning | not started | `p1/cleaning.py` is a stub; the rules are written in `docs/eda_findings.md` §5 and §8 |
| Part 3: transforms, models, pipeline, CV, CLI | not started | stubs |
| `run.py`, `configs/best.toml` | not started | |
| Report | not started | EDA figures ready in `report/figures/eda_*.pdf` |
| AIcrowd submissions | none yet | |

Implemented so far in `p1/`:
- `data.load_raw()`: extracts the CSVs from the zip if missing, parses them with `helpers.load_csv_data`, and caches them in `data/processed/raw.npz` (about 0.5 s to load, 37 s the first time). `load_clean` and `write_submission` are still stubs.
- `eda.py`: helpers for the EDA notebook: `use_style`, `missing_fraction`, `positive_rate_by_value`, `roc_auc`, `univariate_auc`, `duplicate_columns`, `correlated_pairs`, `onehot_encode`, `ridge_probe`, `f1_by_threshold`, `print_table`, `save_figure`. Part-3 code must not import it, but `roc_auc` and `f1_by_threshold` are worth copying into `evaluation.py`.

## Key findings (details: `docs/eda_findings.md`)

- 328,135 train and 109,379 test rows, 321 features, 8.83% positives. Train and test are a random split, so shuffled k-fold CV is reliable (fold noise ±0.004 F1).
- **Leak**: `HAREHAB1` is answered only by heart-attack patients, and all 773 train rows that answered it are positive. 238 test rows answered it → predict 1. Never drop it.
- Answers are codes (7/77/777 don't know, 9/99/999 refused, 88/888 none, 555 never, mixed units such as `ALCDAY5`). "Don't know" often behaves like "yes" → one-hot encode, keep the codes as categories.
- 45% NaN, mostly skip logic: a NaN is informative → NaN as a category or indicator. Dropping sparse features hurts.
- Drop 14 survey-design columns (ids, dates, weights) and 2 exact copies (`CELLFON2`, `_CPRACE`).
- Measured in scratch experiments (logistic regression, best-threshold F1 on an 80/20 split): raw values 0.417 → codes cleaned 0.433 → + NaN indicators 0.439 → one-hot categorical 0.440 (5-fold: 0.4398 ± 0.0044, AUC 0.864).
- **Threshold** ≈ 0.20 for logistic regression (F1 at 0.5 is only about 0.27): tune it in CV. Class weights and undersampling only shift it.
- Logistic regression is enough: an MLP and interactions gained ≤ 0.002 F1.
- The one-hot design is singular, so least squares via `np.linalg.solve` fails: use ridge (λ > 0) or reference levels.
- Plain GD for logistic regression on the standardized one-hot features: γ ≈ 0.07, 300+ iterations.
- With 25% of the rows F1 drops by 0.013: compare pipelines with `subsample` ≥ 0.5.

## Decisions

- Part 2 does the fixed rules only (drop columns, code rules for the numeric features, skip-logic fills, drop the raw mixed-unit features that have a computed version). One-hot encoding, NaN indicators, imputation and standardization are part-3 transforms (they learn levels and statistics from the training fold).
- The core model is L2 logistic regression with a threshold tuned in CV.
- Notebooks are executed with `uv run --with nbconvert jupyter nbconvert --to notebook --execute --inplace notebooks/<name>.ipynb` (nbconvert is not a project dependency).

## Next steps (in priority order)

1. Fix `implementations.py` (`STRUCTURE.md` §7, ask the user first): return `(w, loss)` everywhere, SGD with the exact signature and batch size 1, the `calculate_mse` calls, no per-iteration prints. Rerun the public tests.
2. Part 2: write `p1/cleaning.py` from `docs/eda_findings.md` §5 and §8, `build_clean_dataset`, `load_clean`, `write_submission`, and the `02_cleaning` notebook.
3. Part 3: transforms (`onehot` with NaN column, NaN indicator + median imputation, `log1p`, `standardize`), the 6 models plus threshold tuning, pipeline, `evaluation.py` (F1, k-fold, threshold choice inside CV), config and CLI. A Newton (IRLS) logistic regression is a cheap exact solver.
4. First AIcrowd submission, `configs/best.toml`, `run.py`.
5. Ablation presets for the report (raw → cleaned codes → NaN indicators → one-hot → threshold), then the report.

## Session log

- 2026-10-27: added `CLAUDE.md` and `PROGRESS.md`. Did the EDA: implemented `data.load_raw` and `p1/eda.py`, wrote and executed `notebooks/01_eda.ipynb` (4 figures saved in `report/figures/`), wrote `docs/eda_findings.md` (rule for every feature, experiment results). Scratch experiment scripts were not kept.
