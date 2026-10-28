# Progress

Last updated: 2026-10-28. Deadline: Thursday 2026-10-29, 16:00.

## Status

| Part | State | Notes |
|---|---|---|
| Repo structure | done | skeleton from `STRUCTURE.md`; the part-3 function bodies in `p1/` are still `pass` |
| Step 2: `implementations.py` | buggy | 6 of 25 public tests fail, see `STRUCTURE.md` §7 |
| Part 1: EDA | **done** | `notebooks/01_eda.ipynb` (executed), details in `docs/eda_findings.md` |
| Part 2: cleaning | **done** | `p1/cleaning.py`, `notebooks/02_cleaning.ipynb` (executed) → `data/processed/clean.npz`: 291 features, 56 numeric (`cleaning.NUMERIC`) and 235 categorical |
| Part 3: transforms, models, pipeline, CV, CLI | not started | stubs |
| `run.py`, `configs/best.toml` | not started | |
| Report | not started | figures ready in `report/figures/`: `eda_*.pdf`, `clean_numeric_auc.pdf` |
| AIcrowd submissions | none yet | |

Implemented so far in `p1/`:
- `data.py`: `load_raw()` extracts the CSVs from the zip if missing, parses them with `helpers.load_csv_data`, and caches them in `data/processed/raw.npz` (about 0.5 s to load, 37 s the first time). `load_clean()` reads `clean.npz` (0.9 s), raises `FileNotFoundError` if it is missing and warns if `p1/cleaning.py` is newer. `save_dataset()` writes either file. `write_submission()` maps {0, 1} to {-1, 1} and calls `helpers.create_csv_submission`.
- `cleaning.py`: `clean(x, feature_names)` (pure, 0.35 s on the training set) and `build_clean_dataset()` (2.3 s, writes the 1 GB `clean.npz`). The rules are module constants (`DROPPED`, `CODES`, the activity lists). For part 3: `NUMERIC` (the 56 features to use as numbers; all others are categorical codes) and `HEAVY_TAILED` (33 of them, counts with long tails, for `log1p`).
- `eda.py`: helpers for the notebooks: `use_style`, `missing_fraction`, `positive_rate_by_value`, `roc_auc`, `univariate_auc`, `duplicate_columns`, `correlated_pairs`, `onehot_encode`, `ridge_probe`, `f1_by_threshold`, `print_table`, `save_figure` (reproducible PDFs). Part-3 code must not import it, but `roc_auc` and `f1_by_threshold` are worth copying into `evaluation.py`.

## Key findings (details: `docs/eda_findings.md`)

- 328,135 train and 109,379 test rows, 321 features, 8.83% positives. Train and test are a random split, so shuffled k-fold CV is reliable (fold noise ±0.004 F1).
- **Leak**: `HAREHAB1` is answered only by heart-attack patients, and all 773 train rows that answered it are positive. 238 test rows answered it → predict 1. Never drop it.
- Answers are codes (7/77/777 don't know, 9/99/999 refused, 88/888 none, 555 never, mixed units such as `ALCDAY5`). "Don't know" often behaves like "yes" → one-hot encode, keep the codes as categories.
- 45% NaN, mostly skip logic: a NaN is informative → NaN as a category or indicator. Dropping sparse features hurts.
- Drop 14 survey-design columns (ids, dates, weights) and 2 exact copies (`CELLFON2`, `_CPRACE`).
- Measured in scratch experiments (logistic regression, best-threshold F1 on an 80/20 split): raw values 0.417 → codes cleaned 0.433 → + NaN indicators 0.439 → one-hot categorical 0.440 (5-fold: 0.4398 ± 0.0044, AUC 0.864). The "codes cleaned" gain comes from turning the *categorical* don't-know/refused codes into NaN, which part 2 does not do (one-hot keeps them).
- **Value of part 2**: it fixes the numeric features (a model on them alone: F1 0.339 → 0.345), but inside the one-hot encoding E it changes nothing measurable (paired 5-fold F1 0.4404 with raw numeric codes vs 0.4401 clean, difference −0.0003 ± 0.0009; `log1p` also neutral). E on `clean.npz` reproduces the EDA's best: 5-fold F1 0.440 ± 0.002, AUC 0.864.
- **Threshold** ≈ 0.20 for logistic regression (F1 at 0.5 is only about 0.27): tune it in CV. Class weights and undersampling only shift it.
- Logistic regression is enough: an MLP and interactions gained ≤ 0.002 F1.
- The one-hot design is singular, so least squares via `np.linalg.solve` fails: use ridge (λ > 0) or reference levels.
- Plain GD for logistic regression on the standardized one-hot features: γ ≈ 0.07, 300+ iterations.
- With 25% of the rows F1 drops by 0.013: compare pipelines with `subsample` ≥ 0.5.

## Decisions

- Part 2 does the fixed rules only (drop columns, code rules for the numeric features, skip-logic fills, drop the raw mixed-unit features that have a computed version). One-hot encoding, NaN indicators, imputation and standardization are part-3 transforms (they learn levels and statistics from the training fold).
- The part-2 rules are those of `docs/eda_findings.md` §5, each checked against the BRFSS 2015 codebook (it corrected the EDA for `SCNTWRK1`/`SCNTLWK1`, `ASTHMAGE`, `LONGWTCH`, the drinking skip, `PAFREQ*`/`STRFREQ_` and the hundredths of `DROCDY3_`/`_DRNKWEK`). Clean features keep their raw names; the converted ones are in times per year (`BLDSUGAR`, `FEETCHK2`), years (`LONGWTCH`) and months before the interview (`FLSHTMY2`, `HIVTSTD3`).
- Skip-logic fills: `POORHLTH` → 0 when `PHYSHLTH` = `MENTHLTH` = 88; drinks → 0 only when `ALCDAY5` = 888 (not when it is unknown); the 10 activity amounts per week → 0 when `EXERANY2` = 2, and the second activity's → 0 when `EXRACT21` = 88.
- Part 3 splits the features with `cleaning.NUMERIC` (numbers: NaN indicator + median, standardize) vs the rest (one-hot with a NaN column); `log1p` on `cleaning.HEAVY_TAILED` is optional.
- The core model is L2 logistic regression with a threshold tuned in CV.
- Notebooks are executed with `uv run --with nbconvert jupyter nbconvert --to notebook --execute --inplace notebooks/<name>.ipynb` (nbconvert is not a project dependency).

## Next steps (in priority order)

1. Fix `implementations.py` (`STRUCTURE.md` §7, ask the user first): return `(w, loss)` everywhere, SGD with the exact signature and batch size 1, the `calculate_mse` calls, no per-iteration prints. Rerun the public tests.
2. Part 3: transforms (`onehot` over the non-`NUMERIC` features with a NaN column, NaN indicator + median imputation for `NUMERIC`, optional `log1p` for `HEAVY_TAILED`, `standardize`), the 6 models plus threshold tuning, pipeline, `evaluation.py` (F1, k-fold, threshold choice inside CV), config and CLI. A Newton (IRLS) logistic regression is a cheap exact solver. Expected 5-fold F1 ≈ 0.440.
3. First AIcrowd submission, `configs/best.toml`, `run.py` (calls `build_clean_dataset()` then `load_clean()`).
4. Ablation presets for the report, then the report. Build the ablation around the encoding (raw numbers → one-hot → NaN indicators → threshold): with one-hot the cleaning adds nothing, and the EDA's "codes cleaned" step needs the categorical don't-know codes turned into NaN, which part 2 does not do (`docs/eda_findings.md` §7–8).

## Session log

- 2026-10-27: added `CLAUDE.md` and `PROGRESS.md`. Did the EDA: implemented `data.load_raw` and `p1/eda.py`, wrote and executed `notebooks/01_eda.ipynb` (4 figures saved in `report/figures/`), wrote `docs/eda_findings.md` (rule for every feature, experiment results). Scratch experiment scripts were not kept.
- 2026-10-27/28: Part 2: implemented `p1/cleaning.py` (every numeric rule checked against the codebook, which corrected several EDA rules), `data.load_clean`, `save_dataset` and `write_submission`, and wrote and executed `notebooks/02_cleaning.ipynb` (figure `clean_numeric_auc.pdf`). Measured that the cleaning helps numeric-only models but not the one-hot encoding (paired 5-fold CV). Scratch scripts were not kept.
