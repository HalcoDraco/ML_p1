# Progress

Last updated: 2026-10-29 (report preparation; experimental measurements are archived). Deadline: Thursday 2026-10-29, 16:00.

**Start here: `docs/part3_results.md`** (part-3 results, what they suggest, next steps with commands).

## Status

| Part | State | Notes |
|---|---|---|
| Repo structure | done | from `STRUCTURE.md`; every module of `p1/` is implemented |
| Step 2: `implementations.py` | **done** | 25/25 public tests pass, also with Python 3.9 + NumPy 1.23.1; fixes listed in `STRUCTURE.md` §7 |
| Step 2: run and evaluate the six methods | **done** | presets `configs/step2/*.toml`, `uv run p1 cv configs/step2/*.toml`; README section "Step 2"; results under Key findings |
| Part 1: EDA | **done** | `notebooks/01_eda.ipynb` (executed), details in `docs/eda_findings.md` |
| Part 2: cleaning | **done** | `p1/cleaning.py`, `notebooks/02_cleaning.ipynb` (executed) → `data/processed/clean.npz`: 291 features, 56 numeric (`cleaning.NUMERIC`) and 235 categorical |
| Part 3: transforms, models, pipeline, CV, CLI | infrastructure done | everything runs (`p1 cv/train/predict`) |
| Part 3: OOF tooling | **done** (2026-10-28) | every CV fold writes `outputs/oof/<run>/seed<s>_fold<k>.npz` (validation + test scores); `p1 cv --fold K --variant I`, `p1 collect`, `compare`, `blend`, `stages`, `submit`; `p1/ensemble.py` |
| Part 3: NumPy GBDT | **done, tuned** | `p1/gbdt.py`; tuned preset `configs/gbdt_tuned.toml` (ordinal splits, feature fraction 0.3, λ 10, depth 5, η 0.05, 550 trees): nested F1 0.4457 per run, 3-seed bag 0.4462 vs LR 0.4386 (+0.0077, t 4.28) |
| Cluster (SCITAS jed) | **set up** | `slurm/submit.sh` + `slurm/p1.sbatch`, see "Cluster" below |
| Deep research | **done** | 3 reports in `docs/deep_research_outputs/`, judged and merged into `docs/deep_research_synthesis.md` |
| Model plan | steps 1–4 done | `docs/model_plan.md`; results in `docs/part3_results.md`; left: report ablations with the tuned settings, documentation |
| `run.py`, `configs/best.toml` | earlier recipe retained | refits the 3-seed `gbdt_tuned` bag, threshold 0.2166. The team-confirmed 0.456 submission has not been mapped to its producing configuration; see `report_draft/HANDOFF.md`. |
| Report | **complete draft, compiled and visually reviewed** | `report_draft/report.pdf`: 2 content pages + 1 reference page, IEEE 10 pt; numerical methods, MLPs, two Seaborn vector figures, source-labelled measurements and build checks. Author names included; email addresses await confirmation. Original `report/` remains the course template. |
| AIcrowd submissions | best reported F1 **0.456** | confirmed by Roger on 2026-10-29; exact CSV/configuration and submission ID not available in this checkout. Earlier recorded scores: logistic 0.449 and `best.csv` 0.454. |

Implemented so far in `p1/`:
- `data.py`: `load_raw()` extracts the CSVs from the zip if missing, parses them with `helpers.load_csv_data`, and caches them in `data/processed/raw.npz` (about 0.5 s to load, 37 s the first time). `load_clean()` reads `clean.npz` (0.9 s), raises `FileNotFoundError` if it is missing and warns if `p1/cleaning.py` is newer. `save_dataset()` writes either file. `write_submission()` maps {0, 1} to {-1, 1} and calls `helpers.create_csv_submission`.
- `cleaning.py`: `clean(x, feature_names)` (pure, 0.35 s on the training set) and `build_clean_dataset()` (2.3 s, writes the 1 GB `clean.npz`). The rules are module constants (`DROPPED`, `CODES`, the activity lists). For part 3: `NUMERIC` (the 56 features to use as numbers; all others are categorical codes) and `HEAVY_TAILED` (33 of them, counts with long tails, for `log1p`).
- `eda.py`: helpers for the notebooks: `use_style`, `missing_fraction`, `positive_rate_by_value`, `roc_auc`, `univariate_auc`, `duplicate_columns`, `correlated_pairs`, `onehot_encode`, `ridge_probe`, `f1_by_threshold`, `print_table`, `save_figure` (reproducible PDFs). Part-3 code must not import it (`metrics.py` has its own copies of `roc_auc` and the F1-by-threshold logic).
- `transforms.py`: `impute_median` (`columns`, `indicators`), `onehot` (`columns`, `min_count=30`, `max_levels=100`, NaN column), `log1p`, `standardize`, `poly`, `whiten`. A `columns` option takes feature names or a group (`numeric`, `categorical`, `heavy_tailed`), resolved by `pipeline.build_pipeline`.
- `models.py`: one class per `implementations.py` function (`least_squares`, `ridge`, `mse_gd`, `mse_sgd`, `logistic`, `reg_logistic`), zero initial weights. `threshold = "auto"` (default) picks the F1-optimal threshold on the training rows' own scores. Fitted state: `w_` (bias first), `loss_`, `threshold_`.
- `pipeline.py` (`Pipeline`, `build_pipeline`), `metrics.py` (`f1_score`, `accuracy`, `roc_auc`, `best_threshold`), `evaluation.py` (`kfold_indices`, `cross_validate` → F1, accuracy, AUC, threshold), `config.py` (presets, `--set` overrides, `[grid]`), `cli.py` (`p1 cv | train | predict`).

## Cluster (SCITAS jed)

- Account `master`, partition and QOS `academic` (free; up to 1,152 CPUs per user, 7,000 MB per CPU, default time limit 5 min, so always pass `-t`). The VS Code session itself is a 2-CPU, 8 GB slurm job: run anything heavy as a job.
- Setup on a fresh clone: install uv (`curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh`), `uv sync --frozen`, then build the data in a job (`p1.cleaning.build_clean_dataset()`, 70 s, 5.5 GB).
- `slurm/submit.sh [sbatch options] -- [p1 arguments]` copies the code to `outputs/code/<time>/` and submits `slurm/p1.sbatch` on it, so later edits do not change queued jobs. `@TASK@`/`@FOLD@`/`@VARIANT@` in the arguments become the array task id. Example: `slurm/submit.sh -J gbdt --array=0-4 -c 1 --mem=6G -t 01:30:00 -- cv configs/gbdt.toml --fold @TASK@`. Logs: `outputs/slurm/`.
- Resources: GBDT fold 1 CPU / 6 GB; logistic-regression fold 4 CPUs / 27 GB.

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
- The one-hot design is singular, so least squares via `np.linalg.solve` fails; `implementations.least_squares` uses the pseudo-inverse instead (same F1 as ridge with λ = 1e-8).
- **Step 2, the six methods** (5-fold CV on all rows, encoding of `configs/step2/`, same folds): logistic 0.4390 ± 0.0027 (AUC 0.864), reg. logistic 0.4388 ± 0.0022, least squares 0.4301 ± 0.0018 (AUC 0.857), ridge 0.4297 ± 0.0031, MSE GD 0.4289 ± 0.0022, SGD 0.4255 ± 0.0033; `baseline` (least squares on the raw values, NaN → median) 0.4095 ± 0.0019 (AUC 0.845). Logistic beats least squares on every fold (+0.009); the encoding gains +0.021 for least squares. The logistic GD result matches the EDA's Newton fit (0.4398), so the pipeline reproduces the scratch experiments.
- Learned thresholds are 0.21–0.23 for every method. At 0.5 the F1 is 0.26 (logistic) and 0.16 (least squares). Logistic on validation fold 1: threshold 0.237, 12.1% of rows flagged, precision 0.376, recall 0.523, F1 0.438.
- `HAREHAB1` is the only leak: among all (feature, value) groups with ≥ 30 rows (NaN included), no other one is above 46% positive.
- GD on the standardized encoding: λ_max(XᵀX/N) = 63.3, so MSE GD needs γ < 0.032 and logistic γ < 0.126 (γ = 0.25 also converged in practice). MSE GD reaches the least-squares F1 in 100 steps. Logistic GD gives the same F1 after 300 and 1000 steps (0.18 s per step on 262k × 1,527). SGD diverges from γ = 1e-4: squared row norms reach 5·10⁵ (rare one-hot levels after standardization).
- Cost: a 5-fold CV of the logistic preset takes 5 min; `train` on all rows takes 74 s and peaks at 13 GB of RAM; the saved pipeline is 59 KB.
- With 25% of the rows F1 drops by 0.013: compare pipelines with `subsample` ≥ 0.5.
- **GBDT, round 1** (2026-10-28, 5 folds of seed 42, 2,000 trees at η 0.05, staged validation metrics every 50 trees; `p1 stages RUN`): the default (depth 5, sorted categorical splits, feature fraction 0.6) overfits fast: best log-loss at **300 trees** (0.21474, AUC 0.8683), nested F1 only 0.4323 at 2,000 trees. **Ordinal splits on the codes beat the Fisher-sorted categorical splits** (log-loss 0.21403 vs 0.21474, AUC 0.8694). Feature fraction 0.3 helps (0.21445), λ 10 slightly (0.21463); depth 3–6 similar, depth 8 worse (0.21542); min_child_hessian 50, row fraction 0.5, 255 bins, cat_smooth 50: no gain. Boosting from the LR logit (`init`) is worse (0.21532). Ablations: depth 1 (additive) nested F1 0.4417 (already > LR 0.4387: binning the numeric features), depth 2 0.4398 at 2,000 trees (0.4445 at its best stage); without `HAREHAB1` F1 0.4372 at the best stage (−0.008).
- **GBDT bag `fc_m{0,1,2}`** (ordinal, feature fraction 0.3, λ 10, depth 5, η 0.05, 550 trees; fold seeds 42/43/44 × model seeds 0/1/2): nested F1 0.4449–0.4469 per run, AUC 0.8698. Bag of the 3 model seeds (`fc_bag`): 0.4462 (AUC 0.8703); **`fc_bag` − logistic = +0.0077, corrected t 4.28, 95% CI [+0.0038, +0.0115]** (15 paired folds). Bagging seeds adds +0.0001 (n.s.); blending with the LR adds +0.0001 (weight 0.05 on the LR): the GBDT already holds what the LR knows. `outputs/submissions/fc_bag.csv` flags 12.93% of the test rows (OOF 12.85%), all 238 `HAREHAB1` rows → 1.
- **GBDT, round 2** (ordinal + λ 10): feature fraction 0.2 → log-loss 0.21372, AUC 0.8699 at 700 trees; min_child_hessian 1 → 0.21380 at 600 trees. The optimum is at 600–800 trees.
- **Deep research** (`docs/deep_research_synthesis.md`): the best results reported on this dataset are NumPy GBDTs at CV F1 0.442–0.447 (AUC 0.870), against 0.435–0.438 for those teams' own logistic regressions, so the realistic ceiling is about 0.445–0.450. Only a GBDT (+0.004–0.007 expected) and an OOF blend with the LR (+0.001–0.003) are worth building; everything else is ≤ 0.002.
- Aspirin module, checked on the data: `CVDASPRN` is answered by 14,046 rows (4.3%), all in Georgia, Iowa, Oregon and Puerto Rico; `RLIVPAIN`/`RDUCHART`/`RDUCSTRK` only when `CVDASPRN` = 1, `ASPUNSAF` otherwise; `HAREHAB1` only inside the module. Module rows without `HAREHAB1` are 4.1% positive. No second leak; Gemini's claims about this module are false.
- GBDT cost, measured with NumPy on one core: the `np.bincount` histograms of one depth-5 tree take 0.33 s (+0.10 s to gather the sample) with 70% of 262k rows and 60% of 291 features, so 500 trees take about 4–5 min per fold.

## Decisions

- Part 2 does the fixed rules only (drop columns, code rules for the numeric features, skip-logic fills, drop the raw mixed-unit features that have a computed version). One-hot encoding, NaN indicators, imputation and standardization are part-3 transforms (they learn levels and statistics from the training fold).
- The part-2 rules are those of `docs/eda_findings.md` §5, each checked against the BRFSS 2015 codebook (it corrected the EDA for `SCNTWRK1`/`SCNTLWK1`, `ASTHMAGE`, `LONGWTCH`, the drinking skip, `PAFREQ*`/`STRFREQ_` and the hundredths of `DROCDY3_`/`_DRNKWEK`). Clean features keep their raw names; the converted ones are in times per year (`BLDSUGAR`, `FEETCHK2`), years (`LONGWTCH`) and months before the interview (`FLSHTMY2`, `HIVTSTD3`).
- Skip-logic fills: `POORHLTH` → 0 when `PHYSHLTH` = `MENTHLTH` = 88; drinks → 0 only when `ALCDAY5` = 888 (not when it is unknown); the 10 activity amounts per week → 0 when `EXERANY2` = 2, and the second activity's → 0 when `EXRACT21` = 88.
- Part 3 splits the features with `cleaning.NUMERIC` (numbers: NaN indicator + median, standardize) vs the rest (one-hot with a NaN column); `log1p` on `cleaning.HEAVY_TAILED` is optional.
- The core model is L2 logistic regression with a threshold tuned in CV.
- The decision threshold is learned with the model (`threshold = "auto"`): the F1-optimal threshold on the training rows' own scores. CV refits it on each training split, so the reported F1 never uses the validation fold, and `train` saves it with the model. On fold 1 it lands within 0.002 F1 of the oracle threshold tuned on the validation fold, for every method, so nested CV is not needed.
- `least_squares` solves the normal equations with `np.linalg.pinv` (allowed; only `lstsq` is banned), so it runs on the singular one-hot design and gives the minimum-norm solution. On well-posed inputs it equals `np.linalg.solve`.
- The step-2 presets share one encoding: `impute_median` (numeric, with indicators) → `onehot` (categorical) → `standardize`, 1,526 columns per training fold. Hyper-parameters, from a sweep on fold 1: ridge λ = 1e-3 (λ barely matters); MSE GD γ = 0.02, 100 iterations; SGD γ = 5e-6, 10⁶ steps; logistic GD γ = 0.1, 300 iterations; reg. logistic λ = 1e-3 (1e-2 starts to hurt).
- Metrics live in `p1/metrics.py`, which imports nothing from `p1`: `models` needs `best_threshold`, and putting it in `evaluation.py` would create the cycle evaluation → pipeline → models → evaluation.
- Notebooks are executed with `uv run --with nbconvert jupyter nbconvert --to notebook --execute --inplace notebooks/<name>.ipynb` (nbconvert is not a project dependency).

## Next steps (in priority order)

Details and commands: `docs/part3_results.md` §6.
1. Identify the CSV/configuration and submission ID behind the team-confirmed 0.456, then verify exact reproduction. Report handoff and candidate email addresses: `report_draft/HANDOFF.md`.
2. Optional: η 0.02 / 1,800-tree bag on fold seeds 42/43/44 (best log-loss, 0.21349); adopt it only if its 15-fold mean nested F1 beats `fc_bag`.
3. Report ablations with the tuned settings on 3 fold seeds: depth 1, depth 2, sorted categorical splits, without `HAREHAB1` (GBDT and LR), the raw baseline.
4. Documentation: `STRUCTURE.md` contracts (`gbdt.py`, `ensemble.py`, CLI, `slurm/`), README results table, report (first half can be written now).
5. Low priority (≤ +0.002 each): bag of configurations, MLP/FM blend member, count features.

## Session log

- 2026-10-27: added `CLAUDE.md` and `PROGRESS.md`. Did the EDA: implemented `data.load_raw` and `p1/eda.py`, wrote and executed `notebooks/01_eda.ipynb` (4 figures saved in `report/figures/`), wrote `docs/eda_findings.md` (rule for every feature, experiment results). Scratch experiment scripts were not kept.
- 2026-10-27/28: Part 2: implemented `p1/cleaning.py` (every numeric rule checked against the codebook, which corrected several EDA rules), `data.load_clean`, `save_dataset` and `write_submission`, and wrote and executed `notebooks/02_cleaning.ipynb` (figure `clean_numeric_auc.pdf`). Measured that the cleaning helps numeric-only models but not the one-hot encoding (paired 5-fold CV). Scratch scripts were not kept.
- 2026-10-28: Step 2: fixed `implementations.py` (25/25 public tests, also on Python 3.9 + NumPy 1.23.1; extra fixes for SGD speed, logistic NaN and overflow); implemented the part-3 modules (`transforms`, `models`, `pipeline`, new `metrics`, `evaluation`, `config`, `cli`); wrote the presets `configs/step2/` after a hyper-parameter sweep on fold 1; ran the 5-fold CV of the six methods plus `baseline` (results in Key findings, rows in `outputs/cv_results.csv`); trained `step2/logistic` and wrote `outputs/submissions/logistic.csv`; added the README section "Step 2" and updated `STRUCTURE.md`. Scratch scripts were not kept.
- 2026-10-28: wrote `docs/deep_research_prompt.md`, a self-contained prompt for an external deep-research engine (constraints, data facts, current results, what was tried, research questions, output format); measured the fold-1 operating point and scanned for other leaks (none). Rebuilt `clean.npz`: the git checkouts had made `p1/cleaning.py` newer than it, which made `load_clean` warn.
- 2026-10-28: merged the three deep-research reports into `docs/deep_research_synthesis.md`: claims judged, and Gemini's aspirin-module claims checked on the data and rejected. Measured the NumPy cost of one GBDT tree, and wrote `docs/model_plan.md` (plan only, nothing implemented). Scratch scripts were not kept.
- 2026-10-28 (cluster session): fresh clone on SCITAS jed; installed uv, rebuilt the data in a job. Submitted `logistic.csv`: AIcrowd F1 0.449 (CV 0.439, offset +0.010 as other teams saw). Implemented the OOF tooling (`p1/ensemble.py`, new `evaluation.py`, CLI commands) and the NumPy GBDT (`p1/gbdt.py`), and the slurm scripts. Launched the LR with fold seeds 42/43/44 and 15 GBDT variants (5 folds each) on the cluster.
- 2026-10-28 (cluster session, continued): GBDT round 1 (15 variants) and round 2 (7 variants) on 5 folds; tuned GBDT bag on 3 fold seeds × 3 model seeds; blends; ordinal splits beat sorted categorical splits; ensembling adds nothing measurable. Wrote `configs/gbdt_tuned.toml`, `configs/best.toml`, `run.py` (ran on the cluster → `best.csv`) and `docs/part3_results.md`.
- 2026-10-28 (late): `best.csv` scored AIcrowd F1 0.454 / acc 0.882. Round 3 (library benchmarks, count + target-order features, MLP, config bags): best `fe2_bag5` CV 0.4474 (+0.0011 vs `fc_bag`, t 1.51). Summary and the 3 CSVs to submit: `docs/round3_summary.md`. `run.py`/`best.toml` not yet switched to the new model.
- 2026-10-29: created `report_draft/` with a full English IEEE manuscript, authors supplied by Roger, two Seaborn vector figures, generated numerical table, six verified primary references, source/protocol provenance, reproducible build scripts and a handoff for Jacobo. Compiled and inspected all pages; 2 content pages + 1 references page, no LaTeX overflow or unresolved references. Modeling code and final prediction recipes were not changed. Remaining team inputs: confirmed email addresses and the exact configuration/CSV for the reported 0.456 submission; MLP code/fold artifacts are absent from this checkout.
