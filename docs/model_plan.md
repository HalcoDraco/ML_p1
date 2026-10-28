# Plan: GBDT and blend (part 3)

Written on 2026-10-28 at 14:30, for the deadline of 2026-10-29 at 16:00. It implements the recommendations of `docs/deep_research_synthesis.md`, referred to below as "the synthesis". Each step ends with a check, and a gate decides whether to continue or fall back.

## Goal

- Raise the 5-fold CV F1 from 0.439 (logistic regression) to about 0.445–0.448, with a histogram GBDT written in NumPy and an out-of-fold (OOF) blend of the GBDT with the logistic regression.
- Get the rows of the report's ablation table from the same runs.
- Keep the logistic regression ready to submit at every moment, as the fallback.

## Overview

| Step | What | Expected ΔF1 | Hours | When |
|---|---|---|---|---|
| 0 | Submit the logistic regression; `best.toml` and `run.py` for it | leaderboard anchor, fallback | 0.5 | Wed 14:30–15:00 |
| 1 | CV tooling: OOF files, per-fold scores, parallel folds, paired test, blend | enables 2–4 | 2 | Wed 15:00–17:00 |
| 2 | `p1/gbdt.py`: the histogram GBDT | +0.004–0.007 | 3–4 | Wed 17:00–21:00 |
| 3 | GBDT tuning and ablations, on the cluster or locally | | 1–2 of wall time | Wed 21:00, runs overnight |
| 4 | Blend of the LR and GBDTs; repeated CV | +0.001–0.003 | 1 | Thu 08:00–09:30 |
| 5 | Optional: GBDT started from the LR logit; factorization machine | 0 to +0.002 | 1–3 | only if ahead of schedule |
| 6 | Final model, `run.py`, documentation | | 2 | Thu 09:30–12:00 |
| 7 | Report | | 3.5 | Thu 12:00–15:30 |

- The first half of the report (data, cleaning, encoding, the step-2 table) does not depend on the GBDT. A teammate can write it today.
- Not built, for the reasons in the synthesis §4 and §8: GBDT leaves fed to a logistic regression, embedding MLPs, hand-made risk features (kept as the fallback of Gate 1), F1-shaped losses, resampling.

## Step 0: the fallback first

- Submit `outputs/submissions/logistic.csv` (CV 0.4390). It measures our CV → leaderboard offset; other teams saw +0.01.
- Copy `configs/step2/logistic.toml` into `configs/best.toml` and write `run.py` as `STRUCTURE.md` §4 describes. Check that its CSV is identical to the submitted one (`cmp`).
- `best.toml` changes only once more, when the final model is chosen (step 6).

## Step 1: CV tooling

Changes to `p1/evaluation.py` and `p1/cli.py`, and a new module `p1/ensemble.py` for blends. Items 1–3 and 6–8 are needed in any case; item 4 only if we use the cluster.
1. **Fold files.** `cross_validate` saves, for each fold, `outputs/oof/<name>/seed<seed>_fold<k>.npz`. It holds the validation row indices, their scores, the fold model's threshold, and the fold model's scores on the test set. Blends and paired tests are computed from these files.
2. **Per-fold scores.** `outputs/cv_results.csv` gets a column `f1_folds`. It also gets the F1 with a nested OOF threshold (synthesis §5.2), next to the F1 with the model's own threshold.
3. **Parallel folds.** `p1 cv PRESET --jobs N` runs the folds in a process pool, and each worker loads `clean.npz` itself (0.9 s). Python 3.14 starts workers with forkserver by default, so they do not share the parent's arrays.
   - Memory: a logistic-regression fold needs about 10 GB, so use `--jobs 2` for it locally. A GBDT fold needs about 3 GB, so `--jobs 5` works.
4. **One task per job, for slurm arrays.** `p1 cv PRESET --variant I --fold K` runs only grid variant I (`expand_grid` order) and fold K, and writes the fold file. `p1 collect NAME...` then turns the fold files into rows of `cv_results.csv`. Only `collect` writes the CSV, so parallel jobs never append to it at the same time.
5. **Repeats.** A repeat is a run with another fold seed (`--set seed=43`), and the fold files are keyed by seed. Use the seeds 42, 43 and 44.
6. **`p1 compare A B`.** It pairs the F1 of the (seed, fold) pairs that both runs have. It prints the mean difference, its standard deviation, the Nadeau–Bengio corrected t and the 95% interval (critical values 2.776 for 4 degrees of freedom and 2.145 for 14).
7. **`p1 blend A B ... [--weights ...]`.** Logit average of the members' OOF scores, which must come from the same folds. It searches the weight on the pooled OOF F1, gives the nested F1, and returns the pooled-OOF threshold. It writes fold files for the blend too, so `compare` works on blends.
8. **Dropping features for ablations.** A preset key `drop = ["HAREHAB1"]` removes features from the dataset before the pipeline is built (in `cv`, `train` and `predict`). It works for any model.

Check: `p1 cv step2/logistic --jobs 2` reproduces 0.4390 ± 0.0027 and writes 5 fold files. `p1 compare logistic least_squares` shows about +0.009 with a large t.

## Step 2: the GBDT

A new module `p1/gbdt.py` holds `GradientBoostedTrees`, registered as `gbdt` in `models.MODELS`. The type hint of `MODELS` becomes a base class common to all models. The preset `configs/gbdt.toml` has `transforms = []`, then `[model] name = "gbdt"` with the hyper-parameters.

Design (synthesis §5.1):
- **Binning** happens inside `fit`, so the bins are part of the model. The option `nominal = "categorical"` is resolved to column indices by `build_pipeline`, like a transform's `columns`. The other features get quantile bins.
- **Trees.** Complete trees grown level by level, with one `np.bincount` per feature and level for the histograms.
  - Numeric splits learn the direction of the NaN bin.
  - Categorical splits sort the bins by G/(H + `cat_smooth`).
  - Every split is stored as a feature plus a mask over its bins, and a leaf's value is −η G/(H + λ).
- **Randomness.** Rows and features are sampled from `np.random.default_rng(seed)`, where `seed` is a model option. Different seeds give a bag of GBDTs.
- **Storage.** The trees are stored as small arrays, so the pipeline pickles as before.
- **Staged scores.** `staged_scores(x, every=50)` returns the scores after every 50 trees. `cross_validate` stores the validation AUC, log-loss and oracle F1 at each step in the fold file, so one run of 1,000 trees shows the best number of trees.
- **Threshold.** `threshold` is either `"auto"` (the training rows' own scores, as for the linear models) or picked on a holdout: with `threshold_holdout = 0.1`, the model is fitted on 90% of the rows and the threshold is chosen on the other 10%.
- **Importance.** Split-gain feature importance (`importance_`), for the report.

Before any CV, run the five sanity checks of the synthesis §5.1 in scratch scripts outside the repo, and check the speed: below 1 s per tree on a full training fold.

**Gate 1**, around Wed 21:00: on fold 1 of seed 42, the default configuration must beat the logistic regression's fold-1 result (F1 0.438) by at least 0.003 in AUC, with an F1 not below it. The default is depth 5, η 0.05, 500 trees, λ 1, `min_child_hessian` 5, 70% of the rows and 60% of the features per tree.
- If the gate fails, debug for at most one hour (the checks above, η and depth).
- If it still fails, stop the GBDT. The final model is then the logistic regression, possibly with the fallback features of the synthesis §7, and the time goes to the report.

## Step 3: tuning and ablations

1. **Grid on fold 1**, 1,000 trees with staged scores, one job per configuration: depth {3, 4, 5, 6} × η {0.05, 0.1} × `min_child_hessian` {5, 20} × λ {1, 10} × features per tree {0.5, 0.8}.
   - That is 64 jobs of about 8 min: 10 min on the cluster, about 1.5 h locally with 6 jobs at a time. Without the cluster, cut it to depth {4, 5, 6} × η {0.05, 0.1} × `min_child_hessian` {5, 20}, which is 12 jobs.
   - Choose by validation log-loss and AUC, which are smooth, then by F1.
2. **Threshold protocol**, on fold 1 with the best configuration: compare the training-row threshold, the 10% holdout and the oracle (the best threshold of the validation fold itself). Keep the cheapest one within 0.001 F1 of the oracle.
3. **5-fold CV** of the 3 best configurations, with the number of trees found in item 1.
4. **Ablations** on 5 folds with the best configuration:
   - depth 1, an additive model without interactions; it should match the logistic regression, and needs about 2,000 trees at η = 0.1, which is cheap at depth 1;
   - depth 2, which adds pairwise interactions like GA²M;
   - all splits ordinal, without the categorical sorting;
   - `drop = ["HAREHAB1"]`, for the logistic regression too.

**Checkpoint:** compare the best GBDT with the logistic regression (`p1 compare`). If the GBDT is not above it, stop tuning it and keep it only as a blend member. Tonight's GBDT submission (5 a day) is worth spending only if it is above.

## Step 4: blend and repeated CV

1. **Members:** `step2/logistic`, the best GBDT, and 2 more GBDTs with other `model.seed` values. This bag is cheap: another team's average of 4 GBDTs gained +0.0015.
2. **Blend:** `p1 blend` picks the weight on the pooled OOF F1, gives the nested estimate, and returns the threshold.
3. **Repeated CV:** run every member with the fold seeds 43 and 44 too, for 15 paired folds in total. Then `compare` the blend, the GBDT and the logistic regression.

**Selection rule**, fixed now, before seeing the results:
- The final model is the candidate (logistic regression, GBDT or blend) with the highest mean F1 over the 15 paired folds. Every candidate is scored with the nested OOF threshold; for the blend, the weight is also chosen on the other 4 folds.
- The leaderboard is only a sanity check: it cannot resolve gains below about 0.005.
- The corrected t only decides what the report may claim.

## Step 5: optional

Only if the GBDT passed the checkpoint and step 4 is done before Thu 09:00:
- **GBDT started from the LR logit.** A model option `init = "step2/logistic"` makes `build_pipeline` build the logistic regression from that preset and hand it to the GBDT, which fits it on the same rows. Compare it with the blend, not with the plain GBDT.
- **Factorization machine** warm-started from the logistic regression (synthesis §5.5), as a third blend member.
- **Newton (IRLS) logistic regression.** It gives the exact optimum; expect +0.001 at most. Only worth it if the logistic regression is the final model.

## Step 6: final model and reproducibility

1. **Fit on all rows.** Train the final members on all training rows: the logistic regression takes 74 s and 13 GB; the GBDT about 10 min.
   - Blend the test scores, then apply the threshold chosen on the OOF scores pooled over the three seeds.
   - Freeze that threshold as a number in `configs/best.toml`, with the command that produced it in a comment.
2. **Checks before submitting.**
   - The 238 test rows that answered `HAREHAB1` are predicted 1.
   - The share of test rows predicted positive is close to the OOF one, about 12%.
3. **`run.py`:** `build_clean_dataset()` → `load_clean()` → fit the members on all rows → blend → threshold → `outputs/submissions/best.csv`. Check with `cmp` that it equals the submitted file.
   - Run it on the local machine, which also produced the submitted file: BLAS sums can differ between machines and flip rows that sit at the threshold.
4. **Documentation.**
   - Update the README (GBDT, blend, CV procedure), `STRUCTURE.md` (contracts of `gbdt.py`, `ensemble.py` and the new commands) and `PROGRESS.md`.
   - Then run `uv run black .`, make sure no "todo" is left, and run the grading tests. `implementations.py` stays untouched.

## Step 7: report

The ablation table, all rows on the same 5 folds (seed 42), and the final candidates also on 3 × 5 folds:

| Row | Shows |
|---|---|
| Least squares on the raw values (0.4095) | the baseline |
| Logistic regression on the one-hot encoding (0.4390) | the effect of the encoding |
| GBDT, depth 1 | non-linear main effects only |
| GBDT, depth 2 | plus pairwise interactions |
| GBDT, best depth | plus higher-order interactions |
| GBDT without the categorical sorting | how categories are split |
| GBDT and logistic regression without `HAREHAB1` | the label proxy |
| Blend of the logistic regression and the GBDT | ensembling |

Figures, if space allows:
- F1 and log-loss as a function of the number of trees, for each depth (from the staged scores).
- The split-gain importance of the top 15 features.

## Compute

- **GPUs are no use.** The code must use NumPy, which runs on CPUs only, and GPU libraries (CuPy, PyTorch) are not allowed.
- **Local machine** (16 threads, 30 GB): development, the fold-1 checks, 5 GBDT folds in parallel, and at most 2 logistic-regression folds in parallel.
- **Cluster (CPU nodes).** Worth it for the step-3 grid, the ablations and the repeated CV, if setting it up takes less than about 30 min. That is about 150 jobs of 5–10 min each.
  - One array task is one (grid variant, seed, fold). Resources and environment:
    - GBDT tasks: 1 CPU, 4 GB, 30 min. `np.bincount` uses a single thread, so set `OMP_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1`.
    - Logistic-regression tasks: 4 CPUs (for BLAS), 16 GB, 30 min.
  - Setup, on the login node, since compute nodes may have no internet:
    - Install uv.
    - Copy the working tree with rsync, which needs no commit. Exclude `.venv`, `data/processed` and `outputs`.
    - Run `uv sync --frozen`.
    - Copy `clean.npz` with `rsync -a`, which keeps its time stamp (otherwise `load_clean` warns), or rebuild it from the zip.
  - Afterwards, rsync `outputs/oof/` back and run `p1 collect`.

```bash
#!/bin/bash
#SBATCH --array=0-63              # 64 grid variants, fold 1 only
#SBATCH --cpus-per-task=1 --mem=4G --time=00:30:00
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
uv run p1 cv configs/gbdt_grid.toml --variant "$SLURM_ARRAY_TASK_ID" --fold 0
```

## Risks

| Risk | Fallback |
|---|---|
| The GBDT has a bug or stays below the logistic regression at Gate 1 | stop at 22:00; the final model is the logistic regression (with the fallback features); more time for the report |
| The GBDT is too slow | histogram subtraction; fewer trees with η = 0.1; the cluster |
| The training-row threshold is biased for trees | the holdout or the pooled OOF threshold (step 3.2) |
| The gains are within the noise | report them as such; choose by mean F1 (selection rule) |
| `run.py` gives a different CSV from the submission | produce both on the same machine; `cmp` before uploading |
| Memory: the logistic regression on all rows needs 13 GB | fit the members one after the other in `run.py` |
| Plagiarism | write the GBDT from the recipe; do not open the other teams' repositories |

## Open questions for the team

- **Use the cluster?** Recommended for step 3 and the repeated CV, if its setup is quick. Everything also fits on the local machine with the smaller grid.
- **Where does the slurm script live?** In the repo (`slurm/`) or outside? Default: outside, since it is specific to our cluster.
- **Submissions, 5 a day:**
  - Wednesday: the logistic regression, plus the GBDT if it passes the checkpoint tonight.
  - Thursday: the blend, the final model, and one spare.
