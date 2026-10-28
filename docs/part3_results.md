# Part 3 results: GBDT, ensembles, leaderboard (2026-10-28)

What the cluster runs of 2026-10-28 found, and what they suggest. The tooling and commands are in the README ("Part 3: GBDT, out-of-fold scores and blends" and "Running on the cluster"). Every run is in `outputs/oof/<run>/` (gitignored); `uv run p1 collect '*'` rebuilds the full table in `outputs/cv_results.csv`.

**Scores used below:**
- **Nested F1**: for each fold, the threshold is the F1-optimal one on the other folds' out-of-fold (OOF) scores. This is the honest score, used for model choice.
- **Best-stage metrics** of a GBDT: the validation log-loss, AUC and F1 after the number of trees with the best fold-averaged log-loss (`p1 stages RUN`). That F1 uses each fold's own best threshold, so it is about 0.001 optimistic.
- **Paired comparisons** (`p1 compare`) use the Nadeau–Bengio corrected t-test over (fold seed, fold) pairs.

## 1. Leaderboard reading

- `logistic.csv` (CV 0.439) scored **F1 0.449, accuracy 0.885** on AIcrowd: an offset of +0.010, the same that other teams saw. Expect it for every model.
- From F1 and accuracy with 8.8% positives, our submission flags about 12.0% of the test rows (precision 0.389, recall 0.531). The top team (0.460 / 0.881) flags about 13.2% (precision 0.384, recall 0.574): their model ranks better, it does not use another threshold rule.
- 0.460 on the leaderboard ≈ CV 0.450, which is at the ceiling reported for this dataset. The leaderboard's standard error is about 0.005, so the ranks near the top are mostly noise.
- LightGBM, XGBoost and CatBoost are excluded by the rules ("No use of external ML libraries is allowed in Project 1"). `p1/gbdt.py` implements their core algorithm in NumPy.

## 2. Main results

| Run | What | Nested F1 | AUC |
|---|---|---|---|
| `logistic` (3 fold seeds) | step-2 one-hot logistic regression | 0.4386 (0.4387 / 0.4387 / 0.4383) | 0.8643 |
| `fc_m0..2` (9 runs) | tuned GBDT, `configs/gbdt_tuned.toml` with model seeds 0, 1, 2, fold seeds 42/43/44 | 0.4449–0.4469, mean 0.4457 | 0.8698 |
| `fc_bag` | mean logit of the 3 model seeds | **0.4462** (0.4454 / 0.4466 / 0.4467) | 0.8703 |
| `blend_v1` | `logistic` + `fc_bag`; searched weight 0.05 on the LR | 0.4463 | 0.8703 |

Paired comparisons, over 15 folds:

| Comparison | Difference | Corrected t | 95% interval |
|---|---|---|---|
| `fc_bag` − `logistic` | **+0.0077** | 4.28 | [+0.0038, +0.0115] |
| `fc_bag` − `fc_m0`: bagging seeds | +0.0001 | 0.12 | [−0.0022, +0.0025] |
| `blend_v1` − `fc_bag`: adding the LR | +0.0001 | 0.25 | [−0.0008, +0.0010] |
| equal bag of 4 GBDT configurations − one of them (seed 42 only, 5 folds) | +0.0020 | 1.88 | [−0.0010, +0.0051] |

So the GBDT gains +0.008 F1 and +0.006 AUC over the LR, which is significant. Ensembling (seeds, configurations, LR) gains nothing measurable: the members are too correlated, and the LR is dominated.

## 3. GBDT tuning (5 folds of fold seed 42, best-stage metrics)

**Round 1**: one change at a time from depth 5, η 0.05, sorted categorical splits, feature fraction 0.6, λ 1, `min_child_hessian` 5, row fraction 0.7, 64 bins.

| Change | Trees | Log-loss | AUC | F1 |
|---|---|---|---|---|
| none (default) | 300 | 0.21474 | 0.8683 | 0.4456 |
| **ordinal splits for every feature** | 450 | **0.21403** | **0.8694** | 0.4467 |
| **feature fraction 0.3** | 350 | 0.21445 | 0.8689 | 0.4467 |
| λ 10 | 300 | 0.21463 | 0.8685 | 0.4462 |
| depth 3 / 4 / 6 / 8 | 650 / 400 / 250 / 150 | 0.21493 / 0.21475 / 0.21488 / 0.21542 | 0.8680 / 0.8684 / 0.8681 / 0.8674 | ≈ 0.445 |
| `min_child_hessian` 50; row fraction 0.5; 255 bins; `cat_smooth` 50 | | 0.2147–0.2151 | | no gain |
| boosted from the LR logit (`init`) | 150 | 0.21532 | 0.8676 | 0.4439: worse, dropped |

- With η 0.05 the default overfits fast: its nested F1 is 0.4323 at 2,000 trees. Always pick the number of trees from `p1 stages`.
- **The Fisher-sorted categorical splits overfit.** Ordinal splits on the answer codes, with the side of the missing values learned, generalize better. This is a finding for the report.

**Round 2**, on top of ordinal splits + feature fraction 0.3 + λ 10:

| Change | Trees | Log-loss | AUC |
|---|---|---|---|
| none | 600 | 0.21377 | 0.8699 |
| feature fraction 0.2 | 700 | 0.21372 | 0.8699 |
| `min_child_hessian` 1 | 600 | 0.21380 | 0.8698 |
| depth 4 / 6 | 950 / 500 | 0.21385 / 0.21382 | 0.8697 / 0.8699 |
| **η 0.02** | 1,750–2,000 (flat from 1,500 to 2,250) | **0.21349** | **0.8703** |

Between 500 and 900 trees the curves are flat at η 0.05, so 550 trees (the choice of `gbdt_tuned.toml`) is inside the optimum. η 0.02 is slightly better in log-loss, but its F1 is within noise (nested 0.4456 at 2,500 trees vs 0.4453–0.4459 for the 550-tree runs on the same folds).

## 4. Ablations so far (for the report)

| Model | Score | Shows |
|---|---|---|
| LR, one-hot encoding | nested F1 0.4386, AUC 0.8643 | the step-2 model |
| GBDT depth 1 (additive, η 0.2) | best stage 0.21633 / AUC 0.8660 / F1 0.4435; nested 0.4417 at 2,000 trees | binned main effects alone already beat the LR (+0.003) |
| GBDT depth 2 (η 0.1) | best stage 0.21544 / AUC 0.8672 / F1 0.4445 | plus pairwise interactions |
| GBDT depth 5, tuned | nested 0.4457 (bag 0.4462), AUC 0.8698 | plus higher-order interactions |
| GBDT without `HAREHAB1` | best stage 0.21867 / AUC 0.8647 / F1 0.4372 | the label proxy is worth about 0.008–0.009 |

These rows used round-1 settings (sorted categorical splits, seed 42). For the final table, rerun them with the tuned settings on 3 fold seeds (§6, step 3).

## 5. What this suggests

- **We are at the plateau.** Our best nested F1 is 0.446. The best result reported on this dataset is 0.4463 nested. With the +0.010 offset, expect **about 0.456 ± 0.005** on the leaderboard.
- Further GBDT tuning (η 0.02, feature fraction, depth) moves log-loss by about 3·10⁻⁴ and F1 by at most 0.001. That is below the leaderboard's resolution, so choose by CV, not by the leaderboard.
- Bagging and blending do not help here. Keep a small seed bag for stability, and report the negative result: it is evidence, and it answers the bagging question.
- The cheap wins are done. What remains is a reproducible final submission, the ablation table, and the report.

## 6. Next steps, in order

1. **Final submission: ready.** Job 40712920 ran `run.py` in 2 min 46 s, refitting the 3-seed bag of `gbdt_tuned` on all rows, with threshold 0.2166.
   - `outputs/submissions/best.csv` flags 12.94% of the test rows (14,158), and all 238 `HAREHAB1` rows are predicted 1.
   - It agrees with the fold-model average `fc_bag.csv` on 99.30% of the rows (767 borderline rows differ; 14,139 positives there).
   - **Submit `best.csv`**: it is the file `run.py` reproduces. Expected ≈ 0.456.
2. **Optional upgrade to η 0.02.** Run `gbdt_tuned` with `model.learning_rate=0.02 model.n_trees=1800` for fold seeds 42/43/44 × model seeds 0/1/2. Compare its bag with `fc_bag` (selection rule: highest mean nested F1 over the 15 folds). If it wins, update `gbdt_tuned.toml` and `best.toml`, with the threshold from `p1 submit`, and rerun `run.py`, which then takes about 3× longer.
3. **Ablations for the report**, all with the tuned settings on fold seeds 42/43/44 (`L` is a shell helper):

   ```bash
   L() { n=$1; s=$2; shift 2; slurm/submit.sh -J $n --array=0-4 -c 1 --mem=6G -t 01:30:00 -- cv gbdt_tuned --set name=$n --set seed=$s "$@" --fold @TASK@; }
   for s in 42 43 44; do
     L abl_d1 $s --set model.depth=1 --set model.learning_rate=0.2 --set model.n_trees=1200
     L abl_d2 $s --set model.depth=2 --set model.learning_rate=0.1 --set model.n_trees=600
     L abl_sorted $s --set model.categorical=categorical --set model.n_trees=400
     L abl_noharehab $s --set 'drop=["HAREHAB1"]'
     slurm/submit.sh -J lr_noharehab --array=0-4 -c 4 --mem=27G -t 00:40:00 -- cv step2/logistic --set name=lr_noharehab --set seed=$s --set 'drop=["HAREHAB1"]' --fold @TASK@
     slurm/submit.sh -J baseline --array=0-4 -c 2 --mem=13G -t 00:20:00 -- cv baseline --set seed=$s --fold @TASK@
   done
   ```

   Then use `p1 collect`, and `p1 compare` against `fc_bag` or `logistic`. Check `abl_sorted`'s best number of trees with `p1 stages`.
4. **Ideas not tried** (each expected ≤ +0.002, low priority):
   - a bag of configurations on 3 fold seeds;
   - an MLP or factorization machine as a blend member (the LR got weight 0.05, so probably 0);
   - GBDT features such as counts of chronic conditions or of don't-know answers.
5. **Documentation.**
   - `STRUCTURE.md`: the contracts of `p1/gbdt.py`, `p1/ensemble.py`, the new commands and `slurm/`.
   - README: a GBDT results table.
   - The report, with the figure "validation log-loss and F1 vs trees by depth", from `ensemble.stages`.

## 7. State of the code (not committed yet)

- **New**: `p1/gbdt.py`, `p1/ensemble.py`, `slurm/p1.sbatch`, `slurm/submit.sh`, `configs/gbdt.toml`, `configs/gbdt_tuned.toml`, this file.
- **Changed**:
  - `p1/evaluation.py`: fold files, `--fold`.
  - `p1/cli.py`: `collect`, `compare`, `blend`, `stages`, `submit`, `--fold`, `--variant`, a `__main__` guard.
  - `p1/pipeline.py`: public `transform`; the `categorical` and `init` model options.
  - `p1/models.py`: `gbdt` registered.
  - `p1/data.py`: `drop_features`; atomic `save_dataset`.
  - `p1/config.py`: `drop` key.
  - `p1/metrics.py`: `log_loss`.
  - `p1/paths.py`: `OOF_DIR`.
  - `run.py`: implemented.
  - `configs/best.toml`: the GBDT bag.
  - `README.md`, `PROGRESS.md`.
- `implementations.py` is untouched. `uv run black --check .` passes, and no `.py` file of ours contains "todo".
- Speed: a depth-5 tree takes 0.45 s on a training fold with feature fraction 0.6, and about 0.25 s with 0.3. A fold of 550 trees takes about 4 min, a fold of 2,500 trees at η 0.02 about 12 min.
