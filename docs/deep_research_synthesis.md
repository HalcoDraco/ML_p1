# Deep research: combined findings

On 2026-10-28, three deep-research engines answered `docs/deep_research_prompt.md`. Their raw reports are in `docs/deep_research_outputs/` (`chatgpt.md`, `claude.md`, `gemini.txt`). This file keeps what is useful for the project, judged against our data and results. The plan that follows from it is `docs/model_plan.md`.

Labels: (Claude), (ChatGPT), (Gemini) mark a claim from that report; (ours) marks our own measurement or judgment. Numbers that other teams report on this dataset come from their repositories and are not verified.

## 1. How far each report can be trusted

| Report | Strengths | Weaknesses | Use |
|---|---|---|---|
| Claude | Benchmarks on this exact dataset; a GBDT recipe written for NumPy; skip logic read from the 2015 questionnaire (confirmed on our data, §6); the right significance test | Its benchmarks are numbers reported by student teams of this year's edition | main source |
| ChatGPT | Sensible caution; the same ranking of methods | Generic; weak sources (an aggregator site); effort estimates far too high (GBDT 15–25 h); advises against using `HAREHAB1` | confirms the ranking |
| Gemini | Useful formulas (factorization-machine gradients, histogram tricks); boosting from the logistic regression's logit | Its claims about the aspirin module are false on our data (§6); the ceiling and the gains have no source (GBDT +0.008–0.012); it attributes to Grinsztajn et al. 2022 a result that paper does not contain; its paired t-test overstates significance | formulas only, after checking |

## 2. Bottom line

- **Ceiling.** With NumPy models on this data, CV F1 tops out around 0.445–0.450, with AUC around 0.870. The best result reported on this dataset is 0.4463 (nested CV) / 0.4473 (pooled out-of-fold), with AUC 0.870 (Claude). Our logistic regression (F1 0.439, AUC 0.864) is 0.006–0.011 below that.
- **GBDT.** Only one model family has evidence of beating a well-encoded logistic regression on this data: histogram gradient-boosted trees. All three reports rank it first. Expected gain for us: +0.004 to +0.007 F1 and about +0.005 AUC (ours: other teams' GBDTs beat their own logistic regressions, which are weaker than ours, by +0.006 and +0.008).
- **Blend.** An out-of-fold (OOF) logit average of the GBDT and the logistic regression should add +0.001 to +0.003 (Claude). Our MLP + logistic rank average gave +0.0017.
- **Everything else** is expected to move F1 by 0.002 or less, below our fold noise: factorization machines, GBDT leaves fed to a logistic regression, embedding MLPs, hand-made risk features, F1-shaped losses, expected-F1 decoding, resampling. Skip these, or try them only if time is left over.
- **Leaks.** There is no second label leak. Claude read this from the questionnaire, and our data confirm it (§6).
- **Validation.** Compare models with paired CV on the same folds and the Nadeau–Bengio corrected t-test. The public leaderboard cannot resolve gains below about 0.005.

## 3. Benchmarks

| Source | Data | Model | F1 | AUC | Note |
|---|---|---|---|---|---|
| Team A, this year's edition (Claude) | this dataset | average of 4 NumPy GBDTs | 0.4473 pooled, 0.4463 nested | 0.8700 | as reported by the team |
| same | | one GBDT, depth 5 | 0.4457 pooled, 0.4445 nested | 0.8691 | without `HAREHAB1`: 0.4356 |
| same | | logistic regression, 2,207 columns | 0.4382 | | |
| Team B, this year's edition (Claude) | this dataset | NumPy GBDT | 0.4422 nested | | AIcrowd public 0.455, a pipeline chosen on the leaderboard |
| same | | logistic regression | 0.4346 | | ridge 0.4195 |
| 3 teams with logistic regression only (Claude) | this dataset | logistic regression | 0.434–0.440 | 0.859–0.865 | one: holdout 0.438 → AIcrowd 0.447 |
| Hasnat et al., arXiv 2511.01947 (Claude) | Kaggle BRFSS 2015, 21 features | LightGBM / XGBoost / LR | 0.377–0.381, best for LR | 0.8385 / 0.8372 / 0.8355 | GBDT +0.003 AUC over LR |
| Noh et al., BMC Public Health 2026 (Claude) | BRFSS 2023, 348 predictors | gradient boosting | 0.433; 0.418 without post-diagnosis proxies | 0.867; 0.858 | closest to our feature set |
| Christodoulou et al., J Clin Epidemiol 2019;110:12–22 | 282 LR vs ML comparisons in clinical prediction | | | logit(AUC) difference 0.00 (−0.18 to 0.18) in the studies at low risk of bias | ML looked better only in the biased ones |
| Studies with AUC 0.95–0.98 and F1 0.94 on Kaggle BRFSS (Claude) | | | | | SMOTE applied before the split: leaked, ignore |
| PLOS ONE 2025 LightGBM (Claude); Salehi et al. 2026 (ChatGPT) | BRFSS 2015; unknown | | 0.48; 0.556 | 0.81; 0.846 | an F1 of 0.48 is implausible at AUC 0.81 and 9% prevalence; unknown setup: ignore both |

- **CV vs leaderboard.** Two teams scored about 0.01 higher on the AIcrowd public leaderboard than in CV (0.438 → 0.447 and 0.444 → 0.455). One notes leaderboard values around 0.46 (Claude). Our first submission will show whether the offset holds for us.
- **Plagiarism.** The other teams are in this year's edition. Use their numbers only as benchmarks. Do not open their code, since the course checks for plagiarism. It is also better not to cite them in the report. Their links are in `docs/deep_research_outputs/claude.md`.

## 4. Methods, ranked

ΔF1 and ΔAUC are relative to our logistic regression (0.439 / 0.864). "Ours" is the estimate we plan with.

| # | Approach | ChatGPT | Claude | Gemini | Ours | Effort | 5-fold CV cost | Verdict |
|---|---|---|---|---|---|---|---|---|
| 1 | Histogram GBDT, depth about 5, NaN bin | +0.005–0.010 | +0.004–0.008 F1, +0.004–0.006 AUC | +0.008–0.012 | +0.004–0.007 F1, +0.005 AUC | 4–6 h | 4–5 min per fold and core for 500 trees (measured, §5.1) | **do** |
| 2 | OOF logit average of GBDT and LR, plus GBDTs with other seeds | +0.001–0.003 (with the MLP) | +0.001–0.003 | +0.003–0.005 (with an FM) | +0.001–0.003 | 1–2 h, with the OOF tooling | seconds, once OOF scores exist | **do** |
| 3 | GBDT started from the LR logit | "not automatically beneficial" | 0 to +0.002; fewer trees | its main recipe | 0 to +0.002 | +1 h after #1 | one more LR fit per fold | optional |
| 4 | Factorization machine (FM) on the one-hot features | +0.002–0.005 | 0 to +0.002 | +0.003–0.006 | 0 to +0.002: the EDA's 190 explicit products of the top-20 features gave +0.0018 | 3–4 h | 10–20 min | only with spare time, as a blend member |
| 5 | GBDT leaf indices, one-hot, fed to an LR (He et al. 2014) | +0.002–0.005 | 0 to +0.002 | +0.005–0.009 | 0 to +0.002 | 1–2 h after #1 | +10 min | skip |
| 6 | MLP with embeddings; ensembles of seeds | 0 to +0.005 | 0 to +0.002 | 0 to +0.002 | ≤ +0.002: our MLP ties the LR and overfits after 1–2 epochs | 3–8 h | 30–60 min | skip |
| 7 | Hand-made risk features: age × sex, count of chronic conditions, `GENHLTH` × `PHYSHLTH`, aspirin flags | "test with a paired ablation" | ≈ 0 on top of a GBDT; ≤ 0.001 for LR | +0.004–0.007 | ≤ +0.002 for LR, 0 for GBDT | 1–2 h | 5 min | only if the GBDT fails (§7) |
| 8 | Expected-F1 decoding, sigmoid-F1 loss, focal loss, class weights | no | ≈ 0 | no | 0 | | | skip (§8) |

## 5. Recipes

### 5.1 Histogram GBDT in NumPy

This merges the three reports, with Claude's recipe as the base. (ours) marks our additions.

**Binning**, fitted on the training rows of each fold:
- Input: the 291 features of `clean.npz`, without one-hot encoding.
- NaN gets its own bin, 0, so a split can send the missing values to either side. Missingness is informative (`BPMEDS`, `DIABAGE2`).
- Numeric features (`cleaning.NUMERIC`): up to 63 quantile bins (Claude; Gemini uses 255, ChatGPT 64–255). A feature with fewer distinct values gets one bin per value.
- Categorical features: one bin per level that has at least `min_count` training rows, in code order. Rarer and unseen levels share an "other" bin. The don't-know and refused codes keep their own bins.
- Storage: `uint8` in Fortran order, so each column is contiguous; 262k × 291 takes 76 MB.
- Keep every feature, `HAREHAB1` included: trees ignore useless features (Grinsztajn et al. 2022). Code cleaning barely matters for trees (Claude: one team got 0.4445 with raw codes and 0.4452 with cleaned ones).

**Boosting** uses Newton steps on the log-loss, as in XGBoost (Chen & Guestrin 2016):
1. Start from F = log(p̄ / (1 − p̄)), or from the LR logit (§5.4).
2. For each tree: p = σ(F), g = p − y, h = p(1 − p). Draw 70% of the rows and 50–80% of the features.
3. Grow the tree level by level to depth D, keeping one array `node` with the node of each sampled row. For each sampled feature f, `np.bincount(node * B + X[:, f], weights=g, minlength=n_nodes * B)` gives the gradient histograms of every node of the level in one pass. Do the same with h.
4. Find the best split of every node at once, from the (features, nodes, bins) histograms:
   - Numeric features: take cumulative sums over bins 1…B−1. Add the NaN bin once to the left side and once to the right, and keep the better of the two: the direction of the missing values is learned, like XGBoost's default direction.
   - Categorical features: within each node, sort the bins by G / (H + `cat_smooth`), then scan the cumulative sums in that order. This finds the best split of the categories into two groups (Fisher 1958; LightGBM). NaN counts as one more category. ChatGPT's alternative, one category against the rest, is a special case.
   - The gain of a split is G_L²/(H_L + λ) + G_R²/(H_R + λ) − G²/(H + λ). Reject candidates where a child has H < `min_child_hessian`.
   - (ours) Store every split as a feature plus a boolean mask over its bins, true where a row goes left. Numeric and categorical splits are then applied the same way: `node = 2 * node + ~left[node, X[row, feature[node]]]`.
   - (ours) Grow complete trees: a node with no valid split sends all its rows to its left child. This removes all leaf bookkeeping and leaves the leaf values unchanged.
5. A leaf's value is −η G / (H + λ). Add it to F for every training row, not only the sampled ones, by passing all rows through the tree, and do the same for the validation rows. If the validation scores are kept after every block of trees, one run gives F1, AUC and log-loss as a function of the number of trees.

**Hyper-parameters.** Both GBDTs that work on this dataset use depth 5.

| Parameter | Start | Search | Source |
|---|---|---|---|
| depth | 5 | 3–6 | Claude |
| learning rate η | 0.05 | 0.05, 0.1 | all three |
| trees | 500 | up to 1,000, picked from the validation scores | Claude (Gemini's 100 is too few at η = 0.05) |
| λ (L2 on the leaf values) | 1 | 1, 10 | all three |
| `min_child_hessian` | 5, about 60 rows since h ≈ 0.08 | 5, 20, 50 | Claude 5, Gemini 20, ChatGPT about 40 (500 rows) |
| row subsample | 0.7 | 0.7 | Claude 0.7, ChatGPT 0.8 |
| features per tree | 0.6 | 0.4–0.8 | one team samples 128 of 295 features per node |
| bins | 63 | 63, 255 | |
| `cat_smooth` | 10 | | LightGBM's default |

**Cost** (ours, measured on this machine with one core, on random `uint8` data of the same shape):
- One depth-5 tree with 70% of the rows and 60% of the features: 0.33 s for the histograms, plus 0.10 s to gather the sample.
- With all rows and features: 0.80 s plus 0.25 s.
- So 500 trees take 4–5 min per fold, and the 5 folds run in parallel. For comparison, another team reports 26 min per fold on one core (Claude).

Speed-ups, and what was not adopted:
- Histogram subtraction (optional): build the smaller child's histogram and get its sibling as parent − child (Ke et al. 2017). This roughly halves the cost of levels 1 and below.
- Gemini's single `bincount` over all features at once (each feature's bins offset into one index array): it removes the Python loop over features but not the work, so it gains little with 291 features.
- Leaf-wise growth (Gemini, LightGBM), not adopted: it reaches a lower loss for a given number of leaves, but it is hard to vectorize in NumPy, and both GBDTs that work on this dataset grow level-wise trees of depth 5.

**Pitfalls**:
- Fit the bins inside each fold.
- Keep the NaN bin out of the numeric cumulative scan.
- Clip p to [1e-6, 1 − 1e-6].
- Seed the row and feature sampling per model (`np.random.default_rng`).
- (ours) The F1-optimal threshold on the training rows' own scores may be optimistic for trees, which fit their training rows more closely than the LR does. Compare it with an out-of-fold threshold (§5.2).

**Sanity checks**:
- Many depth-1 trees on one categorical feature converge to the per-level log-odds. Claude says one tree with η = 1 reproduces them, but that is a single Newton step, so the match is only approximate.
- On a synthetic XOR target, depth 2 learns it and depth 1 does not.
- The training scores recomputed by `predict_score` equal the F accumulated during training.
- The same seed gives identical scores.
- When all the NaN rows of a feature are positive, the trees split the NaN bin off.

### 5.2 Out-of-fold blending and the threshold

- **OOF scores.** Save each model's OOF scores on the same folds, plus its test scores: either the mean of the 5 fold models' test scores, or the scores of a model refitted on all rows (Claude).
- **Blend on the logit scale**: s = w · logit(p_GBDT) + (1 − w) · logit(p_LR), with w ∈ {0, 0.1, …, 1} picked on the pooled OOF F1. One or two weights cannot overfit 328k rows. Rank averaging is the alternative when scores are not calibrated, but it needs the OOF score distribution to rank the test scores. Stacking with a logistic regression on top gives about the same result for more work. The team with the 4-GBDT average used equal weights on the logits.
- **Threshold.** Use the threshold that maximizes F1 on the pooled OOF blended scores. To estimate its F1 honestly, use a nested variant: choose the threshold on 4 folds and apply it to the 5th. The team above saw pooled and nested F1 differ by 0.001.
- **Sanity bound.** The predicted-positive rate should stay near 12%. Our LR flags 12.1% of the rows of fold 1, and the other team's final submission flagged 12.24% of the test rows.
- **Theory** (all three): with calibrated probabilities, the F1-optimal threshold is F1*/2 (Lipton et al. 2014). Our thresholds of 0.21–0.24 ≈ 0.44 / 2, so our scores are calibrated and a single threshold is the right decision rule. Choosing the decisions that maximize expected F1 gives the same result at our sample sizes (Nan et al. 2012).

### 5.3 Validation protocol

- **Paired folds.** Every model uses the same folds (same `seed`). Compare two models by the per-fold differences d_i of their F1.
- **Corrected t-test.** Use the Nadeau–Bengio corrected resampled t-test (Nadeau & Bengio 2003; Bouckaert & Frank 2004 found it the most replicable test): t = mean(d) / sqrt((1/J + n_test/n_train) · var(d)). Here J is the number of (repeat, fold) pairs, n_test/n_train = 1/4 for 5 folds, and there are J − 1 degrees of freedom.
- Gemini's plain paired t-test divides by sqrt(J). It ignores that the training sets overlap, which overstates significance.
- **Power** (ours). The smallest gain that is significant at 5%:

  | Paired standard deviation | 1 × 5 folds | 3 × 5 folds | 10 × 5 folds |
  |---|---|---|---|
  | 0.001, what we measured between two encodings | 0.0019 | 0.0012 | 0.0011 |
  | 0.002, likely between two model families | 0.0037 | 0.0024 | 0.0021 |

  The correction term 1/4 dominates, so 3 × 5 folds is the useful maximum.
- **Leaderboard** (Claude's own estimate). With 2,000–5,000 positives in the public part, one F1 score has a standard error of 0.005–0.008. Choose the final model by CV, decided before the last submissions, and use the leaderboard only as a sanity check, with at most 2 submissions per candidate. The final ranking uses the private part of the test set.

### 5.4 Boosting from the logistic regression's logit

- Initialize F with the logit of a logistic regression fitted on the same training rows, refitted in each fold. An LR fitted on all rows would leak the validation labels into the initial scores.
- Expected effect: fewer trees (150–300 instead of 500) and 0 to +0.002 F1. There is no published evidence (Claude, ChatGPT), and Gemini's +0.008–0.012 has no source.
- The blend of §5.2 captures much of the same complementarity, so compare this against the blend, not against the plain GBDT.

### 5.5 Factorization machine (only with spare time)

- Model (Rendle 2010): logit = w₀ + Σᵢ wᵢxᵢ + ½ Σ_f [(Σᵢ v_if xᵢ)² − Σᵢ v_if² xᵢ²]. It costs O(N · D · k) instead of O(N · D²).
- Gradient of the log-loss for V: Xᵀ(e ⊙ A) − ((X ⊙ X)ᵀ e) ⊙ V, with A = XV and e = p − y (Gemini's formula, checked).
- (ours) Start w₀ and w from the fitted logistic regression and draw V from N(0, 0.01²). Then the gain over the LR is exactly that of the interaction term.
- Settings: k ∈ {4, 8}; Adam with a learning rate of 1e-3 to 1e-2; batches of about 8,192 rows; L2 on V from 1e-4 to 1e-2; early stopping on a 10% holdout, since our MLP overfit after 1–2 epochs.

## 6. Questionnaire skip logic: no second leak

Claude read Module 9 ("Cardiovascular Health") of the 2015 questionnaire:
- `HAREHAB1` is asked only if `CVDINFR4` (heart attack) = yes: the leak we already use. `STREHAB1` is asked only after a stroke.
- `CVDASPRN` (aspirin) is asked of every module respondent. `RLIVPAIN`, `RDUCHART` and `RDUCSTRK` are asked only if `CVDASPRN` = yes, and `ASPUNSAF` otherwise.
- No question depends on `CVDCRHD4` (angina or coronary heart disease). The sodium module (`WTCHSALT`, `LONGWTCH`, `DRADVISE`) does not depend on any cardiovascular answer.
- The label: `_MICHD` = 1 if `CVDINFR4` = 1 or `CVDCRHD4` = 1 (CDC, calculated variables, version 4).

Checked on our training data (ours, 2026-10-28):
- `CVDASPRN` is answered by 14,046 rows (4.3%), all in Georgia, Iowa, Oregon and Puerto Rico, where 81–98% of each state's rows answered it. These are the module states that CDC lists.
- `RLIVPAIN` is answered exactly when `CVDASPRN` = 1 (one exception), and `ASPUNSAF` exactly when `CVDASPRN` ≠ 1.
- `HAREHAB1` is answered only inside the module: 773 rows, all positive.
- Inside the module, a missing `HAREHAB1` means "no heart attack": those 13,273 rows are 4.1% positive, against 8.8% overall. A one-hot LR represents this additively (the `CVDASPRN` levels plus `HAREHAB1`), and trees find it, so there is nothing to add.
- Gemini's claims are false:
  - `CVDASPRN` is 95.7% NaN, not 65.9%.
  - It is 9.4% positive when answered and 8.8% when missing, not 21.4% vs 2.3%.
  - It is not limited to people with high blood pressure or cholesterol: 7,599 of its 14,046 respondents answered `BPHIGH4` = no.
  - The follow-up questions are 20.8% positive (`RLIVPAIN` and the other two) and 4.5% (`ASPUNSAF`), not above 30%.

About `HAREHAB1`: ChatGPT advises against using it, because it is post-diagnosis information. For this competition it is an input like any other, and dropping it costs 0.007–0.009 F1. The report should say that it is a proxy for the label and show the model without it (another team's GBDT: 0.4445 → 0.4356).

## 7. Features from epidemiology

- The established risk factors of the Framingham and Pooled Cohort Equations scores (age, sex, blood pressure, cholesterol, diabetes, smoking, kidney disease) are already one-hot main effects.
- Proposed interactions (ChatGPT, Gemini): age × sex (in the EDA, men have 1.4–2.1 times the risk of women above 40), age × hypertension, age × diabetes, hypertension × cholesterol, BMI × hypertension, `GENHLTH` × `PHYSHLTH`.
- A count of chronic conditions or of "don't know" answers adds nothing to the LR as a linear term, because it is a sum of existing one-hot columns. It needs binning or an interaction (ours, as the prompt already noted).
- Gemini's aspirin flags add nothing: "answered `CVDASPRN`" is already the NaN column of the encoding, and its "recommended aspirin" flag concerns 4% of the rows.
- Trees learn these interactions by themselves. For the LR alone, our EDA bounds the gain: the 190 products of the top-20 features gave +0.0018. So keep these features as the fallback if the GBDT fails, and skip them otherwise.

## 8. Don't-bother list

At least two reports agree on each item, and so do our experiments:
- **Class weights, undersampling, SMOTE.** They only move the threshold (EDA). Published BRFSS results with SMOTE applied before the train/test split are leaked.
- **Dropping high-NaN or low-AUC features.** It cost 0.008–0.010 (EDA), and trees are robust to useless features.
- **More L2 tuning, `log1p`, finer quantile bins for the LR.** Effects within ±0.0004 (EDA).
- **F1-shaped losses and expected-F1 decoding** (sigmoid-F1 surrogates, focal loss). A tuned threshold on calibrated probabilities is already optimal.
- **Deeper MLPs and tabular transformers** without a large regularization search. Kadra et al. 2021 needed a per-dataset search over 13 regularizers to beat GBDTs. Trees do better on irregular, heavy-tailed tabular data (Grinsztajn et al. 2022; McElfresh et al. 2023).
- **Target or weight-of-evidence encoding.** They risk leaking the labels, and the one-hot main effects are saturated: the learning curve is flat.
- **Kernel methods and exhaustive interactions.** An RBF SVM is impractical at 328k rows, and all pairwise products of the 1,527 columns give 1.16 M columns, which does not fit in memory.
- **Survey weights and design columns.** −0.0008 (EDA).
- **Searching for another leak** (§6).

## 9. Claims not kept

| Claim | From | Why not |
|---|---|---|
| The aspirin module depends on blood pressure or cholesterol; `CVDASPRN` is 65.9% NaN, 21.4% vs 2.3% positive | Gemini | false on our data (§6) |
| A ceiling of 0.448–0.452 from "meta-analytic evidence"; Kaggle numbers for LR and LightGBM | Gemini | no source (the same-dataset numbers of §3 give a similar ceiling anyway) |
| Grinsztajn et al. 2022 show that the gap between GBDTs and linear models on binned, one-hot features narrows | Gemini | that paper compares trees with neural nets, not with linear models |
| GBDT +0.008–0.012, leaves fed to an LR +0.005–0.009, risk features +0.004–0.007 | Gemini | no evidence; the GBDTs on this dataset gained +0.006–0.008 over weaker LRs |
| A plain paired t-test on 15 folds | Gemini | the training sets overlap, so it overstates significance (§5.3) |
| Do not use `HAREHAB1` | ChatGPT | a legitimate input for this task; discuss it in the report instead (§6) |
| F1 0.556 (Salehi et al. 2026) as a possible ceiling | ChatGPT | unknown setup, not comparable |
| A GBDT takes 15–25 h to implement | ChatGPT | the NumPy recipe is a few hundred lines: 4–6 h |
| One depth-1 tree with η = 1 reproduces the univariate log-odds | Claude | only many trees do; a single Newton step is approximate |

## 10. References

For the report. "(added)" marks standard references that no report gave. Check the details of the 2026 papers before citing them.
- **Gradient boosting and its NumPy design.**
  - Friedman, *Greedy function approximation: a gradient boosting machine*, Annals of Statistics 2001 (added).
  - Chen & Guestrin, *XGBoost*, KDD 2016: Newton leaf values, split gain, learned direction for missing values (added).
  - Ke et al., *LightGBM*, NeurIPS 2017: histograms, histogram subtraction, categorical splits.
  - Fisher, *On grouping for maximum homogeneity*, JASA 1958: the sorted categorical split (added).
  - Lou, Caruana, Gehrke & Hooker, *Accurate intelligible models with pairwise interactions* (GA²M), KDD 2013: boosting trees of depth 1 or 2 gives additive or pairwise models (added).
- **Other models.** He et al., *Practical lessons from predicting clicks on ads at Facebook*, ADKDD 2014; Rendle, *Factorization machines*, ICDM 2010.
- **Trees vs neural nets on tabular data.** Grinsztajn, Oyallon & Varoquaux, NeurIPS 2022 (arXiv 2207.08815); McElfresh et al., NeurIPS 2023 (arXiv 2305.02997); Kadra et al., NeurIPS 2021 (arXiv 2106.11189).
- **Logistic regression vs machine learning in clinical prediction.** Christodoulou et al., J Clin Epidemiol 2019;110:12–22.
- **F1 and thresholds.** Lipton, Elkan & Narayanaswamy, ECML-PKDD 2014 (arXiv 1402.1892); Nan, Chai, Lee & Chieu, *Optimizing F-measure: a tale of two approaches*, ICML 2012; Jansche, ACL 2007.
- **Comparing models.** Nadeau & Bengio, *Inference for the generalization error*, Machine Learning 2003 (added); Bouckaert & Frank, PAKDD 2004; Dietterich, Neural Computation 1998.
- **BRFSS documentation.**
  - 2015 questionnaire: https://www.cdc.gov/brfss/questionnaires/pdf-ques/2015-brfss-questionnaire-12-29-14.pdf
  - Calculated variables, version 4: https://www.cdc.gov/brfss/annual_data/2015/pdf/2015_calculated_variables_version4.pdf
  - Modules by state: https://www.cdc.gov/brfss/questionnaires/modules/category2015.htm
- **Prediction studies on BRFSS.** Noh et al., BMC Public Health 2026 (BRFSS 2023); Hasnat et al., arXiv 2511.01947 (Kaggle BRFSS 2015).
