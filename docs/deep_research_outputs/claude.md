# Beating 0.440 F1 on BRFSS 2015 `_MICHD` with NumPy only: build one histogram GBDT, then blend it with your logistic regression

A histogram gradient-boosted tree model written in NumPy is the one method with direct evidence of beating a well-encoded logistic regression on this exact course dataset. Expect about +0.004 to +0.008 cross-validated F1 and +0.005 AUC. A logit average of the GBDT and your current LR should add roughly +0.001 to +0.003 more. Nothing credible points to a jump beyond about 0.450 CV F1. Your 0.440 is already within about 0.007 of the best documented result on this data.

## TL;DR

- **Ceiling.** The best documented NumPy results on this exact dataset (328,135 × 321, `_MICHD`) come from 2026 CS-433 teams. One team's average of four GBDTs reached pooled 5-fold F1 0.4473 (nested 0.4463, AUC 0.8700). Another team's single GBDT reached nested F1 0.4422. Their own logistic regressions scored 0.4346–0.4382. Expect your 0.440 / 0.864 to top out near 0.445–0.450 CV F1. Public AIcrowd scores run about 0.01 above CV.
- **Do two things, in this order.** (1) A depth-5, level-wise histogram GBDT on integer-coded raw features with a dedicated NaN bin: about 4–6 h to build, 15–40 min for 5-fold CV in parallel, expected +0.004 to +0.008 F1. (2) A logit or rank average of GBDT + LR (+ your MLP) on out-of-fold scores, with one threshold picked on the pooled OOF blend: 1 h, +0.001 to +0.003.
- **Skip the rest.** Factorization machines, GBDT-leaf → LR, bigger MLPs, F1-surrogate losses, expected-F1 decoding, class weights and resampling are all expected to move F1 by less than your 0.003 noise floor. The 2015 questionnaire shows that `HAREHAB1` is the only feature conditioned on the label questions, so no second leak exists to find.

## Key Findings (summary)

1. **Realistic ceiling: about 0.445–0.450 CV F1 and AUC about 0.870** (BRFSS-based, same dataset). Team F1-Cardiaque's four-GBDT NumPy ensemble reports pooled F1 0.4473, nested 0.4463, AUC 0.8700. Their best LR, on 2,207 encoded columns, reached 0.4382.\[1\] Antonio Arbizzani Tonelli's NumPy booster reports nested F1 0.4422 against 0.4346 for LR with the same cleaning (+0.0077). It placed an author-reported 3rd on AIcrowd on 14 Oct 2026, with a public F1 of 0.455.\[2\]
2. **Your LR is already stronger than those teams' LRs** (0.440 vs 0.435–0.438). The GBDT gap over your LR is therefore probably at the low end of the +0.006 to +0.009 they observed (own estimate: +0.004 to +0.007).
3. **Outside this dataset, the GBDT advantage on BRFSS-like data is small but consistent in AUC.**
   - On the 21-feature Kaggle BRFSS 2015 set: LightGBM 0.8385, XGBoost 0.8372, LR 0.8355 AUC; LR had the best F1 (0.3813).\[3\]
   - On a 37-feature BRFSS 2023 set: HistGradientBoosting AUC 0.851 / F1 0.405 vs LR 0.847 / 0.399.\[4\]
   - In clinical prediction generally, Christodoulou, Ma, Collins, Steyerberg, Verbakel & Van Calster (J Clin Epidemiol 2019;110:12–22) report that "for 145 comparisons at low risk of bias, the difference in logit(AUC) between LR and ML was 0.00 (95% confidence interval, -0.18 to 0.18)"; ML looked better only in the 137 high-risk-of-bias comparisons (logit(AUC) +0.34).
4. **Best gain per hour:** GBDT (about +0.006 for about 5 h), then OOF blending (about +0.002 for 1 h), then a rigorous repeated-CV protocol, which buys no F1 but protects you from leaderboard overfitting.
5. **The skip-logic check is clean.** In the 2015 questionnaire, only `HAREHAB1` is gated on `CVDINFR4` (heart attack = yes). No question is gated on `CVDCRHD4`. `CVDASPRN` is asked of all module respondents; `RLIVPAIN`/`RDUCHART`/`RDUCSTRK` are asked only if `CVDASPRN` = yes; `ASPUNSAF` is asked otherwise.\[5\]

## Ranked table

| Rank | Approach | Expected ΔF1 / ΔAUC vs 0.440 / 0.864 | Evidence behind the number | NumPy effort | 5-fold CV runtime (your 5800X) | Risk | Sources |
|---|---|---|---|---|---|---|---|
| 1 | Histogram GBDT, depth-5 level-wise, about 500 trees, NaN bin, integer-coded raw features | +0.004 to +0.008 / +0.004 to +0.006 | Same dataset: GBDT nested 0.4445 vs LR 0.4382 (+0.006) at AUC 0.8691; GBDT 0.4422 vs LR 0.4346 (+0.008). Discounted for your stronger LR (own estimate) | 4–6 h | about 15–40 min with 5 folds in parallel (own estimate; one team reports about 26 min per fold single-core on a laptop for 500 trees) | Medium: implementation bugs and the missing-value split direction | F1-Cardiaque repo (SaFouNaldo/projects-ML-2026); Arbizzani Tonelli repos; Ke et al., NeurIPS 2017 |
| 2 | Blend: logit or rank average of GBDT + LR (+ MLP), threshold on pooled OOF blend | +0.001 to +0.003 over the best member / +0.000 to +0.002 | Same dataset: average of 4 GBDTs pooled 0.4473 vs best member 0.4458 (+0.0015), AUC +0.0005 to +0.0009. Your own MLP+LR rank average gave +0.0017. Kaggle BRFSS: Hasnat et al. (arXiv 2511.01947, 229,781 patients) report that their weighted LightGBM/XGBoost/CNN ensemble "achieves a statistically significant improvement over the best individual model, with a Test AUC of 0.8371 (p=0.003)", yet their LightGBM alone had test AUC 0.8385, so the real effect is tiny | 0.5–1 h | Minutes (reuses OOF scores) | Low | F1-Cardiaque; Hasnat et al., arXiv 2511.01947 |
| 3 | GBDT boosted from the LR margin (init score = LR logit) | 0 to +0.002 over plain GBDT (own estimate); mainly saves trees | No direct published evidence on this data; it is the standard "boost from a base margin" idea | +0.5 h on top of #1 | Fewer trees (150–300), so about half of #1 | Low–medium: LR overfit leaks into the init if the LR is not refit per fold | Own estimate |
| 4 | GBDT leaf indices one-hot → L2 LR | 0 to +0.002 | He et al. (ADKDD 2014) reported a >3% normalized-entropy gain on ad clicks, a domain with far richer interactions; your 190 pairwise products gave only +0.0018 | 1–2 h after #1 | +10 min | Medium: overfits with many leaves | He et al., 2014\[6\]\[7\] |
| 5 | Factorization machine on one-hot features | 0 to +0.002 | Rendle (ICDM 2010) designed FMs for extremely sparse ID features; your explicit top-20 products already capture most of the pairwise signal (+0.0018) | 3–4 h | 10–20 min | Medium | Rendle, 2010\[8\] |
| 6 | Embedding MLP, multi-seed ensemble | 0 to +0.002 | Your MLP already ties LR (0.440). Kadra et al. (NeurIPS 2021) need per-dataset searches over 13 regularizers to beat XGBoost; Grinsztajn et al. (NeurIPS 2022) and McElfresh et al. (NeurIPS 2023) find trees better on irregular, heavy-tailed data | 3–5 h | 30–60 min | High (tuning time) | Kadra 2021; Grinsztajn 2022; McElfresh 2023\[9\]\[10\] |
| 7 | Expected-F1 decoding, F1 surrogate loss, focal loss | about 0 | Nan et al. (ICML 2012): EUM and decision-theoretic approaches are asymptotically equivalent with large train and test sets; your threshold is already within 0.002 of the oracle | 1–3 h | — | Low gain | Lipton et al., ECML-PKDD 2014; Nan et al., 2012\[11\] |

## Recipes for the top 3

### Recipe 1: Histogram GBDT in vectorized NumPy

**Data preparation (once, outside the folds; binning uses only training-fold statistics):**
- **Categoricals and raw codes.** Map each raw value to a small integer index. Keep 7/77/777 (don't know) and 9/99/999 (refused) as their own codes, since don't-know behaves like yes. Merge levels with fewer than about 30 rows. Do not one-hot encode: trees split on integer codes directly.
- **Numerics.** Apply your code rules (88→0, unit conversion). Then use up to 63 quantile bins fitted on the training fold. F1-Cardiaque cut raw columns into 63 quantile bins. Their v1 (raw codes) and v2 ("none" → 0, alcohol in days per month) binnings gave nested 0.4445 vs 0.4452, so heavy code cleaning is optional for trees.\[1\]
- **Missing values.** NaN gets its own bin, index 0. Missingness is informative (BPMEDS, DIABAGE2), and a dedicated bin lets the tree send it to either side.
- **Storage.** Store X as a Fortran-ordered `uint8` array of 328k × 291 (about 95 MB), so each feature column is contiguous.
- **Columns to drop.** Keep all 291 features, including `HAREHAB1`. Dropping columns hurt you before, and trees are robust to uninformative features (Grinsztajn et al. 2022).\[12\]

**Training loop (binary log-loss, Newton boosting):**
1. Set `F = log(p̄/(1−p̄))` (or the LR margin; see Recipe 3).
2. For each tree:
   - Compute `p = σ(F)`, `g = p − y`, `h = p(1 − p)`.
   - Draw a row subsample (0.7) and a column subsample (0.5–0.8 per tree).
3. Grow the tree level-wise to depth D. Keep one array, `node_id`, with each row's node at the current level.
   - For each sampled feature f, build all node histograms of the level in one call: `Gh = np.bincount(node_id*B + X[:, f], weights=g, minlength=n_nodes*B)`, and the same with `h`.
   - This is one linear pass per feature per level, with no fancy-index gathers. At the root that is 291 passes over 262k uint8 values, about 76M element operations per level.
   - Optional: histogram subtraction. Build only the smaller child and get the sibling as parent − child. This roughly halves the work (Ke et al., NeurIPS 2017; LightGBM docs), but it complicates the single-bincount trick.\[13\] Skip it in version 1.
4. Find splits on reshaped `(n_nodes, B)` histograms.
   - For ordinal and numeric features: take cumulative sums over bins 1..B−1 and score `GL²/(HL+λ) + GR²/(HR+λ) − G²/(H+λ)`.
   - Evaluate the NaN bin twice, once added to the left side and once to the right, and keep the better.
   - For nominal features (state, employment, marital status, …): sort the bins by `G/H` within the node, then scan the cumulative sums. This is the LightGBM-style gradient-sorted categorical split.
   - Mask out candidates where either child has `H < min_child_hessian`.
5. Leaf value is `−G/(H+λ)`. Update `F += η · leaf_value[node_id]` for all training rows (not only the subsample) and for the validation rows.

**Hyper-parameters** (start / search range):
- Depth: 5 / {4, 5, 6}. Both same-dataset repos chose depth 5.\[1\]\[14\]
- Learning rate η: 0.05 / {0.03, 0.05, 0.1}.
- Trees: 500 / 300–1000. Use early stopping on inner-validation log-loss, or fix the count from fold 1.
- λ (L2 on leaves): 1 / {0.1, 1, 10}. Arbizzani Tonelli's final model uses leaf L2 = 1.\[2\]
- Min child hessian: 5 / {1, 5, 20}. Since h ≈ 0.08 on average, 5 ≈ 60 rows.
- Row subsample: 0.7 / {0.5, 0.7, 1.0}.
- Columns per tree: 0.6 / {0.4–0.8}. Arbizzani Tonelli sampled 128 of about 295 features per node.\[14\]
- Bins: 63.

**Speed and memory** (own estimate):
- Per tree: about 5 levels × 0.6 × 291 features × 2 bincounts of 0.7 × 262k rows. That should be about 1–2 s, giving about 8–15 min per fold for 500 trees.
- Run the 5 folds as 5 `multiprocessing` workers. Each holds about 0.5 GB, well within 30 GB.
- Cast g and h to float64 once per tree. Precompute `node_id*B` as int32.
- For context: F1-Cardiaque reports about 26 min per fold for 500 trees, single core on a laptop. Arbizzani Tonelli's full 200-tree fit on all 328k rows took 18 min.\[1\]\[14\]

**Pitfalls:**
- Fit bins inside each fold.
- Do not let the NaN bin take part in the ordinal cumulative scan.
- Clip p to [1e-6, 1 − 1e-6].
- Check that depth-1 trees with η = 1 reproduce univariate log-odds for one feature (unit test).
- Use deterministic seeds per fold.

**Validation:**
- Same 5 folds as your LR; threshold picked on the training-fold scores, as now.
- Report the paired per-fold ΔF1 and the ΔAUC.
- For the report's ablation: show GBDT with vs without `HAREHAB1`. On this data, F1-Cardiaque saw 0.4445 → 0.4356 without it.\[1\]

### Recipe 2: Out-of-fold blending and threshold choice

1. For each model (LR, GBDT, optionally the MLP), save the 5-fold OOF logits on the same folds, plus test logits (the mean of the 5 fold models, or a refit on all rows).
2. Convert each model's scores to a common scale. Either use logits (good when both models are calibrated; your LR is) or ranks normalized to [0, 1].
3. Blend with weights w ∈ {0.3, 0.5, 0.7} for GBDT vs LR. Pick w on pooled OOF F1. With only 1–2 free weights, overfitting is negligible at 328k rows. Stacking with a meta-LR on OOF logits gives about the same result and costs more; F1-Cardiaque used equal-weight logit averaging.\[1\]
4. Threshold: maximize F1 on the pooled OOF blended scores. Cross-check with a "nested" variant (choose on 4 folds, apply to the 5th). F1-Cardiaque's pooled and nested F1 differ by about 0.001.\[1\] Then apply that threshold to the test blend.
   - Fallback: match the OOF predicted-positive rate. F1-Cardiaque's final submission flagged 12.24% of test rows;\[1\] your LR flags 12%.
5. Validate the gain as the paired difference against GBDT alone on the same folds.

### Recipe 3: Validation protocol that can detect +0.002 to +0.005

- **Repeats.** Run 3 × 5-fold repeated CV with fixed fold seeds shared by all models. That gives 15 paired differences dᵢ in F1.
- **Test.** Use the Nadeau–Bengio corrected resampled t-test: `t = d̄ / sqrt((1/15 + n_test/n_train) · s_d²)` with `n_test/n_train = 0.25` and df = 14. Bouckaert & Frank (PAKDD 2004) found this corrected repeated-CV test the most replicable; they recommend 10 × 10 CV, but 3 × 5 is a pragmatic compromise for your deadline.\[15\]\[16\]
- **Power** (own calculation). With your paired SD of about 0.001: SE ≈ 0.001 × sqrt(0.317) ≈ 0.00056. A +0.002 gain then gives t ≈ 3.5, which is detectable. A +0.001 gain is not.
- **Leaderboard discipline.** Pick the final model by CV, not by the public leaderboard. Use at most 2 submissions per candidate: GBDT alone, then the blend.

## Answers to the research questions

### Q1. Benchmarks and ceiling

**Same course dataset (BRFSS-based, 2026 CS-433 public repos):**
- **GBDT results.**
  - F1-Cardiaque, depth-5 NumPy GBDT: pooled 0.4457, nested 0.4445, AUC 0.8691.\[1\]
  - F1-Cardiaque, average of 4: pooled 0.4473, nested 0.4463, AUC 0.8700. Their LR was 0.4382; without `HAREHAB1`, their GBDT dropped to 0.4356.\[1\]
  - Arbizzani Tonelli: GBDT nested 0.4422 vs LR 0.4346 vs ridge 0.4195. The 3-outer × 2-inner run took 46.6 min. Their final pipeline scored AIcrowd F1 0.455 (accuracy 0.882), author-reported 3rd on 14 Oct 2026.\[2\]\[14\]
- **LR-only teams.** Kenzoud: holdout 0.438, AIcrowd 0.447; they note leaderboard values of about 0.46.\[17\] BYT18: 5-fold F1 0.440 ± 0.008, AUC 0.865, the same as yours.\[18\] hedihidri1: F1 0.434, AUC 0.8586.\[19\]
- **CV-to-leaderboard offset.** AIcrowd scores sit about 0.009–0.011 above CV for two teams (0.438 → 0.447; 0.444 → 0.455). Plausible causes are refitting on 100% of the rows (your learning curve suggests about +0.002) and the public split's noise.
- **Earlier editions.** I could not retrieve the Fall 2023–2025 AIcrowd leaderboards or their median team scores. AIcrowd lists the challenge as completed in 2019–2025,\[20\] but the leaderboard was not fetchable. So I give no median-team number.

**Kaggle "Heart Disease Health Indicators" (BRFSS 2015, 21 features):**
- Hasnat et al. (arXiv 2511.01947): test AUC LightGBM 0.8385, XGBoost 0.8372, LR 0.8355, RF 0.8327; test F1 0.3774–0.3813, with LR the highest.\[3\]
- The best credible numbers are therefore AUC about 0.84 and F1 about 0.38–0.41, with GBDT ≤ +0.003 AUC over LR.
- Papers reporting AUC 0.95–0.98 / F1 0.94 on this set used SMOTE before splitting or similar leakage, and are not credible. A 2026 medRxiv paper (EXHEART) explicitly notes that many studies apply SMOTE before the train/test split.\[21\]\[22\]

**"Personal Key Indicators" (BRFSS 2020):**
- The literature reviewed in a 2024 medRxiv paper cites LR AUC 0.84 (Mamun et al.). XGBoost AUC 0.9487 (Akkaya et al.) on a heavily preprocessed split looks optimistic.\[23\]
- I found no rigorous LR-vs-GBDT comparison there.

**Peer-reviewed BRFSS studies:**
- Noh et al. (BMC Public Health 2026; BRFSS 2023, 348 predictors) is the closest analogue to your feature richness. Gradient boosting reached ROC-AUC 0.867, PR-AUC 0.389, best F1 0.433.\[24\] After excluding post-diagnosis proxies (medications, functional limitations, general health), it reached 0.858 / 0.372 / 0.418.\[25\]
- "An interpretable LightGBM model for predicting coronary heart disease" (PLOS ONE, 12 Sep 2025, doi:10.1371/journal.pone.0330377) reports "an accuracy of 90.60% and an AUROC of 81.06% on the BRFSS_2015 dataset", with F1 of 47.6% at the original parameters and 48.04% after tuning (AUROC 81.11%). That F1 is hard to square with an AUROC of 0.81 at 9% prevalence, which suggests a different evaluation set-up, so do not treat it as a ceiling.
- Christodoulou, Ma, Collins, Steyerberg, Verbakel & Van Calster (J Clin Epidemiol 2019;110:12–22): "For 145 comparisons at low risk of bias, the difference in logit(AUC) between LR and ML was 0.00 (95% confidence interval, -0.18 to 0.18)."

**Verdict:**
- 0.440 / 0.864 is about 0.006–0.008 F1 and 0.005 AUC below the best demonstrated NumPy GBDT ensembles on the identical data.
- A tuned GBDT's realistic gap over a well-encoded LR on survey data is +0.003 to +0.008 F1 and +0.002 to +0.006 AUC.

### Q2. Models

- **Histogram GBDT.** This is the only method with same-dataset evidence (see Recipe 1).
  - Depth-wise vs leaf-wise: both successful repos used depth-5 trees. F1-Cardiaque grew them level by level.\[1\] Level-wise growth vectorizes best with the `node_id*B + bin` bincount.
  - Leaf-wise growth (LightGBM) gives lower loss for a fixed number of leaves but is harder to batch in NumPy and riskier to overfit (LightGBM docs).\[13\]
  - Categorical handling: integer codes with gradient-sorted splits for nominal features; NaN as its own bin, sent to the better side (F1-Cardiaque: "missing values go to the better side of each split").\[1\]
  - Boosting from the LR margin: worth trying as an ablation (Recipe 3 in the table). There is no published evidence, so expect mostly fewer trees, not much more F1.
- **GBDT leaves → LR** (He et al., ADKDD 2014; >3% NE gain on ad clicks):\[26\] low priority. Your pairwise products already show that interactions are worth only about +0.002 in linear form.
- **Factorization machines** (Rendle, ICDM 2010): they compute all pairwise interactions in linear time, but their advantage is in very sparse ID data.\[8\] Expect ≤ +0.002.
- **Better neural nets.**
  - Grinsztajn et al. (NeurIPS 2022, 45 datasets): trees win on medium-sized data, and NNs must "be robust to uninformative features" (you have 57 with no signal).\[9\]
  - McElfresh et al. (NeurIPS 2023, 19 algorithms × 176 datasets): GBDTs are "much better than NNs at handling skewed or heavy-tailed feature distributions" (you have 33 heavy-tailed numerics), and light GBDT tuning often matters more than the model family.\[27\]
  - Kadra et al. (NeurIPS 2021): MLPs beat XGBoost only with per-dataset searches over 13 regularizers, which is infeasible by tomorrow.\[10\]
- **Ensembles.** Equal-weight logit averaging worked on this data (+0.0015 pooled F1).\[1\] Choose the ensemble threshold on pooled OOF blended scores (Recipe 2).
- **GA²M / EBM.** This is essentially a restricted GBDT: depth-1 or depth-2 cyclic boosting. If you build Recipe 1, you get it for free by setting depth = 2. Do not build it separately.
- **Target or WoE encoding** for LR: with one-hot main effects already saturated (your learning curve is flat at 100%), expect about 0.

### Q3. Features from the documentation

**Skip patterns** (CDC 2015 BRFSS Questionnaire, final 12/29/2014, Module 9 "Cardiovascular Health", pp. 55–57):
- `HAREHAB1`: "If Core Q6.1 = 1 (Yes), ask Q1", where Q6.1 = `CVDINFR4`.\[5\] This is the leak.\[28\]
- `STREHAB1`: gated on Q6.3, the stroke question.\[5\] Not a leak.
- `CVDASPRN`: "asked for all respondents" of the module.\[5\]
- `ASPUNSAF`: asked when `CVDASPRN` ≠ yes.\[5\]
- `RLIVPAIN`, `RDUCHART`, `RDUCSTRK`: asked only when `CVDASPRN` = 1.\[5\]
- No question anywhere in the 2015 core or optional modules is conditioned on `CVDCRHD4`. Sodium module (`WTCHSALT`, `LONGWTCH`, `DRADVISE`): `WTCHSALT` and `DRADVISE` are asked of all respondents; `LONGWTCH` only if `WTCHSALT` = yes.\[5\] No cardiovascular gating.
- Module coverage: CDC lists the Cardiovascular Health module in Georgia, Iowa, Oregon and Puerto Rico in the combined file (Maine in a version file).\[29\]
- Fang, Ayala, Luncheon, Ritchey & Loustalot (MMWR 2017;66(33):869–873) report that "in 2015, a total of 20,776 participants from four states completed the module, 4.3% (3.9–4.7) of whom were heart attack survivors". That is a weighted percentage, not a count, but it is the same order of magnitude as your 773 + 238 = 1,011 `HAREHAB1` answers.

**Implication for the module states:**
- Inside module states, "`CVDASPRN` answered but `HAREHAB1` NaN" means the respondent did not report a heart attack, so they can be positive only through angina/CHD.
- LR already represents this additively (a module-respondent indicator plus `HAREHAB1`-observed), and trees capture it automatically. No new leak is available.

**Label definition** (CDC "Calculated Variables in the 2015 Data File", v4):
- `_MICHD` = 1 if `CVDINFR4` = 1 OR `CVDCRHD4` = 1; = 2 if both = 2; missing otherwise.\[30\]
- So don't-know/refused respondents on both questions should not be in your labeled data unless the course recoded them.

**Risk-factor engineering:**
- Pooled-cohort-style factors (age, sex, diabetes, smoking, hypertension, cholesterol, kidney disease) are already one-hot main effects in your data.
- Your measured interaction gain (+0.0018) and the GBDT gain (+0.006) bound what engineered interactions can add.
- A GBDT finds age × sex, age × BP and similar interactions automatically, so hand-made counts and risk scores add about 0 on top of it (own estimate).
- For LR alone, a binned count of chronic conditions or "don't know" answers is a cheap ≤ 0.001 experiment, below your noise floor.
- A 2026 PR on the Arbizzani Tonelli repo added target-feature interactions ("phase 16"). Their final pipeline instead credits diet representation and conditional imputation,\[2\]\[31\] so I found no evidence that interactions mattered.

### Q4. From probabilities to F1

- Lipton, Elkan & Narayanaswamy (ECML-PKDD 2014, LNCS 8725, pp. 225–239): "if the classifier outputs are well-calibrated conditional probabilities, then the optimal threshold is half the optimal F1 value." Your 0.24 ≈ 0.48/2 matches, so your scores are calibrated and your thresholding is already optimal.
- Nan et al. (ICML 2012): the empirical-utility approach (threshold tuning) and the decision-theoretic approach (expected-F1 decoding) are asymptotically equivalent with large training and test sets.\[11\] With 109k test rows, expect a difference of about 0.
- Lipton et al. also warn of a "winner's curse" when thresholds are tuned on small samples.\[32\] At 262k rows that effect is below 0.002, as your oracle gap shows.
- Recommendation: threshold on pooled OOF scores of the final blend, with the predicted-positive-rate check (about 12%) as a sanity bound.
- Do not spend time on sigmoid-F1 surrogates, focal loss or cost weights. They only re-shape the score-to-threshold mapping, as your class-weight experiments showed.

### Q5. Validating small gains

- Use the 3 × 5 repeated, paired CV with the Nadeau–Bengio correction (Recipe 3). Bouckaert & Frank (2004) show this corrected repeated-CV test gives the best replicability; 10 × 10 is their gold standard.\[15\]\[33\]
- Leaderboard noise (own estimate): with roughly 2,000–5,000 positives in the public portion, the standard error of an unpaired F1 is about 0.005–0.008. The leaderboard cannot resolve gains below about 0.005, so treat it only as a sanity check.
- Avoid public-leaderboard overfitting by fixing the final choice from CV before your last submissions. The final ranking uses the private split.

## Suggested schedule (about 24 h to deadline)

1. **Hours 0–5:** implement Recipe 1, unit-test it on 10% of the rows, then run 5-fold CV in parallel. In parallel, save the LR's OOF and test logits.
2. **Hours 5–6:** Recipe 2 blend. Submission 1: GBDT alone. Submission 2: GBDT + LR blend.
3. **Hours 6–8:** optional LR-margin init variant and 3 × 5 repeated CV for the final pair (run it overnight).
4. **Remaining time:** write the report. The ablation table could read: raw least squares 0.410 → LR one-hot 0.440 → GBDT → GBDT + LR blend → GBDT without `HAREHAB1`.

## Don't-bother list

- **Class weights, undersampling or SMOTE.** These only move the threshold (your experiments; Hasnat et al. used class weights and still got F1 about 0.38).\[3\] SMOTE before splitting inflates BRFSS results to implausible AUC 0.98 / F1 0.94 levels.\[21\]
- **Dropping high-NaN or low-AUC features.** This cost you 0.008–0.010, and trees are robust to uninformative features anyway (Grinsztajn 2022).\[9\]
- **More L2 tuning, `log1p`, finer quantile bins for LR.** All within ±0.0004 in your runs.
- **Expected-F1 decoding, F1 surrogate losses, focal loss.** Asymptotically equivalent to tuned thresholds (Nan et al. 2012).\[11\]
- **Bigger or deeper MLPs without heavy regularization search.** Kadra 2021 needs per-dataset tuning; your MLP already overfits after 1–2 epochs.
- **Survey weights or design columns.** Removing them was neutral to slightly negative for you (−0.0008 when kept).
- **Hunting for another label leak.** The questionnaire shows only `HAREHAB1` is gated on the label questions.

## Caveats

- **Unverified benchmark numbers.** All same-dataset benchmarks come from public student repositories with author-reported numbers, not peer review. Arbizzani Tonelli explicitly states that their 0.455 leaderboard score informed model choice and that no nested evaluation exists for their exact final pipeline.\[2\]
- **Plagiarism.** Use those repos only for ideas and benchmarks, as you planned. Do not read their code before writing yours.
- **Own estimates.** Runtime and gain numbers labeled "own estimate" are extrapolations: the GBDT runtime on your machine, the discount for your stronger LR, and the leaderboard standard error.
- **Missing data.** I could not retrieve the 2015 LLCP codebook PDF or earlier-year AIcrowd leaderboards, so there is no median-team number.
- **Other studies aren't directly comparable.** The BRFSS 2023 study used a 0.5 threshold for LR and inverse-frequency weights.\[25\]

## Sources

1. [GitHub - SaFouNaldo/projects-ML-2026](https://github.com/SaFouNaldo/projects-ML-2026)
2. [GitHub - antonio-arbizzani-tonelli/cardiovascular-classification-numpy: Cardiovascular disease classification from survey data using gradient boosting built with NumPy, with model comparisons and error analysis.](https://github.com/antonio-arbizzani-tonelli/cardiovascular-classification-numpy)
3. [Interpretable Heart Disease Prediction via a Weighted Ensemble Model: A Large-Scale Study with SHAP and Surrogate Decision Trees](https://arxiv.org/html/2511.01947)
4. [GitHub - Fadhili-N/heart-disease-brfss · GitHub](https://github.com/Fadhili-N/heart-disease-brfss)
5. <https://cdc.gov/brfss/questionnaires/pdf-ques/2015-brfss-questionnaire-12-29-14.pdf>
6. [Decoding Ad-Clicks with Machine Learning — Part 1](https://medium.com/@thevisionaryvectorsblog/decoding-ad-clicks-with-machine-learning-part-1-ae37499d7155)
7. [\[PDF\] Practical Lessons from Predicting Clicks on Ads at Facebook](https://www.semanticscholar.org/paper/Practical-Lessons-from-Predicting-Clicks-on-Ads-at-He-Pan/daf9ed5dc6c6bad5367d7fd8561527da30e9b8dd)
8. [Factorization Machines (2010)](https://scispace.com/papers/factorization-machines-44urbwb9em)
9. [Why do tree-based models still outperform deep learning on tabular data?](https://arxiv.org/pdf/2207.08815)
10. [\[2106.11189\] Well-tuned Simple Nets Excel on Tabular Datasets](https://arxiv.org/abs/2106.11189)
11. [Optimizing F-measures: A tale of two approaches](https://eprints.qut.edu.au/107528/)
12. [Why do tree-based models still outperform deep learning on tabular data?](https://www.researchgate.net/publication/362123616_Why_do_tree-based_models_still_outperform_deep_learning_on_tabular_data)
13. [Features — LightGBM 4.7.0.99 documentation](https://lightgbm.readthedocs.io/en/latest/Features.html)
14. [GitHub - antonio-arbizzani-tonelli/ML-project-1](https://github.com/antonio-arbizzani-tonelli/ML-project-1)
15. [Evaluating the Replicability of Signiﬁcance Tests for](https://ml.cms.waikato.ac.nz/publications/2004/bouckaert-frank.pdf)
16. [The Variance Brain Foundation Models Forgot: Third-Order Statistics Predict Cognition Where Billion-Parameter Models Fail](https://arxiv.org/pdf/2606.04010)
17. [GitHub - Kenzoud/ML\_project1](https://github.com/Kenzoud/ML_project1)
18. [GitHub - BYT18/ML\_proj1: Project 1 for EPFL ML course · GitHub](https://github.com/BYT18/ML_proj1)
19. [GitHub - hedihidri1/Coronary-Heart-Disease-Prediction: Using regression techniques to predict Coronary Heart Disease. First project of CS-433. · GitHub](https://github.com/hedihidri1/Coronary-Heart-Disease-Prediction)
20. [AIcrowd](https://www.aicrowd.com/challenges/epfl-machine-learning-project-1)
21. [(PDF) Comparative Analysis of XGBoost, KNN, and SVM Algorithms for Heart Disease Prediction Using SMOTE-Tomek Balancing](https://www.researchgate.net/publication/399756341_Comparative_Analysis_of_XGBoost_KNN_and_SVM_Algorithms_for_Heart_Disease_Prediction_Using_SMOTE-Tomek_Balancing)
22. [EXHEART: A Fairness-Aware Explainable Stacked Ensemble for Cardiovascular Disease Classification with Cross-Instrument Disparity Attribution](https://www.medrxiv.org/content/10.64898/2026.06.03.26354879v1.full)
23. [Improving Heart Disease Probability Prediction Sensitivity with a Grow Network Model](https://www.medrxiv.org/content/10.1101/2024.02.28.24303495v1.full)
24. [Occupational and socioeconomic predictors of myocardial infarction and coronary heart disease: a machine learning analysis](https://link.springer.com/article/10.1186/s12889-026-27570-3)
25. [Occupational and socioeconomic predictors of myocardial infarction and coronary heart disease: a machine learning analysis - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC13326371/)
26. [Practical Lessons from Predicting Clicks on Ads at Facebook](https://andrewclark.co.uk/papers-for-product-managers/practical-lessons-from-predicting-clicks-on-ads-at-facebook)
27. [\[2305.02997\] When Do Neural Nets Outperform Boosted Trees on Tabular Data?](https://arxiv.org/abs/2305.02997)
28. [Ablation: exclude HAREHAB1 by AlessiaMeroni · Pull Request #1 · antonio-arbizzani-tonelli/ML-project-1 · GitHub](https://github.com/antonio-arbizzani-tonelli/ML-project-1/pull/1)
29. [Error processing SSI file](https://cdc.gov/brfss/questionnaires/modules/category2015.htm)
30. <https://www.cdc.gov/brfss/annual_data/2015/pdf/2015_calculated_variables_version4.pdf>
31. [Add target feature interactions (phase 16) and finetuning (phase 17) by lucaasabatini · Pull Request #3 · antonio-arbizzani-tonelli/ML-project-1 · GitHub](https://github.com/antonio-arbizzani-tonelli/ML-project-1/pull/3)
32. [Optimal thresholding of classifiers to maximize F1 measure](https://scispace.com/pdf/optimal-thresholding-of-classifiers-to-maximize-f1-measure-u2df3qv98r.pdf)
33. [Statistical Significance Tests for Comparing Machine Learning Algorithms - MachineLearningMastery.com](https://machinelearningmastery.com/statistical-significance-tests-for-comparing-machine-learning-algorithms/)