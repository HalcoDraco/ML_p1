# Deep research request: maximize F1 for heart-disease prediction from BRFSS 2015, with NumPy only

## 1. Goal

We are a student team in EPFL's Machine Learning course (CS-433), Project 1. The task is to predict `_MICHD` (ever diagnosed with coronary heart disease or a heart attack) from answers to the 2015 BRFSS health survey. Predictions are scored on a hidden test set by **F1 of the positive class**, so F1 is what we maximize. Accuracy is secondary: with 8.8% positives, predicting all negatives already gives 0.912 accuracy, while our F1-optimal predictions give 0.880.

Our best pipeline reaches a 5-fold CV F1 of 0.440 (AUC 0.864) with L2 logistic regression on one-hot features. Since then, every refinement we tried has moved F1 by less than 0.002. We need to know three things: whether a different approach can raise F1 meaningfully, which approach, and how to implement it. **We can build at most one or two new methods before the deadline, so we need a ranked, evidence-based recommendation, not a survey.**

## 2. Hard constraints

- **Code:** only the Python standard library and NumPy, for all training and prediction. No scikit-learn, pandas, XGBoost, LightGBM, CatBoost, PyTorch, etc. Every algorithm must be implementable from scratch in vectorized NumPy.
- **No external code or data.** Code may not be copied from anywhere, because the course checks for plagiarism. No data beyond the provided files may be used. That rules out the public BRFSS 2015 files and the Kaggle datasets derived from them, which contain `_MICHD` for these same respondents. The BRFSS documentation (codebook, questionnaire) and ideas from papers and write-ups are allowed; we will cite them.
- **Compute:** one desktop with an AMD Ryzen 7 5800X (8 cores, 16 threads), 30 GB of RAM and no GPU. For scale: one gradient step of logistic regression on 262k × 1,527 float64 takes 0.18 s, and a full 5-fold CV of our model takes 5 min.
- **Time:** the deadline is 2026-10-29 at 16:00, about a day away, and the 2-page report is not written yet. A candidate method must be implementable in a few hours, and its 5-fold CV should finish within an hour.
- **Leaderboard:** at most 5 submissions per day. The final ranking uses a private part of the test set.
- **Secondary:** part of the report grade is an ablation study against baselines. Prefer methods whose effect can be measured in isolation.

## 3. The data

- **Source.** The course release of BRFSS 2015, a CDC telephone survey of US adults. Train has 328,135 rows and test has 109,379 rows, with 321 features. Feature names follow the 2015 LLCP codebook (all listed in §8). Every value is a float, and NaN means no answer.
- **Split.** Train and test are a random split: per-feature NaN rates differ by at most 0.004, and state shares by at most 0.0013. Shuffled k-fold CV is therefore a faithful proxy for the test score.
- **Label.** `_MICHD` is 1 if the respondent reported a heart attack (`CVDINFR4`) or angina / coronary heart disease (`CVDCRHD4`).
  - Those two questions are not among the features. `CVDSTRK3` (stroke) is.
  - 8.83% of rows are positive (28,975).
  - Predicting all positives gives an F1 of 0.162.
- **Feature kinds.** Two sources: raw answers (core sections, plus optional modules asked only in some states) and CDC computed variables (names starting with `_` or ending in `_`).
  - Raw answers are codes: 7/77/777 don't know, 9/99/999 refused, 8/88/888 none or never, 555 never, and 1xx/2xx/3xx for per day / week / month (mixed units).
  - Computed variables mostly use 9 for missing.
  - After our cleaning there are 56 numeric features and 235 categorical ones. The numeric ones are days of poor health, drinks, fruit and vegetable frequencies, activity minutes, age, height, weight, BMI, ...; 33 of them are heavy-tailed.
- **Missing values.** 45% of all cells are NaN, and no row is complete.
  - NaN share per feature: 82 features have none, 57 have ≤ 10%, 35 have 10–50%, 48 have 50–90%, 69 have 90–99%, and 30 have more than 99%.
  - Most NaNs come from skip logic (the question did not apply) or from optional modules (about 100 questions asked only in some states).
  - So a NaN carries information. `BPMEDS`, asked only of people with high blood pressure, is NaN in 60% of rows: 3.8% of those rows are positive, against 16.3% of the rows that answered. For `DIABAGE2` the rates are 6.9% vs 22.1%.
- **"Don't know" behaves like "yes".** Positive rate for yes / no / don't know: stroke 37.1 / 7.6 / 31.7%; COPD 26.7 / 7.2 / 23.8%; kidney disease 28.8 / 8.1 / 23.0%; high blood pressure 16.3 / 3.7 / 15.3%.
- **One label leak.** `HAREHAB1` (outpatient cardiac rehabilitation after a heart attack) is asked only of people who reported a heart attack.
  - All 773 train rows that answered it are positive, and 238 test rows answered it.
  - Our model learns it, which is worth about 0.007 F1.
  - We scanned every (feature, value) group with at least 30 rows, NaN included. No other group is above 46% positive.
  - `STREHAB1`, the stroke version, is not a leak (36% positive).
- **Signal.** Univariate AUC, with each value treated as a category:
  - Strongest: age 0.736, `GENHLTH` 0.729, `EMPLOY1` 0.715 (retired, unable to work), `BPMEDS` 0.701, `BPHIGH4` 0.688, the pneumonia vaccine 0.679 and the flu shot 0.673 (both proxies for age), `TOLDHI2` (high cholesterol) 0.676, `PHYSHLTH` 0.647, `DIFFWALK` 0.643, arthritis 0.640.
  - Then, in order: poor-health days, marital status (17% for widowed people), diabetes (22% for diabetics), income, use of special equipment, internet use, smoking, COPD, alcohol (11.4% for non-drinkers), education, stroke, veteran status, sex, BMI.
  - 57 features have no signal (AUC < 0.505), mostly from the optional modules.
  - Risk rises from 0.5% at ages 18–24 to 22% at 80+. Above 40, men have 1.4–2.1 times the risk of women (an age × sex interaction). By `GENHLTH`, it goes from 2.1% (excellent) to 31.9% (poor).
- **Redundancy.** 46 feature pairs have |r| > 0.95, mostly raw answers and their CDC recodes. Age appears in 6 variables; BMI, smoking, alcohol, blood pressure, cholesterol, arthritis and race also have several versions.
- **Learning curve.** Logistic regression on our encoding, trained on a share of the 262k training rows: 5% → F1 0.371, 10% → 0.405, 25% → 0.427, 50% → 0.438, 100% → 0.440. The curve is nearly flat at full size.

## 4. Current pipeline and results

**Encoding** (every statistic is refitted on each training fold):

- Drop 30 columns, leaving 291 features: 14 survey-design columns (ids, interview date, strata, sampling weights), 2 exact copies, and 14 raw mixed-unit answers that have a computed version.
- 56 numeric features: apply code rules first. These map 88 → 0 and 77/99 → NaN, convert units, and fill skip logic (for example "no exercise → 0 minutes"). Then impute the median and add a NaN indicator.
- 235 categorical features, with the don't-know and refused codes kept as levels: one-hot encode the levels that have at least 30 training rows (at most 100 per feature), plus a NaN column.
- Standardize. The result has about 1,526 columns.

**Threshold:** we use the F1-optimal threshold on the training rows' own scores, refitted in each fold. It never sees the validation fold and lands within 0.002 F1 of the oracle threshold.

**5-fold CV on all rows, same folds for every model:**

| Model | F1 | AUC |
|---|---|---|
| L2 logistic regression (gradient descent, 300 steps; an exact Newton fit gives the same F1) | 0.439 ± 0.003 | 0.864 |
| Least squares / ridge on the 0/1 label | 0.430 ± 0.002 | 0.857 |
| Linear regression by SGD (batch size 1) | 0.426 ± 0.003 | 0.855 |
| Baseline: least squares on the raw values, NaN → median | 0.410 ± 0.002 | 0.845 |

**Operating point on validation fold 1:**

- Threshold 0.24 on the predicted probability.
- 12% of rows flagged; precision 0.38, recall 0.52, F1 0.438, accuracy ≈ 0.88.

The threshold is close to F1/2, the known optimum for calibrated probabilities, so the probabilities are well calibrated.

**Noise:**

- From fold to fold, F1 varies by ±0.003.
- The paired difference between two pipelines on the same folds has a standard deviation of about 0.001.
- Gains below about 0.003 F1 will not show reliably on the leaderboard.

## 5. Already tried

Results are on an 80/20 split (noise about ±0.004 F1) unless noted. The reference F1 is 0.440.

| Change | Result |
|---|---|
| Raw values with NaN → median, then don't-know/refused codes → NaN, then + NaN indicators, then one-hot categoricals (our encoding) | 0.417 → 0.433 → 0.439 → 0.440 |
| One-hot everything, with numeric features as 10 or 20 quantile bins (≈ 2,200 columns) | 0.4387 / 0.4376 |
| `log1p` on heavy-tailed features; code cleaning of the numeric features inside the one-hot encoding (paired 5-fold) | changes of 0.0003 or less |
| One-hot `_STATE` (53 states); keeping the survey-design columns | +0.0009; −0.0008 |
| Balanced class weights; undersampling negatives 1:1 and 3:1 | 0.438; 0.435 and 0.438 (they only move the threshold) |
| L2 strength λ from 1e-6 to 1e-3 | ±0.0004 |
| Dropping features with more than 70% (up to more than 99%) NaN | −0.008 to −0.010, mostly from losing the leak |
| Keeping the top-k features by univariate AUC (k = 30 … 150) | 0.400 … 0.438 |
| 190 pairwise products of the top-20 features | +0.0018 F1, +0.0005 AUC |
| MLP in NumPy (1 hidden ReLU layer, Adam, early stopping): 64 units; 256 units with dropout 0.3 | 0.434; 0.440 (overfits after 1–2 epochs) |
| Rank average of that MLP with logistic regression | 0.4417 (+0.0017) |

## 6. Research questions, in priority order

1. **Benchmarks and ceiling.** What F1 (positive class) and AUC have been reported on this exact task and its close relatives, and with which methods?
   - We believe earlier editions of this course project used the same data on AIcrowd (EPFL CS-433 Project 1, at least Fall 2023: BRFSS 2015, `_MICHD`, 328,135 × 321). What leaderboard F1 did the top and median teams reach? What do the public team reports say worked? Use them only for benchmarks and ideas, since we cannot reuse their code.
   - Two Kaggle datasets with a similar target:
     - "Heart Disease Health Indicators": BRFSS 2015, 253,680 rows, 21 features, target `HeartDiseaseorAttack` = `_MICHD`.
     - "Personal Key Indicators of Heart Disease": BRFSS 2020, about 320k rows, 17 features, similar target.

     For both: what are the best credible validation AUC and F1? How much did gradient boosting, neural nets or ensembles gain over logistic regression?
   - Peer-reviewed studies that predict CHD, MI or CVD from BRFSS or similar self-report surveys (NHIS, NHANES). Also systematic comparisons of ML vs logistic regression for clinical prediction, e.g. Christodoulou et al. 2019.
   - From all of this: is F1 ≈ 0.44 / AUC ≈ 0.865 close to the ceiling for these features? How large is the realistic gap between a tuned GBDT and a well-encoded logistic regression on this kind of data? Our 321 features are much richer than the Kaggle versions' 17–21, so their absolute numbers are lower bounds for us, but their relative gains should transfer.
2. **Models that could beat our logistic regression, implementable in NumPy.** For each one, give:
   - the evidence for a gain on large (~300k rows), mostly categorical survey data with informative missingness and ~9% prevalence;
   - the expected ΔAUC and ΔF1;
   - an implementation recipe.

   Candidates:
   - **Histogram gradient-boosted trees (LightGBM-style).**
     - Design choices: binning, gradient/hessian histograms with `np.bincount`, histogram subtraction, leaf-wise vs depth-wise growth, categorical codes vs one-hot vs ordinal, NaN handling.
     - Hyper-parameters: learning rate, number of trees and leaves, minimum hessian per leaf, L2 on leaf values, row and column subsampling.
     - The expected runtime in vectorized NumPy on 328k × 291 integer-coded features.
     - Is it worth starting the boosting from the logistic regression's margin (as the initial score), so that fewer trees are needed?
   - GBDT leaf indices as one-hot inputs to logistic regression (He et al. 2014).
   - Factorization machines, or another low-rank pairwise-interaction model, on the one-hot features (Rendle 2010).
   - Better neural nets: an embedding per categorical feature, regularization, ensembling several seeds. When do neural nets win on tabular data at all (Grinsztajn et al. 2022; McElfresh et al. 2023; Kadra et al. 2021)?
   - Ensembles: probability averaging vs rank averaging vs stacking on out-of-fold predictions. Typical gains, and how to choose the threshold of an ensemble.
   - Anything else with credible evidence, with its expected gain. Examples: GA²M / EBM-style additive models with pairwise terms, target or weight-of-evidence encoding, kernel approximations.
3. **Features from the BRFSS 2015 documentation and CHD epidemiology.**
   - Skip patterns. Is any of our features (§8) asked only when `CVDINFR4` or `CVDCRHD4` is yes, like `HAREHAB1`? Does the missingness of any feature depend on those two questions? Check the skip instructions of the 2015 questionnaire and codebook, especially for the cardiovascular-health and aspirin questions (`CVDASPRN`, `ASPUNSAF`, `RLIVPAIN`, `RDUCHART`, `RDUCSTRK`).
   - Established CHD risk factors (Framingham, Pooled Cohort Equations, ...) and their known interactions: age × sex, diabetes, smoking, hypertension, cholesterol, kidney disease, ... Which engineered features or interaction terms have improved BRFSS-based models beyond one-hot main effects?
   - Counts and scores: the number of chronic conditions, of "don't know" answers or of skipped modules, or an approximate risk score. Do they help a model that already has one-hot main effects? For logistic regression, a linear count adds nothing unless it is binned or interacted.
4. **From probabilities to F1.** Theory says that a single threshold on calibrated probabilities is F1-optimal in the large-sample limit (Lipton et al. 2014; Narasimhan et al. 2014; Koyejo et al. 2014). Does anything beat it measurably on a 109k-row test set? Candidates:
   - decisions that maximize expected F1 (Jansche 2007; Nan et al. 2012);
   - choosing the threshold out-of-fold, or by the predicted positive rate;
   - losses that target F1 directly (sigmoid-F1 surrogates, focal loss, cost-sensitive weights).
5. **Validating small gains.** Fold noise is ±0.003 and we get 5 submissions a day. Which protocol reliably detects a gain of +0.002 to +0.005 F1: repeated or paired k-fold, how many repeats, which statistical test? How do we avoid overfitting the public leaderboard?

## 7. What to return

1. **Summary** (at most 10 lines): the realistic F1 ceiling for our setup, and the 3–5 actions with the best expected F1 gain per hour of work under the constraints of §2.
2. **Ranked table** with these columns:
   - approach;
   - expected ΔF1 and ΔAUC over our 0.440 / 0.864, with the evidence behind the number;
   - effort to implement it in NumPy (hours);
   - 5-fold CV runtime on our machine;
   - risk;
   - sources.
3. **Recipes for the top 3:** the algorithm's steps, recommended hyper-parameters with search ranges, tips on NumPy vectorization and memory, pitfalls, and how to validate the gain. Pseudocode is enough; no full code.
4. **Answers to questions 1–5**, with numbers.
5. **Don't-bother list:** popular ideas with evidence that they do not help on this kind of data.

Cite every claim with a URL, or a venue and year. Label each number by origin: BRFSS-based work, general tabular benchmarks, or your own estimate. Prefer peer-reviewed papers, the CDC BRFSS 2015 documentation, and write-ups with reproducible validation numbers over blog posts. Do not propose anything that breaks §2: other libraries, external data, or looking up labels.

## 8. Appendix: the 321 features, by how we use them

**Dropped (30).**
- Survey design: SEQNO, _PSU, FMONTH, IDATE, IMONTH, IDAY, IYEAR, _STSTR, _STRWT, _RAWRAKE, _WT2RAKE, _LLCPWT, _CLLCPWT, _DUALCOR.
- Exact copies: CELLFON2, _CPRACE.
- Raw mixed-unit answers replaced by their computed version: WEIGHT2, HEIGHT3, ALCDAY5, FRUITJU1, FRUIT1, FVBEANS, FVGREEN, FVORANG, VEGETAB1, EXEROFT1, EXEROFT2, EXERHMM1, EXERHMM2, STRENGTH.

**Numeric (56)**, with code rules, median imputation and a NaN indicator: PHYSHLTH, MENTHLTH, POORHLTH, DIABAGE2, CHILDREN, AVEDRNK2, DRNK3GE5, MAXDRNKS, JOINPAIN, FLSHTMY2, HIVTSTD3, BLDSUGAR, FEETCHK2, DOCTDIAB, CHKHEMO3, FEETCHK, LONGWTCH, ASTHMAGE, ASERVIST, ASDRVIST, ASRCHKUP, ASACTLIM, SCNTWRK1, SCNTLWK1, _AGE80, HTIN4, HTM4, WTKG3, _BMI5, DROCDY3_, _DRNKWEK, FTJUDA1_, FRUTDA1_, BEANDAY_, GRENDAY_, ORNGDAY_, VEGEDA1_, _FRUTSUM, _VEGESUM, METVL11_, METVL21_, MAXVO2_, FC60_, PADUR1_, PADUR2_, PAFREQ1_, PAFREQ2_, _MINAC11, _MINAC21, STRFREQ_, PAMIN11_, PAMIN21_, PA1MIN_, PAVIG11_, PAVIG21_, PA1VIGM_.

**Categorical (235)**, one-hot with a NaN level: _STATE, DISPCODE, CTELENUM, PVTRESD1, COLGHOUS, STATERES, CELLFON3, LADULT, NUMADULT, NUMMEN, NUMWOMEN, CTELNUM1, CADULT, PVTRESD2, CCLGHOUS, CSTATE, LANDLINE, HHADULT, GENHLTH, HLTHPLN1, PERSDOC2, MEDCOST, CHECKUP1, BPHIGH4, BPMEDS, BLOODCHO, CHOLCHK, TOLDHI2, CVDSTRK3, ASTHMA3, ASTHNOW, CHCSCNCR, CHCOCNCR, CHCCOPD1, HAVARTH3, ADDEPEV2, CHCKIDNY, DIABETE3, SEX, MARITAL, EDUCA, RENTHOM1, NUMHHOL2, NUMPHON2, CPDEMO1, VETERAN3, EMPLOY1, INCOME2, INTERNET, PREGNANT, QLACTLM2, USEEQUIP, BLIND, DECIDE, DIFFWALK, DIFFDRES, DIFFALON, SMOKE100, SMOKDAY2, STOPSMK2, LASTSMK2, USENOW3, EXERANY2, EXRACT11, EXRACT21, LMTJOIN3, ARTHDIS2, ARTHSOCL, SEATBELT, FLUSHOT6, IMFVPLAC, PNEUVAC3, HIVTST6, WHRTST10, PDIABTST, PREDIAB1, INSULIN, EYEEXAM, DIABEYE, DIABEDU, CAREGIV1, CRGVREL1, CRGVLNG1, CRGVHRS1, CRGVPRB1, CRGVPERS, CRGVHOUS, CRGVMST2, CRGVEXPT, VIDFCLT2, VIREDIF3, VIPRFVS2, VINOCRE2, VIEYEXM2, VIINSUR2, VICTRCT4, VIGLUMA2, VIMACDG2, CIMEMLOS, CDHOUSE, CDASSIST, CDHELP, CDSOCIAL, CDDISCUS, WTCHSALT, DRADVISE, ASATTACK, ASYMPTOM, ASNOSLEP, ASTHMED3, ASINHALR, HAREHAB1, STREHAB1, CVDASPRN, ASPUNSAF, RLIVPAIN, RDUCHART, RDUCSTRK, ARTTODAY, ARTHWGT, ARTHEXER, ARTHEDU, TETANUS, HPVADVC2, HPVADSHT, SHINGLE2, HADMAM, HOWLONG, HADPAP2, LASTPAP2, HPVTEST, HPLSTTST, HADHYST2, PROFEXAM, LENGEXAM, BLDSTOOL, LSTBLDS3, HADSIGM3, HADSGCO1, LASTSIG3, PCPSAAD2, PCPSADI1, PCPSARE1, PSATEST1, PSATIME, PCPSARS1, PCPSADE1, PCDMDECN, SCNTMNY1, SCNTMEL1, SCNTPAID, SCNTLPAD, SXORIENT, TRNSGNDR, RCSGENDR, RCSRLTN2, CASTHDX2, CASTHNO2, EMTSUPRT, LSATISFY, ADPLEASR, ADDOWN, ADSLEEP, ADENERGY, ADEAT1, ADFAIL, ADTHINK, ADMOVE, MISTMNT, ADANXEV, QSTVER, QSTLANG, MSCODE, _CHISPNC, _CRACE1, _DUALUSE, _RFHLTH, _HCVU651, _RFHYPE5, _CHOLCHK, _RFCHOL, _LTASTH1, _CASTHM1, _ASTHMS1, _DRDXAR1, _PRACE1, _MRACE1, _HISPANC, _RACE, _RACEG21, _RACEGR3, _RACE_G1, _AGEG5YR, _AGE65YR, _AGE_G, _BMI5CAT, _RFBMI5, _CHLDCNT, _EDUCAG, _INCOMG, _SMOKER3, _RFSMOK3, DRNKANY5, _RFBING5, _RFDRHV5, _MISFRTN, _MISVEGN, _FRTRESP, _VEGRESP, _FRTLT1, _VEGLT1, _FRT16, _VEG23, _FRUITEX, _VEGETEX, _TOTINDA, ACTIN11_, ACTIN21_, PAMISS1_, _PACAT1, _PAINDX1, _PA150R2, _PA300R2, _PA30021, _PASTRNG, _PAREC1, _PASTAE1, _LMTACT1, _LMTWRK1, _LMTSCL1, _RFSEAT2, _RFSEAT3, _FLSHOT6, _PNEUMO2, _AIDTST3.
