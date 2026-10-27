# EDA findings

Reference for part 2 (cleaning) and part 3 (transforms, models). The notebook `notebooks/01_eda.ipynb` shows the main results with plots; this file has the details, the rule for every feature, and the experiments run during the EDA.

Data: BRFSS 2015 (CDC telephone health survey), as released for the course. Codebook: https://www.cdc.gov/brfss/annual_data/2015/pdf/codebook15_llcp.pdf (not in the repo).

## 1. Facts

- Train: 328,135 rows × 321 features, ids 0–328134. Test: 109,379 rows, ids 328135–437513. Every value is a float; NaN = no answer.
- Label `_MICHD` (heart attack or coronary heart disease): 28,975 positives, 8.83%. In the code labels are {0, 1}; `y_train.csv` and submissions use {-1, 1}.
- Baselines: all-negative accuracy 91.2%, all-positive F1 0.162.
- 45% of all cells are NaN and no row is complete. Per-feature NaN rate: 82 features have none, 57 have ≤10%, 35 have 10–50%, 48 have 50–90%, 69 have 90–99%, and 30 have more than 99%.
- Train and test are a random split: the NaN rate of any feature differs by at most 0.004, state shares by at most 0.0013. The row order is random (AUC of the row index = 0.503). There are no duplicate rows. → Shuffled k-fold CV is a faithful proxy for the leaderboard.

## 2. Label leak

- `HAREHAB1` (cardiac rehab after a heart attack) is asked only to people who reported a heart attack, which is part of MICHD. All 773 train rows that answered it (any code) are positive. **238 test rows answered it: predict 1 for them.**
- Logistic regression on one-hot features learns it on its own (those rows score 0.99–1.0). Removing `HAREHAB1` costs about 0.007 F1. A rule "answered `HAREHAB1` → 1" on top of a model without the feature recovers most of it (0.4332 → 0.4386).
- Never let a "drop features with many NaNs" filter remove it (99.8% NaN).
- `STREHAB1` is the same question for strokes, which are not part of MICHD (answered only by stroke patients, 36% positive). It is not a leak.

## 3. Code conventions

| Code | Meaning | Where |
|---|---|---|
| 7, 77, 777, 7777, 777777 | don't know / not sure | width matches the question's field |
| 9, 99, 999, 9999, 999999 | refused | same |
| 8, 88, 888 | none / never / zero (when the question counts something) | e.g. `PHYSHLTH` 88 = 0 days, `ALCDAY5` 888 = no drinks, `CHECKUP1` 8 = never |
| 555 | never | fruit/vegetable questions, `LONGWTCH`, `FEETCHK2` |
| 300 | less than once a month | fruit/vegetable questions |
| 1xx / 2xx / 3xx / 4xx | per day / week / month / year | `ALCDAY5` (1xx per week, 2xx per month), fruit and vegetables, `EXEROFT*`, `STRENGTH`, `BLDSUGAR`, `FEETCHK2`, `LONGWTCH` |
| 9 or 14 or 3 | don't know / refused / missing | computed `_` variables: 9 for most, `_AGEG5YR` 14, `_AGE65YR` 3 |

The value 7, 8 or 9 is a valid answer in some questions (`EMPLOY1` 7 = retired and 8 = unable to work, `_AGEG5YR` 7–9 = age groups, `INCOME2` 7–8, `DIABAGE2` 77 = age 77). One-hot encoding avoids having to decide.

"Don't know" carries signal, often close to "yes":

| Question | yes | no | don't know | refused |
|---|---|---|---|---|
| `CVDSTRK3` stroke | 37.1% | 7.6% | 31.7% | 25% (n=20) |
| `BPHIGH4` high blood pressure | 16.3% | 3.7% | 15.3% | 4.3% |
| `CHCCOPD1` COPD | 26.7% | 7.2% | 23.8% | 13.3% |
| `CHCKIDNY` kidney disease | 28.8% | 8.1% | 23.0% | 21.7% |
| `TOLDHI2` high cholesterol | 16.1% | 5.2% | 14.0% | 7.6% |

→ Mapping "don't know" to NaN and then to the median (usually "no") loses this. Keep the codes as categories (one-hot).

## 4. Missing values are mostly skip logic

A NaN usually means the question did not apply. Positive rate when NaN vs when answered:

| Feature | NaN | rate if NaN | rate if answered | NaN means (agreement checked on the data) |
|---|---|---|---|---|
| `BPMEDS` | 60% | 3.8% | 16.3% | `BPHIGH4` ≠ 1, no high blood pressure (100%) |
| `TOLDHI2`, `CHOLCHK`, `_RFCHOL` | 13% | 2.5% | 9.8% | `BLOODCHO` ≠ 1, cholesterol never checked (100%) |
| `POORHLTH` | 49% | 6.2% | 11.3% | `PHYSHLTH` = `MENTHLTH` = 88 (100%) → the value is 0 |
| `DIABAGE2` | 87% | 6.9% | 22.1% | `DIABETE3` ≠ 1, not diabetic (100%) |
| `_FLSHOT6`, `_PNEUMO2` | 64% | 4.6% | 16.4% | age < 65 (98.8%) |
| `AVEDRNK2`, `DRNK3GE5`, `MAXDRNKS` | 52% | 11.1% | 6.4% | no drink in 30 days, or unknown (99.9%) → the value is 0 |
| `SMOKDAY2` | 58% | 6.3% | 12.4% | `SMOKE100` ≠ 1, never smoked 100 cigarettes (99.98%) |
| landline block (5 features, see §5) | 42.5% | 6.2% | 10.7% | interviewed by cell phone |
| cell block (`CTELNUM1`, `PVTRESD2`, ...) | 57.5% | 10.7% | 6.2% | interviewed by landline |
| `HAREHAB1` | 99.8% | 8.6% | 100% | no heart attack (leak) |

About 100 questions belong to optional modules asked in some states only (diabetes, caregiving, vision, cognitive decline, asthma call-back, cancer screening, sexual orientation, ...): there a NaN means "state did not ask".

→ Keep NaN as a category or add a NaN indicator. Fill numeric skips with the value they imply (0 days, 0 drinks). Dropping sparse features hurts (§8).

## 5. Rule for every feature

Groups cover the 321 features exactly once (checked by script). "One-hot" means: keep the raw codes in part 2, and one-hot encode them in part 3 (NaN as its own column).

**Drop: survey design (14).** `SEQNO`, `_PSU` (record id, identical columns), `FMONTH`, `IDATE`, `IMONTH`, `IDAY`, `IYEAR` (interview date), `_STSTR` (state × stratum), `_STRWT`, `_RAWRAKE`, `_WT2RAKE`, `_LLCPWT`, `_CLLCPWT`, `_DUALCOR` (sampling weights). Keeping them changed F1 by -0.0008.

**Drop: exact copies (2).** `CELLFON2` (= `CTELNUM1`), `_CPRACE` (= `_CRACE1`).

**Keep, one-hot: survey context (6).** `_STATE` (53 states; one-hot added +0.0009 F1, within noise), `DISPCODE` (1100 complete, 1200 partial interview), `QSTVER` (10–13 landline, 20–23 cell; rates 10.7% vs 6%), `QSTLANG`, `MSCODE` (metro status, NaN for cell), `_DUALUSE`.

**Keep, one-hot: phone blocks (19).** Landline block, NaN for cell interviews: `CTELENUM`, `PVTRESD1`, `CELLFON3`, `NUMHHOL2`, `CPDEMO1` (identical NaN pattern), `STATERES`, `NUMADULT`, `NUMMEN`, `NUMWOMEN` (counts, no codes), `NUMPHON2` (97% NaN), `COLGHOUS`, `LADULT` (32 rows answered). Cell block, NaN for landline interviews: `CTELNUM1`, `PVTRESD2`, `CADULT`, `CSTATE`, `LANDLINE`, `HHADULT` (1–76; 77, 99 → NaN), `CCLGHOUS`. They mostly encode landline vs cell (and household size).

**Numeric with codes (18).** Fixed rules, then use as numbers (or bin):

| Feature | Valid | → 0 | → NaN | Skip-logic NaN |
|---|---|---|---|---|
| `PHYSHLTH`, `MENTHLTH`, `POORHLTH` | 1–30 days | 88 | 77, 99 | `POORHLTH` NaN → 0 |
| `CHILDREN` | 1–87 | 88 | 99 | |
| `AVEDRNK2`, `MAXDRNKS` | 1–76 | | 77, 99 | NaN (no drinking) → 0 |
| `DRNK3GE5` | 1–76 | 88 | 77, 99 | NaN (no drinking) → 0 |
| `JOINPAIN` | 0–10 | | 77, 99 | NaN = no arthritis or not asked |
| `DIABAGE2` | 1–97 years | | 98, 99 | NaN = not diabetic |
| `SCNTWRK1`, `SCNTLWK1` | 1–96 hours | 98 = none | 77, 99 (97 = 97+?) | 92% NaN |
| `DOCTDIAB`, `CHKHEMO3`, `FEETCHK` | 1–76 | 88 | 77, 99, `CHKHEMO3` 98 | diabetes module, 93% NaN |
| `ASTHMAGE` | 11–96 years | | 98, 99 (97 = ≤ 10) | 99.8% NaN, no signal |
| `ASERVIST`, `ASDRVIST`, `ASRCHKUP` | 1–87 | 88 | 98, 99 | 99.9% NaN, no signal |

**Mixed units, replaced by a computed variable (14): drop the raw one.** `WEIGHT2` → `WTKG3` (kg), `HEIGHT3` → `HTM4` (m) / `HTIN4` (in), `ALCDAY5` → `DROCDY3_` and `DRNKANY5`, `FRUITJU1` → `FTJUDA1_`, `FRUIT1` → `FRUTDA1_`, `FVBEANS` → `BEANDAY_`, `FVGREEN` → `GRENDAY_`, `FVORANG` → `ORNGDAY_`, `VEGETAB1` → `VEGEDA1_` (times per day), `EXEROFT1`/`EXEROFT2` → `PAFREQ1_`/`PAFREQ2_` (per week), `EXERHMM1`/`EXERHMM2` → `PADUR1_`/`PADUR2_` (minutes), `STRENGTH` → `STRFREQ_` (per week).

**Mixed units without a computed version (4).** `BLDSUGAR`, `FEETCHK2` (diabetes module: 1xx–4xx per day/week/month/year, 555 no feet, 888 never, 777/999), `LONGWTCH` (1xx–4xx days/weeks/months/years, 555 never, 777/999), `ASACTLIM` (1–365 days, 888 none, 777/999). Convert to one unit, or quantile-bin; all ≥ 93% NaN, weak signal.

**Dates MMYYYY (2).** `FLSHTMY2` (last flu shot), `HIVTSTD3` (last HIV test; 77YYYY = month unknown); 777777 and 999999 → NaN. Convert to months before 01/2016, or quantile-bin. Their signal is mostly in the NaN pattern.

**Computed numeric (32).** Already clean except for these codes: `DROCDY3_` 900 → NaN, `_DRNKWEK` 99900 → NaN, `MAXVO2_` and `FC60_` 999 → NaN. List: `_AGE80` (18–80), `HTIN4`, `HTM4`, `WTKG3`, `_BMI5` (BMI, already divided by 100), `DROCDY3_`, `_DRNKWEK`, `FTJUDA1_`, `FRUTDA1_`, `BEANDAY_`, `GRENDAY_`, `ORNGDAY_`, `VEGEDA1_`, `_FRUTSUM`, `_VEGESUM`, `METVL11_`, `METVL21_`, `MAXVO2_`, `FC60_`, `PADUR1_`, `PADUR2_`, `PAFREQ1_`, `PAFREQ2_`, `_MINAC11`, `_MINAC21`, `STRFREQ_`, `PAMIN11_`, `PAMIN21_`, `PA1MIN_`, `PAVIG11_`, `PAVIG21_`, `PA1VIGM_`. The activity minutes and fruit/vegetable frequencies are heavy-tailed (99th percentile far above the median; max 54,000 minutes): use `log1p` or clipping, or bin. `MAXVO2_` and `FC60_` are functions of age and sex.

**Nominal with many codes (2).** `EXRACT11`, `EXRACT21` (type of main activities, codes 1–76; 77, 99 DK/refused; 88 = no second activity; 98 = other): one-hot with a minimum count, needs `max_levels` ≥ 80.

**Categorical answers, one-hot (148).** All other raw questions. They use 1–5 or 1–2 answers with 7 = DK and 9 = refused (sometimes 8 = never/none), or 2-digit fields (`INCOME2` 1–8, `LASTSMK2` 1–8, `IMFVPLAC`, `WHRTST10`, `CRGVREL1`, `CRGVPRB1`, `VINOCRE2`, `HPVADSHT`, `PCDMDECN`; 77/99 DK/refused). The `AD*` depression items (`ADPLEASR` ... `ADMOVE`) are 1–14 days with 88 = none. Includes `HAREHAB1`.

**Computed categorical, one-hot (60).** All `_` recodes not listed above (`_RFHLTH`, `_HCVU651`, `_RFHYPE5`, `_CHOLCHK`, `_RFCHOL`, asthma, race, `_AGEG5YR`, `_AGE65YR`, `_AGE_G`, `_BMI5CAT`, `_RFBMI5`, `_CHLDCNT`, `_EDUCAG`, `_INCOMG`, `_SMOKER3`, `_RFSMOK3`, alcohol flags, fruit/vegetable flags, physical-activity categories, `_LMT*`, `_RFSEAT*`, `_FLSHOT6`, `_PNEUMO2`, `_AIDTST3`, ...). Note that `_HCVU651` = 9 includes everyone aged 65+, so it is an age signal.

### Redundant groups

46 pairs have |r| > 0.95 (pairwise-complete rows). Main groups, for choosing one representation per concept if needed (one-hot of all of them works with regularization):

- age: `_AGEG5YR`, `_AGE80`, `_AGE_G`, `_AGE65YR`, and `MAXVO2_`, `FC60_` (from age and sex)
- body: `HEIGHT3`/`HTIN4`/`HTM4`, `WEIGHT2`/`WTKG3`, `_BMI5`/`_BMI5CAT`/`_RFBMI5`
- smoking: `SMOKE100`, `SMOKDAY2`, `_SMOKER3`, `_RFSMOK3`
- alcohol: `ALCDAY5`, `DRNKANY5`, `DROCDY3_`, `AVEDRNK2`, `_DRNKWEK`, `_RFDRHV5`, `DRNK3GE5`, `_RFBING5`, `MAXDRNKS`
- blood pressure: `BPHIGH4`/`_RFHYPE5`; cholesterol: `TOLDHI2`/`_RFCHOL`, `BLOODCHO`/`CHOLCHK`/`_CHOLCHK`
- arthritis: `HAVARTH3` = `_DRDXAR1`, `LMTJOIN3`/`_LMTACT1`, `ARTHDIS2`/`_LMTWRK1`, `ARTHSOCL`/`_LMTSCL1`
- asthma: `ASTHMA3`/`_LTASTH1`, `ASTHNOW`/`_CASTHM1`/`_ASTHMS1`
- race: `_PRACE1`/`_MRACE1`, `_RACE`/`_RACEG21`/`_RACEGR3`/`_RACE_G1`, `_HISPANC`
- education, income, children: `EDUCA`/`_EDUCAG`, `INCOME2`/`_INCOMG`, `CHILDREN`/`_CHLDCNT`
- vaccines, HIV, seat belt, health: `FLUSHOT6`/`_FLSHOT6`, `PNEUVAC3`/`_PNEUMO2`, `HIVTST6`/`_AIDTST3`, `SEATBELT`/`_RFSEAT2`/`_RFSEAT3`, `GENHLTH`/`_RFHLTH`
- fruit, vegetables, physical activity: the raw answers, their computed versions and about 25 computed flags and categories

## 6. Signal

Univariate AUC, folded so that 0.5 = no signal (`eda.univariate_auc`: each value or quantile bin is a category, scored by its positive rate on the other half of the rows):

- Top: `_AGE80` 0.736, `_AGEG5YR` 0.735, `FC60_`/`MAXVO2_` 0.735, `GENHLTH` 0.729, `_AGE_G` 0.716, `EMPLOY1` 0.715, `BPMEDS` 0.701, `BPHIGH4` 0.688, `_RFHYPE5` 0.687, `_PNEUMO2` 0.679, `TOLDHI2` 0.676, `_FLSHOT6` 0.673, `_AGE65YR` 0.670, `_HCVU651` 0.668, `_RFHLTH` 0.661, `PNEUVAC3` 0.661, `_LMT*` ~0.65, `PHYSHLTH` 0.647, `DIFFWALK` 0.643, `QLACTLM2` 0.642, `HAVARTH3` 0.640.
- Then: `POORHLTH`, `MARITAL` (widowed 17%), `CHOLCHK`, `DIABETE3` (22% for diabetes), `INCOME2`, `USEEQUIP`, `INTERNET` (no internet 17%), `_SMOKER3` (former smokers 13.5%), `SMOKE100`, `CHILDREN`, `CHCCOPD1`, alcohol (non-drinkers 11.4%), `EDUCA`, `CVDSTRK3`, `VETERAN3`, `SEX` (men 11.2%, women 7.1%), `_BMI5`.
- Raw numbers lose the signal for skip-logic NaNs, codes and nominal answers (encoded → raw): `BPMEDS` 0.70 → 0.50, `_PNEUMO2` 0.68 → 0.53, `MARITAL` 0.62 → 0.50, `_FLSHOT6` 0.67 → 0.56, `DIABAGE2` 0.61 → 0.51, `JOINPAIN` 0.64 → 0.55, `PHYSHLTH` 0.65 → 0.58.
- 57 features have AUC < 0.505: mostly optional-module questions (vision, asthma call-back, caregiving, PSA/HPV screening, ...) plus `_FRT16`, `_VEG23`, `_FRUITEX`, `_FRTRESP`, `COLGHOUS`, `LADULT`.
- Shape: risk goes from 0.5% (18–24) to 22% (80+) and accelerates after 40. Men and women are equal below 40, then men have 1.4–2.1 times the risk (an age × sex interaction). `GENHLTH`: 2.1%, 4.2%, 9.1%, 19.3%, 31.9% from excellent to poor.

## 7. Experiments run during the EDA

Scratch scripts, not kept in the repo; to be redone with `p1 cv` in part 3 for the report. Setup unless noted: a fixed 80/20 split (`np.random.default_rng(42).permutation`), logistic regression fitted by Newton's method with an L2 penalty λ on the mean loss (λ = 1e-5 or 1e-4, which makes no difference), features standardized on the train part, and F1 at the best threshold on the 20% part ("best F1"). Encoding "E": the 14 design columns dropped; features with ≤ 30 distinct values one-hot (levels with ≥ 30 rows, plus a NaN column); the numeric features of §5 cleaned with their code rules, `log1p` for the heavy-tailed ones, NaN → median plus a NaN indicator; the raw mixed-unit features skipped. About 1,340 columns.

| Encoding | AUC | best F1 | F1 at 0.5 |
|---|---|---|---|
| A: raw values, NaN → median (321 columns) | 0.8528 | 0.4172 | 0.196 |
| B: A without the 14 design columns | 0.8527 | 0.4171 | |
| C: codes cleaned, NaN → median | 0.8618 | 0.4332 | 0.246 |
| D: C + NaN indicators | 0.8646 | 0.4387 | 0.269 |
| **E: one-hot categorical + cleaned numerics + NaN indicators** | **0.8650** | **0.4400** | 0.275 |
| E with the design columns | 0.8649 | 0.4392 | |
| E + `_STATE` one-hot | 0.8656 | 0.4409 | |
| G1: everything one-hot (≤ 60 values) or 10 quantile bins, no code rules (about 2,200 columns) | 0.8635 | 0.4387 | 0.283 |
| G1 + cleaned numerics | 0.8632 | 0.4382 | |
| G1 with 20 bins | 0.8630 | 0.4376 | |

Other results on encoding E:

- **Noise**: 5-fold CV gives AUC 0.8641 ± 0.0022, F1 0.4398 ± 0.0044, best threshold 0.202 ± 0.005.
- **Threshold**: the best threshold is about 0.20 and flags about 13% of rows. At 0.5, F1 is only 0.20–0.28.
- **Class weights**: balanced weights give F1 0.4380 with the threshold moved to 0.74. They only shift the threshold.
- **Undersampling negatives**: 1:1 → 0.4345 (threshold 0.75), 3:1 → 0.4380 (threshold 0.49). Use it only for speed.
- **λ**: 1e-6 → 0.4399, 1e-4 → 0.4402, 1e-3 → 0.4403. Negligible with 262k rows.
- **Dropping sparse features**: NaN > 0.99 (30 features) → 0.4324, > 0.95 (68) → 0.4299, > 0.9 (99) → 0.4302, > 0.7 (121) → 0.4304. Most of the loss is `HAREHAB1`.
- **Keeping the top-k features by univariate AUC**: 30 → 0.3995, 60 → 0.4160, 100 → 0.4294, 150 → 0.4382. Univariate ranking is a bad selector.
- **Interactions**: the 190 pairwise products of the top-20 features (cleaned, standardized) → 0.4420 (+0.0018, AUC +0.0005).
- **MLP** (numpy, 1 hidden ReLU layer, Adam, early stopping on validation): 64 units → 0.4341 (AUC 0.8622); 256 units with dropout 0.3 → 0.4402 (AUC 0.8644); it overfits after 1–2 epochs. A rank average with the logistic regression → 0.4417. Little headroom beyond a linear model.
- **Learning curve** (share of the 262k training rows): 5% → 0.371, 10% → 0.405, 25% → 0.427, 50% → 0.438, 100% → 0.440. Compare encodings with `subsample` ≥ 0.5.
- **Least squares**: the standardized one-hot design (with the bias column) has 293 zero eigenvalues out of 1,342, so `np.linalg.solve` on XᵀX raises `LinAlgError: Singular matrix`. Use ridge (λ > 0) or drop one level per one-hot group. The raw design is singular too, because of the exact copies.
- **Plain GD for logistic regression** (as in `implementations.logistic_regression`), on standardized E: λ_max(XᵀX/N) = 56.4, so the step size γ = 4/λ_max ≈ 0.07. 50 iterations → 0.4312, 100 → 0.4357, 300 → 0.4400, 1000 → 0.4424, 2000 → 0.4403, at about 0.15 s per iteration on 262k rows. A Newton fit takes about 35 s.
- **Ridge regression probe** (closed form, scores used as probabilities): raw 0.8452 / 0.4154, E 0.8581 / 0.4366, G1 0.8564 / 0.4346 (AUC / best F1). The same ranking of encodings as logistic regression, 0.002–0.004 F1 lower.

## 8. Recommendations

Part 2 (`p1/cleaning.py`, fixed rules applied to train and test alike):
1. Drop the 14 survey-design columns and the 2 exact copies.
2. Apply the code rules of §5 to the numeric features, including the skip-logic fills (`POORHLTH`, drinking). Drop the 14 raw mixed-unit features that have a computed version. Convert or leave the 4 without one, and convert or leave the 2 date features, for part-3 binning.
3. Keep everything else unchanged, codes included (part 3 one-hot encodes them). Keep `HAREHAB1`.

Part 3.1 (transforms, refitted on each training fold):
- `onehot` with levels learned on the training fold (`max_levels` around 30, `min_count` around 30, NaN column), for the categorical features.
- NaN indicators plus median imputation for numeric features, `log1p` for the heavy-tailed ones (or quantile-bin them into the one-hot).
- `standardize` last.
- The generic encoding G1 (one-hot or bin everything, no code rules) is a strong fallback, only 0.0013 F1 below E.

Part 3.2 (models):
- Logistic regression (L2) is the core model. A Newton/IRLS solver is exact in about 10 iterations with no step size to tune. Plain GD needs γ ≈ 4/λ_max and about 300+ iterations.
- The decision threshold must be tuned. Choose it inside CV (for example the F1-optimal threshold on the training folds' out-of-fold scores, or the average best threshold across folds); it is about 0.2 for unweighted logistic regression.
- Least squares and ridge: λ > 0 or reference levels (singular design). Their best threshold on the scores is about 0.21–0.23.
- Expected: 5-fold F1 ≈ 0.44, AUC ≈ 0.865. Only small extra gains are likely (interactions +0.002, `_STATE` +0.001, an ensemble with an MLP +0.0015); each needs a paired CV comparison, given the ±0.004 fold noise.
