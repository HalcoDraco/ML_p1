# Maximizing F1 for Heart-Disease Prediction on BRFSS 2015

## Summary

A positive-class F1 score of approximately 0.44–0.45 is typical for this dataset. For example, on the BRFSS 2015 Heart Disease Health Indicators dataset, LightGBM achieved an AUC of approximately 0.837 and an F1 score of approximately 0.42. Your logistic regression already achieves F1 = 0.440.

A 2026 study by Salehi et al. reports that an optimized model can achieve F1 = 0.556, suggesting there may be room for improvement, although its evaluation setup and metric comparability need to be verified. A practical target for your project is around F1 = 0.45–0.46, rather than assuming that 0.50 is readily achievable.

The five actions with the best expected return on implementation time are:

1. Histogram gradient boosting (LightGBM-style): estimated ΔF1 ≈ +0.005 to +0.010; ΔAUC ≈ +0.005 to +0.010. Approximately 20 hours to implement; five-fold CV could take about one hour. Medium-to-high risk.
2. Factorization Machines (pairwise interactions): estimated ΔF1 ≈ +0.002 to +0.005. Approximately 8 hours to implement; five-fold CV could take about one hour. Medium risk.
3. Neural networks with categorical embeddings: estimated ΔF1 ≈ +0.000 to +0.005. Approximately 8 hours to implement; five-fold CV could take about 30 minutes. Medium risk.
4. Ensembling logistic regression and an MLP: estimated improvement around +0.002 F1. Approximately 1–2 hours to implement; five-fold CV takes about five minutes. Low risk.
5. Boosting followed by one-hot leaf encoding and logistic regression: estimated ΔF1 ≈ +0.002 to +0.005. Approximately 10–15 hours to implement, with potentially more than one hour for five-fold CV. High risk.

These priorities balance possible improvement against implementation cost. Popular approaches such as oversampling, arbitrary binning, and simple ordinal encodings rarely help much in this setting.

## 1. Ranked approaches

| Method                                                             | Expected ΔF1 (ΔAUC)                                                | Implementation effort | Five-fold CV runtime | Risk   | Sources                                                                                                                                                                                                                                              |
| ------------------------------------------------------------------ | ------------------------------------------------------------------ | --------------------- | -------------------- | ------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Histogram gradient boosting, LightGBM-style                        | +0.005 to +0.010 (+0.005 to +0.010)                                | 15–25 h               | \~1 h                | High   | [Heart Disease Health Indicators](https://www.emergentmind.com/topics/heart-disease-health-indicators-dataset?utm_source=chatgpt.com); [Salehi et al., 2026](https://journals.sagepub.com/doi/full/10.1177/14604582261444612?utm_source=chatgpt.com) |
| Factorization Machines, rank-2 interactions                        | +0.002 to +0.005 (+0.002 to +0.005)                                | 5–10 h                | \~1 h                | Medium | [AWS Factorization Machines](https://docs.aws.amazon.com/sagemaker/latest/dg/fact-machines.html?utm_source=chatgpt.com)                                                                                                                              |
| Neural network with categorical embeddings                         | +0.000 to +0.005 (AUC approximately unchanged, potentially +0.002) | 5–10 h                | \~30 min             | Medium | Grinsztajn et al., 2022                                                                                                                                                                                                                              |
| Ensemble of logistic regression and MLP                            | +0.001 to +0.003 (AUC approximately unchanged)                     | 1–2 h                 | \~5 min              | Low    | [Heart Disease Health Indicators](https://www.emergentmind.com/topics/heart-disease-health-indicators-dataset?utm_source=chatgpt.com); your existing experiments                                                                                     |
| Boosting followed by one-hot leaf encoding and logistic regression | +0.002 to +0.005 (+0.002)                                          | 10–15 h               | More than 1 h        | High   | He et al., 2014                                                                                                                                                                                                                                      |

Evidence and interpretation. Gradient-boosted trees are often strong on heterogeneous tabular datasets, which makes histogram boosting the most promising new model family. Factorization Machines explicitly capture pairwise interactions in sparse, high-dimensional inputs. Neural embeddings may learn nonlinear relationships between categorical variables, but their gains on tabular problems are often modest. Ensembling is cheap and has already produced a small gain in your experiments.

The numerical improvements in this table are estimates, not guaranteed results. The literature supports the general ranking of methods more strongly than it supports any precise F1 gain on your exact task.

## 2. Recipes for the three main methods

### Method 1: Histogram gradient boosting

Objective: Learn nonlinear effects and interactions that logistic regression cannot represent using main effects alone.

Algorithm

1. Bin the features. Convert each feature to integer bin IDs. Use approximately 64–255 bins for numerical features. For categorical variables, preserve categories rather than interpreting category IDs as continuous numerical values.
2. Initialize the predictions. Start with the log-odds of the training-set prevalence:\
   \\[\
   F_0=\log\left(\frac{\bar y}{1-\bar y}\right).\
   \\]
3. Compute gradients and Hessians. For logistic loss, with \\(p_i=\sigma(F_i)\\):\
   \\[\
   g_i=p_i-y_i,\qquad h_i=p_i(1-p_i).\
   \\]
4. Build trees. For each candidate split, accumulate gradient and Hessian histograms with `np.bincount`. Evaluate the gain from each split and select the best one.
5. Grow leaves. Prefer leaf-wise growth if you can correctly maintain a priority queue of candidate splits. Limit the number of leaves to control overfitting.
6. Update the predictions. Apply a learning-rate-scaled leaf value to the observations that reach each leaf.
7. Repeat. Train the selected number of trees and calculate probabilities with the sigmoid function.
8. Tune the F1 threshold. Select the threshold using training-only or out-of-fold predictions, never the held-out fold's labels.

Recommended hyperparameters

| Parameter              | Initial value | Search range |
| ---------------------- | ------------- | ------------ |
| Learning rate          | 0.05          | 0.02–0.10    |
| Number of trees        | 200           | 100–500      |
| Maximum leaves         | 31            | 16–50        |
| Minimum leaf size      | 500           | 200–2,000    |
| L2 leaf regularization | 1.0           | 0.1–10       |
| Row subsampling        | 0.8           | 0.7–1.0      |
| Feature subsampling    | 0.8           | 0.7–1.0      |

NumPy and memory considerations

Store the binned matrix in a compact integer type, such as `uint8` when the number of bins permits it. For each node and feature, use histogram accumulation rather than sorting all observations repeatedly. Histogram subtraction can accelerate child-node histogram construction when the parent histogram is already known.

For categorical variables, a naive binary split over category IDs is inappropriate: the integer ordering of category codes has no semantic meaning. A practical first implementation should either use categorical-versus-rest splits or treat carefully selected low-cardinality variables separately.

Important risks

A correct gradient-boosting implementation is substantially more work than it first appears. Split search, leaf assignment, missing-value routing, regularization, and stopping rules all affect performance. The runtime estimate of about one hour for five-fold CV is only plausible if the implementation is efficient and uses a constrained tree configuration; it is not a benchmark that has been established on your machine.

Starting from a logistic-regression margin is an interesting experiment, but it is not automatically beneficial. Standard boosting initialized with the prevalence log-odds learns residual structure from scratch. Using a fitted logistic margin instead creates a residual-correction model and requires a consistent procedure across folds.

Validation

Begin with one fixed split and a small configuration. If the model beats logistic regression by at least 0.003 F1 without an unexplained AUC regression, run the full paired five-fold comparison. Do not spend the remaining deadline implementing a large hyperparameter search.

### Method 2: Factorization Machines

Objective: Add low-rank pairwise interactions to the existing one-hot logistic model.

A second-order Factorization Machine models the logit as

\\[ \hat y=\sigma\left( w_0+\sum_i w_i x_i+ \sum\_{i\<j}\langle v_i,v_j\rangle x_i x_j \right), \\]

where \\(x\\) is the approximately 1,526-dimensional feature vector and each feature has a low-dimensional latent factor \\(v_i\\).

The pairwise term can be computed efficiently without enumerating all feature pairs:

\\[ \frac12\sum\_{f=1}^{K} \left[ \left(\sum_i v\_{if}x_i\right)^2 -\sum_i v\_{if}^2x_i^2 \right]. \\]

Algorithm

1. Reuse the existing one-hot encoding and missingness indicators.
2. Initialize the linear weights and a latent-factor matrix \\(V\in\mathbb R^{d\times K}\\) with small random values.
3. Compute the linear logit and pairwise interaction term using the identity above.
4. Minimize binary cross-entropy using minibatch gradient descent or Adam.
5. Regularize both the linear weights and the latent factors.
6. Choose the F1 threshold from training-only or out-of-fold predictions.

Suggested search

- Latent rank \\(K\\): 3, 5, 10.
- Learning rate: \\(10^{-3}\\) to \\(10^{-2}\\).
- Regularization: 0.001 to 0.1.
- Batch size: 1,024–8,192.

The implementation must exploit sparsity or use minibatches. Materializing a full dense \\(328{,}135\times1{,}526\\) matrix in float64 consumes several gigabytes, while repeatedly allocating large intermediate arrays can increase memory use considerably.

Factorization Machines are designed for sparse, high-dimensional data and may capture interactions such as diabetes × hypertension or age × sex. However, the expected ΔF1 of +0.002 to +0.005 is an estimate based on their modelling capacity, not a verified BRFSS benchmark.

### Method 3: Neural networks with categorical embeddings

Objective: Replace large one-hot vectors with learned low-dimensional representations of categorical features.

Algorithm

1. Map each categorical feature to integer IDs, keeping distinct levels for informative responses such as “don't know,” “refused,” and missing values.
2. Create a small embedding table per categorical feature.
3. Concatenate the embeddings with the cleaned numerical features and missingness indicators.
4. Use one or two fully connected hidden layers with ReLU activation.
5. Train with binary cross-entropy and Adam.
6. Apply regularization through weight decay, dropout, or early stopping.
7. Tune the F1 threshold on training-only or out-of-fold predictions.

Suggested hyperparameters

| Parameter           | Search range                |
| ------------------- | --------------------------- |
| Embedding dimension | 2–8                         |
| Hidden units        | 64–256                      |
| Hidden layers       | 1–2                         |
| Dropout             | 0.1–0.5                     |
| Learning rate       | \\(10^{-4}\\)–\\(10^{-3}\\) |
| Weight decay        | \\(10^{-5}\\)–\\(10^{-3}\\) |

Embeddings can reduce input dimensionality, but they do not guarantee better performance. Your existing MLP already overfits quickly, and its rank-averaged ensemble with logistic regression improved F1 by only 0.0017. Consequently, this method is lower priority than boosting or a lightweight ensemble.

Ablate the embedding model against the existing MLP and logistic regression separately. Use early stopping and compare out-of-fold predictions, rather than choosing the model based solely on training loss.

## 3. Detailed answers to the research questions

### 1. Benchmarks and the practical performance ceiling

BRFSS-related results. Public evaluations on the reduced-feature Heart Disease Health Indicators dataset report an AUC around 0.8371 for LightGBM and an F1 score around 0.42 in one report. Some ensembles report F1 around 0.376 with recall as high as 80%. These numbers vary with thresholds, validation setup, feature preprocessing, and class imbalance. See the [Heart Disease Health Indicators overview](https://www.emergentmind.com/topics/heart-disease-health-indicators-dataset?utm_source=chatgpt.com).

A 2026 study by Salehi and colleagues reports that logistic regression and gradient boosting achieved an AUC of approximately 0.8462 on their evaluation, while a proposed FCA-constrained logistic model achieved higher precision and F1. The paper's findings suggest that alternative feature representations and decision rules can matter even when AUC is unchanged. See [Salehi et al., 2026](https://journals.sagepub.com/doi/full/10.1177/14604582261444612?utm_source=chatgpt.com).

These benchmarks are not directly comparable with your model: your feature set is much richer than the 17–21-feature Kaggle variants, your training sample is larger than the 253,680-row BRFSS 2015 Kaggle release, and your F1 threshold is explicitly optimized rather than fixed at 0.5.

Clinical prediction literature. Christodoulou et al. (2019), in Journal of Clinical Epidemiology, concluded that machine-learning methods did not show a consistent predictive-performance advantage over logistic regression in clinical prediction models. This does not prove that boosting will fail here, but it cautions against assuming that a more complex model must perform substantially better.

Interpretation for your project.

Your F1 = 0.440 and AUC = 0.864 are already strong results for this class-imbalanced self-report problem. AUC is particularly competitive relative to the cited reduced-feature benchmarks. However, the available evidence does not establish a theoretical F1 ceiling of 0.45 or 0.50. An F1 around 0.45–0.46 is a reasonable short-term target, while gains substantially beyond that would need to come from better features, interactions, or a genuinely better ranking of positive cases.

### 2. Other implementable models

Histogram gradient boosting: The strongest new modelling candidate. Tree models learn thresholds, nonlinear effects, and interactions automatically. The estimated gain is ΔF1 ≈ +0.005 to +0.010 and ΔAUC ≈ +0.005 to +0.010, but these ranges are speculative until tested on your folds.

Factorization Machines: Suitable for sparse one-hot features and capable of learning low-rank pairwise interactions. Estimated ΔF1 ≈ +0.002 to +0.005 and ΔAUC ≈ +0.002 to +0.005. Their main advantage over your existing model is explicit interaction modelling.

Categorical embeddings: Potentially useful for representing categories compactly and learning nonlinear relationships. Estimated ΔF1 ≈ 0.000 to +0.005 and ΔAUC ≈ 0 to +0.002. The evidence for a meaningful gain on your particular dataset is weak, especially given your existing MLP results.

Ensembling: Averaging predictions from models with different errors may increase F1 even if neither model has a substantially higher AUC. You have already observed a +0.0017 F1 gain by rank-averaging the MLP and logistic regression. This remains the best low-effort experiment.

Boosting followed by leaf encoding: The method popularized by He et al. (2014) maps tree leaf memberships into sparse binary features and then fits a linear model. It can combine tree-learned interactions with linear-model regularization. The expected benefit is small, and the approach inherits the implementation burden of the tree model.

Other ideas. GA²M/EBM-style pairwise additive models, weight-of-evidence encoding, target encoding, and kernel approximations are possible alternatives. However, there is insufficient evidence to expect a sufficiently large gain from implementing them under your time constraints. Target encoding in particular must be constructed without leakage, ideally through out-of-fold statistics.

### 3. Questionnaire skip logic and epidemiological features

Potential leakage. You have already identified `HAREHAB1` as a strong label leak: it records outpatient cardiac rehabilitation after a heart attack, and every training respondent who answered it is positive. It is not appropriate as a predictor for a model intended to generalize without relying on this post-diagnosis information. The presence of a corresponding question in the test data does not make it legitimate evidence of pre-diagnosis risk.

The questionnaire's skip logic is important because missing values can encode which questions respondents were eligible to answer. Variables asked after a diagnosis may leak information through their response values or through whether they were answered.

The research report identified `CVDASPRN`, `ASPUNSAF`, `RLIVPAIN`, `RDUCHART`, and `RDUCSTRK` as cardiovascular-health or aspirin-related variables requiring special attention. Their exact routing conditions should be checked against the official [CDC BRFSS 2015 questionnaire](https://www.cdc.gov/brfss/annual_data/annual_2015.html?utm_source=chatgpt.com) before including them. The report's broad statement that these variables are asked only after a cardiovascular event should not be treated as verified for every listed variable without checking the individual questionnaire instructions.

Established risk factors. Age, sex, hypertension, high cholesterol, diabetes, smoking, obesity, and kidney disease are established cardiovascular risk factors. The Framingham Risk Score and Pooled Cohort Equations also illustrate the importance of combining demographic and clinical risk factors rather than considering them independently.

Your features already capture many of these risk factors. The most plausible interactions to test are:

- Age × sex, which is consistent with the differing age-specific risks you observed.
- Age × hypertension and age × diabetes.
- Hypertension × cholesterol and BMI × hypertension.
- Counts of chronic conditions, provided the counts are binned or interacted with other features rather than simply added as redundant linear terms.

A BRFSS-related report also describes engineered features such as a health-risk score formed from hypertension, high cholesterol, and diabetes, and a BMI × hypertension interaction. See the [Heart Disease Health Indicators overview](https://www.emergentmind.com/topics/heart-disease-health-indicators-dataset?utm_source=chatgpt.com).

Missingness and counts. Since your logistic regression already includes missingness indicators for every numerical feature and a missing category for categorical features, a count of missing values may add limited new information. It could still help if it captures patterns across multiple features that are not effectively represented by the individual indicators.

Likewise, a count of “don't know” responses or skipped modules may capture survey-response behaviour, health literacy, healthcare access, or disease burden. But this should be tested with a paired ablation rather than assumed to help. A linear count adds a new direction to the model, while a binned count can express nonlinear effects.

### 4. F1 decision theory and threshold selection

For a calibrated probability estimator, the F1-optimal decision threshold is related to the optimal F1 itself:

\\[ t^\*=\frac{F_1^\*}{2}. \\]

This result is discussed by Lipton, Elkan, and Narayanaswamy in Thresholding Classifiers to Maximize F1 Score (2014). See the [paper](https://arxiv.org/html/1402.1892v2?utm_source=chatgpt.com).

Your observed threshold of approximately 0.24 is consistent with an F1 score around 0.44, so your operating point is plausible. The result is not proof of perfect calibration, however, and the threshold relation is a population-level result under appropriate assumptions.

Recommendation: Continue optimizing a single threshold using training-only or out-of-fold predictions. Compare models using the same folds and the same threshold-selection protocol. Do not expect focal loss, class weighting, or a direct F1 surrogate to deliver a large gain automatically. You have already found that balanced class weights and undersampling mostly change the decision threshold rather than improve the underlying ranking.

Expected-F1 decision rules, such as those studied by Jansche (2007), Nan et al. (2012), and other work on non-decomposable metrics, are theoretically relevant. But for your setup, threshold tuning is simpler and better justified than replacing binary cross-entropy with a custom F1 loss.

### 5. Validation of small improvements

Your observed fold-to-fold variation is approximately ±0.003 F1, while the paired difference between two pipelines on the same folds has a standard deviation around 0.001.

That means:

- A +0.001 F1 change should not be treated as reliable.
- A +0.002 F1 gain is worth investigating, but may be too small to distinguish confidently with only five folds.
- A +0.003 to +0.005 paired gain is more promising, but still needs careful validation.

Recommended protocol

1. Keep the same folds for every candidate model.
2. Produce out-of-fold predictions for all training observations.
3. Tune each model's decision threshold without using its held-out labels.
4. Compare the paired F1 differences across folds.
5. For promising candidates, repeat cross-validation using several different random partitions.
6. Report the mean and dispersion of paired differences, not just each model's best score.
7. Use the public leaderboard sparingly and retain a final decision rule that does not depend on repeated leaderboard experimentation.

Dietterich (1998) discusses the risks of statistical testing when comparing learning algorithms, including the 5×2 cross-validation approach. Repeated cross-validation can improve the evidence available, but the scores from different repetitions are correlated; treating every fold as an independent observation will overstate statistical significance.

Given the deadline, a practical compromise is to run the existing five folds first, screen candidates using paired differences, and repeat only the strongest candidate. A large 10×5 cross-validation search would probably be too expensive for a newly implemented boosting method.

## 4. What not to spend time on

| Approach                                                          | Why it is low priority                                                                                                     |
| ----------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| Class weighting and oversampling                                  | Already tested, with little or no improvement after threshold optimization.                                                |
| Arbitrary numerical quantile bins                                 | Your tests showed no meaningful improvement over the existing encoding.                                                    |
| Dropping highly missing features                                  | Already hurt F1 by roughly 0.008–0.010, partly because useful information was lost.                                        |
| Simple deeper MLPs without a stronger representation              | Your MLP experiments overfit quickly and did not consistently beat logistic regression.                                    |
| Changing L2 regularization over a wide range                      | Your observed effects were around 0.0004 F1.                                                                               |
| Univariate feature selection                                      | Selecting a small set of features removed useful conditional information and generally hurt performance.                   |
| Naive target or weight-of-evidence encoding                       | Potential leakage risk and no demonstrated benefit over your carefully regularized one-hot baseline.                       |
| Kernel methods such as RBF SVMs                                   | Computationally unattractive at your sample size and unlikely to be worth implementing under the deadline.                 |
| Complex boosting implemented from scratch without early screening | Potentially promising but high engineering risk; spend the time only if a simple prototype demonstrates a meaningful gain. |

## Final recommendation

Given the deadline and your current results, I would prioritize one low-risk experiment and one higher-upside model.

First, retain the existing logistic regression and MLP predictions and test a carefully validated probability average or rank average. This has minimal implementation cost and is supported by your measured +0.0017 F1 improvement.

Second, implement a constrained histogram gradient-boosting model only if you can finish a working prototype quickly enough to validate it. It offers the most credible opportunity to capture nonlinearities and interactions absent from logistic regression. Use a small search, paired folds, and a strict stopping rule: if it does not show a meaningful improvement over the baseline early, do not continue expanding it.

Factorization Machines are the alternative if you want a more controlled second-order interaction model that reuses your existing sparse one-hot representation.

The central conclusion is that your current F1 = 0.440 is already competitive, and a large improvement is not guaranteed. The most defensible approach is to seek a measurable gain through model diversity or nonlinear interactions, while preserving your strong baseline and keeping validation leakage out of the threshold-selection process.