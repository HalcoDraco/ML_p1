# Report handoff

The report has two content pages and a third reference page. It
contains two result tables and two exploratory figures. It covers the six
required implementations, the BRFSS preprocessing, the selected NumPy GBDT,
two distinct validation protocols, and the limits of interpreting the target.
Build it with the command in report/README.md.

The final review recovered and checked the NumPy MLP implementation and its
configuration in `outputs/cv_results.csv`: five 512/256 networks, dropout 0.3,
AdamW at initial rate 0.001, batch size 512, and two epochs, with inner
checkpoint selection. This model is now described explicitly and appears
in Table II (CV F1 0.4399, AUC 0.8655).

## Sources of the numbers

| Report result | Source | Protocol |
|---|---|---|
| Data counts, prevalence, cleaning | docs/eda_findings.md, p1/cleaning.py | Course data and fixed codebook rules |
| Six methods and raw-input baseline | Root README.md; configs/step2/*.toml | Five folds, seed 42; thresholds fitted on training scores |
| Representation figure | docs/eda_findings.md; report/data/results.json | One 80/20 holdout; threshold fitted on validation scores, so F1 is optimistic |
| GBDT tuning figure | docs/part3_results.md, round 1; report/scripts/plot_tuning.py | Five folds, seed 42; one change at a time, each at its best tree count |
| Logistic and selected GBDT | docs/part3_results.md | Three split seeds, five folds each; threshold from other out-of-fold scores |
| Logistic AIcrowd F1 0.449, accuracy 0.885 | docs/part3_results.md | Documented submission of logistic.csv against test labels |
| Selected GBDT AIcrowd F1 0.454, accuracy 0.882 | docs/round3_summary.md | Documented submission of best.csv against test labels |
| GBDT ensembles and feature variants | docs/part3_results.md; docs/round3_summary.md | Mostly 15-fold nested F1; the four-configuration bag uses five folds only. No AIcrowd score is documented for these variants |
| GBDT depth and feature ablations | docs/part3_results.md, sections 3--4 | Five folds, split seed 42; best-stage F1 optimizes a threshold on each validation fold and is slightly optimistic |
| Paired F1 interval | docs/part3_results.md | Corrected paired comparison, conditional on the selected configurations |
| Final prediction recipe | configs/best.toml; run.py | Three refitted GBDTs, seeds 0--2, threshold 0.2166 |

The five-fold numerical-method F1 values and the later 15-fold model-comparison
F1 values must remain separate. The later CV selected model configurations
using the same validation data, so it is not a fully nested estimate of the
full search.

## Final submission correspondence

A team member reported an AIcrowd F1 of 0.456, but this checkout has no
submission ID or CSV tying that score to a specific recipe. The report includes
the documented 0.449 and 0.454 submissions and leaves 0.456 out. Before the
final AIcrowd selection, verify that the chosen submission was produced by the
exact current run.py output.

The final review has now verified that correspondence for the recovered
`best.csv`: a complete retraining with the locked environment produces a
byte-identical CSV (zero differing predictions; SHA-256
`41b4219359b13278fd43fc28ef90aae5be2d48b59157db0d3916dde8d124ce80`).
This establishes reproduction of the documented 0.454 CSV, not a mapping
to the unidentified 0.456 upload. See `SUBMISSION.md` for exact upload steps.
