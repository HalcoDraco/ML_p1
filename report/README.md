# Project 1 report

The manuscript is in report.tex; the compiled copy is report.pdf. The current
draft has three content pages and a fourth page of references. Results use two tables
and two figures about data representation and GBDT tuning.
The paper separates local validation results from two documented AIcrowd
test-submission scores: F1 0.449 for logistic.csv and F1 0.454 for best.csv.

## Build

From the repository root, with uv, latexmk, pdflatex, and bibtex installed:

    uv run --no-project --python 3.12 \
      --with-requirements report/requirements-report.txt \
      python -B report/scripts/build.py

Add --render to save page previews under report/build/. The final build checks
for two portrait content pages and a portrait reference page,
and has no unresolved citations or overflowing boxes. It copies the verified
PDF to report/report.pdf. It does not train models or generate figures.

To compile only the LaTeX:

    cd report
    latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build report.tex

This direct command updates report/build/report.pdf, not report/report.pdf.

## Evidence and files

- metadata.tex: author names, affiliation, and addresses.
- generated/baseline_table.tex: the six required methods and a raw-input
  baseline, from the five-fold experiment documented in the root README.md.
- figures/representation.pdf: exploratory 80/20 representation study from
  docs/eda_findings.md. Its threshold was fitted on the plotted holdout,
  so its F1 values are optimistic and separate from the CV tables.
- figures/gbdt_tuning_draft.pdf: one-factor, best-stage log-loss comparison
  from docs/part3_results.md; scripts/plot_tuning.py regenerates it.
- docs/part3_results.md and docs/round3_summary.md: the repeated five-fold
  comparison of logistic regression, GBDT, and later variants, plus the
  documented AIcrowd scores for logistic.csv and best.csv.
- data/results.json: archived measurements and the protocol for each group.
- scripts/build.py: compilation and checks.

scripts/plot_results.py can regenerate the representation figure from
data/results.json when NumPy, Matplotlib, and Seaborn are installed.
scripts/plot_tuning.py needs Matplotlib. The other existing model_families
figure is not used in the paper. The selected GBDT's CV F1 and AIcrowd F1
are reported separately; run.py refits that recipe on all labeled rows to
create test predictions.
