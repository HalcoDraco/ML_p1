# Numerical Optimization and Nonlinear Models for Heart-Disease Classification

The manuscript is in `report.tex`; the compiled reading copy is `report.pdf`.
It uses the course's IEEE template at its original 10-point body size, with
two pages of content and a separate reference page. All figures are generated
with Seaborn and Matplotlib as vector PDFs, with PNG copies for previews.

## Build

From the repository root, with a TeX distribution providing `latexmk`,
`pdflatex`, and `bibtex`:

```bash
uv run --no-project --python 3.12 \
  --with-requirements report_draft/requirements-report.txt \
  python -B report_draft/scripts/build.py
```

Add `--render` after `build.py` to render all pages to PNG with Ghostscript
(`gs`). Rendering is optional; compilation, page-limit checks, citation checks,
and source-value checks always run. The command does not train any model.

When the four Python packages are already installed in a local interpreter,
the environment used for this draft can be selected without downloading:

```bash
UV_CACHE_DIR=/private/tmp/mlp1-report-uv \
uv run --no-project --no-config --offline \
  --python /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 \
  python -B report_draft/scripts/build.py --render
```

To compile only the LaTeX using the committed vector figures and table:

```bash
cd report_draft
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build report.tex
```

The Python build also copies the verified PDF from `build/report.pdf` to
`report.pdf`. Direct LaTeX compilation leaves its output in `build/`.

## Files

| File | Purpose |
|---|---|
| `metadata.tex` | Author names and affiliation |
| `report.tex` | Complete manuscript |
| `references.bib` | Verified primary references |
| `figures/*.pdf` | Vector figures embedded in the paper |
| `figures/*.png` | High-resolution previews of the same figures |
| `data/results.json` | Measurements, source paths, and evaluation protocols |
| `generated/baseline_table.tex` | Table generated from the numerical evidence |
| `generated/measurements.csv` | Plot and table measurements in plain CSV |
| `scripts/plot_results.py` | Rebuilds both figures and the table |
| `scripts/build.py` | Builds and checks the document |
| `HANDOFF.md` | Evidence map and remaining submission details for the team |

The plotting packages are report tooling. The existing model code and its
runtime dependencies are unchanged. The IEEE class and bibliography style are
copied from the original course template, with their notices preserved.

## Evidence policy

Numbers are transcribed from the archived experiment summaries, not presented
as freshly rerun experiments. The build checks that each plotted or tabulated
F1 remains present in the named source file. The initial five-fold comparison,
exploratory holdout, and repeated five-fold comparison are distinct protocols.
The plots do not infer distributions or error bars from absent fold records.

The report includes the team-confirmed public F1 of 0.456 without assigning
it to an unverified model. See `HANDOFF.md` before selecting the final AIcrowd
submission or claiming exact end-to-end reproduction.
