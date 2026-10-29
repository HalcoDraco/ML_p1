# Archived NumPy experiments

These scripts supplement the final pipeline in `p1/`; the final prediction
entrypoint remains `run.py`. Python launchers resolve the repository from
their own location, without depending on the original cluster paths.

Build the clean dataset once before running a training experiment:

```bash
uv run python -c 'from p1.cleaning import build_clean_dataset; build_clean_dataset()'
```

## Neural networks

`mlp.py` implements the network and its encoding using NumPy. `mlp_run.py`
runs one outer validation fold and saves its predictions. To reproduce the
recorded `m5` configuration on split seed 42, fold 0:

```bash
uv run python experiments/mlp_run.py --name m5 --task 0 \
  --model '{"hidden":[512,256],"dropout":0.3,"lr":0.001,"epochs":2,"n_models":5}'
```

Tasks 0--14 cover split seeds 42--44 and folds 0--4. The defaults retained
by this configuration are batch size 512, weight decay 0.00001, inner
validation fraction 0.1, and model seed 0. The implementation selects the
lowest inner-validation loss checkpoint within the two-epoch budget and
averages five networks' logits. Inputs use 16-bin piecewise-linear numeric
encoding, missingness indicators, categorical one-hot encoding, and scaling.

## Feature engineering and summaries

```bash
uv run python experiments/fe.py --help
uv run python experiments/report.py 'fc_*' --seed 42
uv run python experiments/pooled.py fc_bag logistic
```

The feature launcher registers extra NumPy transforms for count features,
target ordering, and logistic-score stacking. The production count/target-order
recipe is also in `configs/gbdt_fe.toml` and can be run with `p1 cv`.

The summaries above require original fold files under `outputs/oof/`, which
are excluded from Git. The included `outputs/cv_results.csv` preserves aggregate
metrics and configuration JSON, and `outputs/slurm/` preserves execution logs.
`outputs/code/` contains historical source snapshots with original cluster
symlinks; use the root entrypoints for portable execution.

## Cluster launcher

Submit from the repository root, after creating the log directory:

```bash
mkdir -p outputs/slurm
sbatch --array=0-14 -c 4 --mem=24G -t 01:00:00 experiments/job.sbatch \
  experiments/mlp_run.py --name m5 \
  --model '{"hidden":[512,256],"dropout":0.3,"lr":0.001,"epochs":2,"n_models":5}'
```

The launcher uses the project's existing NumPy environment. External-library
benchmark scripts have been retained locally in ignored
`experiments/external_benchmarks/` and are excluded from the submitted tree.
