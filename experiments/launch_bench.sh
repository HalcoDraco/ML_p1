#!/bin/bash
# Submit the external-library benchmark grid (fold seed 42, 5 folds each).
set -euo pipefail
cd /scratch/pgonzale/p1x
J() { name=$1; cpus=$2; mem=$3; time=$4; shift 4; sbatch --parsable -J $name --array=0-4 -c $cpus --mem=$mem -t $time job.sbatch bench.py --name $name "$@"; }
MIMIC='"learning_rate": 0.05, "num_leaves": 32, "max_depth": 5, "min_data_in_leaf": 1, "min_sum_hessian_in_leaf": 5.0, "lambda_l2": 10.0, "feature_fraction": 0.3, "bagging_fraction": 0.7, "bagging_freq": 1, "max_bin": 63, "num_threads": 8, "seed": 0'
J bench_lgb_mimic 8 16G 00:45:00 --model lgb --params "{$MIMIC}" --trees 1500 --save 550,1000,1500
J bench_lgb_bynode 8 16G 00:45:00 --model lgb --params "{$MIMIC, \"feature_fraction\": 1.0, \"feature_fraction_bynode\": 0.3}" --trees 1500 --save 550,1000,1500
J bench_lgb_cat 8 16G 00:45:00 --model lgb --cat --params "{$MIMIC, \"max_cat_to_onehot\": 4, \"cat_smooth\": 50, \"cat_l2\": 10, \"min_data_per_group\": 200, \"max_cat_threshold\": 16}" --trees 1500 --save 550,1000,1500
J bench_lgb_extra 8 16G 01:00:00 --model lgb --params "{$MIMIC, \"extra_trees\": true}" --trees 3000 --save 1000,2000,3000
J bench_lgb_leaf31 8 16G 01:00:00 --model lgb --params '{"learning_rate": 0.03, "num_leaves": 31, "max_depth": -1, "min_data_in_leaf": 100, "lambda_l2": 10.0, "feature_fraction": 0.3, "bagging_fraction": 0.7, "bagging_freq": 1, "max_bin": 63, "num_threads": 8, "seed": 0}' --trees 3000 --save 1000,2000,3000
J bench_lgb_leaf15 8 16G 01:00:00 --model lgb --params '{"learning_rate": 0.03, "num_leaves": 15, "max_depth": -1, "min_data_in_leaf": 200, "lambda_l2": 10.0, "feature_fraction": 0.2, "bagging_fraction": 0.7, "bagging_freq": 1, "max_bin": 63, "num_threads": 8, "seed": 0}' --trees 3000 --save 1000,2000,3000
J bench_lgb_slow 8 16G 01:30:00 --model lgb --every 100 --params '{"learning_rate": 0.01, "num_leaves": 31, "max_depth": -1, "min_data_in_leaf": 100, "lambda_l2": 10.0, "feature_fraction": 0.2, "bagging_fraction": 0.7, "bagging_freq": 1, "max_bin": 63, "num_threads": 8, "seed": 0}' --trees 6000 --save 3000,4500,6000
J bench_lgb_dart 8 16G 02:00:00 --model lgb --every 100 --params "{$MIMIC, \"boosting\": \"dart\", \"learning_rate\": 0.1, \"drop_rate\": 0.1, \"skip_drop\": 0.5}" --trees 1000 --save 1000
J bench_xgb_d5 8 16G 01:00:00 --model xgb --params '{"max_depth": 5, "eta": 0.05, "subsample": 0.7, "colsample_bytree": 0.3, "lambda": 10, "min_child_weight": 5, "max_bin": 64, "nthread": 8, "seed": 0}' --trees 1500 --save 550,1000,1500
J bench_cat_c 16 64G 04:00:00 --model cat --cat --every 100 --params '{"depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 3, "thread_count": 16, "random_seed": 0}' --trees 3000 --save 1000,2000,3000
J bench_cat_n 16 64G 04:00:00 --model cat --every 100 --params '{"depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 3, "thread_count": 16, "random_seed": 0}' --trees 3000 --save 1000,2000,3000
