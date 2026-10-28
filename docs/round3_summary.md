# Round 3 summary (2026-10-28)

CV metrics: mean over 15 folds (3 fold seeds × 5), nested F1 (threshold taken from the other folds), accuracy at that threshold. The submitted `best.csv` (= `fc_bag`) scored **F1 0.454 / acc 0.882** on AIcrowd at CV 0.4462.

## Best things tried (ranked by CV F1)

| # | Run | What it is | F1 | Acc | Δ F1 vs `fc_bag` |
|---|---|---|---|---|---|
| 1 | `fe2_bag5` | 5 GBDTs with **count features + target-ordered codes**, η 0.02, 2000 trees (`configs/gbdt_fe.toml`, 3 seeds + ff 0.2 + min_child_hessian 1) | **0.4474** | **0.8822** | +0.0011 (t 1.51) |
| 2 | `fe2_bag` | same, 3 seeds only | 0.4474 | 0.8805 | +0.0011 |
| 3 | `blend_fc_m5` | `fc_bag` + NumPy MLP (logit blend) | 0.4469 | 0.8792 | +0.0007 (t 1.65) |
| 4 | `bag_cfg7` | bag of 7 tuned GBDT configs | 0.4469 | 0.8795 | +0.0006 |
| 5 | `x_tord_cnt_care` / `x_cnt` / `x_tord_cnt` | single GBDT + FE variants | 0.4466–0.4468 | 0.878–0.881 | +0.0005 |
| 6 | `fe2_cfg7` | `fe2_bag5` + `bag_cfg7` | 0.4467 | 0.8803 | +0.0005 |
| 7 | `fc_bag` (submitted) | 3-seed tuned GBDT bag | 0.4462 | 0.8794 | 0 |
| – | `x_lrf*` | LR-logit stacked as GBDT feature | 0.4456–0.4460 | 0.877–0.879 | ≤ 0 |
| – | LightGBM / XGBoost / CatBoost (scratch benchmarks only, banned for submission) | | ≤ 0.4469 (1 seed) | | ≈ our GBDT |
| – | MLP alone (`m5`) | | 0.4399 | 0.8784 | −0.006 |

Takeaways: our NumPy GBDT already matches LightGBM/XGBoost/CatBoost; the only gain came from feature engineering (counts + target-ordered categorical codes) and bagging. All gains are within noise (t < 2).

## Submit these three

| Priority | File | Why |
|---|---|---|
| 1 | `outputs/submissions/fe2_bag5.csv` | best CV F1 and accuracy (thr 0.220, 12.72% flagged) |
| 2 | `outputs/submissions/blend_fc_m5.csv` | different ingredient (MLP), second-best paired t (thr 0.214, 12.92%) |
| 3 | `outputs/submissions/bag_cfg7.csv` | config diversity, no FE (thr 0.211, 13.33%) |

All three put all 238 `HAREHAB1` test rows at 1. They are averages of the 15 CV fold models (not refits on all rows). If one is chosen as final, `configs/best.toml` + `run.py` must be switched to it and rerun so `run.py` reproduces the CSV (for `fe2_bag5`: members `gbdt_fe` seeds 0–2, `feature_fraction=0.2`, `min_child_hessian=1`).
