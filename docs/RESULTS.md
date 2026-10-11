# Results

Every number here comes from `docs/results/*.json`, written by the `fraudops` step named in each
section, on an M-series MacBook (24 GB, models capped at 4 threads). Run date: 2026-10-12.
Reasoning behind each choice: [DESIGN.md](DESIGN.md).

## 1. Data and split (`fraudops prepare`, `train`)

| | rows | days | fraud rate | fraud share of dollars |
|---|---|---|---|---|
| all | 590,540 | 1-182 | 3.50% | 3.87% |
| train | 410,601 | 1-119 | 3.51% | 3.79% |
| validation | 90,613 | 120-151 | 3.45% | 4.18% |
| test | 89,326 | 152-182 | 3.49% | 3.90% |

434 raw columns after the identity join; 453 model inputs after feature engineering (the uid and
`account_start` are built but are not inputs). CSV to Parquet plus all features: about 6 s.

## 2. Models on the test month (`fraudops train`)

PR-AUC is the headline (3.5% positives). All models are early-stopped and tuned on validation;
test was scored once per model.

| model | val PR-AUC | test PR-AUC | test ROC-AUC | test recall at 1% FPR |
|---|---|---|---|---|
| logistic regression (rank-scaled, non-Vesta columns) | 0.573 | 0.528 | 0.914 | 45.0% |
| LightGBM, raw columns only (no engineered features) | 0.621 | 0.534 | 0.898 | 44.4% |
| LightGBM, no known-fraud features | 0.652 | 0.551 | 0.904 | 46.0% |
| LightGBM, no velocity/history features | 0.713 | 0.635 | 0.932 | 55.2% |
| LightGBM, all features, hand-picked settings | 0.717 | 0.637 | 0.931 | 56.0% |
| **LightGBM, Optuna-tuned (champion, registry v2)** | **0.717** | **0.644** | **0.936** | **55.3%** |

* **Tuned LightGBM vs logistic:** +0.116 test PR-AUC, paired bootstrap 95% CI **+0.105 to
  +0.128** (300 resamples of test rows).
* **What the features are worth:** the known-fraud features (30-day label delay) carry the gain:
  removing them costs 0.093 test PR-AUC. The velocity/history features (counts, mean amount,
  time since previous, 24-hour activity) add only 0.002: Vesta's 339 V columns and the C counts
  already encode most of that. They stay because they cost under a millisecond to serve, but
  this is the honest result.
* **Optuna:** 30 trials in 8.3 minutes, 20 pruned early by the median rule. Best validation
  PR-AUC 0.7172 vs 0.7167 for the hand-picked setting (enqueued as trial 0). Tuning barely
  matters here; the features do.
* **Val to test gap** (0.717 to 0.644): validation was used for early stopping, tuning,
  calibration and the threshold, and test is further from training in time. Section 7 shows the
  decay week by week.

**First-version numbers kept for the record** (registry v1, now `retired`): test PR-AUC 0.568
with `account_start` as an input and no known-fraud features; logistic with `StandardScaler`
0.166 test vs 0.347 validation (recall at 1% FPR 2.2%). DESIGN.md section 3 and 5 explain both.

## 3. From score to decision (`fraudops train`, test month)

Costs: a review $10; a missed fraud = amount + $20 chargeback fee. Fraud in the test month: $540k
of cost if nothing is flagged.

| rule (all chosen on validation) | cost | saved vs no model | flagged | precision | recall | fraud $ caught |
|---|---|---|---|---|---|---|
| flag nothing | $539,636 | 0% | 0% | - | 0% | 0% |
| raw score ≥ 0.5 | $356,551 | 33.9% | 1.9% | 84.2% | 46.3% | 35.9% |
| best single threshold on validation | $209,753 | 61.1% | 11.8% | 23.8% | 80.4% | 80.7% |
| **expected cost: review if p × (amount + $20) > $10** | **$169,886** | **68.5%** | **9.0%** | **27.5%** | **70.9%** | **85.0%** |

The expected-cost rule catches *fewer* frauds than the best threshold (70.9% vs 80.4%) but more
fraud **dollars** (85.0% vs 80.7%) with fewer reviews (9.0% vs 11.8%): it spends reviews on
large transactions and lets small low-risk ones through. On validation it saves 74.1%.

**Calibration on the test month** (isotonic, fitted on validation), by decile of predicted
probability:

| decile | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|
| predicted | 0.00% | 0.05% | 0.15% | 0.25% | 0.46% | 0.83% | 1.28% | 3.01% | 26.2% |
| observed | 0.03% | 0.08% | 0.28% | 0.29% | 0.49% | 1.07% | 1.79% | 2.88% | 24.5% |

(Decile 1 merges into 2: the isotonic map gives them the same value.)

## 4. Leakage, measured (`fraudops leakage`)

Same LightGBM settings throughout, test PR-AUC:

| | time split (train past, test future) | random split |
|---|---|---|
| past-only features (production-like) | **0.637** | 0.873 |
| future-peeking aggregates (whole-table group stats) | 0.641 | 0.887 |

A random split overstates the model by 37% (0.637 to 0.873): the same cards' later transactions
sit in training. The future-peeking aggregates add little on top here, so the split is where
the leak bites in this dataset.

## 5. What a faster chargeback feed is worth (`fraudops label-delay`)

Known-fraud features with labels usable after N days, same model settings:

| label delay | 7 days | 30 days (used) | 60 days | 90 days |
|---|---|---|---|---|
| val PR-AUC | 0.759 | 0.717 | 0.683 | 0.645 |
| test PR-AUC | **0.707** | 0.637 | 0.607 | 0.573 |

Every month of label delay costs about 0.03-0.07 PR-AUC. At 90 days the known-fraud features are
worth little more than nothing (0.573 vs 0.551 without them). Getting chargebacks or fraud
confirmations faster is worth as much as any modelling change in this project.

## 6. Serving (`fraudops build-store`, `fraudops bench`)

**Feature parity.** The online store replayed all 590,540 transactions (with chargebacks reported
after scoring) and reproduced the offline value of every one of 24 online-computed columns:
**14,172,960 values, 0 mismatches** (exact equality, 6.2 s). The same test runs in CI on
synthetic data.

**Store snapshot** at the end of validation: 501,214 transactions replayed in 4.5 s; 12,914 cards
and 175,045 uids; 21.6 MB.

**HTTP benchmark:** uvicorn with the registry champion and that snapshot; the first 5,200 test
transactions sent one at a time in stream order (first 200 dropped as warm-up); fraud rows
reported through `/v1/chargeback` after scoring.

| | client p50 | client p95 | client p99 | server p95 |
|---|---|---|---|---|
| with top-5 SHAP reasons | 7.9 ms | **8.2 ms** | 8.3 ms | 7.6 ms |
| without reasons | 1.0 ms | **1.1 ms** | 1.2 ms | 0.7 ms |

Every one of the 5,000 API probabilities equals the offline batch prediction for the same row
(max absolute difference 0.0). The review rate in this slice is 7.8%. Explanations cost about
7 ms: TreeSHAP walks every path of 605 trees over 453 features. A real deployment would explain
only the transactions it sends to review (9%).

## 7. Drift and decay (`fraudops drift`)

Reference: the training period. Each row is one week of validation + test (weeks 17-20 are
validation).

| week | transactions | fraud rate | PR-AUC | score PSI | features with PSI > 0.2 (of top 30) |
|---|---|---|---|---|---|
| 17 | 20,524 | 3.56% | 0.814 | 0.106 | 4 |
| 18 | 19,622 | 3.53% | 0.732 | 0.070 | 3 |
| 19 | 18,630 | 3.46% | 0.699 | 0.089 | 4 |
| 20 | 18,460 | 3.59% | 0.656 | 0.087 | 3 |
| 21 | 20,378 | 3.17% | 0.680 | 0.107 | 4 |
| 22 | 20,740 | 2.85% | 0.622 | 0.104 | 4 |
| 23 | 20,944 | 3.25% | 0.563 | 0.059 | 2 |
| 24 | 20,206 | 3.67% | 0.671 | 0.059 | 3 |
| 25 | 17,681 | 4.16% | 0.671 | 0.087 | 3 |
| 26 | 2,754 | 4.32% | 0.733 | 0.090 | 3 |

* **The score never drifted** (PSI 0.06-0.11, under the 0.2 alarm every week) **while PR-AUC fell
  from 0.81 to 0.56-0.67.** Input and score monitoring alone would have reported "stable".
  Performance monitoring on matured labels is what catches decay, and it lags by the label
  delay. That is the argument for a scheduled retrain rather than a drift-triggered one.
* 25 of 453 features exceed PSI 0.2 between train and val+test. The worst is
  `card1_fraud_rate_known` (1.34), and that drift is structural: in the first 30 days of
  training no label has matured yet, so the feature is missing there. Next: start training at
  day 31 so the feature is populated for every training row. Real drift: `id_31` (browser and
  version, 0.36) as new browser versions ship; the M1-M9 match flags (0.30-0.32) change their
  missing rate.

## 8. Engineering

* 38 tests (pytest), 95% line coverage, ruff + mypy (strict) clean; CI runs everything on a
  synthetic dataset with the same schema (the real data cannot be redistributed).
* Full `train` (logistic, 5 LightGBM fits, 30 Optuna trials, SHAP, bootstrap, registry): 14.7
  minutes at `nice 10` on 4 threads.
* MLflow (SQLite + local artifacts): runs for every model, registry `fraud-detector` v1
  (`retired`) and v2 (`champion`).
