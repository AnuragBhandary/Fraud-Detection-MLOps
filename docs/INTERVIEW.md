# Interview guide

How to learn this project, pitch it, and defend it in an ML engineer, data scientist or MLOps
interview. Numbers: [RESULTS.md](RESULTS.md). Reasoning: [DESIGN.md](DESIGN.md).

## 1. Learning path (read in this order)

| # | File | What to be able to explain afterwards |
|---|---|---|
| 1 | `config.py`, `data.py`, `split.py` | The dataset (what C, D, M, V, id columns are), why only the labelled half, the day cut points, why whole days |
| 2 | `features.py` `row_features`, `history_features` | Stream order; `cum_sum().shift(1).over(key)` = sum of strictly earlier rows; the 24-hour count with one `searchsorted` over (key, time); why `global` mode leaks |
| 3 | `features.py` `label_features` | Matured vs unknown labels; why "no chargeback yet" must not mean legitimate; the cumulative-sum trick for "fraud among rows older than t - 30 days" |
| 4 | `online.py` | The per-entity state; the 24-hour deque; the pending queue and maturation; why scoring reads before it writes; the float32 rounding that makes parity exact |
| 5 | `tests/test_parity.py`, `test_features.py` | The replay test; the tamper-the-future test; the brute-force 24-hour check |
| 6 | `model.py` | `prepare` (encoder fitted on train only); early stopping on validation PR-AUC; the logistic pipeline and why `QuantileTransformer`; the Optuna objective, pruning callback, enqueued trial 0; `FraudModel` bundle |
| 7 | `metrics.py` | Average precision vs ROC-AUC; exact cost-minimising threshold in one sort + cumsum; expected-cost rule; paired bootstrap |
| 8 | `registry.py` | Pyfunc wrapping a pickled bundle; registering; champion/challenger/retired; loading by alias |
| 9 | `api.py`, `serve.py`, `bench.py` | Request flow; sync handler in a threadpool; chargeback endpoint; how the benchmark proves parity over HTTP |
| 10 | `drift.py`, `pipeline.py drift` | PSI formula, bins from reference deciles, missing as a bin; why score PSI stayed flat while PR-AUC fell |
| 11 | `pipeline.py leakage`, `label_delay` | The 2x2 leakage experiment; the delay sweep |

Exercise after each file: close it and rewrite the core function on paper. Three to drill:
PSI by hand for two 3-bin histograms; the expected-cost rule for a $500 transaction at 3% risk
(0.03 × 520 = $15.60 > $10, so review); and why `cum_sum().shift(1)` and not `cum_sum() - x`
(the subtraction rounds differently from the online running sum, and parity is exact equality).

## 2. The 60-second pitch

> "I built a card-fraud model on Vesta's IEEE-CIS data, 590,000 real e-commerce transactions with
> 3.5% fraud, and focused on what breaks fraud models in production rather than on leaderboard
> score. I split by time: train on four months, tune on the next, report on the month after. A
> random split would have said PR-AUC 0.87; out of time it's 0.64. The biggest gain came from
> each customer's known fraud history, but counted only once a label is 30 days old, because
> chargebacks arrive late; I measured that a 7-day label feed would lift PR-AUC to 0.71 and a
> 90-day one drops it to 0.57. I turned scores into decisions with a dollar cost rule on
> calibrated probabilities: on the test month it cuts fraud cost by 68%, against 34% for a 0.5
> cutoff. The model is versioned in MLflow with champion/challenger promotion and served by
> FastAPI with SHAP reasons in 1.1 ms at p95, or 8 ms with explanations. I wrote the features twice,
> vectorised for training and streaming for serving, and proved they agree on all 14 million
> values. The drift report showed the score distribution never moved while accuracy fell from
> 0.81 to about 0.6, so drift metrics alone would have missed it."

## 3. Numbers to know cold

| | |
|---|---|
| Data | 590,540 txns, 182 days, 3.50% fraud; 410,601 / 90,613 / 89,326 (days 1-119 / 120-151 / 152-182) |
| Champion | test PR-AUC 0.644, ROC-AUC 0.936, recall at 1% FPR 55.3%; 605 trees; 453 features |
| Logistic | 0.528 (rank-scaled); 0.166 with StandardScaler |
| Gain vs logistic | +0.116, CI +0.105 to +0.128 |
| Ablations (test) | no known-fraud 0.551; no velocity 0.635; neither 0.534 |
| Label delay | 7d 0.707, 30d 0.637, 60d 0.607, 90d 0.573 |
| Leakage | time 0.637, random 0.873; future aggregates 0.641 / 0.887 |
| Cost | no model $539,636; 0.5 cutoff 33.9% saved; best threshold 61.1%; expected cost **68.5%** ($169,886), 9.0% reviewed, 85.0% of fraud $ caught |
| Optuna | 30 trials, 20 pruned, 8.3 min; 0.7167 → 0.7172 val |
| Parity | 14,172,960 values, 0 mismatches; API 5,000/5,000 identical |
| Latency | p95 1.1 ms; 8.2 ms with SHAP |
| Drift | score PSI 0.06-0.11 weekly; PR-AUC 0.81 (week 17) to 0.56-0.67; `account_start` PSI 1.84 |

## 4. Likely questions

**Problem framing**

- *Why PR-AUC and not accuracy or ROC-AUC?* Accuracy is 96.5% for "never fraud". ROC-AUC
  counts how well you rank the 96.5% easy negatives; PR-AUC measures precision among what you
  flag, which is what the review team experiences. I also report recall at 1% FPR, a number an
  ops team can act on.
- *Why not SMOTE or class weights?* They change the score scale, not the ranking much, and I need
  probabilities for the cost rule. I keep the true base rate and calibrate instead.
- *What is your cost model and where do the numbers come from?* $10 per review, amount + $20 per
  missed fraud. They are assumptions in one config class; the rule takes them as inputs, so a
  business can plug in its own. The point is the rule, not the exact dollars.

**Leakage and validation**

- *How do you know there is no leakage?* (1) Time split on whole days. (2) Every history feature
  uses strictly earlier rows; a test multiplies all future amounts by 50 and checks no earlier
  feature changes. (3) Labels count only after 30 days. (4) Category codes fitted on train only.
  (5) Every model decision (features, tuning, threshold) made on validation, test scored once.
- *Your validation is 0.72 and test 0.64. Why?* Validation chose early stopping, 30 trials, the
  calibrator and the threshold, so it is optimistic; and test is further in time, and the
  weekly table shows decay with distance from training.
- *Isn't the known-fraud feature target leakage?* Leakage is using information you would not have
  at decision time. A chargeback from 30+ days ago is information the issuer does have. The leak
  would be counting recent transactions as legitimate because no chargeback has arrived yet. That
  is why young transactions are "unknown" and why the delay sweep exists.
- *Why did `account_start` look important but hurt generalisation?* It is a date: test accounts
  have start days the model never saw (PSI 1.84). It was acting as an account identifier that let
  the model remember accounts that defrauded in training, a proxy for past labels. Dropping it
  alone made validation worse; once known fraud existed explicitly, validation preferred the model
  without it.

**Modelling**

- *Why LightGBM?* Tabular, 450 mixed columns, heavy missingness (it learns which branch missing
  values take), native categoricals, fast. The logistic baseline tells me how much non-linearity
  buys: 0.116 PR-AUC with a tight bootstrap interval.
- *Why did tuning barely help?* Early stopping already fits the number of trees, and LightGBM's
  defaults are robust on this size; the gains were in features. Knowing when tuning is not the
  bottleneck is a result, and 20 of 30 trials were pruned, so the search was cheap.
- *Why did the velocity features not help?* Vesta's V columns are their own engineered
  aggregates (counts and amounts over windows per entity), so mine overlap. I kept them because
  they cost under a millisecond and they make the model less dependent on a vendor's opaque
  columns.
- *Why isotonic, not Platt?* 90k validation rows is plenty for isotonic, and it makes no shape
  assumption. The decile table shows predicted vs observed within a few tenths of a percent at the
  low end and 26.2% vs 24.5% in the top decile.

**MLOps and serving**

- *How do you prevent training/serving skew?* One definition, two implementations, one exact
  test: replay the stream through the online store and compare every value; then send real HTTP
  requests and compare every probability to the batch prediction. Both run in CI on synthetic
  data.
- *What happens with two API replicas?* Each has its own in-memory store, so history splits. Next
  step: a shared store, e.g. Redis hashes per card and uid with the same update logic. The parity
  test would then run against Redis.
- *How does a new model reach production?* `train` registers a new version; it takes the
  `champion` alias only if validation PR-AUC beats the current champion's; the API loads
  `models:/fraud-detector@champion`. The old one is tagged `retired`, so rollback is moving the
  alias back.
- *Why is SHAP 7 ms when scoring is 1 ms?* TreeSHAP walks every root-to-leaf path of 605 trees
  for 453 features. In production you explain only flagged transactions (9%), or explain
  asynchronously.
- *How would you monitor this?* Score PSI and feature PSI daily (cheap, immediate), PR-AUC and
  cost on matured labels weekly (truth, but 30 days late), retrain on a schedule. My drift table
  is the reason: the score stayed stable while performance fell.

**Questions to ask back:** what their label delay is; whether they decide on probability or rank;
how they handle a second model version in shadow.

## 5. Honest limits

- Column meanings are masked; I cannot say what C13 measures, only that it matters.
- The uid is a reconstruction, not a real customer id; 11% of rows have none.
- One time period, one dataset; no live traffic.
- The cost numbers are assumptions.
- The API keeps state in one process; scaling needs a shared store.
- The known-fraud features are missing for the first 30 days of training; starting training at day
  31 is the next experiment.
