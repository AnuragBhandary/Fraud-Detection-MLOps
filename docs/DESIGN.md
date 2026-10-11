# Design

Why each piece is the way it is. Numbers live in [RESULTS.md](RESULTS.md).

## 1. The data and what it can honestly tell us

IEEE-CIS (Vesta, 2019): 590,540 labelled card-not-present transactions over 182 days, 3.5% fraud,
434 columns after joining the identity table. The column meanings are masked (C1-C14 are counts,
D1-D15 time deltas, V1-V339 Vesta's own engineered features), and the competition's test labels
were never published, so only the labelled half is used and the project carves its own validation
and test periods out of it.

`TransactionDT` is seconds from an unpublished reference point. Day 1 to 182 is all we know; the
hour of day is derived assuming the reference is midnight (the hourly fraud pattern supports it,
but it is an assumption).

## 2. Split by time, not at random

| set | days | rows | why |
|---|---|---|---|
| train | 1-119 | 410,601 | fit models, fit category codes |
| validation | 120-151 | 90,613 | early stopping, Optuna, calibration, threshold |
| test | 152-182 | 89,326 | touched once per model, for the numbers we report |

A fraud model is always trained on the past and used on the future. A random split lets the
model see a card's next-week transactions while training, which a deployed model never can.
`fraudops leakage` measures the gap instead of asserting it. Cuts fall on whole days so one day
never straddles two sets.

The validation month is used for several decisions (early stopping, 30 Optuna trials, the
isotonic calibrator, the cost threshold), so validation scores are optimistic. That is what the
untouched test month is for.

## 3. Features: only the past

**Reconstructed customer.** There is no customer id. `D1` is "days since the card was first
seen", so `day - D1` is a constant per account: the account's start day. `uid = card1 + addr1 +
account start` is the standard reconstruction from the competition's top solutions. It is a
proxy, not ground truth; 11% of rows have no `addr1` or `D1` and get no uid features.

**History per entity** (`card1` and `uid`): number of earlier transactions, their mean and
standard deviation of amount, this amount divided by that mean, seconds since the previous one,
and how many arrived in the last 24 hours. "Earlier" means strictly before in stream order
`(TransactionDT, TransactionID)`. It never includes the row itself or anything after it.

Many public notebooks compute these aggregates over train+test at once (group mean over the whole
table, frequency encoding over everything). That leaks the future into the past. The `global`
mode exists only so the leakage experiment can measure how much it inflates scores.

**Known fraud, with the label delay modelled.** For each card and uid: how many earlier
transactions have a *matured* label, how many of those were fraud, and the rate. A label matures
`LABEL_DELAY_DAYS` (30) after the transaction, because that is roughly when a chargeback has
arrived. A transaction younger than that is *unknown*, not legitimate. Labelling a fresh
transaction "not fraud" just because no chargeback has come yet is the classic way this feature
leaks. `fraudops label-delay` reruns the model with labels usable after 7, 30, 60 and 90 days:
the price of a slow chargeback feed.

Vesta labelled later transactions linked to a charged-back card (same account, email or billing address) as fraud too (the competition host explained this in the Kaggle data discussion
thread), which is why account-level fraud history is so predictive here. In a real
issuer the same thing happens operationally: a compromised account keeps getting hit until it is
blocked.

**A feature that was a date in disguise.** The first version used `account_start` (day - D1) as
a model input. The drift check put it at PSI 1.84 between train and test, the worst of all 450
columns, and SHAP had it second most important. It is a calendar date: test-month accounts have
start days the model never saw. Dropping it alone made validation *worse* (0.665 to 0.649): it was
helping by letting the model recognise accounts that had committed fraud in training, i.e.
standing in for past labels. Once the honest known-fraud features existed, validation preferred
the model without it, so it is now used only to build the uid. All three decisions were made on
validation, not test.

Raw `TransactionDT` and `day` are excluded for the same reason: a time trend learnt on days
1-119 extrapolates nonsense onto day 160.

**Categoricals:** codes learnt on train only. Values seen fewer than 20 times, and values never
seen, share one "rare" code; missing stays missing (LightGBM learns which side missing values
go). Unseen values in production are normal (new email domains, new devices), so this path is
exercised in tests.

## 4. One definition, two implementations, one test

The offline builder is vectorised Polars (window functions plus a sorted `searchsorted` for the
24-hour counts) because it has to process 590k rows in about a second. The API cannot run window
functions over a table, so `OnlineFeatureStore` keeps a small running state per entity (count,
sum, sum of squares, last timestamp, a deque of the last day's timestamps) and reads it before
adding the new transaction.

Two implementations of one definition is how training/serving skew happens. The guard is a
replay: push every transaction through the online store in stream order and compare every value
with the offline column. Both sides do the same float64 arithmetic in the same order and round
through float32 the same way, so the comparison is exact equality, not a tolerance. The test runs
in CI on synthetic data; the full 590k-row replay was run on the real data (RESULTS.md).

The benchmark then goes one level further: the real HTTP API, warmed with train+val history,
scores test transactions in order, and every returned probability is compared with the offline
batch prediction for the same row.

## 5. Models

* **Logistic regression** on the ~100 non-Vesta columns (median impute + missing indicators,
  rank scaling with `QuantileTransformer`, one-hot categoricals). The first version used
  `StandardScaler` and its test PR-AUC was half its validation PR-AUC (0.17 vs 0.35; recall at 1%
  FPR fell to 2%): counts and time deltas in the test month sit far outside the training range,
  and a linear model extrapolates them without limit. Rank scaling caps every input at the
  training distribution's tails. Trees never had the problem; they only split.
* **LightGBM** with hand-picked settings, then the **same model without each engineered family**
  (velocity/history, known fraud, both: are they worth their serving cost?), then an **Optuna** search (TPE, 30 trials,
  early stopping on validation PR-AUC, median pruning of hopeless trials; the hand-picked
  setting is enqueued as trial 0 so the search must beat it).

**Metric: PR-AUC (average precision)**, not ROC-AUC. With 3.5% positives, ROC-AUC is dominated by
the easy negatives; precision-recall reflects what a fraud team sees. Also reported: recall at
1% false-positive rate. The tuned-vs-logistic gap gets a paired bootstrap interval over test rows.

## 6. From a score to a decision

A model ranks; the business needs approve or review. Costs (assumptions, in `config.Costs`): a
review costs $10 (analyst time plus friction for a good customer); a missed fraud costs its amount
plus a $20 chargeback fee. Compared on the test month:

1. flag nothing (the cost of having no model),
2. the naive `score >= 0.5`,
3. the single threshold that minimised cost on validation (exact search over every distinct
   score),
4. the **expected-cost rule**: review when `p * (amount + fee) > review cost`. A $2,000
   transaction at 1% risk is worth reviewing; a $5 one at 30% is not. This needs a probability
   that means what it says, hence:

**Calibration.** LightGBM scores are ranks, not probabilities. An isotonic regression fitted on
validation maps them to probabilities; RESULTS.md shows predicted vs observed fraud by decile on
the test month. Ranking metrics are computed on the raw score (isotonic steps create ties).

## 7. Explanations

Every API response carries the top five features by TreeSHAP contribution (log-odds). LightGBM
computes exact TreeSHAP natively (`pred_contrib=True`), so serving needs no `shap` dependency; a
test checks the values equal `shap.TreeExplainer` and that contributions sum to the model's
margin. The global top-20 is the mean absolute contribution on 20k test rows.

## 8. MLflow registry: champion and challenger

Each `train` logs runs (params, metrics, results JSON) to a local SQLite-backed MLflow and
registers the tuned model as a new version of `fraud-detector`. The deployable artifact is one
pyfunc wrapping the whole bundle (category codes, feature order, booster, calibrator, cost
rule), so the API serves exactly what was evaluated. A new version takes the `champion` alias
only if its validation PR-AUC beats the current champion's; otherwise it is tagged
`challenger`. The API loads `models:/fraud-detector@champion`.

## 9. Serving

FastAPI, one transaction per request: online features, encode, predict, decide, explain, then
add the transaction to the store. `POST /v1/chargeback` reports a fraud; the store holds each
entity's last 30 days of transactions in a pending queue and, when one passes the delay, counts
it as fraud if a chargeback was reported and as legitimate otherwise. (A chargeback that arrives
after the delay is missed; the 30-day assumption is exactly that.) The handler is sync, so FastAPI runs it in a worker thread and
the event loop stays free. The store has a lock; with several workers each process would need a
shared store (Redis, keyed by entity), which is the obvious next step and out of scope.
Single-row LightGBM prediction runs with one thread: threading overhead outweighs the work.

## 10. Drift

PSI per feature (bins at the training deciles, missing as its own bin) for each week after
training, on the 30 features the model relies on most, plus PSI of the score itself and the
weekly fraud rate and PR-AUC. Labels arrive late in reality, so score PSI and feature PSI are the
early warning; PR-AUC is the lagging truth.

## 11. Laptop budget

24 GB unified memory, shared with everything else: float32 storage, models capped at 4 threads,
heavy steps under `nice`. No Docker for this project.
