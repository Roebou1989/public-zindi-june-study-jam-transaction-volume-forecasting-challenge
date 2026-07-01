# Solution write-up - Nedbank Transaction Volume Forecasting (#4)

Public LB **0.369841**, private leaderboard **#4**. Metric: RMSLE. This is the narrative behind
[`solution.ipynb`](solution.ipynb), which runs end-to-end and reproduces the exact submission.

## The problem

For each customer, predict the number of banking transactions in the 3 months Nov 2015 - Jan 2016,
given ~18M historical transactions (Dec 2012 - Oct 2015) plus financial snapshots and demographics.
RMSLE grades error on a log scale, so getting the quiet and medium customers right matters far more
than the rare very-busy ones.

## What makes it hard: disjoint customers

The training customers (8,360, with labels) and the test customers (3,584, to predict) do not
overlap. So holding out training customers to validate yourself measures how well you predict
*people like the ones you already saw*, not *unseen strangers* - and strangers are what the
leaderboard grades. Concretely, for most of the competition **better cross-validation gave a worse
public leaderboard score**. Every decision below is a response to that.

## A scoring quirk worth knowing

The live platform scores plain RMSE against a reference stored as `log1p(true_count)`. In effect it
takes the log of the true answers before comparing, so you must submit `log1p(prediction)`, not raw
counts. We verified this early: raw counts scored ~199 (nonsense); the identical predictions as
`log1p` scored ~0.38.

## The idea that worked: a multi-anchor panel

The obvious setup is one row per customer, described as of 31 Oct 2015 - 8,360 rows. With a single
snapshot the model can quietly memorise those specific people, which does not transfer to disjoint
strangers.

A **multi-anchor panel** fixes this. It is a standard rolling-origin technique for time series: pick
many historical "anchor" dates, describe each customer using only what was known by that date, and
set the target to their realized next-3-month count (which you can simply count, because that window
is already in the past). One customer becomes many (customer, anchor) rows across seasons and
activity levels, so the model learns the general "given this recent history, expect this many next
quarter" rule instead of customer identity.

Our winning panel:

- **Real anchor 2015-10-31** (scored month): rows are the 8,360 train customers, labels from
  `Train.csv`. Test customers are excluded here (their future is the hidden target).
- **24 historical anchors** (month-ends 2013-10-31 to 2015-07-31): rows are **all 11,944 customers**
  (train and test), targets counted from observed transactions. This is not leakage - the whole
  3-month window at a historical anchor lies in the past relative to the Oct 2015 data cutoff - and
  it lets the model see the test customers' own historical behaviour, directly attacking the
  disjoint-customer problem.
- **Calendar features per anchor** (anchor month/year; whether the next 3 months include
  Nov/Dec/Jan; number of days/weekends) so seasonality is preserved across the stacked months.
- **Sample weights**: real 2015-10 anchor 3.0, other Octobers 1.5, everything else 1.0.

Result: ~250k rows x ~530 features, versus the 8,360-row snapshot we started from.

## The model, and the core finding

One **LightGBM** regressor, **L1 objective**, trained on `log1p(target)` with sample weights, and
heavily regularised (slow learning rate, humble trees, half the features/rows per tree, strong
L1/L2 penalties). Every knob fights memorisation, because of the finding that ran through all our
experiments:

> **Adding same-distribution data helps the leaderboard; adding model capacity (extra features,
> heavier tuning, ensembles of correlated models) improves cross-validation but hurts the
> leaderboard.**

On a disjoint-customer problem, extra capacity just gives the model more room to memorise training
customers. So once the panel was in place, the only leaderboard-safe lever left was **variance
reduction**: train the identical model over many random seeds and average. It converged as expected -
3 seeds 0.37069, 24 seeds 0.369928, 36 seeds **0.369841** (the variance floor). The winning file is
that 36-seed average.

## What we tried - worked and did not

**Worked (in the final solution):** the panel; including all customers at historical anchors;
seed averaging 3 -> 36; `log1p` submission; L1 objective; heavy regularisation.

**Did not work (CV said yes, leaderboard said no):**

| Change | Leaderboard |
| --- | --- |
| EWMA-decay recency features on the clean base | 0.373984 (worse) |
| Heavier tuning / regularisation sweep | looked like 0.369992 at 3 seeds, but 0.370582 at 24 (seed luck) |
| Passing missing values as NaN instead of median/0 fill | 0.377782 (+0.007 worse) |
| Closing-balance "capacity to spend" features | 0.374-0.378 (worse) |
| Deeper tenure + raw-space averaging | 0.375698 (worse) |

**Deliberately not pursued:** blending several correlated boosted-tree models (once each was 0.9+
correlated, blending stopped helping), and a fitted saturation-growth-curve baseline plus residual
(a different modelling approach we judged out of scope for the final days).

## The transferable lesson

On a disjoint-group transfer problem: trust the direction of the leaderboard over local
cross-validation, expand your *data* rather than your model's *capacity*, and denoise with seeds
before believing a small gain.

## Reproduce it

See [`solution.ipynb`](solution.ipynb) (set `RUN_TRAINING = True` and Run All) and the runtime /
reproducibility notes in the [README](README.md). Only the competition-provided data was used.
