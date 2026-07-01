# #4 Solution write-up (Roebou) - Transaction Volume Forecasting

Thanks to Nedbank and Zindi for a genuinely interesting problem. Sharing my approach in case it
helps others. Full code + a single self-contained notebook that reproduces the submission is here:
**[GitHub repo](https://github.com/Roebou1989/public-zindi-june-study-jam-transaction-volume-forecasting-challenge)**.

## The thing that shaped everything: train and test customers are different people

The train and test customers are disjoint, so normal cross-validation is misleading - it tells you
how well you predict people like the ones you trained on, not unseen customers. For a long time my
better-CV models were my worse-leaderboard models. Once I accepted that, the whole strategy became
"do the things that help you generalise to strangers, and ignore the things that only help CV."

## What actually moved the needle: a multi-anchor panel

Instead of one row per customer as of Oct 2015, I slid the "as-of" month backwards through history
and made one row per (customer, month), each with the realized next-3-month count as its target
(you can just count it, since those windows are in the past). That turns ~8k rows into ~250k and
forces the model to learn a general "given this recent history, expect this many next quarter" rule
instead of memorising individual customers. I also included all customers (train + test) at the
historical anchors - completely legitimate, since only the Oct 2015 future label is hidden - which
puts the test customers' own past behaviour into training and directly attacks the disjoint-customer
problem. Small calendar features per anchor (does the next quarter include Nov/Dec/Jan, etc.) keep
the festive seasonality, and I weighted the real Oct-2015 anchor more heavily.

## The model: boring on purpose

A single LightGBM, L1 objective, trained on log1p of the target, heavily regularised. No blend, no
deep learning. My consistent experience was: more *data* (anchors, customers) helped the
leaderboard; more *capacity* (fancy features, heavy tuning, correlated ensembles) helped CV but hurt
the leaderboard. So the only extra lever I trusted at the end was pure variance reduction -
averaging the same model over 36 random seeds, which nudged the score down to its floor (0.369841).

## Things that looked good but lost on the leaderboard

Sharing these because the negative results were the most useful part:

- EWMA / decay recency features: better CV, worse LB.
- Heavier hyperparameter tuning: looked like a win at 3 seeds, evaporated at 24 (it was seed luck).
- Passing missing values as NaN instead of filling count features with 0: clearly worse (0 really
  does mean "no activity" here).
- Closing-balance "capacity" features: worse.

## Two practical reminders

- Submit `log1p(prediction)` - the platform scores against a log1p reference (raw counts score ~199).
- Manually select your two final submissions; the auto-pick is the best *public* score, which is not
  what you want on a shake-up-prone competition.

Happy to answer questions. Code and a fuller write-up are in the repo linked above. Thanks again to
Nedbank and Zindi, and congrats to everyone who stuck with this one - the shake-up was brutal.
