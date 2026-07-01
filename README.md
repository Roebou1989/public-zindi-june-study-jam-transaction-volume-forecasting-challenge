# Nedbank Transaction Volume Forecasting - #4 Solution

Predicting how many banking transactions each customer will make in the next 3 months
(Nov 2015 - Jan 2016) from ~18M historical transactions. Zindi June Study Jam. Metric: **RMSLE**.

**Result: #4 on the private leaderboard.** Public LB **0.369841** (`panel_allcust_36seed.csv`).

Everything is in one place: a single, self-contained, well-commented notebook that explains the
approach in plain language and reproduces the exact submission file.

- **[`solution.ipynb`](solution.ipynb)** - read this. Runs top to bottom, no hidden imports.
- **[`SOLUTION.md`](SOLUTION.md)** - the written write-up (approach, and what worked / did not).

## The idea in 30 seconds

The competition is hard because the **training customers and the test customers are different
people** (disjoint). Ordinary cross-validation then misleads you - and in practice, better CV kept
producing worse leaderboard scores. The fix is a **multi-anchor panel** (a standard rolling-origin
time-series technique): instead of one row per customer as of Oct 2015, slide the "as-of" date
backward through history and make one row per (customer, month), each with its own realized
next-3-month target counted from the data. That turns ~8k rows into ~250k and forces the model to
learn a general "recent behaviour -> next quarter" rule instead of memorising specific customers.

On that panel we train one heavily-regularised **LightGBM** (L1 objective, `log1p` target) and
average it over **36 random seeds** to cancel noise. No GPU, no deep learning, no multi-model blend.

## Repo layout

```
solution.ipynb        # the self-contained solution notebook (start here)
SOLUTION.md           # written write-up + what worked / did not
requirements.txt      # pinned dependencies
submissions/
  panel_allcust_36seed.csv   # the exact #4 submission (public LB 0.369841)
data/README.md        # how to get the competition data (not bundled)
LICENSE               # MIT
```

## How to run

1. Put the competition data in `data/` (not bundled - see [`data/README.md`](data/README.md)).
2. `pip install -r requirements.txt` (Python 3.12).
3. Open `solution.ipynb` and **Run All**.
   - Default `RUN_TRAINING = False` loads and verifies the committed submission in seconds.
   - Set `RUN_TRAINING = True` to rebuild the panel from raw data and retrain all 36 seeds.

## Runtime and environment

<!-- RUNTIME -->
- **Environment:** Apple M5, 10-core CPU, 16 GB RAM, macOS, Python 3.12. **CPU only, no GPU.**
- **Full reproduction runtime:** ~107 minutes end-to-end - 10.7 min to build the panel/features +
  ~96 min for the 36 LightGBM seed fits + inference.

## Reproducibility note

LightGBM CPU training with multiple threads is not guaranteed bit-identical across runs, so a fresh
run reproduces the **leaderboard score** (the 36-seed mean is at the variance floor) and matches the
committed predictions to a small tolerance rather than byte-for-byte. We verified this: a full clean
re-run matched the committed file with **correlation 0.99991**, median per-prediction difference
**0.004** and RMS difference **0.016** (log space); a few customers differ more (max 0.46) as small
per-seed differences compound over a deep 3,000-tree ensemble. The committed
`submissions/panel_allcust_36seed.csv` is the exact file that scored on the leaderboard.

## Notes

- Only the competition-provided data was used - no external datasets.
- The multi-anchor panel is a standard rolling-origin technique for time-series forecasting; all
  findings in `SOLUTION.md` are from our own leaderboard experiments on this competition.
- Licence: MIT (see [`LICENSE`](LICENSE)).
