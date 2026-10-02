# Project 21 — A Forecast That Survives Backtesting

**Do this after the eight lessons.**

You forecast something real, prove it beats the baseline across **eight backtest
folds**, give it calibrated intervals, and run it for a month against the
baseline in production.

As always, **"use the baseline"** is a passing conclusion — and on a forecasting
project it is a common correct one.

---

## Requirements

### 1. The decision, and the three numbers
- What decision the forecast feeds, who makes it, and what an error costs **in
  each direction**
- **Horizon, cadence and folds**, each justified by the business, not the data
- At least **two years** of history, or two full seasonal cycles
- `MIN USEFUL`: the MASE below which you will recommend the baseline, written
  first

### 2. All four baselines
- Mean, naive, drift, seasonal naive — each implemented and backtested
- The per-fold table for each
- **Which baseline is hardest to beat on your series, and why**

### 3. The model, evaluated honestly
- At least two models through the same pipeline
- **Backtested over 8+ rolling windows**, never a single split
- Report: **per-fold MASE, the mean, and folds-won** against the best baseline
- A paired comparison between your two models, with the per-fold differences
- An explicit statement of whether the difference exceeds the fold spread

### 4. Features without leakage
- Every feature derived from `y` is `shift`ed before rolling — show the code
- A deliberate leak: build the un-shifted version and **report how much it
  improves backtest error**. That number is your leak detector's calibration
- Every external feature: a line saying **how you will have it at forecast time**
- Recursive **and** direct compared at your real horizon

### 5. Seasonality
- The early/late spread ratio: additive or multiplicative?
- An autocorrelation plot to at least 2x your longest suspected period
- **One cycle you did not know about before plotting it** — or a statement that
  there wasn't one
- If relevant: a moving-holiday feature (Eid, Ramadan), with its measured effect

### 6. Intervals
- Built from **backtest** residuals, per step-ahead
- **Empirical coverage** reported at 50, 80, 90 and 95%
- The in-sample version built too, with its width, to show the gap
- The quantile the business decision actually needs (lesson 06), and why

### 7. Production, for at least a month
- The daily job, with input validation that **holds** rather than forecasting
  from broken history
- `forecast_date`, `target_date`, `model_version` and the input hash on every row
- **The baseline running alongside**, every day
- Forecasts scored h days later, against actuals
- A heartbeat alert, tested
- The monitoring table: rolling MASE, coverage, and the baseline comparison

### 8. The verdict

```text
RECOMMENDATION   Ship / Ship narrower / Use the baseline / Stop
AGAINST BASELINE MASE over 8 folds, and folds won
PRODUCTION       realised MASE over the month, against the baseline
INTERVALS        nominal vs empirical coverage
HORIZON          where the forecast stops being useful, with the number
WHAT IT COSTS    to run and to maintain
WHAT WOULD CHANGE THIS
OWNER AND REVIEW DATE
```

---

## Deliverables

```text
project-21/
├── VERDICT.md             the one page
├── framing.md             decision, horizon, cadence, MIN USEFUL — committed first
├── README.md
├── src/
│   ├── features.py        every y-derived feature shifted
│   ├── backtest.py        rolling-origin evaluation
│   ├── intervals.py       per-horizon residual quantiles
│   └── daily_job.py       validate, forecast, write, score
├── reports/
│   ├── baselines.md       all four, per fold
│   ├── backtest.md        per-fold table, folds won, paired comparison
│   ├── leakage.md         the un-shifted version's "improvement"
│   ├── seasonality.md     ratio, autocorrelation, the cycle you found
│   ├── coverage.md        nominal vs empirical, backtest vs in-sample
│   └── production.md      one month, model vs baseline
└── forecasts/*.parquet    with forecast_date, model_version, input hash
```

---

## Marking

| Weight | Criterion |
|---|---|
| 10% | Decision, three numbers justified, `MIN USEFUL` written first |
| 10% | All four baselines backtested, hardest one identified |
| 20% | 8+ fold backtest with per-fold table, folds-won, and a paired comparison |
| 15% | No leakage, with the deliberate-leak number as evidence |
| 10% | Seasonality analysed, with a cycle found or ruled out |
| 15% | Intervals from backtest residuals, with empirical coverage |
| 15% | A month in production, with the baseline alongside and forecasts scored |
| 5% | The verdict, with "use the baseline" available |

Automatic deductions: a single train/test split; random or shuffled CV; a
rolling feature without `shift`; MAPE as the only metric; intervals from
in-sample residuals; no baseline in production; forecasts overwritten rather
than appended; a claimed improvement smaller than the fold spread.

---

## Milestones

| Day | Done by end of day |
|---|---|
| 1 | Framing committed. Data acquired. All four baselines backtested |
| 2 | Features with shifts; the deliberate-leak measurement |
| 3 | Seasonality: ratio, autocorrelation, moving holidays |
| 4 | Two models, 8-fold backtest, paired comparison |
| 5 | Recursive vs direct at the real horizon; intervals and coverage |
| 6 | The daily job, validation, monitoring, baseline alongside |
| 7 | Verdict and handover (then one month of production before final marking) |

---

## Before you submit

- [ ] `framing.md` predates every modelling commit
- [ ] All four baselines are backtested, not just mentioned
- [ ] The backtest has **8+ folds**, with a per-fold table
- [ ] Folds-won is reported, not only the mean
- [ ] The un-shifted leak was measured, and the number is in the report
- [ ] Recursive and direct were compared at the real horizon
- [ ] Intervals come from backtest residuals, **per step-ahead**
- [ ] Empirical coverage is reported at four levels
- [ ] The baseline runs in production alongside the model
- [ ] Every forecast row has `forecast_date`, `model_version` and an input hash
- [ ] Forecasts from h days ago have been scored against actuals
- [ ] `VERDICT.md` opens with the recommendation, and "use the baseline" was available
