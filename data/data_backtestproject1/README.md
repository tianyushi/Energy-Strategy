# Refinery LS Strategy — Data Export for External Model

Self-contained data bundle for an external AI model to fit / backtest a US refiner long/short strategy. Per-contract futures format (no roll discontinuities), plus equity OHLCV.

---

## Folder layout

```
data/
├── train/                  trade_date ≤ 2021-12-31  (≈ 7 years)
│   ├── futures/            274 per-contract CSVs (CL_*, RB_*, HO_*)
│   ├── VLO_daily.csv       7 refiner equities (OHLCV daily)
│   ├── MPC_daily.csv
│   ├── PSX_daily.csv
│   ├── DINO_daily.csv
│   ├── PBF_daily.csv
│   ├── DK_daily.csv
│   ├── CVI_daily.csv
│   ├── XLE_daily.csv       energy sector ETF
│   └── SPY_daily.csv       broad market
├── test/                   trade_date ≥ 2022-01-01  (≈ 4.3 years through 2026-05-05)
│   ├── futures/            284 per-contract CSVs
│   └── <same 9 equity files>
└── expiry_list.csv         master list of all 537 contracts + expiry dates
```

Total bundle size: **8.3 MB**.

---

## Per-contract futures format (CL / RB / HO)

**Filename convention**: `{commodity}_{YYYY}_{MM}.csv`
e.g. `CL_2025_02.csv` = WTI Crude futures contract expiring in February 2025.

**Each file**:
- Trimmed to the **last 150 trading days before expiry** (or fewer if the contract is still active or starts after 2015-01-01)
- Split by `trade_date`: rows ≤ 2021-12-31 go to `train/futures/`, rows ≥ 2022-01-01 go to `test/futures/`
- A contract whose 150-day window straddles the boundary appears in **both folders** with disjoint trade-date rows (e.g. `CL_2022_04.csv`: 95 rows in train, 55 rows in test). See `expiry_list.csv` for which files exist where.

**Columns**:

| Column | Type | Description |
|---|---|---|
| `date` (index) | datetime | Trading date (NYMEX session) |
| `settlement` | float | EOD settlement price in native units (see below) |
| `expiry_date` | str (ISO) | Last trade date of this contract (constant per file) |
| `days_to_expiry` | int | Calendar days from `date` to `expiry_date` |
| `open_interest` | float | Open interest at EOD |
| `total_volume` | int | Daily volume |

**Native units**:

| Commodity | Unit | To convert to $/bbl |
|---|---|---|
| CL (WTI Crude) | USD / barrel | already $/bbl |
| RB (RBOB Gasoline) | USD / gallon | multiply by 42 |
| HO (ULSD Heating Oil) | USD / gallon | multiply by 42 |

**3:2:1 NYMEX crack spread in $/bbl** (for a given delivery month):
```
crack_321 = (2 × RB_settle × 42 + HO_settle × 42) / 3 − CL_settle
```
This gives a $/bbl refining margin. Typically $5-$25/bbl.

**Sample file** — `train/futures/CL_2020_07.csv`:

```
date,settlement,expiry_date,days_to_expiry,open_interest,total_volume
2019-11-15,55.67,2020-06-22,220,67598.0,12671
2019-11-18,55.02,2020-06-22,217,67698.0,3540
...
2020-06-19,39.75,2020-06-22,3,19690.0,64113
2020-06-22,40.46,2020-06-22,0,877.0,29891
```

---

## Why per-contract format (and not continuous M3/M6/M9 promptness)

Continuous-promptness series (M3/M6/M9 = 3rd/6th/9th-nearby contract) periodically *re-target* to a different forward delivery month as the front contract expires. Computing returns or rolling means on a raw constant-promptness series injects **non-economic jumps at every roll date** (~monthly per leg).

**Roll-jump magnitudes on continuous-promptness data** (which is why we're NOT using that format):

| Series | # Rolls (7y) | Median \|pct_change\| at roll | p90 | Max |
|---|---:|---:|---:|---:|
| CL_M6 promptness | 86 | 1.30% | 3.25% | 12.6% |
| RB_M6 promptness | 83 | 2.36% | 8.09% | 14.9% |
| HO_M6 promptness | 83 | 1.58% | 3.05% | 6.8% |

Worse, **CL, RB, HO roll on different days** (CL expires ~25th of prior month; RB/HO last business day of prior month). On 17.7% of trading days the three legs are pointing at different forward months — so even building the crack spread first does not eliminate the issue. Only 1 of 168 rolls in 7 years had all 3 legs roll simultaneously.

**Per-contract format eliminates all of this** because every file contains one and only one physical contract — there is nothing to roll.

---

## Reconstructing M3 / M6 / M9 (if needed)

For each `trade_date`:
- Identify the contracts active on that date (where `date >= trade_date` and `date <= expiry_date`)
- Sort by `expiry_date` ascending → indices 0, 1, 2, ... correspond to M1, M2, M3, ...
- M3 = the 3rd-soonest expiry. M6 = 6th. M9 = 9th.

Pseudocode:
```python
def get_M_settlement(trade_date, commodity, M):
    """Returns the settlement of the M-th nearby contract on a given trade_date."""
    candidates = []
    for path in glob(f"data/{split}/futures/{commodity}_*.csv"):
        df = pd.read_csv(path, parse_dates=["date", "expiry_date"]).set_index("date")
        if trade_date in df.index and df.loc[trade_date, "expiry_date"] >= trade_date:
            candidates.append((df.loc[trade_date, "expiry_date"], df.loc[trade_date, "settlement"]))
    candidates.sort()  # by expiry ascending
    return candidates[M-1][1] if len(candidates) >= M else None
```

`expiry_list.csv` provides expiry dates for all 537 contracts so you can pre-build a lookup table.

---

## Equity files — `{ticker}_daily.csv`

OHLCV daily bars, split-adjusted (not dividend-adjusted). Same in `train/` and `test/` folders, just sliced by `date`.

```
date,Open,High,Low,Close,Volume
2015-01-02,32.115308,32.705114,31.804201,32.627338,5897000
```

**B7 cap-weighted refiner basket** (production weights):

| Ticker | Name | Weight |
|---|---|---:|
| **VLO** | Valero | 25% |
| **MPC** | Marathon Petroleum | 25% |
| **PSX** | Phillips 66 | 25% |
| DINO | HF Sinclair | 10% |
| PBF | PBF Energy | 5% |
| DK | Delek US | 5% |
| CVI | CVR Energy | 5% |
| | **Total** | **100%** |

Daily basket return = Σ wᵢ × pct_change(Closeᵢ). All 7 names have full coverage 2015+.

`XLE_daily.csv` and `SPY_daily.csv` are included for sector and market hedging respectively.

---

## `expiry_list.csv` — master contract index

Columns:

| Column | Description |
|---|---|
| `commodity` | CL / RB / HO |
| `contract_year`, `contract_month` | Identifies the contract |
| `expiry_date` | Last trade date |
| `first_trade_date_in_bundle` | First date in the 150-day-trimmed file |
| `last_trade_date_in_bundle` | Last date (≤ expiry, typically equals expiry) |
| `n_rows_total` | 150 except for contracts that start after 2015 or extend past 2026-05 |
| `n_rows_train` | rows with `trade_date ≤ 2021-12-31` |
| `n_rows_test` | rows with `trade_date ≥ 2022-01-01` |

**Contract counts**:

| Commodity | # Contracts | Total Rows | Train Rows | Test Rows |
|---|---:|---:|---:|---:|
| CL | 201 | 28,240 | 12,638 | 15,602 |
| RB | 167 | 24,554 | 12,604 | 11,950 |
| HO | 169 | 24,688 | 12,604 | 12,084 |

CL has more contracts than RB/HO because it lists Dec contracts ~10 years forward; RB/HO list ~3-4 years forward.

---

## In-house production strategy (for context / benchmark)

The deterministic baseline that the in-house team built using this same data:

**Setup**:
1. Per `trade_date`, build the M6 (6th-nearby) 3:2:1 crack spread: `(2*RB_M6 + HO_M6)*42/3 − CL_M6`
2. Compute the 10-day SMA of the M6 crack — *grouped by (CL_contract, RB_contract, HO_contract) triple* so the SMA doesn't smear across roll dates
3. Raw signal: `crack_M6_today > SMA10_today` (boolean)
4. T1 hold-through confirmation: confirm a signal flip after 2 consecutive same-sign days
5. Position: +1 long basket / −1 short basket, lagged 1 day vs signal (MOC entry next day)
6. Hedge: short β × SPY, where β is rolling 60-120d daily β (or finer-grained intraday-β if you have it)

**Headline (2012-2026, $10M notional, in-house intraday-β hedge)**:
- Sharpe **+1.49**, Total **+$63.95M**, MaxDD −19.5%
- **14 of 15 positive years**; only 2019 was slightly negative (−$0.22M)
- 2022 specifically: SR +2.73 / +$10.5M (the year the colleague's hypothesis about refiner-vs-crack divergence is most testable)

Production OOS slice (2020-2026 only, where we have a clean walk-forward read): SR +1.80 / +$41.97M / 7 of 7 positive years.

---

## Suggested usage for an external model

1. **Build features per trade_date** by looking up M1...M9 contracts (using `expiry_list.csv` for the contract ordering). The simplest target is to predict the *sign* of the next-day basket return, or its magnitude.
2. **Hedge** by regressing basket returns on SPY returns over a rolling 60-120d window and subtracting β × SPY return.
3. **Train** on 2015-2021 data, **test** on 2022-2026.
4. **Report metrics** on test set: Sharpe (on invested days only — `pnl[pnl != 0]`), Total $ PnL on $10M notional, MaxDD, count of positive years.
5. **Walk-forward** (if your model supports it): the in-house default is 5-year rolling train / 1-year test; we've separately validated that this is the right cadence for picker-class models against 30 candidate signals.

---

## Reproducing the bundle

```bash
python _build_export.py
```

Idempotent. Reads from the in-house `data_cache/`, deletes & rebuilds `data/`.

---

## Data lineage

- **Futures**: internal CMDTYA EOD Explorer (CME official settlement series). Per-contract by `(contract_year, contract_month)`; expiry from CME's `last_trade_date` field (Globex calendar).
- **Equities**: Yahoo Finance daily bars, split-adjusted close.
- All files are timezone-naive, indexed on session date.

---

## Contact

Houston Products Desk — refiner basket strategy team.
