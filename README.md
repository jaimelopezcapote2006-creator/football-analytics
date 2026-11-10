# Football Analytics — research pipeline

This repository runs a reproducible, chronological football forecasting backtest. It does **not** claim betting profitability or provide bookmaker odds.

## Data
- Openplay release `data-v0.1.0`: https://github.com/nlbair/openplay/releases/tag/data-v0.1.0
- Downloaded at run time by GitHub Actions; Parquet files are not committed to this repository.
- Initial scope: Premier League, LaLiga, Bundesliga, Serie A, Ligue 1, Champions League and Europa League (where present in the release).
- Uses 90-minute scores for forecasting targets. Extra-time/shootout outcomes are not used.
- Data coverage and known issues are included in the run report. Openplay attribution: Noah Bair / Openplay, CC BY 4.0 (verify upstream terms before redistribution).

## Models
- **V1.0 baseline:** fixed-rate Poisson forecast from each team's previous 10 matches, blending attack and opponent defensive concession rates.
- **V1.1 experimental:** 55% last five + 45% previous five rolling form, with fitted PoissonRegressor models for home and away goals.
- Features are generated strictly before each match. Matches are sorted chronologically; no future match statistics enter a prediction.
- Chronological split: first 70% of eligible matches for fitting, final 30% for out-of-time evaluation. V1.0 remains a baseline; V1.1 is shadow/experimental.

## Run
GitHub Actions -> **Historical backtest** -> Run workflow. It also runs on pushes to the default branch and weekly.

The run creates:
- `summary.md`: aggregate metrics and data audit
- `predictions.csv`: out-of-time predictions and actual outcomes
- `football_analytics.sqlite`: local SQLite database of predictions and metrics

These are uploaded as a GitHub Actions artifact named `football-analytics-results`. The workflow does not commit raw datasets or model outputs into git.

## Metrics
1X2 multiclass Brier and log loss; goal MAE; O/U 2.5 and BTTS Brier/log loss. Calibration bins are reported. No ROI/CLV is calculated because this dataset does not provide verified historical Bet365 Spain prices.

## Important limitations
This is a first reproducible baseline, not the final multi-market model. It does not currently model corners, cards, fouls, lineups/injuries, referee effects, player props or odds. The dataset has incomplete match coverage and known issues; the report must be read with its coverage audit. Do not use cup or cross-division matches to infer professional-league performance.
