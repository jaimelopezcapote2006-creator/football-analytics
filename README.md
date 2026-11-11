# Football Analytics — research pipeline

This repository runs a reproducible, chronological football forecasting backtest. It does **not** claim betting profitability or scrape bookmaker prices; it can evaluate verified odds supplied in the documented CSV format.

## Data
- Openplay release `data-v0.1.0`: https://github.com/nlbair/openplay/releases/tag/data-v0.1.0
- Downloaded at run time by GitHub Actions; Parquet files are not committed to this repository.
- Initial scope: Premier League, LaLiga, Bundesliga, Serie A, Ligue 1, Champions League and Europa League (where present in the release).
- Uses 90-minute scores for forecasting targets. Extra-time/shootout outcomes are not used.
- Data coverage and known issues are included in the run report. Openplay attribution: Noah Bair / Openplay, CC BY 4.0 (verify upstream terms before redistribution).

## Models
- **V1.4 Elo:** chronological Elo ratings with home advantage, draw factor fitted on training only.
- **V1.5 ensemble:** fixed 50/50 blend of V1.4 Elo and V1.2 Gradient Boosting for 1X2; the blend is not tuned on the holdout.
- **Multi-market count models:** separate Poisson gradient-boosting forecasts for shots, shots on target, corners, cards and fouls, using each team's and opponent's recent match-stat history.
- **V1.0 baseline:** fixed-rate Poisson forecast from each team's previous 10 matches, blending attack and opponent defensive concession rates.
- **V1.1 experimental:** fitted PoissonRegressor models for home and away goals.
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
1X2 multiclass Brier and log loss; goal MAE; O/U 2.5 and BTTS Brier/log loss; MAE and Over/Under Brier/log loss for shots, shots on target, corners, cards and fouls. All models use a chronological holdout. If `data/odds.csv` contains verified bookmaker prices, the pipeline calculates fair odds, minimum entry odds for a 3% expected edge, flat-stake ROI and hit rate on matched test fixtures. It does not fabricate prices.

### Odds import format
Add one row per match and selection to `data/odds.csv` with columns `match_id,market,selection,odds,bookmaker,captured_at`. Example market keys: `1x2` with `home/draw/away`; `over25`, `under25`, `btts_yes`, `btts_no`; and `shots_22.5`, `sot_8.5`, `corners_9.5`, `cards_4.5`, `fouls_22.5` with `over/under`. Only use odds whose source, bookmaker and capture time are known. The current pipeline imports odds; it does not scrape Bet365.

## Important limitations
This is a first reproducible baseline, not the final multi-market model. It does not currently model lineups/injuries, referee effects or player props. Historical Bet365 prices and CLV require a licensed source or verified odds imports. The dataset has incomplete match coverage and known issues; the report must be read with its coverage audit. Do not use cup or cross-division matches to infer professional-league performance.
