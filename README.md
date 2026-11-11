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
1X2 multiclass Brier and log loss; goal MAE; O/U 2.5 and BTTS Brier/log loss; MAE and Over/Under Brier/log loss for shots, shots on target, corners, cards and fouls. All models use a chronological holdout. If `data/odds.csv` contains verified bookmaker prices, the pipeline calculates fair odds and minimum entry odds for a 3% expected edge for every supported market, then calculates flat-stake ROI and hit rate on matched test fixtures. The prediction export includes fair-odds columns even before a bookmaker price is supplied. It does not fabricate prices.

### Odds import format
Add one row per match and selection to `data/odds.csv` with columns `match_id,market,selection,odds,bookmaker,captured_at`. Example market keys: `1x2` with `home/draw/away`; `over25`, `under25`, `btts_yes`, `btts_no`; and `shots_22.5`, `sot_8.5`, `corners_9.5`, `cards_4.5`, `fouls_22.5` with `over/under`. Only use odds whose source, bookmaker and capture time are known. The current pipeline imports odds; it does not scrape Bet365.

## Important limitations
This is a first reproducible baseline, not the final multi-market model. It does not currently model lineups/injuries, referee effects or player props. Historical Bet365 prices and CLV require a licensed source or verified odds imports. The dataset has incomplete match coverage and known issues; the report must be read with its coverage audit. Do not use cup or cross-division matches to infer professional-league performance.


## New research phase: xG and current-season data
- The workflow additionally downloads Openplay `player_match_stats.parquet` and aggregates player xG/npxG and shot-related measures to team-match level.
- `src/xg_enhanced_backtest.py` builds richer pre-match features: rolling 5/10/20-match goals, xG, shots, shots on target, corners, fouls, variability, rest days and Elo strength. It compares an Elo/goals-only challenger against an xG/stat challenger on a validation period, selects only on validation, then reports results on a later untouched test period. It does not automatically replace the existing model.
- The research challenger now also evaluates both feature sets (goals/Elo and xG/stats/Elo) on the same untouched chronological test, under independent Poisson and Dixon-Coles score grids. Dixon-Coles rho is fitted from validation forecasts only. This is a shadow comparison: it does not change the production model or the existing picks.
- `src/open_matchday_ingest.py` downloads the current 2026/27 top-five-league results and fixture CSVs daily, writes an audit, and preserves missing statistics as null. These data are a separate current-season feed; they are not silently mixed into the Openplay historical training set because the provider IDs/schema differ and coverage is partial.
- Current-season files: `results/open_matchday_2627.csv`, `results/upcoming_fixtures_2627.csv`, `results/open_matchday_audit.json`.
- Additional challenger outputs: `results/xg_enhanced_metrics.json`, `results/xg_enhanced_predictions.csv`.
- Open Matchday is provided for non-commercial informational use; provider rights and source attribution remain applicable. Do not redistribute commercially.
- A model is only promoted if it beats simple baselines on a genuinely out-of-time test and remains calibrated. More training rows alone are not evidence of improvement.


## Optional odds snapshot (free tier)
- `src/fetch_eu_odds.py` can pull current EU bookmaker snapshots for 1X2 and totals from The Odds API.
- To enable it, add a GitHub Actions repository secret named `ODDS_API_KEY`. Without it, the workflow records a clear `not_configured` audit and continues.
- This is an EU market-consensus feed, **not Bet365 Spain**. The free feed's bookmaker coverage does not provide Bet365 Spain, and these snapshots are not yet converted into historical match-id rows for ROI. Bet365 prices must still be imported separately from verified captures.
- The provider advertises a free allowance; check current plan limits before enabling frequent pulls. API usage is reported in `results/odds_api_audit.json`.


## Upcoming fixture predictions
- `src/predict_upcoming.py` selects the xG/goals/Elo challenger on a validation period, refits it on all historical rows, replays 2026/27 current-season results and predicts upcoming top-five-league fixtures.
- Team-name matching is fuzzy but audited; unresolved clubs and fixtures without five historical matches are marked and excluded from recommendations.
- Outputs: `results/upcoming_predictions.csv`, `results/upcoming_prediction_audit.json`.
- Fair odds and minimum odds for a 3% expected edge are included for 1X2, O/U 1.5/2.5/3.5 and BTTS. These are model prices, not bookmaker prices.
- Verified odds can be entered in `data/upcoming_odds.csv`. Accepted markets: `1x2` + `home/draw/away`; `over_2.5` + `over`; `under_2.5` + `under`; `btts` + `yes/no`. The pipeline outputs expected value and a 3% edge flag in `results/upcoming_value_candidates.csv`. Never enter guessed prices.
