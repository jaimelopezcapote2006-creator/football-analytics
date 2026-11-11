#!/usr/bin/env python3
"""Chronological Poisson baseline backtest; all features are pre-match only."""
from __future__ import annotations
import argparse, json, math, sqlite3
from collections import defaultdict, deque
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, log_loss

ROOT = Path("data")
OUT = Path("results")
LEAGUES = {"eng", "esp", "ger", "ita", "fra", "ucl", "uel"}
MIN_TEAM_MATCHES = 5
MAX_GOALS = 12

def safe_float(x, default):
    try:
        v=float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default

def weighted_form(values):
    vals=list(values)
    if len(vals) >= 10:
        return 0.55*float(np.mean(vals[-5:])) + 0.45*float(np.mean(vals[-10:-5]))
    if vals:
        return float(np.mean(vals))
    return float("nan")

def score_probs(lh, la):
    h=np.arange(MAX_GOALS+1)
    ph=poisson.pmf(h, max(0.05,lh))
    pa=poisson.pmf(h, max(0.05,la))
    grid=np.outer(ph,pa)
    # Account for the tiny truncated tail by renormalizing.
    grid=grid/grid.sum()
    p1=float(np.tril(grid,-1).sum()) # row home goals > away? lower triangle actually row index > col
    px=float(np.trace(grid))
    p2=float(np.triu(grid,1).sum())
    return p1,px,p2,grid

def market_probs(grid):
    over=0.0; btts=0.0
    for h in range(grid.shape[0]):
        for a in range(grid.shape[1]):
            if h+a >= 3: over += grid[h,a]
            if h>0 and a>0: btts += grid[h,a]
    return float(over), float(1-over), float(btts), float(1-btts)

def brier_binary(p,y):
    return float(np.mean((np.asarray(p)-np.asarray(y))**2))

def calibration_rows(p,y,label):
    p=np.asarray(p); y=np.asarray(y); rows=[]
    edges=np.linspace(0,1,11)
    for i in range(10):
        mask=(p>=edges[i]) & ((p<edges[i+1]) if i<9 else (p<=edges[i+1]))
        if mask.any():
            rows.append({"market":label,"bin":f"{edges[i]:.1f}-{edges[i+1]:.1f}",
                         "n":int(mask.sum()),"mean_probability":float(p[mask].mean()),
                         "observed_rate":float(y[mask].mean())})
    return rows

def metrics_for(df, prefix):
    y1=(df.actual=="H").astype(int).to_numpy()
    yd=(df.actual=="D").astype(int).to_numpy()
    ya=(df.actual=="A").astype(int).to_numpy()
    probs=df[[f"{prefix}_p_home",f"{prefix}_p_draw",f"{prefix}_p_away"]].to_numpy()
    yidx=np.where(y1==1,0,np.where(yd==1,1,2))
    out={
      "matches":int(len(df)),
      "goals_mae":float(mean_absolute_error(df.actual_home_goals+df.actual_away_goals,
                                             df[f"{prefix}_lambda_home"]+df[f"{prefix}_lambda_away"])),
      "1x2_brier":float(np.mean(np.sum((probs-np.eye(3)[yidx])**2,axis=1))),
      "1x2_log_loss":float(log_loss(yidx,probs,labels=[0,1,2])),
    }
    for name,actual in [("over25",(df.actual_home_goals+df.actual_away_goals>=3).astype(int)),
                        ("btts",( (df.actual_home_goals>0)&(df.actual_away_goals>0)).astype(int))]:
        p=df[f"{prefix}_{name}_yes"].to_numpy()
        out[name+"_brier"]=brier_binary(p,actual)
        out[name+"_log_loss"]=float(log_loss(actual,np.clip(p,1e-8,1-1e-8),labels=[0,1]))
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--matches",default=str(ROOT/"matches.parquet"))
    ap.add_argument("--team-stats",default=str(ROOT/"team_match_stats.parquet"))
    ap.add_argument("--coverage",default=str(ROOT/"coverage.csv"))
    args=ap.parse_args()
    OUT.mkdir(exist_ok=True)
    matches=pd.read_parquet(args.matches)
    # Restrict to well-defined professional league/UEFA competitions in this first pass.
    matches=matches[matches["comp"].astype(str).isin(LEAGUES)].copy()
    if "status_code" in matches: matches=matches[matches.status_code==6]
    matches["date"]=pd.to_datetime(matches["date"],errors="coerce")
    matches=matches.dropna(subset=["date","home_score_90","away_score_90"])
    matches=matches.sort_values(["date","match_id"]).reset_index(drop=True)
    if len(matches)<500: raise RuntimeError(f"Insufficient eligible matches: {len(matches)}")

    team_stats=pd.read_parquet(args.team_stats)
    team_stats["date"]=pd.to_datetime(team_stats["date"],errors="coerce")
    # Only use stats for the same selected competition scope; aggregate to a team-match row.
    team_stats=team_stats[team_stats["comp"].astype(str).isin(LEAGUES)].copy()
    stat_lookup={}
    for r in team_stats.itertuples(index=False):
        stat_lookup[(int(r.match_id),int(r.team_id))]={
            "gf":float(r.goals_for),"ga":float(r.goals_against),
            "shots":float(r.shots),"sot":float(r.shots_on_target),
            "corners":float(r.corners_taken),"cards":float(r.yellow_cards+r.second_yellows+r.red_cards),
            "fouls":float(r.fouls_committed)
        }

    history=defaultdict(lambda: deque(maxlen=10))
    comp_history=defaultdict(lambda: deque(maxlen=300))
    records=[]
    # Process all fixtures on a calendar date as one batch. No result from a
    # same-day match is allowed to enter another same-day fixture's features.
    for day, day_matches in matches.groupby("date",sort=True):
        day_rows=list(day_matches.itertuples(index=False))
        for m in day_rows:
            hid=int(m.home_team_id); aid=int(m.away_team_id); comp=str(m.comp)
            hh=list(history[hid]); ah=list(history[aid])
            if len(hh)>=MIN_TEAM_MATCHES and len(ah)>=MIN_TEAM_MATCHES:
                ch=list(comp_history[comp])
                comp_home=float(np.mean([x[0] for x in ch])) if len(ch)>=20 else 1.45
                comp_away=float(np.mean([x[1] for x in ch])) if len(ch)>=20 else 1.20
                h_gf=float(np.mean([x["gf"] for x in hh])); h_ga=float(np.mean([x["ga"] for x in hh]))
                a_gf=float(np.mean([x["gf"] for x in ah])); a_ga=float(np.mean([x["ga"] for x in ah]))
                h_wgf=weighted_form([x["gf"] for x in hh]); h_wga=weighted_form([x["ga"] for x in hh])
                a_wgf=weighted_form([x["gf"] for x in ah]); a_wga=weighted_form([x["ga"] for x in ah])
                v10_h=max(.10, .5*h_gf + .5*a_ga)
                v10_a=max(.10, .5*a_gf + .5*h_ga)
                v10_h=max(.10,v10_h * (comp_home/max(.1,(comp_home+comp_away)/2)))
                v10_a=max(.10,v10_a * (comp_away/max(.1,(comp_home+comp_away)/2)))
                xh=[math.log1p(h_wgf),math.log1p(h_wga),math.log1p(a_wgf),math.log1p(a_wga),
                    math.log1p(comp_home),math.log1p(comp_away)]
                xa=[math.log1p(a_wgf),math.log1p(a_wga),math.log1p(h_wgf),math.log1p(h_wga),
                    math.log1p(comp_away),math.log1p(comp_home)]
                hg=int(m.home_score_90); ag=int(m.away_score_90)
                records.append({"match_id":int(m.match_id),"date":m.date.date().isoformat(),"competition":str(m.competition),
                    "comp":comp,"home_team":str(m.home_team),"away_team":str(m.away_team),
                    "actual_home_goals":hg,"actual_away_goals":ag,
                    "actual":"H" if hg>ag else ("D" if hg==ag else "A"),
                    "v10_lambda_home":v10_h,"v10_lambda_away":v10_a,
                    "v11_lambda_home":np.nan,"v11_lambda_away":np.nan,
                    "v12_lambda_home":np.nan,"v12_lambda_away":np.nan,
                    "features_home":xh,"features_away":xa})
        # Update histories only after every fixture on this date has been forecast.
        for m in day_rows:
            hid=int(m.home_team_id); aid=int(m.away_team_id); comp=str(m.comp)
            history[hid].append({"gf":float(m.home_score_90),"ga":float(m.away_score_90)})
            history[aid].append({"gf":float(m.away_score_90),"ga":float(m.home_score_90)})
            comp_history[comp].append((float(m.home_score_90),float(m.away_score_90)))

    df=pd.DataFrame(records)
    if len(df)<500: raise RuntimeError(f"Too few matches after warm-up: {len(df)}")
    # Split on a unique calendar date, never through fixtures played on the same day.
    unique_dates=sorted(df["date"].unique())
    split_date=unique_dates[int(len(unique_dates)*0.70)]
    train_mask=(df["date"]<split_date).to_numpy()
    test_mask=~train_mask
    split=int(train_mask.sum())
    if split<200 or int(test_mask.sum())<100: raise RuntimeError("Not enough rows for chronological train/test split")
    # Fit only on the first 70%; the final 30% is strictly out-of-time evaluation.
    Xh=np.asarray([r["features_home"] for r in records],dtype=float)
    Xa=np.asarray([r["features_away"] for r in records],dtype=float)
    yh=df.actual_home_goals.to_numpy(); ya=df.actual_away_goals.to_numpy()
    model_h=PoissonRegressor(alpha=0.15,max_iter=500).fit(Xh[train_mask],yh[train_mask])
    model_a=PoissonRegressor(alpha=0.15,max_iter=500).fit(Xa[train_mask],ya[train_mask])
    df["v11_lambda_home"]=np.clip(model_h.predict(Xh),.05,5.5)
    df["v11_lambda_away"]=np.clip(model_a.predict(Xa),.05,5.5)
    # V1.2: nonlinear Poisson boosting, fitted on the same chronological training dates.
    # It is a challenger, not an automatically promoted replacement.
    boost_h=HistGradientBoostingRegressor(loss="poisson",learning_rate=0.06,max_iter=180,
        max_leaf_nodes=15,l2_regularization=2.0,min_samples_leaf=40,random_state=42)
    boost_a=HistGradientBoostingRegressor(loss="poisson",learning_rate=0.06,max_iter=180,
        max_leaf_nodes=15,l2_regularization=2.0,min_samples_leaf=40,random_state=42)
    boost_h.fit(Xh[train_mask],yh[train_mask]); boost_a.fit(Xa[train_mask],ya[train_mask])
    df["v12_lambda_home"]=np.clip(boost_h.predict(Xh),.05,5.5)
    df["v12_lambda_away"]=np.clip(boost_a.predict(Xa),.05,5.5)

    for prefix in ["v10","v11","v12"]:
        p1=[];px=[];p2=[];o=[];u=[];b=[];nb=[]
        for lh,la in zip(df[f"{prefix}_lambda_home"],df[f"{prefix}_lambda_away"]):
            ph,pd_,pa,grid=score_probs(lh,la)
            po,pu,pb,pnb=market_probs(grid)
            p1.append(ph);px.append(pd_);p2.append(pa);o.append(po);u.append(pu);b.append(pb);nb.append(pnb)
        df[f"{prefix}_p_home"]=p1;df[f"{prefix}_p_draw"]=px;df[f"{prefix}_p_away"]=p2
        df[f"{prefix}_over25_yes"]=o;df[f"{prefix}_under25_yes"]=u
        df[f"{prefix}_btts_yes"]=b;df[f"{prefix}_btts_no"]=nb

    test=df.loc[test_mask].copy()
    metrics={"dataset":"Openplay data-v0.1.0","scope":sorted(LEAGUES),"eligible_matches_after_warmup":int(len(df)),
             "train_matches":int(split),"test_matches":int(len(test)),
             "train_start":str(df.iloc[0].date),"train_end":str(df.iloc[split-1].date),
             "test_start":str(df.iloc[split].date),"test_end":str(df.iloc[-1].date),
             "V1.0":metrics_for(test,"v10"),"V1.1_experimental":metrics_for(test,"v11"),"V1.2_gradient_boosting":metrics_for(test,"v12")}
    cal=[]
    for prefix in ["v10","v11","v12"]:
        cal += calibration_rows(test[f"{prefix}_over25_yes"],(test.actual_home_goals+test.actual_away_goals>=3),prefix+"_over2.5")
        cal += calibration_rows(test[f"{prefix}_btts_yes"],((test.actual_home_goals>0)&(test.actual_away_goals>0)),prefix+"_BTTS")
    cal += calibration_rows(test["v13_over25_yes"],(test.actual_home_goals+test.actual_away_goals>=3),"v13_over2.5")
    cal += calibration_rows(test["v13_btts_yes"],((test.actual_home_goals>0)&(test.actual_away_goals>0)),"v13_BTTS")
    if Path(args.coverage).exists():
        cov=pd.read_csv(args.coverage)
        cov=cov[cov.comp.astype(str).isin(LEAGUES)]
        metrics["coverage_rows"]=cov.to_dict(orient="records")
    metrics["data_quality_note"]="Openplay coverage is incomplete; see coverage.csv and data_issues.csv upstream. This first run evaluates only matches with available match/event-derived team stats and a 5-match team warm-up."
    metrics["not_in_scope"]=["corners/cards/fouls model calibration","lineups/injuries/referee effects","player props","Bet365 Spain odds/ROI/CLV"]
    (OUT/"metrics.json").write_text(json.dumps(metrics,indent=2,default=str),encoding="utf-8")
    (OUT/"calibration.csv").write_text(pd.DataFrame(cal).to_csv(index=False),encoding="utf-8")
    cols=[c for c in df.columns if not c.startswith("features_")]
    test[cols].to_csv(OUT/"predictions.csv",index=False)
    # Save a compact, queryable database with the out-of-time forecasts and metrics.
    with sqlite3.connect(OUT/"football_analytics.sqlite") as con:
        test[cols].to_sql("predictions",con,if_exists="replace",index=False)
        pd.DataFrame([{"model":"V1.0",**metrics["V1.0"]},{"model":"V1.1_experimental",**metrics["V1.1_experimental"]},{"model":"V1.2_gradient_boosting",**metrics["V1.2_gradient_boosting"]}]).to_sql("metrics",con,if_exists="replace",index=False)
        pd.DataFrame(cal).to_sql("calibration",con,if_exists="replace",index=False)

    def fmt(d):
        return "\n".join(f"- **{k}:** {v:.4f}" if isinstance(v,float) else f"- **{k}:** {v}" for k,v in d.items())
    report=f"""# Historical backtest results

- Dataset: Openplay `data-v0.1.0`
- Competitions: {', '.join(sorted(LEAGUES))}
- Eligible matches after team warm-up: **{len(df):,}**
- Chronological training set: **{split:,}** ({df.iloc[0].date} to {df.iloc[split-1].date})
- Out-of-time test set: **{len(test):,}** ({df.iloc[split].date} to {df.iloc[-1].date})

## V1.0 — fixed rolling Poisson baseline
{fmt(metrics['V1.0'])}

## V1.1 — experimental fitted Poisson model
{fmt(metrics['V1.1_experimental'])}

## Interpretation
Lower is better for MAE, Brier and log loss. This is a chronological holdout, not a random split. V1.1 is experimental and is not promoted automatically. Compare all three models and calibration; do not choose a winner from accuracy alone.

## Data and scope limitations
{metrics['data_quality_note']}

Not evaluated in this run: corners/cards/fouls calibration, referee/injury/lineup effects, player props, historical Bet365 Spain odds, ROI or CLV. No betting-value claims can be made from this report.

Openplay attribution: Noah Bair / Openplay. Source release: https://github.com/nlbair/openplay/releases/tag/data-v0.1.0
"""
    (OUT/"summary.md").write_text(report,encoding="utf-8")
    print(report)

if __name__=="__main__":
    main()
