#!/usr/bin/env python3
"""Chronological Poisson baseline backtest; all features are pre-match only."""
from __future__ import annotations
import argparse, json, math, sqlite3
from collections import defaultdict, deque
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import poisson
from scipy.optimize import minimize_scalar
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
    vals=[float(v) for v in values if v is not None and np.isfinite(safe_float(v,np.nan))]
    if len(vals) >= 10:
        return 0.55*float(np.mean(vals[-5:])) + 0.45*float(np.mean(vals[-10:-5]))
    if vals:
        return float(np.mean(vals))
    return float("nan")

STAT_TARGETS = {
    "shots": "shots",
    "sot": "shots_on_target",
    "corners": "corners",
    "cards": "cards",
    "fouls": "fouls",
}
STAT_LINES = {
    "shots": [20.5, 21.5, 22.5, 23.5, 24.5],
    "sot": [6.5, 7.5, 8.5, 9.5],
    "corners": [7.5, 8.5, 9.5, 10.5, 11.5],
    "cards": [2.5, 3.5, 4.5, 5.5],
    "fouls": [19.5, 21.5, 22.5, 23.5, 25.5],
}

def stat_feature_vector(home_hist, away_hist):
    out=[]
    for key in STAT_TARGETS:
        out.extend([
            weighted_form([x.get(key+"_for",np.nan) for x in home_hist]),
            weighted_form([x.get(key+"_against",np.nan) for x in home_hist]),
            weighted_form([x.get(key+"_for",np.nan) for x in away_hist]),
            weighted_form([x.get(key+"_against",np.nan) for x in away_hist]),
        ])
    return out

def add_stat_market_probabilities(frame, stat, total_lambda):
    for line in STAT_LINES[stat]:
        suffix=str(line).replace(".","_")
        over=f"{stat}_over_{suffix}"
        prob=np.clip(poisson.sf(int(math.floor(line)),np.maximum(0.01,total_lambda)),0,1)
        frame[over]=prob
        frame[f"{stat}_under_{suffix}"]=1.0-prob

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

def dc_grid(lh, la, rho):
    """Dixon-Coles low-score correction over a truncated independent-Poisson grid."""
    h=np.arange(MAX_GOALS+1)
    ph=poisson.pmf(h, max(0.05,float(lh)))
    pa=poisson.pmf(h, max(0.05,float(la)))
    grid=np.outer(ph,pa)
    # tau(x,y): 00, 01, 10 and 11 are the only adjusted scorelines.
    corrections={
        (0,0): 1.0-float(lh)*float(la)*float(rho),
        (0,1): 1.0+float(lh)*float(rho),
        (1,0): 1.0+float(la)*float(rho),
        (1,1): 1.0-float(rho),
    }
    for (hg,ag),factor in corrections.items():
        grid[hg,ag] *= max(1e-10,factor)
    total=float(grid.sum())
    return grid/total if total>0 else np.outer(ph/ph.sum(),pa/pa.sum())

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
    ap.add_argument("--odds",default=str(ROOT/"odds.csv"))
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
            "gf":safe_float(r.goals_for,np.nan),"ga":safe_float(r.goals_against,np.nan),
            "shots":safe_float(r.shots,np.nan),"sot":safe_float(r.shots_on_target,np.nan),
            "corners":safe_float(r.corners_taken,np.nan),
            "cards":safe_float(r.yellow_cards,np.nan)+safe_float(r.second_yellows,0.0)+safe_float(r.red_cards,0.0),
            "fouls":safe_float(r.fouls_committed,np.nan)
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
                hs=stat_lookup.get((int(m.match_id),hid),{})
                aws=stat_lookup.get((int(m.match_id),aid),{})
                sfh=stat_feature_vector(hh,ah)
                sfa=stat_feature_vector(ah,hh)
                rec={"match_id":int(m.match_id),"date":m.date.date().isoformat(),"competition":str(m.competition),
                    "comp":comp,"home_team":str(m.home_team),"away_team":str(m.away_team),
                    "actual_home_goals":hg,"actual_away_goals":ag,
                    "actual":"H" if hg>ag else ("D" if hg==ag else "A"),
                    "v10_lambda_home":v10_h,"v10_lambda_away":v10_a,
                    "v11_lambda_home":np.nan,"v11_lambda_away":np.nan,
                    "v12_lambda_home":np.nan,"v12_lambda_away":np.nan,
                    "features_home":xh,"features_away":xa,
                    "features_stat_home":sfh,"features_stat_away":sfa}
                for sk in STAT_TARGETS:
                    rec[f"actual_{sk}_home"]=safe_float(hs.get(sk),np.nan)
                    rec[f"actual_{sk}_away"]=safe_float(aws.get(sk),np.nan)
                records.append(rec)
        # Update histories only after every fixture on this date has been forecast.
        for m in day_rows:
            hid=int(m.home_team_id); aid=int(m.away_team_id); comp=str(m.comp)
            hs=stat_lookup.get((int(m.match_id),hid),{})
            aws=stat_lookup.get((int(m.match_id),aid),{})
            home_hist={"gf":float(m.home_score_90),"ga":float(m.away_score_90)}
            away_hist={"gf":float(m.away_score_90),"ga":float(m.home_score_90)}
            for sk in STAT_TARGETS:
                home_hist[sk+"_for"]=safe_float(hs.get(sk),np.nan)
                home_hist[sk+"_against"]=safe_float(aws.get(sk),np.nan)
                away_hist[sk+"_for"]=safe_float(aws.get(sk),np.nan)
                away_hist[sk+"_against"]=safe_float(hs.get(sk),np.nan)
            history[hid].append(home_hist)
            history[aid].append(away_hist)
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

    # V1.5: fixed 50/50 blend of Elo and Gradient Boosting probabilities.
    # The blend weight is fixed in advance to avoid tuning on the holdout.
    # (The Elo probabilities are assigned below; this section is completed after Elo fit.)

    # Multi-market count models: separate Poisson boosting models for each side.
    Xstat_h=np.column_stack([Xh,np.asarray([r["features_stat_home"] for r in records],dtype=float),
                            np.asarray([r["features_stat_away"] for r in records],dtype=float)])
    Xstat_a=np.column_stack([Xa,np.asarray([r["features_stat_away"] for r in records],dtype=float),
                            np.asarray([r["features_stat_home"] for r in records],dtype=float)])
    stat_model_metrics={}
    for sk in STAT_TARGETS:
        ysh=df[f"actual_{sk}_home"].to_numpy(dtype=float)
        ysa=df[f"actual_{sk}_away"].to_numpy(dtype=float)
        mh=train_mask & np.isfinite(ysh)
        ma=train_mask & np.isfinite(ysa)
        if mh.sum()<200 or ma.sum()<200:
            stat_model_metrics[sk]={"status":"insufficient_data","train_home":int(mh.sum()),"train_away":int(ma.sum())}
            continue
        sh=HistGradientBoostingRegressor(loss="poisson",learning_rate=0.05,max_iter=140,
            max_leaf_nodes=15,l2_regularization=3.0,min_samples_leaf=35,random_state=42)
        sa=HistGradientBoostingRegressor(loss="poisson",learning_rate=0.05,max_iter=140,
            max_leaf_nodes=15,l2_regularization=3.0,min_samples_leaf=35,random_state=42)
        sh.fit(Xstat_h[mh],ysh[mh]); sa.fit(Xstat_a[ma],ysa[ma])
        df[f"{sk}_lambda_home"]=np.clip(sh.predict(Xstat_h),0.05,45.0)
        df[f"{sk}_lambda_away"]=np.clip(sa.predict(Xstat_a),0.05,45.0)
        df[f"{sk}_lambda_total"]=df[f"{sk}_lambda_home"]+df[f"{sk}_lambda_away"]
        add_stat_market_probabilities(df,sk,df[f"{sk}_lambda_total"].to_numpy())
        mt=test_mask & np.isfinite(ysh) & np.isfinite(ysa)
        actual_total=ysh+ysa
        pred_total=df[f"{sk}_lambda_total"].to_numpy()
        stat_model_metrics[sk]={"status":"ok","test_matches":int(mt.sum()),
            "home_mae":float(mean_absolute_error(ysh[mt & test_mask],df.loc[mt,f"{sk}_lambda_home"])),
            "away_mae":float(mean_absolute_error(ysa[mt & test_mask],df.loc[mt,f"{sk}_lambda_away"])),
            "total_mae":float(mean_absolute_error(actual_total[mt],pred_total[mt]))}
        for line in STAT_LINES[sk]:
            suffix=str(line).replace(".","_")
            p=df[f"{sk}_over_{suffix}"].to_numpy()[mt]
            actual=(actual_total[mt]>line).astype(int)
            stat_model_metrics[sk][f"over_{line}_brier"]=brier_binary(p,actual)
            stat_model_metrics[sk][f"over_{line}_log_loss"]=float(log_loss(actual,np.clip(p,1e-8,1-1e-8),labels=[0,1]))
    for prefix in ["v10","v11","v12"]:
        p1=[];px=[];p2=[];o=[];u=[];b=[];nb=[]
        for lh,la in zip(df[f"{prefix}_lambda_home"],df[f"{prefix}_lambda_away"]):
            ph,pd_,pa,grid=score_probs(lh,la)
            po,pu,pb,pnb=market_probs(grid)
            p1.append(ph);px.append(pd_);p2.append(pa);o.append(po);u.append(pu);b.append(pb);nb.append(pnb)
        df[f"{prefix}_p_home"]=p1;df[f"{prefix}_p_draw"]=px;df[f"{prefix}_p_away"]=p2
        df[f"{prefix}_over25_yes"]=o;df[f"{prefix}_under25_yes"]=u
        df[f"{prefix}_btts_yes"]=b;df[f"{prefix}_btts_no"]=nb

    # V1.3 Dixon-Coles: fit rho on training dates only.
    # Import local defensivo para evitar problemas de resolución del import global en runners.
    from scipy.optimize import minimize_scalar
    def dc_nll(rho):
        loss=0.0
        for row in df.loc[train_mask].itertuples(index=False):
            g=dc_grid(row.v10_lambda_home,row.v10_lambda_away,float(rho))
            loss-=math.log(max(1e-12,g[int(row.actual_home_goals),int(row.actual_away_goals)]))
        return loss
    dc_rho=float(minimize_scalar(dc_nll,bounds=(-0.15,0.15),method="bounded").x)
    dch=[]; dcd=[]; dca=[]; dco=[]; dcb=[]
    for row in df.itertuples(index=False):
        g=dc_grid(row.v10_lambda_home,row.v10_lambda_away,dc_rho)
        dch.append(float(np.tril(g,-1).sum())); dcd.append(float(np.trace(g))); dca.append(float(np.triu(g,1).sum()))
        ov,_,bt,_=market_probs(g); dco.append(ov); dcb.append(bt)
    df["v13_lambda_home"]=df["v10_lambda_home"]; df["v13_lambda_away"]=df["v10_lambda_away"]
    df["v13_p_home"]=dch; df["v13_p_draw"]=dcd; df["v13_p_away"]=dca
    df["v13_over25_yes"]=dco; df["v13_btts_yes"]=dcb
    # V1.4 Elo snapshots: all matches on a date see pre-date ratings.
    elo=defaultdict(lambda:1500.0); pre={}
    for day, batch_df in matches.groupby("date",sort=True):
        batch=list(batch_df.itertuples(index=False))
        for m in batch: pre[int(m.match_id)]=(elo[int(m.home_team_id)],elo[int(m.away_team_id)])
        for m in batch:
            h=int(m.home_team_id); a=int(m.away_team_id); rh=elo[h]; ra=elo[a]
            gd=abs(float(m.home_score_90)-float(m.away_score_90))
            score=1.0 if m.home_score_90>m.away_score_90 else (0.5 if m.home_score_90==m.away_score_90 else 0.0)
            exp=1.0/(1.0+10.0**(-((rh+65.0)-ra)/400.0))
            change=20.0*(1.0+0.2*math.log1p(gd))*(score-exp)
            elo[h]=rh+change; elo[a]=ra-change
    eh=np.asarray([pre[int(mid)][0] for mid in df.match_id]); ea=np.asarray([pre[int(mid)][1] for mid in df.match_id])
    strength=10.0**((eh+65.0-ea)/400.0)
    yi=np.where(df.actual.to_numpy()=="H",0,np.where(df.actual.to_numpy()=="D",1,2))
    def elo_loss(log_draw):
        draw=math.exp(float(log_draw)); h=strength[train_mask]; d=draw*np.sqrt(h)
        p=np.column_stack([h,d,np.ones_like(h)]); p/=p.sum(axis=1,keepdims=True)
        return float(-np.log(np.clip(p[np.arange(len(p)),yi[train_mask]],1e-12,1)).sum())
    draw_factor=math.exp(float(minimize_scalar(elo_loss,bounds=(-3,1),method="bounded").x))
    ed=draw_factor*np.sqrt(strength); ep=np.column_stack([strength,ed,np.ones_like(strength)]); ep/=ep.sum(axis=1,keepdims=True)
    df["v14_p_home"]=ep[:,0]; df["v14_p_draw"]=ep[:,1]; df["v14_p_away"]=ep[:,2]
    df["v15_lambda_home"]=df["v12_lambda_home"]; df["v15_lambda_away"]=df["v12_lambda_away"]
    df["v15_p_home"]=0.5*df["v12_p_home"]+0.5*df["v14_p_home"]
    df["v15_p_draw"]=0.5*df["v12_p_draw"]+0.5*df["v14_p_draw"]
    df["v15_p_away"]=0.5*df["v12_p_away"]+0.5*df["v14_p_away"]
    df["v15_over25_yes"]=df["v12_over25_yes"]; df["v15_btts_yes"]=df["v12_btts_yes"]

    test=df.loc[test_mask].copy()
    for col in [c for c in df.columns if c.startswith(("v13_","v14_","v15_")) or
                c.endswith(("_lambda_home","_lambda_away","_lambda_total")) or
                any(c.startswith(sk+"_") for sk in STAT_TARGETS)]:
        test[col]=df.loc[test_mask,col].to_numpy()
    metrics={"dataset":"Openplay data-v0.1.0","scope":sorted(LEAGUES),"eligible_matches_after_warmup":int(len(df)),
             "train_matches":int(split),"test_matches":int(len(test)),
             "train_start":str(df.iloc[0].date),"train_end":str(df.iloc[split-1].date),
             "test_start":str(df.iloc[split].date),"test_end":str(df.iloc[-1].date),
             "V1.0":metrics_for(test,"v10"),"V1.1_experimental":metrics_for(test,"v11"),"V1.2_gradient_boosting":metrics_for(test,"v12"), "V1.3_dixon_coles":metrics_for(test,"v13"), "V1.4_elo_1x2":{"matches":int(len(test)),"1x2_brier":float(np.mean(np.sum((test[["v14_p_home","v14_p_draw","v14_p_away"]].to_numpy()-np.eye(3)[np.where(test.actual.to_numpy()=="H",0,np.where(test.actual.to_numpy()=="D",1,2))])**2,axis=1))),"1x2_log_loss":float(log_loss(np.where(test.actual.to_numpy()=="H",0,np.where(test.actual.to_numpy()=="D",1,2)),test[["v14_p_home","v14_p_draw","v14_p_away"]].to_numpy(),labels=[0,1,2]))},
             "V1.5_elo_gradient_ensemble":metrics_for(test,"v15"),"stat_models":stat_model_metrics}
    cal=[]
    for prefix in ["v10","v11","v12"]:
        cal += calibration_rows(test[f"{prefix}_over25_yes"],(test.actual_home_goals+test.actual_away_goals>=3),prefix+"_over2.5")
        cal += calibration_rows(test[f"{prefix}_btts_yes"],((test.actual_home_goals>0)&(test.actual_away_goals>0)),prefix+"_BTTS")
    cal += calibration_rows(test["v13_over25_yes"],(test.actual_home_goals+test.actual_away_goals>=3),"v13_over2.5")
    cal += calibration_rows(test["v13_btts_yes"],((test.actual_home_goals>0)&(test.actual_away_goals>0)),"v13_BTTS")
    odds_path=Path(args.odds)
    odds_summary={"status":"not_provided","message":"No verified bookmaker odds file supplied; ROI/CLV not calculated."}
    if odds_path.exists() and odds_path.stat().st_size>0:
        odds=pd.read_csv(odds_path)
        required={"match_id","market","selection","odds"}
        if not required.issubset(odds.columns):
            raise ValueError(f"Odds file must contain columns: {sorted(required)}")
        odds["match_id"]=pd.to_numeric(odds["match_id"],errors="coerce")
        odds["odds"]=pd.to_numeric(odds["odds"],errors="coerce")
        odds["market"]=odds["market"].astype(str).str.lower().str.strip()
        odds["selection"]=odds["selection"].astype(str).str.lower().str.strip()
        pred=test.set_index("match_id")
        value_rows=[]
        for q in odds.itertuples(index=False):
            if pd.isna(q.match_id) or q.match_id not in pred.index or not np.isfinite(q.odds) or q.odds<=1:
                continue
            row=pred.loc[q.match_id]
            market=q.market; sel=q.selection
            p=np.nan; won=False; actual_label=""
            if market=="1x2" and sel in {"home","draw","away"}:
                p=float(row[{"home":"v15_p_home","draw":"v15_p_draw","away":"v15_p_away"}[sel]])
                actual_label=str(row.actual).lower(); won=(actual_label==sel)
            elif market in {"over25","under25","btts_yes","btts_no"}:
                if market=="over25": p=float(row.v15_over25_yes); won=(row.actual_home_goals+row.actual_away_goals>=3)
                elif market=="under25": p=1-float(row.v15_over25_yes); won=(row.actual_home_goals+row.actual_away_goals<3)
                elif market=="btts_yes": p=float(row.v15_btts_yes); won=(row.actual_home_goals>0 and row.actual_away_goals>0)
                else: p=1-float(row.v15_btts_yes); won=(row.actual_home_goals==0 or row.actual_away_goals==0)
            else:
                parts=market.rsplit("_",1)
                if len(parts)==2 and parts[0] in STAT_TARGETS and sel in {"over","under"}:
                    sk=parts[0]
                    try: line=float(parts[1].replace("_","."))
                    except Exception: continue
                    suffix=str(line).replace(".","_")
                    probcol=f"{sk}_{sel}_{suffix}"
                    actual_total=row.get(f"actual_{sk}_home",np.nan)+row.get(f"actual_{sk}_away",np.nan)
                    if probcol not in row.index or not np.isfinite(actual_total): continue
                    p=float(row[probcol]); won=(actual_total>line) if sel=="over" else (actual_total<line)
            if not np.isfinite(p) or p<=0 or p>=1: continue
            fair=1.0/p; min_entry=1.03/p; ev=p*float(q.odds)-1.0
            value_rows.append({"match_id":int(q.match_id),"date":row.date,"home_team":row.home_team,"away_team":row.away_team,
                "market":market,"selection":sel,"bookmaker":getattr(q,"bookmaker","unknown"),
                "odds":float(q.odds),"model_probability":p,"fair_odds":fair,"minimum_odds_3pct_edge":min_entry,
                "expected_value":ev,"won":int(won),"value_bet":bool(float(q.odds)>=min_entry)})
        value_df=pd.DataFrame(value_rows)
        value_df.to_csv(OUT/"odds_evaluation.csv",index=False)
        chosen=value_df[value_df.value_bet] if len(value_df) else value_df
        if len(chosen):
            profit=np.where(chosen.won.to_numpy()==1,chosen.odds.to_numpy()-1.0,-1.0)
            odds_summary={"status":"evaluated","matched_odds_rows":int(len(value_df)),"qualifying_bets":int(len(chosen)),
                "wins":int(chosen.won.sum()),"win_rate":float(chosen.won.mean()),
                "flat_stake_profit_units":float(profit.sum()),"roi":float(profit.mean()),
                "mean_expected_value":float(chosen.expected_value.mean())}
        else:
            odds_summary={"status":"evaluated_no_value_bets","matched_odds_rows":int(len(value_df)),"qualifying_bets":0}
    metrics["odds_evaluation"]=odds_summary
    if Path(args.coverage).exists():
        cov=pd.read_csv(args.coverage)
        cov=cov[cov.comp.astype(str).isin(LEAGUES)]
        metrics["coverage_rows"]=cov.to_dict(orient="records")
    metrics["data_quality_note"]="Openplay coverage is incomplete; see coverage.csv and data_issues.csv upstream. This first run evaluates only matches with available match/event-derived team stats and a 5-match team warm-up."
    metrics["not_in_scope"]=["verified live Bet365 Spain feed and closing-line value (requires licensed provider/key)","lineups/injuries/referee effects","player props"]
    (OUT/"metrics.json").write_text(json.dumps(metrics,indent=2,default=str),encoding="utf-8")
    (OUT/"calibration.csv").write_text(pd.DataFrame(cal).to_csv(index=False),encoding="utf-8")
    cols=[c for c in df.columns if not c.startswith("features_")]
    test[cols].to_csv(OUT/"predictions.csv",index=False)
    # Save a compact, queryable database with the out-of-time forecasts and metrics.
    with sqlite3.connect(OUT/"football_analytics.sqlite") as con:
        test[cols].to_sql("predictions",con,if_exists="replace",index=False)
        pd.DataFrame([{"model":"V1.0",**metrics["V1.0"]},{"model":"V1.1_experimental",**metrics["V1.1_experimental"]},{"model":"V1.2_gradient_boosting",**metrics["V1.2_gradient_boosting"]},{"model":"V1.3_dixon_coles",**metrics["V1.3_dixon_coles"]},{"model":"V1.4_elo_1x2",**metrics["V1.4_elo_1x2"]},{"model":"V1.5_elo_gradient_ensemble",**metrics["V1.5_elo_gradient_ensemble"]}]).to_sql("metrics",con,if_exists="replace",index=False)
        pd.DataFrame([{"market":k,**v} for k,v in stat_model_metrics.items()]).to_sql("stat_market_metrics",con,if_exists="replace",index=False)
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

## V1.2 — gradient boosting
{fmt(metrics['V1.2_gradient_boosting'])}

## V1.3 — Dixon-Coles
{fmt(metrics['V1.3_dixon_coles'])}
- rho ajustado solo con entrenamiento: {dc_rho:.5f}

## V1.4 — Elo (solo 1X2)
{fmt(metrics['V1.4_elo_1x2'])}
- Factor de empate ajustado solo con entrenamiento: {draw_factor:.5f}

## V1.5 — ensemble Elo + Gradient Boosting
{fmt(metrics['V1.5_elo_gradient_ensemble'])}
- Mezcla fija 50% Elo + 50% Gradient Boosting; peso no optimizado sobre el conjunto de prueba.

## Multi-market models: shots, shots on target, corners, cards and fouls
{json.dumps(stat_model_metrics,indent=2)}

## Odds and value evaluation
{json.dumps(odds_summary,indent=2)}
- Odds CSV schema: match_id, market, selection, odds, bookmaker, captured_at.
- Markets: 1x2 (home/draw/away), over25/under25, btts_yes/btts_no, plus shots_22.5, sot_8.5, corners_9.5, cards_4.5, fouls_22.5 with selection over/under.
- A value bet requires estimated expected return >= 3% at the supplied price. This is a research threshold, not a guarantee.

## Interpretation
Lower is better for MAE, Brier and log loss. This is a chronological holdout, not a random split. V1.1 is experimental and is not promoted automatically. Compare all three models and calibration; do not choose a winner from accuracy alone.

## Data and scope limitations
{metrics['data_quality_note']}

Not evaluated unless odds.csv is supplied: bookmaker ROI/CLV. A CSV import allows verified Bet365 prices to be tested, but this workflow does not scrape Bet365 or invent prices. Current odds require a permitted provider/API credential.

Openplay attribution: Noah Bair / Openplay. Source release: https://github.com/nlbair/openplay/releases/tag/data-v0.1.0
"""
    (OUT/"summary.md").write_text(report,encoding="utf-8")
    print(report)

if __name__=="__main__":
    main()
