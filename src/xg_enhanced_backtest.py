#!/usr/bin/env python3
"""Leakage-safe xG-enhanced challenger with temporal model selection and untouched final test."""
from __future__ import annotations
from collections import defaultdict, deque
from pathlib import Path
import json, math
import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, log_loss

LEAGUES={"eng","esp","ger","ita","fra","ucl","uel"}
ROOT=Path("data"); OUT=Path("results"); OUT.mkdir(exist_ok=True)
MIN_GAMES=5

def finite(v):
    try:
        z=float(v)
        return z if np.isfinite(z) else np.nan
    except Exception:
        return np.nan

def safe_mean(vals, default):
    a=np.asarray([finite(x) for x in vals],dtype=float); a=a[np.isfinite(a)]
    return float(a.mean()) if len(a) else float(default)

def summarize(hist,key,n,default):
    vals=[x.get(key,np.nan) for x in list(hist)[-n:]]
    return safe_mean(vals,default)

def make_features(h,a,day,elo_h,elo_a,league_h,league_a):
    out=[]
    for hist in (h,a):
        for n in (5,10,20):
            for key,default in (("gf",1.3),("ga",1.3),("xg",1.3),("xga",1.3),("shots",12),("sot",4),("corners",5),("fouls",12)):
                out.append(summarize(hist,key,n,default))
        for key,default in (("gf",1.3),("ga",1.3),("xg",1.3),("xga",1.3)):
            v=[finite(x.get(key,np.nan)) for x in list(hist)[-10:]]
            v=[x for x in v if np.isfinite(x)]
            out.append(float(np.std(v)) if len(v)>1 else 0.0)
        out.append(float(np.mean([x.get("home",0) for x in list(hist)[-10:]])) if hist else 0.5)
        last=hist[-1].get("date") if hist else None
        out.append(min(30,max(0,(day-last).days)) if last is not None else 7.0)
    out.extend([elo_h,elo_a,elo_h-elo_a,league_h,league_a])
    # Relative attack/defence and xG-vs-goals residuals are explicit features.
    out.extend([summarize(h,"xg",10,1.3)-summarize(h,"gf",10,1.3),
                summarize(a,"xg",10,1.3)-summarize(a,"gf",10,1.3),
                summarize(h,"xga",10,1.3)-summarize(h,"ga",10,1.3),
                summarize(a,"xga",10,1.3)-summarize(a,"ga",10,1.3)])
    return out

def outcome_probs(lh,la):
    vals=np.arange(11)
    ph=poisson.pmf(vals,max(.05,lh)); pa=poisson.pmf(vals,max(.05,la))
    g=np.outer(ph,pa); g/=g.sum()
    return [float(np.tril(g,-1).sum()),float(np.trace(g)),float(np.triu(g,1).sum())]

def main():
    matches=pd.read_parquet(ROOT/"matches.parquet")
    matches=matches[matches.comp.astype(str).isin(LEAGUES)].copy()
    if "status_code" in matches: matches=matches[matches.status_code==6]
    matches["date"]=pd.to_datetime(matches.date,errors="coerce")
    matches=matches.dropna(subset=["date","home_score_90","away_score_90"]).sort_values(["date","match_id"])
    pstats=pd.read_parquet(ROOT/"player_match_stats.parquet")
    pstats=pstats[pstats.comp.astype(str).isin(LEAGUES)].copy()
    numeric=[c for c in ["xg","npxg","shots","shots_on_target","fouls_committed","corners_taken"] if c in pstats]
    agg=pstats.groupby(["match_id","team_id"],as_index=False)[numeric].sum()
    stat={}
    for r in agg.itertuples(index=False):
        d=r._asdict()
        stat[(int(d["match_id"]),int(d["team_id"]))]={k:finite(d.get(k)) for k in numeric}
    history=defaultdict(lambda:deque(maxlen=20)); elo=defaultdict(lambda:1500.0)
    comp_hist=defaultdict(lambda:deque(maxlen=250)); rows=[]
    for day,batchdf in matches.groupby("date",sort=True):
        batch=list(batchdf.itertuples(index=False))
        for m in batch:
            hid=int(m.home_team_id); aid=int(m.away_team_id)
            h=list(history[hid]); a=list(history[aid])
            if len(h)<MIN_GAMES or len(a)<MIN_GAMES: continue
            lh,la=elo[hid],elo[aid]
            ch=list(comp_hist[str(m.comp)])
            league_h=float(np.mean([x[0] for x in ch])) if len(ch)>=15 else 1.45
            league_a=float(np.mean([x[1] for x in ch])) if len(ch)>=15 else 1.20
            f=make_features(h,a,day,lh,la,league_h,league_a)
            # compact goal-only feature control, to measure whether xG adds information
            f_goal=[]
            for hist in (h,a):
                for n in (5,10,20):
                    for key,default in (("gf",1.3),("ga",1.3)):
                        f_goal.append(summarize(hist,key,n,default))
            f_goal += [lh,la,lh-la,league_h,league_a]
            rows.append({"date":day,"match_id":int(m.match_id),"home":str(m.home_team),"away":str(m.away_team),
                         "comp":str(m.comp),"yh":float(m.home_score_90),"ya":float(m.away_score_90),
                         "features":f,"goal_features":f_goal,"actual":"H" if m.home_score_90>m.away_score_90 else ("D" if m.home_score_90==m.away_score_90 else "A")})
        # no same-day results can affect another match on that date
        for m in batch:
            hid=int(m.home_team_id); aid=int(m.away_team_id)
            hs=stat.get((int(m.match_id),hid),{}); aws=stat.get((int(m.match_id),aid),{})
            history[hid].append({"date":day,"home":1,"gf":float(m.home_score_90),"ga":float(m.away_score_90),
                "xg":hs.get("xg",np.nan),"xga":aws.get("xg",np.nan),"shots":hs.get("shots",np.nan),
                "sot":hs.get("shots_on_target",np.nan),"corners":hs.get("corners_taken",np.nan),"fouls":hs.get("fouls_committed",np.nan)})
            history[aid].append({"date":day,"home":0,"gf":float(m.away_score_90),"ga":float(m.home_score_90),
                "xg":aws.get("xg",np.nan),"xga":hs.get("xg",np.nan),"shots":aws.get("shots",np.nan),
                "sot":aws.get("shots_on_target",np.nan),"corners":aws.get("corners_taken",np.nan),"fouls":aws.get("fouls_committed",np.nan)})
            rh,ra=elo[hid],elo[aid]; gd=abs(float(m.home_score_90)-float(m.away_score_90))
            score=1.0 if m.home_score_90>m.away_score_90 else (0.5 if m.home_score_90==m.away_score_90 else 0.0)
            expected=1/(1+10**(-((rh+65-ra)/400)))
            change=20*(1+.2*math.log1p(gd))*(score-expected)
            elo[hid]=rh+change; elo[aid]=ra-change
            comp_hist[str(m.comp)].append((float(m.home_score_90),float(m.away_score_90)))
    df=pd.DataFrame(rows)
    if len(df)<2000: raise RuntimeError(f"Too few eligible xG rows: {len(df)}")
    dates=sorted(df.date.unique()); n=len(dates)
    d1=dates[int(n*.65)]; d2=dates[int(n*.82)]
    train=(df.date<d1).to_numpy(); val=((df.date>=d1)&(df.date<d2)).to_numpy(); test=(df.date>=d2).to_numpy()
    X=np.asarray(df.features.tolist(),dtype=float); Xg=np.asarray(df.goal_features.tolist(),dtype=float)
    yh=df.yh.to_numpy(); ya=df.ya.to_numpy()
    def fit_pair(mat,mask):
        opts={"loss":"poisson","learning_rate":.05,"max_iter":180,"max_leaf_nodes":15,
              "l2_regularization":3.0,"min_samples_leaf":35,"random_state":42}
        mh=HistGradientBoostingRegressor(**opts).fit(mat[mask],yh[mask])
        ma=HistGradientBoostingRegressor(**opts).fit(mat[mask],ya[mask])
        return mh,ma
    # Compare a richer xG/stat model against a goals/Elo-only challenger on validation, never test.
    candidates={}
    for name,mat in [("goals_elo",Xg),("xg_stats_elo",X)]:
        mh,ma=fit_pair(mat,train)
        ph=np.clip(mh.predict(mat[val]),.05,5.5); pa=np.clip(ma.predict(mat[val]),.05,5.5)
        candidates[name]={"home_mae":float(mean_absolute_error(yh[val],ph)),
                          "away_mae":float(mean_absolute_error(ya[val],pa)),
                          "total_mae":float(mean_absolute_error(yh[val]+ya[val],ph+pa))}
    winner=min(candidates,key=lambda k:candidates[k]["total_mae"])
    mat=X if winner=="xg_stats_elo" else Xg
    fit=train|val
    mh,ma=fit_pair(mat,fit)
    ph=np.clip(mh.predict(mat[test]),.05,5.5); pa=np.clip(ma.predict(mat[test]),.05,5.5)
    rows_out=[]
    for idx,(i,j) in enumerate(zip(np.where(test)[0],range(int(test.sum())))):
        p=outcome_probs(ph[j],pa[j]); actual=df.iloc[i]
        rows_out.append({"match_id":int(actual.match_id),"date":actual.date.date().isoformat(),"home":actual.home,"away":actual.away,
                         "actual":actual.actual,"lambda_home":ph[j],"lambda_away":pa[j],
                         "p_home":p[0],"p_draw":p[1],"p_away":p[2],
                         "fair_home":1/max(1e-9,p[0]),"fair_draw":1/max(1e-9,p[1]),"fair_away":1/max(1e-9,p[2])})
    pred=pd.DataFrame(rows_out); pred.to_csv(OUT/"xg_enhanced_predictions.csv",index=False)
    yidx=np.where(df.loc[test,"actual"].to_numpy()=="H",0,np.where(df.loc[test,"actual"].to_numpy()=="D",1,2))
    probs=pred[["p_home","p_draw","p_away"]].to_numpy()
    onehot=np.eye(3)[yidx]
    # Constant training-frequency baseline and ML scores, evaluated on same untouched test dates.
    freq=pd.Series(df.loc[train|val,"actual"]).value_counts(normalize=True)
    base=np.tile([freq.get("H",0),freq.get("D",0),freq.get("A",0)],(len(pred),1))
    result={"dataset":"Openplay v0.1.0 + player_match_stats","rows":int(len(df)),
      "train":int(train.sum()),"validation":int(val.sum()),"untouched_test":int(test.sum()),
      "train_end":str(d1),"validation_end":str(d2),"test_start":str(d2),
      "validation_candidates":candidates,"selected_on_validation_only":winner,
      "test":{"home_goals_mae":float(mean_absolute_error(df.loc[test,"yh"],pred.lambda_home)),
              "away_goals_mae":float(mean_absolute_error(df.loc[test,"ya"],pred.lambda_away)),
              "total_goals_mae":float(mean_absolute_error(df.loc[test,"yh"]+df.loc[test,"ya"],pred.lambda_home+pred.lambda_away)),
              "1x2_brier":float(np.mean(np.sum((probs-onehot)**2,axis=1))),
              "1x2_log_loss":float(log_loss(yidx,probs,labels=[0,1,2])),
              "constant_baseline_1x2_brier":float(np.mean(np.sum((base-onehot)**2,axis=1))),
              "constant_baseline_1x2_log_loss":float(log_loss(yidx,base,labels=[0,1,2]))},
      "warning":"This is a challenger, not automatically promoted. The final test is untouched during candidate selection. xG coverage is partial; missing xG is retained as missing then defaulted only within historical feature summaries."}
    (OUT/"xg_enhanced_metrics.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))
if __name__=="__main__": main()
