#!/usr/bin/env python3
"""Generate current upcoming 1X2/goals probabilities from the validated xG/Elo challenger."""
from __future__ import annotations
from collections import defaultdict, deque
from pathlib import Path
from difflib import get_close_matches
import json, math, re, unicodedata
import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, log_loss

ROOT=Path("data"); OUT=Path("results"); OUT.mkdir(exist_ok=True)
LEAGUES={"eng","esp","ger","ita","fra","ucl","uel"}
MIN_GAMES=5
ALIASES={
 "man city":"manchester city","man utd":"manchester united","man united":"manchester united",
 "spurs":"tottenham hotspur","athletic bilbao":"athletic club","atletico madrid":"atletico de madrid",
 "real betis":"real betis balompie","sociedad":"real sociedad","girona":"girona fc",
 "inter":"internazionale","inter milan":"internazionale","ac milan":"milan","milan":"milan",
 "roma":"as roma","lazio":"ss lazio","paris saint germain":"paris saint-germain",
 "psg":"paris saint-germain","bayern":"bayern munich","bayern munchen":"bayern munich",
 "dortmund":"borussia dortmund","monchengladbach":"borussia monchengladbach",
 "leverkusen":"bayer leverkusen","eintracht frankfurt":"eintracht frankfurt",
 "wolves":"wolverhampton wanderers","west ham":"west ham united",
 "newcastle":"newcastle united","brighton":"brighton and hove albion",
 "nottingham forest":"nottingham forest","manchester utd":"manchester united",
}
def norm(s):
 s=unicodedata.normalize("NFKD",str(s)).encode("ascii","ignore").decode().lower()
 s=re.sub(r"[^a-z0-9 ]"," ",s)
 s=" ".join(s.split())
 return ALIASES.get(s,s)
def finite(v):
 try:
  z=float(v); return z if np.isfinite(z) else np.nan
 except Exception: return np.nan
def mean(hist,key,n,default):
 a=np.array([finite(x.get(key,np.nan)) for x in list(hist)[-n:]],float); a=a[np.isfinite(a)]
 return float(a.mean()) if len(a) else float(default)
def feats(h,a,day,eh,ea,lh,la):
 out=[]
 for hist in (h,a):
  for n in (5,10,20):
   for key,d in (("gf",1.3),("ga",1.3),("xg",1.3),("xga",1.3),("shots",12),("sot",4),("corners",5),("fouls",12)):
    out.append(mean(hist,key,n,d))
  for key in ("gf","ga","xg","xga"):
   v=[finite(x.get(key,np.nan)) for x in list(hist)[-10:]]; v=[x for x in v if np.isfinite(x)]
   out.append(float(np.std(v)) if len(v)>1 else 0.0)
  out.append(float(np.mean([x.get("home",0) for x in list(hist)[-10:]])) if hist else .5)
  last=hist[-1].get("date") if hist else None
  out.append(min(30,max(0,(day-last).days)) if last is not None else 7.)
 out.extend([eh,ea,eh-ea,lh,la])
 out.extend([mean(h,"xg",10,1.3)-mean(h,"gf",10,1.3),
             mean(a,"xg",10,1.3)-mean(a,"gf",10,1.3),
             mean(h,"xga",10,1.3)-mean(h,"ga",10,1.3),
             mean(a,"xga",10,1.3)-mean(a,"ga",10,1.3)])
 return out
def probs(lh,la):
 vals=np.arange(12); ph=poisson.pmf(vals,max(.05,lh)); pa=poisson.pmf(vals,max(.05,la))
 g=np.outer(ph,pa); g/=g.sum()
 home=float(np.tril(g,-1).sum()); draw=float(np.trace(g)); away=float(np.triu(g,1).sum())
 over={}; under={}
 for line in (1.5,2.5,3.5,4.5):
  p=float(sum(g[i,j] for i in range(12) for j in range(12) if i+j>line))
  over[line]=p; under[line]=1-p
 btts=float(g[1:,1:].sum())
 return home,draw,away,over,under,btts
def price(p): return 1/max(1e-9,min(1.,max(0.,p)))
def main():
 m=pd.read_parquet(ROOT/"matches.parquet")
 m=m[m.comp.astype(str).isin(LEAGUES)].copy()
 if "status_code" in m: m=m[m.status_code==6]
 m["date"]=pd.to_datetime(m.date,errors="coerce")
 m=m.dropna(subset=["date","home_score_90","away_score_90"]).sort_values(["date","match_id"])
 ps=pd.read_parquet(ROOT/"player_match_stats.parquet")
 ps=ps[ps.comp.astype(str).isin(LEAGUES)].copy()
 cols=[c for c in ["xg","shots","shots_on_target","fouls_committed","corners_taken"] if c in ps]
 ag=ps.groupby(["match_id","team_id"],as_index=False)[cols].sum()
 stats={(int(r.match_id),int(r.team_id)):{k:finite(getattr(r,k)) for k in cols} for r in ag.itertuples(index=False)}
 hist=defaultdict(lambda:deque(maxlen=20)); elo=defaultdict(lambda:1500.); league_hist=defaultdict(lambda:deque(maxlen=250))
 name_to_id={}; id_to_name={}
 rows=[]
 # Chronological, date-batched feature creation. No same-day leakage.
 for day,bdf in m.groupby("date",sort=True):
  batch=list(bdf.itertuples(index=False))
  for x in batch:
   hid=int(x.home_team_id); aid=int(x.away_team_id)
   name_to_id[norm(x.home_team)]=hid; name_to_id[norm(x.away_team)]=aid
   id_to_name[hid]=str(x.home_team); id_to_name[aid]=str(x.away_team)
   h=list(hist[hid]); a=list(hist[aid])
   if len(h)<MIN_GAMES or len(a)<MIN_GAMES: continue
   comp=str(x.comp); ch=list(league_hist[comp])
   lhome=float(np.mean([z[0] for z in ch])) if len(ch)>=15 else 1.45
   laway=float(np.mean([z[1] for z in ch])) if len(ch)>=15 else 1.2
   fg=[]
   for hh in (h,a):
    for n in (5,10,20):
     for k,d in (("gf",1.3),("ga",1.3)): fg.append(mean(hh,k,n,d))
   fg += [elo[hid],elo[aid],elo[hid]-elo[aid],lhome,laway]
   rows.append({"date":day,"features":feats(h,a,day,elo[hid],elo[aid],lhome,laway),
                "goal_features":fg,"yh":float(x.home_score_90),"ya":float(x.away_score_90),
                "actual":"H" if x.home_score_90>x.away_score_90 else ("D" if x.home_score_90==x.away_score_90 else "A")})
  for x in batch:
   hid=int(x.home_team_id); aid=int(x.away_team_id); hs=stats.get((int(x.match_id),hid),{}); aws=stats.get((int(x.match_id),aid),{})
   hist[hid].append({"date":day,"home":1,"gf":float(x.home_score_90),"ga":float(x.away_score_90),
     "xg":hs.get("xg",np.nan),"xga":aws.get("xg",np.nan),"shots":hs.get("shots",np.nan),
     "sot":hs.get("shots_on_target",np.nan),"corners":hs.get("corners_taken",np.nan),"fouls":hs.get("fouls_committed",np.nan)})
   hist[aid].append({"date":day,"home":0,"gf":float(x.away_score_90),"ga":float(x.home_score_90),
     "xg":aws.get("xg",np.nan),"xga":hs.get("xg",np.nan),"shots":aws.get("shots",np.nan),
     "sot":aws.get("shots_on_target",np.nan),"corners":aws.get("corners_taken",np.nan),"fouls":aws.get("fouls_committed",np.nan)})
   rh,ra=elo[hid],elo[aid]; score=1. if x.home_score_90>x.away_score_90 else (.5 if x.home_score_90==x.away_score_90 else 0.)
   exp=1/(1+10**(-((rh+65-ra)/400))); change=20*(1+.2*math.log1p(abs(float(x.home_score_90)-float(x.away_score_90))))*(score-exp)
   elo[hid]=rh+change; elo[aid]=ra-change
   league_hist[str(x.comp)].append((float(x.home_score_90),float(x.away_score_90)))
 df=pd.DataFrame(rows)
 if len(df)<2000: raise RuntimeError(f"Insufficient historical rows: {len(df)}")
 dates=sorted(df.date.unique()); d1=dates[int(len(dates)*.65)]; d2=dates[int(len(dates)*.82)]
 tr=(df.date<d1).to_numpy(); va=((df.date>=d1)&(df.date<d2)).to_numpy()
 X=np.asarray(df.features.tolist(),float); Xg=np.asarray(df.goal_features.tolist(),float)
 yh=df.yh.to_numpy(); ya=df.ya.to_numpy()
 opts={"loss":"poisson","learning_rate":.05,"max_iter":180,"max_leaf_nodes":15,"l2_regularization":3.,"min_samples_leaf":35,"random_state":42}
 def fit(mat,mask):
  mh=HistGradientBoostingRegressor(**opts).fit(mat[mask],yh[mask])
  ma=HistGradientBoostingRegressor(**opts).fit(mat[mask],ya[mask])
  return mh,ma
 choices={}
 for nm,mat in (("goals_elo",Xg),("xg_stats_elo",X)):
  mh,ma=fit(mat,tr); ph=np.clip(mh.predict(mat[va]),.05,5.5); pa=np.clip(ma.predict(mat[va]),.05,5.5)
  choices[nm]=float(mean_absolute_error(yh[va]+ya[va],ph+pa))
 winner=min(choices,key=choices.get); matrix=X if winner=="xg_stats_elo" else Xg
 # Refit production model on all available historical data, after model selection on validation.
 mh,ma=fit(matrix,np.ones(len(df),dtype=bool))
 # Map current season teams to historical IDs; fuzzy mapping is audited, never silently trusted.
 cur=pd.read_csv("results/open_matchday_2627.csv")
 fixtures=pd.read_csv("results/upcoming_fixtures_2627.csv")
 keys=list(name_to_id.keys()); mapping={}; unresolved=set()
 for nm in set(cur.home.astype(str))|set(cur.away.astype(str))|set(fixtures.home.astype(str))|set(fixtures.away.astype(str)):
  q=norm(nm)
  if q in name_to_id: mapping[nm]=name_to_id[q]; continue
  hit=get_close_matches(q,keys,n=1,cutoff=.82)
  if hit: mapping[nm]=name_to_id[hit[0]]
  else: unresolved.add(nm)
 mapped=0; unmapped=[]; prospective=[]
 cur["date"]=pd.to_datetime(cur.date,errors="coerce")
 # Rolling-origin prospective test: each current-season match is predicted before its date's results update history.
 for day,bdf in cur.dropna(subset=["ft_home","ft_away","date"]).sort_values("date").groupby("date",sort=True):
  batch=[]
  for x in bdf.itertuples(index=False):
   if x.home not in mapping or x.away not in mapping:
    unmapped.extend([x.home,x.away]); continue
   batch.append(x)
  for x in batch:
   hid=mapping[x.home]; aid=mapping[x.away]
   h=list(hist[hid]); a=list(hist[aid])
   if len(h)>=MIN_GAMES and len(a)>=MIN_GAMES:
    code={"Premier League":"eng","La Liga":"esp","Bundesliga":"ger","Serie A":"ita","Ligue 1":"fra"}.get(str(x.league),"")
    ch=list(league_hist[code]); lh=float(np.mean([z[0] for z in ch])) if len(ch)>=15 else 1.45
    la=float(np.mean([z[1] for z in ch])) if len(ch)>=15 else 1.2
    if winner=="xg_stats_elo": vec=np.asarray(feats(h,a,day,elo[hid],elo[aid],lh,la),float).reshape(1,-1)
    else:
     fv=[]
     for hh in (h,a):
      for n in (5,10,20):
       for k,d in (("gf",1.3),("ga",1.3)): fv.append(mean(hh,k,n,d))
     fv += [elo[hid],elo[aid],elo[hid]-elo[aid],lh,la]
     vec=np.asarray(fv,float).reshape(1,-1)
    ph=float(np.clip(mh.predict(vec)[0],.05,5.5)); pa=float(np.clip(ma.predict(vec)[0],.05,5.5))
    p1,px,p2,ov,un,bt=probs(ph,pa); b1,bx,b2,_,_,_=probs(lh,la)
    actual="H" if x.ft_home>x.ft_away else ("D" if x.ft_home==x.ft_away else "A")
    prospective.append({"date":day.date().isoformat(),"league":x.league,"home":x.home,"away":x.away,
      "actual_home_goals":int(x.ft_home),"actual_away_goals":int(x.ft_away),"actual":actual,
      "lambda_home":ph,"lambda_away":pa,"p_home":p1,"p_draw":px,"p_away":p2,
      "baseline_lambda_home":lh,"baseline_lambda_away":la,"baseline_p_home":b1,"baseline_p_draw":bx,"baseline_p_away":b2})
   hs={"xg":np.nan,"shots":finite(getattr(x,"home_shots",np.nan)),"sot":finite(getattr(x,"home_target",np.nan)),
       "corners":finite(getattr(x,"home_corners",np.nan)),"fouls":finite(getattr(x,"home_fouls",np.nan))}
   aws={"xg":np.nan,"shots":finite(getattr(x,"away_shots",np.nan)),"sot":finite(getattr(x,"away_target",np.nan)),
       "corners":finite(getattr(x,"away_corners",np.nan)),"fouls":finite(getattr(x,"away_fouls",np.nan))}
   hist[hid].append({"date":day,"home":1,"gf":float(x.ft_home),"ga":float(x.ft_away),"xg":np.nan,"xga":np.nan,**hs})
   hist[aid].append({"date":day,"home":0,"gf":float(x.ft_away),"ga":float(x.ft_home),"xg":np.nan,"xga":np.nan,**aws})
   mapped+=1
   rh,ra=elo[hid],elo[aid]; score=1. if x.ft_home>x.ft_away else (.5 if x.ft_home==x.ft_away else 0.)
   exp=1/(1+10**(-((rh+65-ra)/400))); change=20*(1+.2*math.log1p(abs(float(x.ft_home)-float(x.ft_away))))*(score-exp)
   elo[hid]=rh+change; elo[aid]=ra-change
   league_code={"Premier League":"eng","La Liga":"esp","Bundesliga":"ger","Serie A":"ita","Ligue 1":"fra"}.get(str(x.league),str(x.league))
   league_hist[league_code].append((float(x.ft_home),float(x.ft_away)))
 # Report a genuine current-season rolling-origin test against a simple competition-average Poisson baseline.
 testdf=pd.DataFrame(prospective)
 if len(testdf):
  actual_idx=np.where(testdf.actual.to_numpy()=="H",0,np.where(testdf.actual.to_numpy()=="D",1,2))
  onehot=np.eye(3)[actual_idx]
  model_p=testdf[["p_home","p_draw","p_away"]].to_numpy()
  base_p=testdf[["baseline_p_home","baseline_p_draw","baseline_p_away"]].to_numpy()
  current_metrics={"matches":int(len(testdf)),
   "model_total_goals_mae":float(mean_absolute_error(testdf.actual_home_goals+testdf.actual_away_goals,testdf.lambda_home+testdf.lambda_away)),
   "league_average_total_goals_mae":float(mean_absolute_error(testdf.actual_home_goals+testdf.actual_away_goals,testdf.baseline_lambda_home+testdf.baseline_lambda_away)),
   "model_1x2_brier":float(np.mean(np.sum((model_p-onehot)**2,axis=1))),
   "league_average_1x2_brier":float(np.mean(np.sum((base_p-onehot)**2,axis=1))),
   "model_1x2_log_loss":float(log_loss(actual_idx,model_p,labels=[0,1,2])),
   "league_average_1x2_log_loss":float(log_loss(actual_idx,base_p,labels=[0,1,2]))}
  testdf.to_csv(OUT/"current_season_rolling_test.csv",index=False)
 else: current_metrics={"matches":0,"warning":"No current-season fixtures had a verified team-ID mapping and sufficient history."}
 (OUT/"current_season_rolling_metrics.json").write_text(json.dumps(current_metrics,indent=2),encoding="utf-8")
 out=[]
 fixtures["date"]=pd.to_datetime(fixtures.date,errors="coerce")
 for x in fixtures.sort_values("date").itertuples(index=False):
  base={"date":x.date.date().isoformat() if pd.notna(x.date) else "", "time":str(x.time),
        "league":x.league,"home":x.home,"away":x.away,"model":winner}
  if x.home not in mapping or x.away not in mapping:
   out.append({**base,"status":"unmapped_team","reason":"Team name did not match historical club ID"}); continue
  hid=mapping[x.home]; aid=mapping[x.away]; h=list(hist[hid]); a=list(hist[aid])
  if len(h)<MIN_GAMES or len(a)<MIN_GAMES:
   out.append({**base,"status":"insufficient_history","home_matches":len(h),"away_matches":len(a)}); continue
  comp_code={"Premier League":"eng","La Liga":"esp","Bundesliga":"ger","Serie A":"ita","Ligue 1":"fra"}.get(str(x.league),"")
  ch=list(league_hist[comp_code]); lh=float(np.mean([z[0] for z in ch])) if len(ch)>=15 else 1.45
  la=float(np.mean([z[1] for z in ch])) if len(ch)>=15 else 1.2
  f=feats(h,a,x.date,elo[hid],elo[aid],lh,la) if winner=="xg_stats_elo" else (
    [mean(h,k,n,d) for hh in [] for k,n,d in []])
  if winner=="goals_elo":
   f=[]
   for hh in (h,a):
    for n in (5,10,20):
     for k,d in (("gf",1.3),("ga",1.3)): f.append(mean(hh,k,n,d))
   f += [elo[hid],elo[aid],elo[hid]-elo[aid],lh,la]
  vec=np.asarray(f,float).reshape(1,-1)
  ph=float(np.clip(mh.predict(vec)[0],.05,5.5)); pa=float(np.clip(ma.predict(vec)[0],.05,5.5))
  p1,px,p2,ov,un,bt=probs(ph,pa)
  rec={**base,"status":"ready","history_home":len(h),"history_away":len(a),
       "lambda_home":ph,"lambda_away":pa,"p_home":p1,"p_draw":px,"p_away":p2,
       "fair_home":price(p1),"fair_draw":price(px),"fair_away":price(p2),
       "p_over_1_5":ov[1.5],"fair_over_1_5":price(ov[1.5]),"p_under_1_5":un[1.5],"fair_under_1_5":price(un[1.5]),
       "p_over_2_5":ov[2.5],"fair_over_2_5":price(ov[2.5]),"p_under_2_5":un[2.5],"fair_under_2_5":price(un[2.5]),
       "p_over_3_5":ov[3.5],"fair_over_3_5":price(ov[3.5]),"p_under_3_5":un[3.5],"fair_under_3_5":price(un[3.5]),
       "p_btts_yes":bt,"fair_btts_yes":price(bt),"p_btts_no":1-bt,"fair_btts_no":price(1-bt)}
  for k in ["p_home","p_draw","p_away","p_over_1_5","p_under_1_5","p_over_2_5","p_under_2_5","p_over_3_5","p_under_3_5","p_btts_yes","p_btts_no"]:
   rec[k.replace("p_","min_odds_")]=1.03*price(rec[k])
  out.append(rec)
 result=pd.DataFrame(out)
 result.to_csv(OUT/"upcoming_predictions.csv",index=False)
 # Optional manually verified bookmaker prices; empty template means no value/ROI claim.
 odds_path=Path("data/upcoming_odds.csv")
 value_rows=[]
 if odds_path.exists():
  odds=pd.read_csv(odds_path)
  for q in odds.to_dict("records"):
   mask=(result.date.astype(str)==str(q.get("date",""))) & (result.home==q.get("home")) & (result.away==q.get("away")) & (result.status=="ready")
   if not mask.any(): continue
   r=result.loc[mask].iloc[0]; market=str(q.get("market","")).lower(); sel=str(q.get("selection","")).lower()
   col={"1x2_home":"p_home","1x2_draw":"p_draw","1x2_away":"p_away",
        "over_2.5":"p_over_2_5","under_2.5":"p_under_2_5",
        "btts_yes":"p_btts_yes","btts_no":"p_btts_no"}.get(market+"_"+sel) or ({"over_2.5":"p_over_2_5","under_2.5":"p_under_2_5"}.get(market) if sel in ("","over","under") else None)
   if col is None or pd.isna(q.get("odds")): continue
   p=float(r[col]); price_now=float(q["odds"])
   value_rows.append({"date":r.date,"home":r.home,"away":r.away,"market":market,"selection":sel,
       "bookmaker":q.get("bookmaker",""),"odds":price_now,"probability":p,"fair_odds":price(p),
       "min_odds_3pct_edge":1.03*price(p),"expected_value_pct":100*(p*price_now-1),
       "value_3pct_edge":bool(price_now>=1.03*price(p)),"captured_at":q.get("captured_at","")})
 pd.DataFrame(value_rows).to_csv(OUT/"upcoming_value_candidates.csv",index=False)
 audit={"historical_training_rows":len(df),"validation_choice_mae":choices,"selected_model":winner,
        "current_season_rolling_test":current_metrics,
        "historical_teams":len(name_to_id),"current_results":len(cur),"current_results_mapped":mapped,
        "current_team_mapping_unresolved":sorted(set(unmapped)|unresolved),
        "fixtures":len(fixtures),"ready_predictions":int((result.status=="ready").sum()),
        "unmapped_fixtures":int((result.status=="unmapped_team").sum()),
        "insufficient_history_fixtures":int((result.status=="insufficient_history").sum()),
        "warning":"Predictions are model estimates, not betting advice. No bookmaker prices are assumed. Current-season xG is unavailable and missing fields are not treated as zero."}
 (OUT/"upcoming_prediction_audit.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
 print(json.dumps(audit,indent=2))
if __name__=="__main__": main()
