#!/usr/bin/env python3
"""Download and audit Open Matchday's current 2026/27 CSV data (non-commercial source)."""
from pathlib import Path
import pandas as pd
import json

BASE="https://raw.githubusercontent.com/oladeeayo/open-matchday/master/data"
LEAGUES={
 "ENG_1_premier-league":"Premier League",
 "ESP_1_la-liga":"La Liga",
 "GER_1_bundesliga":"Bundesliga",
 "ITA_1_serie-a":"Serie A",
 "FRA_1_ligue-1":"Ligue 1",
}
OUT=Path("results"); OUT.mkdir(exist_ok=True)
frames=[]; fixtures=[]; audit={}
for folder,name in LEAGUES.items():
    url=f"{BASE}/{folder}/results_2627.csv"
    try:
        df=pd.read_csv(url)
        df["league"]=name
        required=["date","home","away","ft_home","ft_away"]
        missing=[c for c in required if c not in df.columns]
        if missing: raise ValueError(f"missing columns {missing}")
        df["date"]=pd.to_datetime(df["date"],errors="coerce")
        for c in ["home_shots","away_shots","home_sot","away_sot","home_fouls","away_fouls",
                  "home_corners","away_corners","home_yellow","away_yellow","home_red","away_red"]:
            if c not in df: df[c]=float("nan")
        frames.append(df)
        audit[name]={"rows":int(len(df)),"finished":int(df.ft_home.notna().sum()),
                     "stats_rows":int(df[["home_shots","away_shots","home_corners","away_corners"]].notna().any(axis=1).sum()),
                     "url":url}
        f_url=f"{BASE}/{folder}/fixtures_2627.csv"
        try:
            fx=pd.read_csv(f_url); fx["league"]=name; fixtures.append(fx)
        except Exception as e:
            audit[name]["fixtures_error"]=str(e)
    except Exception as e:
        audit[name]={"error":str(e)}
if frames:
    all_results=pd.concat(frames,ignore_index=True)
    all_results=all_results.drop_duplicates(subset=["league","date","home","away"],keep="last")
    all_results.to_csv(OUT/"open_matchday_2627.csv",index=False)
else:
    all_results=pd.DataFrame()
if fixtures:
    all_fx=pd.concat(fixtures,ignore_index=True).drop_duplicates(subset=["league","date","home","away"],keep="last")
    all_fx.to_csv(OUT/"upcoming_fixtures_2627.csv",index=False)
else:
    all_fx=pd.DataFrame()
summary={"source":"open-matchday","season":"2026/27","leagues":audit,
         "finished_matches":int(all_results.ft_home.notna().sum()) if len(all_results) else 0,
         "upcoming_fixtures":int(len(all_fx)),
         "stats_rows":int(all_results[["home_shots","away_shots","home_corners","away_corners"]].notna().any(axis=1).sum()) if len(all_results) else 0,
         "warning":"Coverage is partial; blanks are missing data, never zero. Source is for non-commercial informational use and provider terms still apply."}
(OUT/"open_matchday_audit.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
print(json.dumps(summary,indent=2))
