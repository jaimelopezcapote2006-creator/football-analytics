#!/usr/bin/env python3
"""Optional The Odds API EU snapshot. Requires ODDS_API_KEY; never substitutes for Bet365 Spain."""
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.parse import urlencode
from urllib.error import HTTPError, URLError
from datetime import datetime, timezone
import os, json, csv

OUT=Path("results"); OUT.mkdir(exist_ok=True)
KEY=os.getenv("ODDS_API_KEY","").strip()
SPORTS=["soccer_epl","soccer_spain_la_liga","soccer_germany_bundesliga",
        "soccer_italy_serie_a","soccer_france_ligue_one","soccer_uefa_champs_league"]
audit={"provider":"The Odds API","regions":"eu","markets":"h2h,totals",
       "bet365_spain_available":False,"status":"not_configured","events":0,"rows":0,"errors":[]}
rows=[]
if KEY:
    audit["status"]="configured"
    for sport in SPORTS:
        params=urlencode({"apiKey":KEY,"regions":"eu","markets":"h2h,totals","oddsFormat":"decimal","dateFormat":"iso"})
        url=f"https://api.the-odds-api.com/v4/sports/{sport}/odds/?{params}"
        try:
            req=Request(url,headers={"User-Agent":"football-analytics-research/1.0"})
            with urlopen(req,timeout=35) as response:
                payload=json.loads(response.read().decode("utf-8"))
                audit["remaining_requests"]=response.headers.get("x-requests-remaining")
                audit["used_requests"]=response.headers.get("x-requests-used")
            for event in payload:
                audit["events"]+=1
                for book in event.get("bookmakers",[]):
                    for market in book.get("markets",[]):
                        for outcome in market.get("outcomes",[]):
                            rows.append({"event_id":event.get("id"),"sport_key":sport,
                                "commence_time":event.get("commence_time"),"home_team":event.get("home_team"),
                                "away_team":event.get("away_team"),"bookmaker":book.get("title"),
                                "bookmaker_key":book.get("key"),"market":market.get("key"),
                                "selection":outcome.get("name"),"line":outcome.get("point"),
                                "odds":outcome.get("price"),"captured_at":datetime.now(timezone.utc).isoformat()})
        except (HTTPError,URLError,TimeoutError,ValueError) as exc:
            audit["errors"].append({"sport":sport,"error":str(exc)})
    if rows:
        with (OUT/"upcoming_odds_eu.csv").open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    else:
        (OUT/"upcoming_odds_eu.csv").write_text("event_id,sport_key,commence_time,home_team,away_team,bookmaker,bookmaker_key,market,selection,line,odds,captured_at\n",encoding="utf-8")
    audit["rows"]=len(rows)
    audit["status"]="downloaded" if rows else ("no_events_or_no_coverage" if not audit["errors"] else "partial_or_failed")
else:
    audit["message"]="Add GitHub Actions secret ODDS_API_KEY to enable free-tier EU odds snapshots. This provider does not list Bet365 Spain; it is a market-consensus reference only."
(OUT/"odds_api_audit.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
print(json.dumps(audit,indent=2))
