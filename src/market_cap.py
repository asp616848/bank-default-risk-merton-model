# src/market_cap.py
import pandas as pd
from pathlib import Path

def get_fyend_price(df, fy_end):
    df = df.sort_values("Date")
    df = df[df["Date"] <= fy_end]
    if df.empty: 
        return None, None
    last = df.iloc[-1]
    return last["Adj Close"], last["Date"]

def compute_marketcap_from_fundamentals(tickers, price_folder, fundamentals_dict, fy_end):
    rows = []
    for t in tickers:
        p = Path(price_folder) / f"{t}.csv"
        if not p.exists(): 
            print("Missing price file:", t); continue
        df = pd.read_csv(p)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.tz_localize(None)
        price, price_date = get_fyend_price(df, fy_end)
        if price is None:
            print("No FY-end price for", t); continue
        if t not in fundamentals_dict:
            print("No fundamentals row for", t); continue
        shares = int(fundamentals_dict[t]["shares_outstanding"])
        marketcap = shares * price
        rows.append({
            "Ticker": t,
            "AdjClose_FYend": price,
            "FYEnd_Price_Date": str(price_date.date()),
            "Shares_Outstanding": shares,
            "Market_Cap_Rupees": marketcap,
            "Market_Cap_Crore": marketcap/1e7
        })
    return pd.DataFrame(rows)
