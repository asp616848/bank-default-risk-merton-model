# src/volatility.py
import pandas as pd
import numpy as np
from pathlib import Path

def compute_annualized_vol_from_df(df):
    df = df.sort_values("Date")
    df["logret"] = np.log(df["Adj Close"] / df["Adj Close"].shift(1))
    daily_vol = df["logret"].dropna().std()
    return float(daily_vol * np.sqrt(252))

def compute_vol_for_tickers(tickers, price_folder, start_date, end_date):
    results = []
    for t in tickers:
        path = Path(price_folder) / f"{t}.csv"
        if not path.exists():
            print(f"Missing price for {t}")
            continue
        df = pd.read_csv(path)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.tz_localize(None)
        df = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)]
        if df.empty:
            print(f"No price rows for {t} in {start_date}..{end_date}")
            continue
        vol = compute_annualized_vol_from_df(df)
        results.append({"Ticker": t, "Sigma_E": vol})
    return pd.DataFrame(results)
