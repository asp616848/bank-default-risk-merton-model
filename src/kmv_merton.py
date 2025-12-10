# src/kmv_merton.py
# KMV-Merton pipeline: compute V, sigma_V, Merton PD, KMV PD

import os
import numpy as np
import pandas as pd
from scipy import optimize, stats
from datetime import datetime, timedelta

# -----------------------------
# Utility functions
# -----------------------------

def get_fy_end_price(df, fy_end_date="2025-03-31"):
    """
    Given a dataframe with a 'Date' column and 'Adj Close', return the last adjusted close
    on or before fy_end_date (string 'YYYY-MM-DD' or datetime).
    df: raw price dataframe, expects Date column parseable by pandas
    Returns: (adj_close, price_date) or (None, None) if not found
    """
    if isinstance(fy_end_date, str):
        fy_end_date = pd.to_datetime(fy_end_date)
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    df = df[df['Date'] <= fy_end_date]
    if df.shape[0] == 0:
        return None, None
    last_row = df.iloc[-1]
    return float(last_row['Adj Close']), pd.to_datetime(last_row['Date'])


def compute_annualized_volatility(df, start_date=None, end_date=None, date_col='Date', price_col='Adj Close'):
    """
    Compute annualized equity volatility from daily log returns between start_date and end_date.
    If start_date or end_date are None, uses full df range.
    Returns annualized volatility (float).
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    if start_date is not None:
        df = df[df[date_col] >= pd.to_datetime(start_date)]
    if end_date is not None:
        df = df[df[date_col] <= pd.to_datetime(end_date)]
    df = df.sort_values(date_col)
    # require at least some observations
    if df.shape[0] < 5:
        return np.nan
    returns = np.log(df[price_col] / df[price_col].shift(1)).dropna()
    daily_std = returns.std(ddof=1)
    annualized = daily_std * np.sqrt(252)
    return float(annualized)


# -----------------------------
# Merton system solver
# -----------------------------

def merton_equations(vars_vec, E, sigma_E, D, r, T):
    """
    System of two equations to solve for V and sigma_V:
    1) E = V*N(d1) - D*exp(-rT)*N(d2)
    2) sigma_E = (V/E)*N(d1)*sigma_V
    where d1 = (ln(V/D) + (r + 0.5*sigma_V^2)T) / (sigma_V * sqrt(T))
          d2 = d1 - sigma_V*sqrt(T)
    vars_vec = [V, sigma_V]
    """
    V, sigma_V = vars_vec
    if V <= 0 or sigma_V <= 0:
        # return something that encourages solver to avoid non-positive values
        return [1e6, 1e6]
    sqrtT = np.sqrt(T)
    d1 = (np.log(V / D) + (r + 0.5 * sigma_V ** 2) * T) / (sigma_V * sqrtT)
    d2 = d1 - sigma_V * sqrtT
    Nd1 = stats.norm.cdf(d1)
    Nd2 = stats.norm.cdf(d2)
    eq1 = V * Nd1 - D * np.exp(-r * T) * Nd2 - E
    eq2 = (V / E) * Nd1 * sigma_V - sigma_E
    return [eq1, eq2]


def solve_for_assets(E, sigma_E, D, r=0.07, T=1.0, V0=None, sigmaV0=None, tol=1e-8):
    """
    Solve the Merton system for V and sigma_V using scipy.optimize.root.
    Returns dict with keys: V, sigma_V, d1, d2, success, message
    """
    # sanity checks and initial guesses
    if E <= 0 or D <= 0 or sigma_E <= 0:
        return {'V': np.nan, 'sigma_V': np.nan, 'd1': np.nan, 'd2': np.nan, 'success': False, 'message': 'bad inputs'}

    # initial guess for V: use market cap + D as starting point (or E + D)
    if V0 is None:
        V0 = max(E + D, D * 1.2)
    if sigmaV0 is None:
        # naive initial guess: asset vol smaller than equity vol scaled by lever (E/V)
        sigmaV0 = max(0.05, sigma_E * E / V0)

    x0 = np.array([V0, sigmaV0])

    try:
        sol = optimize.root(lambda x: merton_equations(x, E, sigma_E, D, r, T), x0, method='hybr', tol=tol)
        if not sol.success:
            # try different initial guesses (some robustness)
            x0_alt = np.array([E + D, max(0.1, sigma_E * 0.6)])
            sol = optimize.root(lambda x: merton_equations(x, E, sigma_E, D, r, T), x0_alt, method='hybr', tol=tol)
        if not sol.success:
            return {'V': np.nan, 'sigma_V': np.nan, 'd1': np.nan, 'd2': np.nan, 'success': False, 'message': sol.message}
        V, sigma_V = sol.x
        sqrtT = np.sqrt(T)
        d1 = (np.log(V / D) + (r + 0.5 * sigma_V ** 2) * T) / (sigma_V * sqrtT)
        d2 = d1 - sigma_V * sqrtT
        return {'V': float(V), 'sigma_V': float(sigma_V), 'd1': float(d1), 'd2': float(d2), 'success': True, 'message': sol.message}
    except Exception as e:
        return {'V': np.nan, 'sigma_V': np.nan, 'd1': np.nan, 'd2': np.nan, 'success': False, 'message': str(e)}


# -----------------------------
# KMV computations
# -----------------------------

def kmv_default_point(short_term_debt, long_term_debt):
    """
    Default point DP = ST + 0.5 * LT
    Inputs assumed in same units as D (e.g., rupees)
    """
    return float(short_term_debt) + 0.5 * float(long_term_debt)


def compute_kmv_metrics(V, sigma_V, short_term_debt, long_term_debt):
    """
    Given V and sigma_V (annualized), compute DP, DD (simple KMV form), and PD_kmv = N(-DD)
    Using DD = (V - DP) / (V * sigma_V)
    If sigma_V=0 or V<=0, returns NaNs.
    """
    DP = kmv_default_point(short_term_debt, long_term_debt)
    if V <= 0 or sigma_V <= 0:
        return {'DP': DP, 'DD': np.nan, 'PD_KMV': np.nan}
    DD = (V - DP) / (V * sigma_V)
    PD_kmv = float(stats.norm.cdf(-DD))
    return {'DP': float(DP), 'DD': float(DD), 'PD_KMV': PD_kmv}


# -----------------------------
# Top-level pipeline
# -----------------------------

def compute_kmv_for_ticker(ticker,
                           prices_folder,
                           fundamentals_df,
                           fy_end_date="2025-03-31",
                           vol_start_date="2020-04-01",
                           vol_end_date="2025-03-31",
                           r=0.07,
                           T=1.0):
    """
    Compute the KMV/Merton outputs for a single ticker.
    Expects:
      - prices_folder contains CSV files named TICKER.csv with Date and 'Adj Close'
      - fundamentals_df contains rows indexed or column 'ticker' with columns:
           'shares_outstanding', 'short_term_debt', 'long_term_debt'
      - amounts in fundamentals should be in rupees (not crores), consistent with market cap units
    Returns a dict with all computed outputs
    """
    fname = os.path.join(prices_folder, f"{ticker}.csv")
    if not os.path.exists(fname):
        return {'Ticker': ticker, 'error': f'Price file not found: {fname}'}

    # read price csv
    df = pd.read_csv(fname)
    # ensure columns exist
    if 'Adj Close' not in df.columns:
        # accept 'Adj_Close' or 'AdjClose' as fallback
        for alt in ['Adj_Close', 'AdjClose', 'AdjClose']:
            if alt in df.columns:
                df.rename(columns={alt: 'Adj Close'}, inplace=True)
        if 'Adj Close' not in df.columns:
            return {'Ticker': ticker, 'error': 'Adj Close column missing'}

    # get FY-end price
    fy_close, price_date = get_fy_end_price(df, fy_end_date)
    if fy_close is None:
        return {'Ticker': ticker, 'error': f'No FY-end price on/before {fy_end_date}'}

    # fetch fundamentals
    # allow fundamentals_df to be indexed by ticker or contain a 'ticker' column
    if ticker in fundamentals_df.index:
        row = fundamentals_df.loc[ticker]
    elif 'ticker' in fundamentals_df.columns:
        row = fundamentals_df[fundamentals_df['ticker'] == ticker].iloc[0]
    else:
        return {'Ticker': ticker, 'error': 'ticker not found in fundamentals file'}

    # required fields
    shares = float(row.get('shares_outstanding', np.nan))
    st_debt = float(row.get('short_term_debt', 0.0))
    lt_debt = float(row.get('long_term_debt', 0.0))

    if np.isnan(shares):
        return {'Ticker': ticker, 'error': 'shares_outstanding missing'}

    # compute market cap (E)
    E = fy_close * shares

    # compute equity vol over the specified period
    sigma_E = compute_annualized_volatility(df, start_date=vol_start_date, end_date=vol_end_date, date_col='Date', price_col='Adj Close')

    # compute face value of debt D (simple KMV / Merton approach) - you may override externally
    # For classic Merton we often use face value = ST + LT; here we will use total debt D = ST + LT
    D = st_debt + lt_debt

    # solve for asset value & vol
    sol = solve_for_assets(E=E, sigma_E=sigma_E, D=D, r=r, T=T)

    if not sol['success']:
        # still attempt compute KMV DP with NaN V/sigma_V
        kmv = compute_kmv_metrics(sol['V'], sol['sigma_V'], st_debt, lt_debt)
        out = {
            'Ticker': ticker,
            'AdjClose_FYend': fy_close,
            'Price_Date': price_date,
            'Shares_Outstanding': shares,
            'MarketCap_E': E,
            'Sigma_E': sigma_E,
            'D_total': D,
            'ST_debt': st_debt,
            'LT_debt': lt_debt,
            'V': sol['V'],
            'Sigma_V': sol['sigma_V'],
            'd1': sol['d1'],
            'd2': sol['d2'],
            'Merton_PD': np.nan,
            'KMV_DP': kmv['DP'],
            'KMV_DD': kmv['DD'],
            'KMV_PD': kmv['PD_KMV'],
            'solve_message': sol['message']
        }
        return out

    # compute Merton PD from d2
    d2 = sol['d2']
    merton_pd = float(stats.norm.cdf(-d2))

    kmv = compute_kmv_metrics(sol['V'], sol['sigma_V'], st_debt, lt_debt)

    out = {
        'Ticker': ticker,
        'AdjClose_FYend': fy_close,
        'Price_Date': pd.to_datetime(price_date).date(),
        'Shares_Outstanding': shares,
        'MarketCap_E': E,
        'Sigma_E': sigma_E,
        'D_total': D,
        'ST_debt': st_debt,
        'LT_debt': lt_debt,
        'V': sol['V'],
        'Sigma_V': sol['sigma_V'],
        'd1': sol['d1'],
        'd2': sol['d2'],
        'Merton_PD': merton_pd,
        'KMV_DP': kmv['DP'],
        'KMV_DD': kmv['DD'],
        'KMV_PD': kmv['PD_KMV'],
        'solve_message': sol['message']
    }
    return out


def compute_kmv_for_all_tickers(tickers, prices_folder, fundamentals_df, **kwargs):
    """
    Convenience function to compute KMV/Merton for a list of tickers.
    Returns a DataFrame with all results.
    """
    results = []
    for t in tickers:
        print(f"Computing: {t}")
        res = compute_kmv_for_ticker(ticker=t,
                                     prices_folder=prices_folder,
                                     fundamentals_df=fundamentals_df,
                                     **kwargs)
        results.append(res)
    return pd.DataFrame(results)


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    # Example: run for several tickers using a fundamentals CSV and a prices folder
    PRICES_FOLDER = "data/prices"   # change to your folder, e.g., "data/prices" or "nse_prices"
    FUND_FILE = "data/fundamentals.csv" # path to your fundamentals CSV with ticker, shares_outstanding, short_term_debt, long_term_debt
    
    # read fundamentals
    fund = pd.read_csv(FUND_FILE)
    # optionally set index to ticker
    if 'ticker' in fund.columns:
        fund.set_index('ticker', inplace=True)

    tickers = ["SBIBANK", "BANKBARODA", "CANBK", "HDFCBANK", "ICICIBANK", "AXISBANK", "KOTAKBANK", "INDUSINDBK", "BAJFINANCE", "PNB"]
    
    out_df = compute_kmv_for_all_tickers(
        tickers=tickers,
        prices_folder=PRICES_FOLDER,
        fundamentals_df=fund,
        fy_end_date="2025-03-31",
        vol_start_date="2020-04-01",
        vol_end_date="2025-03-31",
        r=0.075,  # adjust risk-free rate as needed (e.g., 7.5%)
        T=1.0
    )
    
    out_df.to_csv("merton_kmv_results_FY2025.csv", index=False)
    print("Saved results to merton_kmv_results_FY2025.csv")
