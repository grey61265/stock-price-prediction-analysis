import yfinance as yf
from pathlib import Path
import pandas as pd

TICKERS = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "ITC.NS", "SBIN.NS", "LT.NS", "BHARTIARTL.NS", "HINDUNILVR.NS",
]


START_DATE = "2016-09-30"
END_DATE = "2026-09-30"

raw_dir = Path(__file__).resolve().parents[1] / "data" / "raw"
raw_dir.mkdir(parents=True, exist_ok=True)

frames = []
for t in TICKERS:
    df = yf.download(t, start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)
    df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    df.insert(1, "Ticker", t)
    frames.append(df)
    print(f"{t}: {len(df)} rows")

prices = pd.concat(frames, ignore_index=True)
prices.to_csv(raw_dir / "prices.csv", index=False)
print(prices.shape)