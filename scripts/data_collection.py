import yfinance as yf
from pathlib import Path

TICKERS = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "ITC.NS", "SBIN.NS", "LT.NS", "BHARTIARTL.NS", "HINDUNILVR.NS",
]


START_DATE = "2016-09-30"
END_DATE = "2026-09-30"

def main():
    output_dir = Path(__file__).parent.parent / "data" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)

    for ticker in TICKERS:
        print(f"Downloading {ticker}...")

        data = yf.download(
            ticker,
            start=START_DATE,
            end=END_DATE,
            auto_adjust=False
        )

        filename = f"{ticker}_{START_DATE}_{END_DATE}.csv"
        data.to_csv(output_dir / filename)

        print(f"Saved {filename}")


if __name__ == "__main__":
    main()