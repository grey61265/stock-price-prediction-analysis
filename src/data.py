import pandas as pd

FEATURE_COLS = [
    "ret_5", "ret_20", "gap", "body",
    "close_pos", "ma20_ratio", "vol_20", "vol_ratio"
]


def load_features(path="data/clean/features.csv"):
    df = pd.read_csv(path, parse_dates=["Date"])
    
    return df.sort_values(["Ticker", "Date"]).reset_index(drop=True)


def time_split(dat, train_frac=0.8):
    train_sets, test_sets = {}, {}
    
    for ticker, t in dat.groupby("Ticker"):
        
        cut = int(len(t) * train_frac)
        train_sets[ticker] = t.iloc[:cut]
        test_sets[ticker] = t.iloc[cut:]
        
        
    return train_sets, test_sets