import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from src.data import FEATURE_COLS
from src.lr import LinearRegressionScratch
from src.baseline import MeanBaseline


def get_models():
    
    # place holder for more models in future
    
    return {
        "LR (scratch)": LinearRegressionScratch(),
        "LR (sklearn)": LinearRegression(),
        "Baseline": MeanBaseline(),
    }


def metrics(y_true, y_pred):
    return {
        "MAE": np.mean(np.abs(y_true - y_pred)),
        "RMSE": np.sqrt(np.mean((y_true - y_pred) ** 2)),
        "R2": r2_score(y_true, y_pred),
        "DirAcc": np.mean((y_pred > 0) == (y_true > 0)),
    }


def run_one(ticker, model_name, train_sets, test_sets):
    """Train one model on one ticker, return the test period with predictions."""
    tr, te = train_sets[ticker], test_sets[ticker]
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(tr[FEATURE_COLS])
    Xte = scaler.transform(te[FEATURE_COLS])

    model = get_models()[model_name]
    model.fit(Xtr, tr["target"].values)

    return pd.DataFrame({
        "Date": te["Date"].values,
        "actual": te["target"].values,
        "predicted": model.predict(Xte),
    })


def evaluate_all(train_sets, test_sets):
    rows = []
    for tk in train_sets:
        for name in get_models():
            out = run_one(tk, name, train_sets, test_sets)
            rows.append({"Ticker": tk, "Model": name,
                         **metrics(out["actual"].values, out["predicted"].values)})
    return pd.DataFrame(rows)

def predict_next_day(ticker, model_name, feat, latest):
    """Train on all history for this ticker, predict the return after the latest day."""
    tr = feat[feat["Ticker"] == ticker]
    row = latest[latest["Ticker"] == ticker]

    scaler = StandardScaler()
    X = scaler.fit_transform(tr[FEATURE_COLS])

    model = get_models()[model_name]
    model.fit(X, tr["target"].values)

    pred = model.predict(scaler.transform(row[FEATURE_COLS]))[0]
    return float(pred), row["Date"].iloc[0]