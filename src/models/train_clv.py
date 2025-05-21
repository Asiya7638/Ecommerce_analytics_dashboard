# src/models/train_clv.py

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import joblib

def prepare_features(df, snapshot_date, label_window_days=90):
    df_hist = df[df['InvoiceDate'] < snapshot_date].copy()
    rfm = (
        df_hist.assign(Revenue=df_hist.Quantity * df_hist.UnitPrice)
              .groupby('CustomerID')
              .agg(
                  Recency=('InvoiceDate', lambda x: (snapshot_date - x.max()).days),
                  Frequency=('InvoiceNo', 'nunique'),
                  Monetary=('Revenue', 'sum')
              )
    )
    df_future = df[
        (df['InvoiceDate'] >= snapshot_date) &
        (df['InvoiceDate'] < snapshot_date + pd.Timedelta(days=label_window_days))
    ].copy()
    labels = (
        df_future.assign(Revenue=df_future.Quantity * df_future.UnitPrice)
                 .groupby('CustomerID')['Revenue']
                 .sum()
                 .rename('FutureRevenue')
    )
    data = rfm.join(labels, how='left').fillna(0)
    return data

def main():
    # ─── locate project root ────────────────────────────────────────────────────
    # train_clv.py is at: project/src/models/train_clv.py
    # we want: project/data/processed/cleaned_retail.csv
    root = Path(__file__).parents[2]  # go up two levels: src/models → src → project
    data_path = root / 'data' / 'processed' / 'cleaned_retail.csv'
    if not data_path.exists():
        raise FileNotFoundError(f"Could not find cleaned data at {data_path}")

    # ─── load cleaned retail data ───────────────────────────────────────────────
    df = pd.read_csv(data_path, parse_dates=['InvoiceDate'])

    # ─── choose snapshot and build features + label ─────────────────────────────
    snapshot = pd.to_datetime('2011-09-10')
    data = prepare_features(df, snapshot, label_window_days=90)

    # ─── train/test split ────────────────────────────────────────────────────────
    X = data[['Recency', 'Frequency', 'Monetary']]
    y = data['FutureRevenue']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ─── train a simple GBM ──────────────────────────────────────────────────────
    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    model.fit(X_train, y_train)

    # ─── evaluate ────────────────────────────────────────────────────────────────
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print(f"Test RMSE: £{rmse:,.2f}")

    # ─── save model artifact ─────────────────────────────────────────────────────
    out_dir = root / 'models'
    out_dir.mkdir(exist_ok=True)
    artifact = {'model': model, 'snapshot': snapshot}
    joblib.dump(artifact, out_dir / 'clv_model.pkl')
    print("Model saved to", out_dir / 'clv_model.pkl')

if __name__ == "__main__":
    main()
