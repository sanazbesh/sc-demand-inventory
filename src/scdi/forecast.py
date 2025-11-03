import pandas as pd

from .data import FEATURES
from .model import QuantileGBM


def train_pooled(train_df: pd.DataFrame) -> QuantileGBM:
    X = train_df[FEATURES].fillna(0)
    y = train_df["sales_units"].astype(float)
    mdl = QuantileGBM().fit(X, y)
    return mdl


def forecast_quantiles(mdl: QuantileGBM, df_future: pd.DataFrame) -> pd.DataFrame:
    Xf = df_future[FEATURES].fillna(0)
    q_preds = mdl.predict(Xf)
    for i, q in enumerate(mdl.quantiles):
        df_future[f"q{int(q*100)}"] = q_preds[:, i]
    return df_future
