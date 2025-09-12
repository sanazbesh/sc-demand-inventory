import pandas as pd

FEATURES = [
    "price", "promo_flag", "week_of_year", "ma_4", "std_4", "si_52",
    "lag_1","lag_2","lag_3","lag_4","lag_8","lag_13","lag_26","lag_52"
]

def build_panel(fact_demand: pd.DataFrame, calendar: pd.DataFrame, params: pd.DataFrame) -> pd.DataFrame:
    df = (fact_demand.merge(calendar, on="week_start", how="left")
                    .merge(params, on=["sku_id","loc_id"], how="left"))
    df = df.sort_values(["sku_id","loc_id","week_start"])
    return df

def add_lags(df: pd.DataFrame, lags=(1,2,3,4,8,13,26,52)) -> pd.DataFrame:
    for l in lags:
        df[f"lag_{l}"] = df.groupby(["sku_id","loc_id"])["sales_units"].shift(l)
    return df

def add_rolls(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["sku_id","loc_id"])["sales_units"]
    df["ma_4"]  = g.transform(lambda s: s.shift(1).rolling(4).mean())
    df["std_4"] = g.transform(lambda s: s.shift(1).rolling(4).std())
    df["si_52"] = df["lag_52"] / (g.transform(lambda s: s.shift(1).rolling(52).mean()))
    return df
