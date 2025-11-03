from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ----------------------------------
# Page configuration
# ----------------------------------
st.set_page_config(page_title="Demand Forecast Dashboard", layout="wide")
PROC = Path("data/processed")

st.title("📦 Demand Forecast Dashboard")


# ----------------------------------
# Load forecast file (robust)
# ----------------------------------
@st.cache_data
def load_forecast() -> pd.DataFrame:
    fp = PROC / "forecast_future_tft.parquet"
    df = pd.read_parquet(fp).copy()

    # ensure plotting date column
    if "week_start" not in df.columns:
        if "time_idx" in df.columns:
            base = pd.Timestamp("2000-01-03")  # Monday anchor
            df["week_start"] = base + pd.to_timedelta(df["time_idx"].astype(int), unit="W")
        else:
            df["week_start"] = pd.to_datetime("2000-01-03") + pd.to_timedelta(
                np.arange(len(df)), unit="W"
            )
    df["week_start"] = pd.to_datetime(df["week_start"])
    return df


# Always stop the app cleanly if loading fails
try:
    df = load_forecast()
    if df is None or df.empty:
        st.error("Forecast file is empty. Re-run `python scripts/forecast_tft.py`.")
        st.stop()
except FileNotFoundError:
    st.error(
        "No forecast file found.\n\n"
        "Run `python scripts/forecast_tft.py` first to generate "
        "`data/processed/forecast_future_tft.parquet`."
    )
    st.stop()
except Exception as e:
    st.error(f"Couldn't load forecast file: {type(e).__name__}: {e}")
    st.stop()

# ----------------------------------
# Metadata & overview metrics
# ----------------------------------
sku_id = df["sku_id"].iloc[0] if "sku_id" in df.columns and len(df) else "n/a"
loc_id = df["loc_id"].iloc[0] if "loc_id" in df.columns and len(df) else "n/a"

hist_df = df[~df["is_future"]].copy()
fut_df = df[df["is_future"]].copy()

recent_window = 8
recent_actual_avg = (
    hist_df["actual_units"].tail(recent_window).mean() if not hist_df.empty else float("nan")
)
future_forecast_avg = fut_df["forecast_units"].mean() if not fut_df.empty else float("nan")
forecast_horizon_weeks = len(fut_df)

st.subheader("Overview")
c1, c2, c3 = st.columns(3)
c1.metric(
    "Avg demand (last 8 wks)",
    f"{recent_actual_avg:0.1f} units/wk" if pd.notna(recent_actual_avg) else "n/a",
)
c2.metric(
    "Avg forecast (next period)",
    f"{future_forecast_avg:0.1f} units/wk" if pd.notna(future_forecast_avg) else "n/a",
)
c3.metric("Forecast horizon", f"{forecast_horizon_weeks} weeks")

st.caption(
    f"SKU `{sku_id}` at Location `{loc_id}` · "
    "Model: Temporal Fusion Transformer · Horizon is multi-week ahead"
)
st.divider()

# ----------------------------------
# Actuals vs Forecast Plot
# ----------------------------------
st.subheader("Actuals vs Forecast")

if hist_df.empty and fut_df.empty:
    st.warning("No forecast data to display.")
    st.stop()

plot_hist = hist_df[["week_start", "actual_units"]].rename(columns={"actual_units": "units"})
plot_hist["series"] = "actual"

plot_fut = fut_df[["week_start", "forecast_units"]].rename(columns={"forecast_units": "units"})
plot_fut["series"] = "forecast"

plot_all = pd.concat([plot_hist, plot_fut], ignore_index=True)
plot_all["series"] = pd.Categorical(
    plot_all["series"], categories=["actual", "forecast"], ordered=True
)

fig_forecast = px.line(
    plot_all,
    x="week_start",
    y="units",
    color="series",
    markers=True,
    labels={"week_start": "Week Start", "units": "Units", "series": "Series"},
    title="Recent Actual Demand vs Predicted Future Demand",
)
fig_forecast.update_layout(legend_title_text="")
st.plotly_chart(fig_forecast, use_container_width=True)
st.divider()

# ----------------------------------
# Raw table
# ----------------------------------
st.subheader("Forecast Table (Recent History + Future Horizon)")
table_df = (
    df[["week_start", "actual_units", "forecast_units", "is_future"]]
    .sort_values("week_start")
    .rename(
        columns={
            "week_start": "Week Start",
            "actual_units": "Actual Units",
            "forecast_units": "Forecast Units",
            "is_future": "Future?",
        }
    )
)
st.dataframe(table_df, use_container_width=True)
st.caption(
    "Note: Actual Units are shown for history only. Forecast Units are shown for future weeks."
)
