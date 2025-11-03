from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data.encoders import GroupNormalizer
from pytorch_forecasting.metrics import RMSE

RAW = Path("data/raw")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

# Longer lookback helps capture seasonality & bursts
MAX_ENCODER_LENGTH = 78  # was 52
MAX_PREDICTION_LENGTH = 8


# -------------------------------
# Helpers
# -------------------------------
def _make_time_index(sub: pd.DataFrame) -> pd.DataFrame:
    sub = sub.copy()
    sub["time_idx"] = ((sub["week_start"] - sub["week_start"].min()).dt.days // 7).astype(int)
    return sub


def _add_calendar_feats(sub: pd.DataFrame) -> pd.DataFrame:
    sub = sub.copy()
    sub["week_of_year"] = sub["week_start"].dt.isocalendar().week.astype(int)
    sub["sin_woy"] = np.sin(2 * np.pi * sub["week_of_year"] / 52.0)
    sub["cos_woy"] = np.cos(2 * np.pi * sub["week_of_year"] / 52.0)
    return sub


def _add_sma_feats(df: pd.DataFrame) -> pd.DataFrame:
    # short moving averages help the decoder stay near recent level/shape
    df = df.copy()
    df["sma_4"] = df["sales_units"].rolling(4, min_periods=1).mean()
    df["sma_8"] = df["sales_units"].rolling(8, min_periods=1).mean()
    return df


# -------------------------------
# Data prep
# -------------------------------
def load_top_series():
    df = pd.read_parquet(RAW / "fact_demand.parquet")
    df["week_start"] = pd.to_datetime(df["week_start"])

    # pick (sku, loc) with largest volume
    sku_id, loc_id = (
        df.groupby(["sku_id", "loc_id"])["sales_units"].sum().sort_values(ascending=False).index[0]
    )
    print(f"[TFT] Using sku_id={sku_id}, loc_id={loc_id}")

    sub = (
        df[(df["sku_id"] == sku_id) & (df["loc_id"] == loc_id)]
        .sort_values("week_start")
        .reset_index(drop=True)
        .copy()
    )

    sub["promo_flag"] = sub["promo_flag"].astype(float)
    sub["price"] = sub["price"].ffill().bfill()
    sub["sales_units"] = sub["sales_units"].astype(float)
    sub["sku_id"] = sub["sku_id"].astype(str)
    sub["loc_id"] = sub["loc_id"].astype(str)

    sub = _make_time_index(sub)
    sub = _add_calendar_feats(sub)
    sub = _add_sma_feats(sub)
    return sub, str(sku_id), str(loc_id)


def make_datasets(series_df):
    df = series_df.copy()
    df["sku_id"] = df["sku_id"].astype(str)
    df["loc_id"] = df["loc_id"].astype(str)
    df["time_idx"] = df["time_idx"].astype(int)

    # targets & known reals
    df["log_sales"] = np.log1p(df["sales_units"])
    df["price_scaled"] = (df["price"] - df["price"].mean()) / (df["price"].std() + 1e-9)
    df["promo_flag"] = df["promo_flag"].astype(int)

    # short moving-averages (NO LAGS for these)
    if "sma_4" not in df.columns:
        df["sma_4"] = df["sales_units"].rolling(4, min_periods=1).mean()
    if "sma_8" not in df.columns:
        df["sma_8"] = df["sales_units"].rolling(8, min_periods=1).mean()

    # lags only for log_sales, bounded by history
    candidate_lags = [1, 2, 3, 4, 6, 8, 12, 16, 26, 52]
    max_time_idx = int(df["time_idx"].max())
    lags = [lag for lag in candidate_lags if lag <= max_time_idx]
    lag_max = max(lags) if lags else 0

    # split
    training_cutoff = max_time_idx - MAX_PREDICTION_LENGTH
    train_df = df[df["time_idx"] <= training_cutoff].copy()

    # include enough history for encoder + lag in val
    val_start = training_cutoff - MAX_ENCODER_LENGTH - lag_max
    val_df = df[df["time_idx"] >= max(0, val_start)].copy()

    needed_hist = MAX_ENCODER_LENGTH + lag_max + 1
    if (train_df["time_idx"].nunique() < needed_hist) or (
        val_df["time_idx"].nunique() < needed_hist
    ):
        enc = min(MAX_ENCODER_LENGTH, max(12, train_df["time_idx"].nunique() // 2))
        lags = [lag for lag in lags if lag <= max(6, enc - 1)]
    else:
        enc = MAX_ENCODER_LENGTH

    train_dataset = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target="log_sales",
        group_ids=["sku_id", "loc_id"],
        max_encoder_length=enc,
        min_encoder_length=1,
        max_prediction_length=MAX_PREDICTION_LENGTH,
        static_categoricals=["sku_id", "loc_id"],
        time_varying_known_reals=[
            "time_idx",
            "price_scaled",
            "promo_flag",
            "sin_woy",
            "cos_woy",
            "sma_4",
            "sma_8",  # <- keep as known reals (no lags)
        ],
        time_varying_unknown_reals=["log_sales"],
        # <- lags ONLY for log_sales to avoid sklearn feature-name mismatch
        lags={"log_sales": lags} if lags else None,
        target_normalizer=GroupNormalizer(groups=["sku_id", "loc_id"]),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )

    val_dataset = TimeSeriesDataSet.from_dataset(
        train_dataset, val_df, predict=False, stop_randomization=True
    )

    train_loader = train_dataset.to_dataloader(train=True, batch_size=64, num_workers=0)
    val_loader = val_dataset.to_dataloader(train=False, batch_size=64, num_workers=0)
    return train_dataset, train_loader, val_loader


def train_tft(dataset, train_loader, val_loader):
    tft = TemporalFusionTransformer.from_dataset(
        dataset,
        learning_rate=1e-3,
        hidden_size=64,
        attention_head_size=4,
        dropout=0.10,
        hidden_continuous_size=32,
        loss=RMSE(),  # <-- use RMSE to avoid quantiles machinery
        log_interval=10,
        log_val_interval=1,
        reduce_on_plateau_patience=6,
    )

    tb_logger = TensorBoardLogger(
        save_dir="lightning_logs", name="tft_run", default_hp_metric=False
    )
    early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=10)
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    trainer = Trainer(
        max_epochs=120,
        gradient_clip_val=0.1,
        enable_checkpointing=False,
        enable_progress_bar=True,
        logger=tb_logger,
        callbacks=[early_stop, lr_monitor],
    )
    trainer.fit(model=tft, train_dataloaders=train_loader, val_dataloaders=val_loader)
    return tft


# -------------------------------
# Forecast (with local calibration)
# -------------------------------
def generate_future_forecast(
    model,
    dataset,
    full_df,
    sku_id,
    loc_id,
    target_h: int | None = None,  # total weeks to forecast; None -> dataset.max_prediction_length
    w_tft: float = 0.20,  # tri-blend weights
    w_woy: float = 0.50,
    w_recent: float = 0.30,
    recent_k: int = 8,  # for calibration & recent pattern
    clip_window: int = 26,  # for guard rails
):
    """
    Blocked prediction with padding: for each block we provide a full decoder
    window (pred_len). If the final block is shorter, pad extra future weeks so
    PF can slice a valid sample, then discard the padded predictions.
    Carries forward predicted log_sales so lags work across blocks.
    """

    pred_len = dataset.max_prediction_length
    enc_len = dataset.max_encoder_length
    H = int(target_h) if target_h is not None else pred_len

    sku_id = str(sku_id)
    loc_id = str(loc_id)

    # ===== slice & prep (match training) =====
    df = (
        full_df.query("sku_id == @sku_id and loc_id == @loc_id")
        .sort_values("time_idx")
        .reset_index(drop=True)
        .copy()
    )
    if df.empty:
        raise ValueError(f"No data for sku_id={sku_id}, loc_id={loc_id}")

    mu, sigma = df["price"].mean(), df["price"].std() + 1e-9
    df["price_scaled"] = (df["price"] - mu) / sigma
    df["promo_flag"] = df["promo_flag"].astype(int)
    df["log_sales"] = np.log1p(df["sales_units"])
    df["sma_4"] = df["sales_units"].rolling(4, min_periods=1).mean()
    df["sma_8"] = df["sales_units"].rolling(8, min_periods=1).mean()

    lag_dict = getattr(dataset, "lags", None)
    lag_max = max(max(v) for v in lag_dict.values() if len(v) > 0) if lag_dict else 0
    tail = df.tail(max(enc_len, enc_len + lag_max)).copy()

    last_t = int(df["time_idx"].iloc[-1])
    last_day = pd.to_datetime(df["week_start"].iloc[-1])

    # full horizon index/dates for convenience
    fut_idx_full = np.arange(last_t + 1, last_t + 1 + H, dtype=int)
    fut_dates_full = [last_day + timedelta(days=7 * i) for i in range(1, H + 1)]

    # ===== add sin/cos to tail & init rolling history buffer =====
    tail = tail.copy()
    tail_woy = pd.to_datetime(tail["week_start"]).dt.isocalendar().week.astype(int)
    tail["sin_woy"] = np.sin(2 * np.pi * tail_woy / 52.0)
    tail["cos_woy"] = np.cos(2 * np.pi * tail_woy / 52.0)

    hist_buf = tail[
        [
            "time_idx",
            "week_start",
            "sku_id",
            "loc_id",
            "price_scaled",
            "promo_flag",
            "sma_4",
            "sma_8",
            "sin_woy",
            "cos_woy",
            "log_sales",
        ]
    ].copy()

    # ===== block-wise TFT with padding on final block =====
    y_tft_all = []
    ptr = 0
    while ptr < H:
        # how many *real* steps we still need
        blk_real = min(pred_len, H - ptr)

        # we always supply a *full* pred_len window to PF
        seg_start_t = last_t + 1 + ptr
        seg_idx_full = np.arange(seg_start_t, seg_start_t + pred_len, dtype=int)

        # dates for the full block (pad beyond original horizon if needed)
        seg_dates_full = [
            last_day + timedelta(days=7 * i)
            for i in range((seg_start_t - (last_t + 1)), (seg_start_t - (last_t + 1)) + pred_len)
        ]
        seg_woy_full = pd.Series(seg_dates_full).dt.isocalendar().week.astype(int)

        fut_known_full = pd.DataFrame(
            {
                "time_idx": seg_idx_full,
                "week_start": seg_dates_full,
                "sku_id": sku_id,
                "loc_id": loc_id,
                "price_scaled": float(df["price_scaled"].iloc[-1]),
                "promo_flag": 0,
                "sma_4": float(df["sma_4"].iloc[-1]),
                "sma_8": float(df["sma_8"].iloc[-1]),
                "sin_woy": np.sin(2 * np.pi * seg_woy_full / 52.0),
                "cos_woy": np.cos(2 * np.pi * seg_woy_full / 52.0),
                # seed unknown real with last finite log_sales
                "log_sales": float(hist_buf["log_sales"].iloc[-1]),
            }
        )

        need = max(enc_len + lag_max, enc_len)
        enc_ctx = hist_buf.tail(need).copy()

        pred_frame = pd.concat([enc_ctx, fut_known_full], ignore_index=True)
        for c in [
            "log_sales",
            "price_scaled",
            "promo_flag",
            "sin_woy",
            "cos_woy",
            "sma_4",
            "sma_8",
            "time_idx",
        ]:
            pred_frame[c] = (
                pred_frame[c].replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)
            )

        predict_ds = TimeSeriesDataSet.from_dataset(
            dataset,
            data=pred_frame,
            predict=True,
            stop_randomization=True,
            min_prediction_idx=int(seg_idx_full[0]),
        )

        y_blk_full = np.asarray(model.predict(predict_ds)).reshape(-1)
        y_blk_full = np.expm1(y_blk_full)  # units

        # take only the *real* portion for this horizon
        y_blk = y_blk_full[:blk_real].copy()
        y_tft_all.append(y_blk)

        # roll history forward with only the real part (on log scale)
        add_hist = fut_known_full.iloc[:blk_real].copy()
        add_hist["log_sales"] = np.log1p(y_blk)
        hist_buf = pd.concat([hist_buf, add_hist], ignore_index=True)

        # bound buffer
        hist_buf = hist_buf.tail(max(enc_len + lag_max + pred_len, 4 * enc_len)).reset_index(
            drop=True
        )
        ptr += blk_real

    y_tft = np.concatenate(y_tft_all, axis=0)

    # ===== WOY profile & repeated recent pattern =====
    woy_full_h = pd.Series(fut_dates_full).dt.isocalendar().week.astype(int)
    df["_woy"] = pd.to_datetime(df["week_start"]).dt.isocalendar().week.astype(int)
    woy_median = df.groupby("_woy")["sales_units"].median()
    y_woy = np.array(
        [float(woy_median.get(int(w), df["sma_8"].iloc[-1])) for w in woy_full_h], dtype=float
    )

    recent = df["sales_units"].tail(max(1, recent_k)).to_numpy(dtype=float)
    if recent.size == 0:
        recent = np.array([float(df["sma_8"].iloc[-1])], dtype=float)
    y_rep = np.resize(recent, H)

    # ===== tri-blend =====
    tot = max(1e-9, w_tft + w_woy + w_recent)
    y_blend = (w_tft / tot) * y_tft + (w_woy / tot) * y_woy + (w_recent / tot) * y_rep

    # ===== level & variance calibration =====
    recent_win = df["sales_units"].tail(max(2, recent_k)).to_numpy(dtype=float)
    mu_r = float(np.mean(recent_win))
    sd_r = float(np.std(recent_win, ddof=1)) if recent_win.size > 1 else 0.0
    mu_h = float(np.mean(y_blend))
    sd_h = float(np.std(y_blend, ddof=1)) if y_blend.size > 1 else 0.0

    if sd_h > 1e-6 and sd_r > 0:
        scale = np.clip(sd_r / sd_h, 0.5, 3.0)
        y_blend = mu_h + (y_blend - mu_h) * scale

    y_final = y_blend + (mu_r - float(np.mean(y_blend)))

    # ===== guard rails =====
    recent_clip = df["sales_units"].tail(clip_window)
    lo = float(np.percentile(recent_clip, 5)) if len(recent_clip) else 0.0
    hi = float(np.percentile(recent_clip, 95)) if len(recent_clip) else np.inf
    y_final = np.clip(y_final, max(0.0, lo * 0.7), hi * 1.6)

    # ===== save parquet =====
    hist_out = tail[["time_idx", "week_start", "sku_id", "loc_id", "sales_units"]].copy()
    hist_out.rename(columns={"sales_units": "actual_units"}, inplace=True)
    hist_out["forecast_units"] = np.nan
    hist_out["is_future"] = False

    fut_out = pd.DataFrame(
        {
            "time_idx": fut_idx_full,
            "week_start": fut_dates_full,
            "sku_id": sku_id,
            "loc_id": loc_id,
            "actual_units": np.nan,
            "forecast_units": y_final.astype(float),
            "is_future": True,
        }
    )

    combined = pd.concat([hist_out, fut_out], ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(OUT / "forecast_future_tft.parquet", index=False)
    print(
        f"[TFT] Saved forecast_future_tft.parquet (H={H}) ✅ [blocks={int(np.ceil(H / pred_len))}]"
    )


def main():
    full_df, sku_id, loc_id = load_top_series()
    dataset, train_loader, val_loader = make_datasets(full_df)
    model = train_tft(dataset, train_loader, val_loader)
    generate_future_forecast(model, dataset, full_df, sku_id, loc_id, target_h=52)


if __name__ == "__main__":
    main()
