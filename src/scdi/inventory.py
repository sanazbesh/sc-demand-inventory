import numpy as np
import pandas as pd
from .config import LEAD_TIME_W, ORDER_CYCLE_W, TARGET_SERVICE

_Z_MAP = {0.90: 1.282, 0.95: 1.645, 0.97: 1.881, 0.98: 2.054, 0.99: 2.326}

def _z(alpha: float) -> float:
    return _Z_MAP.get(round(alpha, 2), 1.645)

def protection_agg(df: pd.DataFrame, horizon: int):
    mu = df["q50"].rolling(horizon).sum()
    sig = (df["q90"] - df["q10"]).rolling(horizon).sum() / 2.56  # crude proxy
    return mu, sig

def reorder_point(df_fore: pd.DataFrame, target_service: float = TARGET_SERVICE,
                  lead_time: int = LEAD_TIME_W, order_cycle: int = ORDER_CYCLE_W) -> pd.Series:
    prot = lead_time + order_cycle
    mu, sig = protection_agg(df_fore, prot)
    s = (mu + _z(target_service) * sig).clip(lower=0)
    return s

def order_up_to(df_fore: pd.DataFrame, s: pd.Series, order_cycle: int = ORDER_CYCLE_W) -> pd.Series:
    mu_next = df_fore["q50"].rolling(order_cycle).sum()
    return (s + mu_next).clip(lower=0)

def round_constraints(order_qty: float, case_pack: int = 1, moq: int = 0) -> int:
    qty = max(order_qty, moq)
    if case_pack > 1:
        qty = int(np.ceil(qty / case_pack) * case_pack)
    return int(qty)
