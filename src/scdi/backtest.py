import pandas as pd

from .forecast import forecast_quantiles, train_pooled
from .inventory import order_up_to, reorder_point, round_constraints


def rolling_backtest(df: pd.DataFrame, start_week, horizons=8, case_pack=1, moq=0, lead_time=2):
    his = df[df["week_start"] < start_week].copy()
    fut = df[df["week_start"] >= start_week].copy()

    mdl = train_pooled(his.dropna(subset=["sales_units"]))
    records = []

    on_hand = his["on_hand"].iloc[-1] if "on_hand" in his else 0
    on_order = 0
    last_order = 0

    weeks = sorted(fut["week_start"].unique())
    for t, wk in enumerate(weeks):
        ctx = pd.concat([his, fut[fut["week_start"] <= wk]]).copy().tail(horizons)
        ctx = forecast_quantiles(mdl, ctx)

        s = reorder_point(ctx, lead_time=lead_time).iloc[-1]
        S = order_up_to(ctx, s).iloc[-1]

        inv_pos = on_hand + on_order
        order_qty = max(S - inv_pos, 0)
        order_qty = round_constraints(order_qty, case_pack=case_pack, moq=moq)

        if t >= lead_time:
            on_hand += last_order
            on_order = 0

        demand_t = fut.loc[fut["week_start"] == wk, "sales_units"].sum()
        sales_t = min(on_hand, demand_t)
        on_hand -= sales_t
        lost_sales = max(demand_t - sales_t, 0)

        last_order = order_qty
        on_order += order_qty

        records.append(
            dict(
                week=wk,
                demand=demand_t,
                sales=sales_t,
                lost_sales=lost_sales,
                order=order_qty,
                on_hand=on_hand,
                s=float(s),
                S=float(S),
            )
        )
    return pd.DataFrame(records)
