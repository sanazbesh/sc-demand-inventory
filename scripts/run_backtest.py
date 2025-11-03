from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data/raw")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

# ----------------------------
# 1. Load input data
# ----------------------------

# Demand history: weekly sales per (store, sku)
demand_df = pd.read_parquet(RAW / "fact_demand.parquet")
# (columns: week_start, loc_id, sku_id, sales_units, price, promo_flag, stockout_flag)

# Inventory snapshot: last known on-hand
inv_snap = pd.read_parquet(RAW / "inv_snap.parquet")
# (columns: loc_id, sku_id, on_hand, week_start)

# Location / SKU policy params
params = pd.read_parquet(RAW / "sku_location_params.parquet")
# (columns: sku_id, loc_id, lead_time_w, moq_units, target_fill_rate, ...)

# Make sure types are what we expect
demand_df["week_start"] = pd.to_datetime(demand_df["week_start"])
demand_df = demand_df.sort_values(["sku_id", "loc_id", "week_start"])

# ----------------------------
# 2. Pick a "high volume" pair to analyze
# ----------------------------
pair = (
    demand_df.groupby(["sku_id", "loc_id"])["sales_units"]
    .sum()
    .sort_values(ascending=False)
    .index[0]
)

sku_id, loc_id = pair
print(f"Analyzing sku_id={sku_id}, loc_id={loc_id}")

hist = (
    demand_df[(demand_df["sku_id"] == sku_id) & (demand_df["loc_id"] == loc_id)]
    .sort_values("week_start")
    .reset_index(drop=True)
)

# attach policy info (lead time, MOQ etc)
row_params = params[(params["sku_id"] == sku_id) & (params["loc_id"] == loc_id)].iloc[0]

lead_time = int(row_params["lead_time_w"])
moq_units = int(row_params["moq_units"])

# get initial on_hand from inv_snap if available, else default
match_inv = inv_snap[(inv_snap["sku_id"] == sku_id) & (inv_snap["loc_id"] == loc_id)]
if len(match_inv) > 0:
    starting_inventory = int(match_inv["on_hand"].iloc[-1])
else:
    starting_inventory = 100  # fallback

print(f"lead_time={lead_time}w, moq={moq_units} units, starting_inv={starting_inventory}")

# ----------------------------
# 3. Build naive demand forecast
# ----------------------------
# rolling average of last 8 weeks; require at least 4 weeks history
hist["forecast_units"] = hist["sales_units"].shift(1).rolling(window=8, min_periods=4).mean()

# if forecast is NaN early in series, fall back to observed demand
hist["forecast_units"] = hist["forecast_units"].fillna(hist["sales_units"])

# ----------------------------
# 4. Simulate weekly inventory behavior
# ----------------------------
# We implement a periodic-review policy:
# - Each week, you look at forecasted demand for the next X weeks of coverage.
# - If your effective projected inventory is below that, you place an order.
# - Orders arrive after lead_time weeks.
#
# Simplifications:
#   - No backorders, unmet demand becomes lost_sales.
#   - We round orders up to meet MOQ.
#   - We assume review every week.

coverage_weeks = 2  # we want enough stock for 2 weeks of forecast
pipeline = [0] * lead_time  # list of scheduled arrivals, length = lead_time

on_hand_list = []
pipeline_list = []
order_qty_list = []
lost_sales_list = []

inv = starting_inventory

for i, row in hist.iterrows():
    demand = float(row["sales_units"])
    fcast = float(row["forecast_units"])

    # 1. Receive arrivals due this week
    arriving = pipeline.pop(0) if len(pipeline) > 0 else 0
    inv += arriving

    # 2. Satisfy demand
    sellable = min(inv, demand)
    inv -= sellable
    lost = demand - sellable

    # 3. Compute reorder point and decide whether to order
    reorder_point_units = fcast * coverage_weeks

    # project inventory position = current inv + pipeline incoming
    projected_position = inv + sum(pipeline)

    if projected_position < reorder_point_units:
        # we want target level = 2x coverage
        target_level = 2 * reorder_point_units
        order_raw = max(target_level - projected_position, 0)

        # respect MOQ
        # round up to nearest multiple of moq_units
        if moq_units > 0:
            order_qty = int(np.ceil(order_raw / moq_units) * moq_units)
        else:
            order_qty = int(np.ceil(order_raw))
    else:
        order_qty = 0

    # 4. Push this order into the pipeline to arrive after lead_time weeks
    if lead_time > 0:
        # extend pipeline if somehow we popped more than length
        while len(pipeline) < lead_time - 1:
            pipeline.append(0)

        pipeline.append(order_qty)
    else:
        # "arrives immediately" if lead_time=0
        inv += order_qty

    # 5. track metrics for this week
    on_hand_list.append(inv)
    pipeline_list.append(sum(pipeline))
    order_qty_list.append(order_qty)
    lost_sales_list.append(lost)

# add simulation outputs back to hist
hist["on_hand"] = on_hand_list
hist["pipeline_units"] = pipeline_list
hist["order_qty"] = order_qty_list
hist["lost_sales"] = lost_sales_list

# ----------------------------
# 5. Compute KPIs
# ----------------------------
total_demand = hist["sales_units"].sum()
total_lost = hist["lost_sales"].sum()
fill_rate = 1 - (total_lost / total_demand if total_demand > 0 else 0)
avg_inventory = hist["on_hand"].mean()
order_volume = hist["order_qty"].sum()

print("------ KPI ------")
print(f"Fill rate        : {fill_rate:.1%}")
print(f"Total demand     : {total_demand:.0f}")
print(f"Total lost sales : {total_lost:.0f}")
print(f"Avg on-hand      : {avg_inventory:.1f}")
print(f"Total ordered    : {order_volume:.0f}")
print("-----------------")

# ----------------------------
# 6. Save output for the dashboard
# ----------------------------
out_detail = OUT / "backtest_single_pair.parquet"
hist.to_parquet(out_detail, index=False)

kpi_summary = pd.DataFrame(
    [
        {
            "sku_id": sku_id,
            "loc_id": loc_id,
            "fill_rate": fill_rate,
            "total_demand": total_demand,
            "total_lost_sales": total_lost,
            "avg_on_hand": avg_inventory,
            "total_ordered_units": order_volume,
            "lead_time_w": lead_time,
            "moq_units": moq_units,
            "coverage_weeks": coverage_weeks,
        }
    ]
)

out_kpi = OUT / "backtest_single_pair_kpi.parquet"
kpi_summary.to_parquet(out_kpi, index=False)

print(f"Saved detail to  {out_detail}")
print(f"Saved KPI to     {out_kpi}")
