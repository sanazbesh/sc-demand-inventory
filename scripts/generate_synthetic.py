from pathlib import Path

import numpy as np
import pandas as pd

np.random.seed(42)

# -----------------
# config
# -----------------
DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

WEEKS = 208  # ~4 years
N_STORES = 50
N_SKUS = 400
dates = pd.date_range("2022-01-03", periods=WEEKS, freq="W-MON")  # Mondays


# -----------------
# dimension tables
# -----------------
stores = pd.DataFrame(
    {
        "loc_id": np.arange(1, N_STORES + 1),
        "loc_type": "STORE",
        "city": np.random.choice(
            ["Toronto", "Vancouver", "Calgary", "Edmonton", "Ottawa", "Montreal", "Quebec City"],
            N_STORES,
        ),
        "region": np.random.choice(["West", "Central", "East"], N_STORES),
    }
)

skus = pd.DataFrame(
    {
        "sku_id": np.arange(10001, 10001 + N_SKUS),
        "category": np.random.choice(
            ["Beverage", "Snacks", "Dairy", "Bakery", "Produce", "Frozen", "Household"],
            N_SKUS,
        ),
        "brand": np.random.choice(
            ["Astra", "Nova", "Zenith", "Pioneer", "Cascade"],
            N_SKUS,
        ),
        "case_pack": np.random.choice([4, 6, 8, 12, 24], N_SKUS, p=[0.15, 0.35, 0.2, 0.2, 0.1]),
        "shelf_life_w": np.random.choice([4, 8, 12, 26, 52], N_SKUS, p=[0.1, 0.2, 0.25, 0.25, 0.2]),
    }
)

# policy-ish knobs per (sku, loc)
params = skus.assign(key=1).merge(stores.assign(key=1), on="key").drop("key", axis=1)

params["lead_time_w"] = np.random.choice([1, 2, 3, 4], len(params), p=[0.4, 0.4, 0.15, 0.05])
params["moq_units"] = np.random.choice([6, 12, 24, 48], len(params), p=[0.3, 0.4, 0.2, 0.1])
params["target_fill_rate"] = np.random.uniform(0.93, 0.99, len(params))

# fake supply path
supply_path = params[["sku_id", "loc_id"]].copy()
supply_path["dc_id"] = np.random.randint(1, 6, size=len(supply_path))  # 5 DCs
supply_path["ship_mode"] = np.random.choice(
    ["truck", "air", "intermodal"], len(supply_path), p=[0.7, 0.1, 0.2]
)

# promo / holiday drivers at calendar level
calendar = pd.DataFrame(
    {
        "week_start": dates,
    }
)

# "corporate promo week?" (global lift baseline)
calendar["is_promo_week"] = np.random.binomial(1, 0.10, size=len(calendar))

# holiday peak: last 2 weeks of year + first week of year
calendar["holiday_uplift"] = np.where(
    calendar["week_start"].dt.isocalendar().week.isin([51, 52, 1]),
    1.3,
    1.0,
)

# --------------------------------------------------
# DEMAND GENERATION
# --------------------------------------------------

# 1. Base attractiveness per SKU
sku_base = np.random.gamma(shape=2.0, scale=5.0, size=N_SKUS)  # ~10 avg

# 2. Store strength (traffic)
store_factor = np.random.uniform(0.5, 1.5, size=N_STORES)

# 3. Slow trend per store (some stores growing/shrinking over time)
#    e.g. +0.2 over ~4 years means ~5% lift by the end
store_trend_slope = np.random.normal(loc=0.0, scale=0.002, size=N_STORES)
# shape: [week, store] trend multiplier like 1.00,1.001,1.002,...
store_trend_curve = np.array(
    [1.0 + store_trend_slope * w for w in range(WEEKS)]
)  # shape (WEEKS, N_STORES)

# 4. Category-level promo memory
#    We'll say certain categories ("Snacks","Beverage") run promos that spike
#    and then decay for 2-3 weeks.
cat_list = skus["category"].unique()
category_promo_bias = {cat: 0.0 for cat in cat_list}
decay_rate = 0.5  # how fast lift decays after promo

# We'll pre-generate per-week "promo_lift_by_category"
promo_lift_by_category = {cat: np.zeros(WEEKS) for cat in cat_list}
for w in range(WEEKS):
    # if corporate promo this week, choose some categories to feature
    featured_cats = []
    if calendar.loc[w, "is_promo_week"] == 1:
        featured_cats = np.random.choice(cat_list, size=2, replace=False)
        for fc in featured_cats:
            category_promo_bias[fc] += 0.5  # spike this week

    # decay all categories each step
    for cat in cat_list:
        category_promo_bias[cat] *= decay_rate
        promo_lift_by_category[cat][w] = 1.0 + category_promo_bias[cat]

# Now build the weekly records
records = []

sku_ids = np.arange(10001, 10001 + N_SKUS)
loc_ids = np.arange(1, N_STORES + 1)

for wk_idx, wk_start in enumerate(dates):
    # 5. Seasonal curve (annual sinusoid)
    seasonal = 1.0 + 0.2 * np.sin(2 * np.pi * (wk_idx / 52.0))

    # 6. Holiday multiplier
    holiday_mult = calendar.loc[wk_idx, "holiday_uplift"]

    # 7. "Promo pressure" at corporate/global level
    promo_uplift_global = 1.0 + 0.15 * calendar.loc[wk_idx, "is_promo_week"]

    # 8. Base "price" per SKU, then apply discount
    base_price = np.log(sku_base + 1.0) * 5.0 + 5.0  # deterministic baseline
    promo_mask = np.random.rand(N_SKUS) < 0.15  # ~15% SKUs on discount
    price_down = np.where(promo_mask, 0.85, 1.0)  # 15% off
    price_effect = (base_price * price_down).astype(float)

    # 9. Build per-(store,sku) category promo lift
    #    Each SKU belongs to a category with a lingering effect.
    cat_for_sku = skus["category"].values  # length N_SKUS
    cat_week_lift = np.array(
        [promo_lift_by_category[cat_for_sku[j]][wk_idx] for j in range(N_SKUS)]
    )  # shape [N_SKUS]

    # Base mean before noise/promo/etc:
    # shape (stores, skus)
    mean_matrix = (
        sku_base[None, :]  # SKU base pull
        * store_factor[:, None]  # store strength
        * store_trend_curve[wk_idx, :, None]  # slow trend over time
        * seasonal
        * promo_uplift_global
        * holiday_mult
        * cat_week_lift[None, :]  # lingering category promo effect
    )

    # 10. Noise-ish overdispersion (Gamma-Poisson ≈ NegBin behavior)
    noise_gamma = np.random.gamma(shape=2.0, scale=0.5, size=mean_matrix.shape)
    lam_noisy = mean_matrix * (0.7 + 0.6 * noise_gamma)

    # raw poisson draw
    raw_demand = np.random.poisson(lam_noisy)

    # 11. Intermittency / zero-inflation
    intermittent_mask = np.random.rand(N_STORES, N_SKUS) < 0.75
    keep_prob = np.ones((N_STORES, N_SKUS))
    keep_prob[intermittent_mask] = 0.25
    keep_draw = (np.random.rand(N_STORES, N_SKUS) < keep_prob).astype(int)
    realized = raw_demand * keep_draw
    realized = realized.astype(int)

    # 12. crude "stockout-like" suppression:
    # for very high last-week sell-thru, sometimes next week collapses.
    # simulate demand that went unmet.
    if wk_idx > 0:
        high_last = realized > (mean_matrix * 1.8)
        suppress_mask = np.random.rand(N_STORES, N_SKUS) < 0.3  # 30% chance
        realized[high_last & suppress_mask] = 0  # looked like we stocked out

    # --- Flatten to rows for this week ---
    week_loc_ids = np.repeat(loc_ids, N_SKUS)
    week_sku_ids = np.tile(sku_ids, N_STORES)
    week_qty = realized.reshape(-1)

    this_week_price = np.tile(price_effect, (N_STORES, 1)).reshape(-1)
    this_week_promo_flag = np.tile(promo_mask, (N_STORES, 1)).reshape(-1).astype(int)

    records.append(
        pd.DataFrame(
            {
                "week_start": wk_start,
                "loc_id": week_loc_ids,
                "sku_id": week_sku_ids,
                "sales_units": week_qty,
                "price": this_week_price,
                "promo_flag": this_week_promo_flag,
            }
        )
    )

# full demand fact table
fact_demand = pd.concat(records, ignore_index=True)

# naive stockout flag placeholder (future: compute actual service)
fact_demand["stockout_flag"] = 0

# inventory snapshot (simple proxy = sum last 4 weeks' sales)
inv_snap = (
    fact_demand.groupby(["loc_id", "sku_id"])["sales_units"]
    .tail(4)
    .groupby([fact_demand["loc_id"], fact_demand["sku_id"]])
    .sum()
    .reset_index()
    .rename(columns={"sales_units": "on_hand"})
)
inv_snap["week_start"] = dates[-1]

# -----------------
# save parquet outputs
# -----------------
stores.to_parquet(DATA_DIR / "dim_location.parquet", index=False)
skus.to_parquet(DATA_DIR / "dim_sku.parquet", index=False)
params.to_parquet(DATA_DIR / "sku_location_params.parquet", index=False)
supply_path.to_parquet(DATA_DIR / "supply_path.parquet", index=False)
calendar.to_parquet(DATA_DIR / "dim_calendar.parquet", index=False)
fact_demand.to_parquet(DATA_DIR / "fact_demand.parquet", index=False)
inv_snap.to_parquet(DATA_DIR / "inv_snap.parquet", index=False)

print("Wrote:", [p.name for p in DATA_DIR.glob("*.parquet")])
print("Done.")
