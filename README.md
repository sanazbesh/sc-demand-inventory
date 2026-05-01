# Supply Chain Demand & Inventory (SCDI)

This repository implements a **lightweight demand forecasting and inventory policy simulation package**. It combines tabular time-series feature engineering, LightGBM quantile forecasting, reorder point / order-up-to inventory logic, and rolling backtest simulation to evaluate SKU-location inventory decisions under demand uncertainty. The current codebase is best described as a **simulation-based inventory decision-support prototype** (a lightweight inventory digital twin prototype), not a full enterprise platform.

## What this project does today

- Builds weekly SKU-location panels and engineered demand features (lags, rolling stats, seasonal-index style features).
- Trains pooled **LightGBM quantile models** (one model per quantile).
- Produces probabilistic demand forecasts (e.g., q10/q50/q90).
- Computes inventory control parameters using reorder-point and order-up-to logic with service-level safety stock approximation.
- Runs a rolling inventory simulation backtest to estimate service and inventory outcomes over time.

## Architecture overview

The implementation is organized as a small Python package (`scdi`) with focused modules:

1. **Data & features**: panel construction and time-series feature engineering.
2. **Forecasting model**: quantile LightGBM training and inference.
3. **Inventory policy**: protection horizon aggregation and replenishment calculations.
4. **Backtesting simulation**: rolling train-forecast-simulate loop for policy evaluation.
5. **UI placeholder**: minimal Streamlit starter page (not yet integrated with package workflows).

## Module-by-module breakdown

### `src/scdi/data.py`
Feature engineering for weekly SKU-location demand:
- Panel building from demand, calendar, and parameter tables.
- Lag features (e.g., 1/2/3/4/8/13/26/52).
- Rolling statistics (moving average and rolling standard deviation).
- Seasonal index style signal (`si_52`) based on lag-52 and rolling baseline.

### `src/scdi/model.py`
`QuantileGBM` model class:
- Trains one LightGBM model per quantile (quantile objective).
- Stores per-quantile models in a dictionary.
- Returns stacked multi-quantile predictions.

### `src/scdi/forecast.py`
Forecasting wrappers:
- `train_pooled`: pooled model training on engineered features.
- `forecast_quantiles`: quantile prediction helper that appends forecast columns (`q10`, `q50`, `q90`, etc.) to future context data.

### `src/scdi/inventory.py`
Inventory policy calculations:
- Service-level-to-z-score approximation.
- Protection-horizon demand aggregation.
- Reorder point computation.
- Order-up-to level computation.
- MOQ and case-pack rounding constraints.

### `src/scdi/backtest.py`
Rolling simulation backtest:
- Trains on historical data before the chosen start week.
- Generates rolling quantile forecasts during simulated periods.
- Applies reorder point and order-up-to policy logic.
- Simulates ordering, lead-time arrivals, sales fulfillment, lost sales, and inventory trajectory over time.

### `app/streamlit_app.py`
Streamlit app status:
- Currently a **starter placeholder page** only.
- Not yet wired to the forecasting, policy, or backtest pipeline.

## Workflow

Typical package-level workflow:

1. Prepare weekly demand history and supporting tables (calendar, SKU-location parameters).
2. Build feature panel (`data.py`).
3. Train pooled quantile model (`forecast.py` + `model.py`).
4. Generate quantile forecasts (`forecast.py`).
5. Convert forecast uncertainty to inventory policy parameters (`inventory.py`).
6. Evaluate decisions in a rolling simulation backtest (`backtest.py`).

At present, this flow is implemented in Python modules/functions rather than a single end-to-end CLI pipeline.

## Configuration assumptions

Current implementation assumptions include:

- Weekly time bucket (`week_start`) by SKU-location.
- Forecast quantiles are modeled with separate LightGBM quantile models.
- Policy logic uses reorder-point and order-up-to style replenishment.
- Lead time / order cycle / service targets are sourced from package configuration defaults when not overridden.
- Optional constraints include MOQ and case-pack rounding.

## Outputs and KPIs

The backtest currently returns period-level simulation outputs such as:

- Demand
- Sales
- Lost sales
- Order quantity
- On-hand inventory
- Policy levels (`s`, `S`)

These outputs support KPI derivation for service, stock availability, and inventory responsiveness under uncertainty.

## How to install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## How to run tests

```bash
pytest
```

## Current status and limitations

- The active modeling approach is **LightGBM quantile forecasting**, not TFT or other deep learning architectures.
- The Streamlit app is a placeholder and is not connected to the modeling/backtest pipeline yet.
- Core functionality is currently package-module based; there is not yet a polished single-command runnable pipeline/CLI.
- Some dependencies declared in `pyproject.toml` may not be used in the current primary code path.
- This repository is a prototype library and simulation/backtesting engine, not yet a production-grade decision-support application.

## Future roadmap

Potential next steps:

- Add a reproducible end-to-end pipeline script or CLI entrypoint.
- Integrate the Streamlit interface with training, forecasting, and simulation outputs.
- Expand KPI reporting (fill rate, cycle service level, inventory turns, cost-to-serve proxies).
- Add richer policy alternatives and experiment tracking for scenario comparison.
- Harden data contracts and testing for production-style usage.
