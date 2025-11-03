**🧠 Supply Chain Demand Forecasting**
This project implements a Temporal Fusion Transformer (TFT) model for multi-week demand forecasting.
It uses PyTorch Forecasting for training and Streamlit for interactive visualization.

**⚙️ Tech Stack**
- Python 3.12
- PyTorch Forecasting
- PyTorch Lightning
- Streamlit
- Pandas / NumPy

#### 🧩 Key Components

| File | Description |
|------|--------------|
| `scripts/generate_synthetic.py` | Generates synthetic weekly demand data |
| `scripts/forecast_tft.py` | Builds, trains, and runs the TFT model |
| `scripts/run_backtest.py` | Evaluates model performance over past horizons |
| `app/streamlit_app.py` | Streamlit dashboard for visualization |
| `data/` | Raw and processed demand data |
| `lightning_logs/` | Training logs and checkpoints |


#### 🚀 Run the Project
```bash
# create environment
conda create -n sc-demand python=3.12
conda activate sc-demand

# install dependencies
pip install -r requirements.txt

# run training + forecasting
python scripts/forecast_tft.py

# launch dashboard
streamlit run app/streamlit_app.py
```


#### 📊 Dashboard Preview

Shows actual vs forecasted weekly demand and forecast tables for each SKU-location combination.

