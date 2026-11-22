# Air Quality Index (AQI) Forecasting System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Pipeline-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Regressor-red.svg)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-Regressor-brightgreen.svg)](https://lightgbm.readthedocs.io/)
[![Optuna](https://img.shields.io/badge/Optuna-TPE-blueviolet.svg)](https://optuna.org/)
[![statsmodels](https://img.shields.io/badge/statsmodels-ADF%20%7C%20KPSS-lightgrey.svg)](https://www.statsmodels.org/)

Time-series regression and multi-horizon AQI forecasting pipeline across five Indian cities (**Delhi, Mumbai, Chennai, Kolkata, and Bangalore**), with model benchmarking, Bayesian hyperparameter tuning, residual diagnostics, and a packaged forecasting class (`AQIForecastSystem`).

---

## Table of Contents

- [Repository Contents](#repository-contents)
- [Dataset & Input Schema](#dataset--input-schema)
- [Key Design Decisions](#key-design-decisions)
- [Pipeline Architecture, Functions & Models](#pipeline-architecture-functions--models)
- [Outputs & Observations](#outputs--observations)
- [How to Use](#how-to-use)
- [Honest Future Improvements](#honest-future-improvements)
- [Setup & Run Instructions](#setup--run-instructions)
- [Contributing](#contributing)

---

## Repository Contents

| File | Description |
|---|---|
| [`aqi-forecasting-code.ipynb`](aqi-forecasting-code.ipynb) | Complete 51-cell notebook covering EDA, stationarity testing, feature engineering, model comparison, Optuna tuning, diagnostics, recursive forecasting prototype, and `AQIForecastSystem`. |
| [`city_day.csv`](city_day.csv) | Daily air quality dataset (`18,265` rows, `16` columns) across 5 cities from `2015-01-01` to `2024-12-31`. |
| [`aqi_forecast_deployment.joblib`](aqi_forecast_deployment.joblib) | Serialized deployment dictionary containing the trained `Tuned_XGBoost` pipeline, `feature_cols`, `preprocessor`, `historical_data`, `cities`, `aqi_categories`, and metadata. |

---

## Dataset & Input Schema

- **Records:** `18,265` daily observations (`3,653` per city, `0` missing values); `18,260` rows after lag creation (`dropna` on `AQI_lag1`).
- **Date Range:** `2015-01-01` to `2024-12-31`.
- **Train / Test Split:** Chronological 6-month holdout (`split_date = 2024-06-30`) — **Train:** `17,335` rows (`2015-01-02` to `2024-06-29`), **Test:** `925` rows (`2024-06-30` to `2024-12-31`).

### Raw Columns (`16`)
| Column | Type | Role | Description |
|---|---|---|---|
| `City` | `object` | Categorical feature | `Delhi`, `Mumbai`, `Chennai`, `Kolkata`, `Bangalore` |
| `Datetime` | `object` | Index (`Date`) | Observation date (`YYYY-MM-DD`) |
| `PM2.5`, `PM10` | `float64` | Numeric features | Particulate matter concentrations |
| `NO`, `NO2`, `NOx`, `NH3`, `CO`, `SO2`, `O3` | `float64` | Numeric features | Gaseous pollutant concentrations |
| `Benzene`, `Toluene`, `Xylene` | `float64` | Numeric features | Volatile organic compound (VOC) concentrations |
| `AQI` | `float64` | Target (`y`) | Daily Air Quality Index (`0.0` to `500.0`) |
| `AQI_Bucket` | `object` | Dropped | CPCB category label (excluded from `X`) |

### Engineered Features (`23` Input Features)
- **Temporal & Cyclical:** `Month`, `Year`, `DayOfWeek`, `IsWeekend`, `Month_sin`, `Month_cos`
- **City-Specific Season:** `Season` via `get_season(city, month)` (`Winter`, `Summer`, `Monsoon`, `Post-Monsoon`, plus `SW-Monsoon` / `NE-Monsoon` for Chennai)
- **Minimal Lags:** `AQI_lag1` (previous day's AQI per city), `AQI_roll_mean_7` (shifted 7-day rolling mean per city)
- **Pollutant Ratio:** `PM_ratio = PM2.5 / (PM10 + 1e-8)`

---

## Key Design Decisions

- **Stationarity & ACF/PACF before lag engineering:** Ran `adfuller` (`p = 0.0000`) and `kpss` (`p = 0.1000`) plus 30-lag ACF/PACF plots first; weak autocorrelation led to keeping only `AQI_lag1` and `AQI_roll_mean_7` instead of deep lag windows.
- **Dropping `AQI_Bucket` from features:** Excluded `AQI_Bucket` from `X` to prevent target leakage, since bucket labels normally derive directly from `AQI`.
- **Cyclical month encoding & city-aware seasons:** Used `Month_sin`/`Month_cos` so December and January stay adjacent, and mapped Indian seasons per city (accounting for Chennai's `NE-Monsoon`).
- **Chronological holdout + `TimeSeriesSplit`:** Held out the final 6 months and used 5-fold `TimeSeriesSplit` in Optuna instead of random shuffling so future observations never leak into training.
- **Pipeline + `joblib` packaging:** Wrapped `ColumnTransformer` preprocessing and the estimator inside a single scikit-learn `Pipeline` so training and inference share identical transformations.
- **Prototype-to-class consolidation:** Built `forecast_aqi_with_variability()` first to test recursive multi-day rollouts, then consolidated forecasting, CPCB categorization, and pollutant attribution into `AQIForecastSystem`.

---

## Pipeline Architecture, Functions & Models

### Preprocessing (`ColumnTransformer`)
- **Numeric (`21` cols):** `SimpleImputer(strategy="median")`
- **Categorical (`City`, `Season`):** `SimpleImputer(strategy="most_frequent")` + `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`

### Models & Tuning
| Model | Configuration / Parameters |
|---|---|
| **Persistence** | Baseline using `AQI_lag1` (`X_test['AQI_lag1'].values`) |
| **RandomForest** | `RandomForestRegressor(n_estimators=300, random_state=41, n_jobs=-1)` |
| **XGBoost** | `XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=41)` |
| **LightGBM** | `LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=41, verbose=-1)` |
| **Ensemble** | Unweighted average `(y_pred_rf + y_pred_xgb + y_pred_lgbm) / 3` |
| **Tuned_XGBoost** | Optuna `TPESampler(seed=41)`, `50` trials, `TimeSeriesSplit(n_splits=5)` tuning `n_estimators` `[100, 500]`, `max_depth` `[3, 10]`, `learning_rate` `[0.01, 0.2]`, `subsample` `[0.6, 1.0]`, `colsample_bytree` `[0.6, 1.0]`, `min_child_weight` `[1, 10]`, `gamma` `[0, 0.5]`, `reg_alpha` `[1e-8, 10]`, `reg_lambda` `[1e-8, 10]`. |

### Core Functions & Classes
- **Evaluation & Diagnostics:** `evaluate_model()`, `plot_error_over_time_simple()`, `plot_feature_importance()`, `plot_actual_vs_predicted()`, `plot_residual_analysis()`
- **Initial Prototype:** `forecast_aqi_with_variability()`, `plot_city_forecast()`, `plot_all_cities_grid()`, `plot_forecast_comparison_line()`
- **Final System (`AQIForecastSystem`):** `forecast(city, start_date, num_days=30)`, `display_forecast(forecast_df)`, `get_forecast_summary(forecast_df)`, plus helpers `predict_aqi()`, `compare_cities_forecast()`, and `plot_forecast()`.

---

## Outputs & Observations

### Test Set Results (`2024-06-30` to `2024-12-31`)
| Model | MAE | RMSE | MAPE (%) | R² | Direction Acc (%) |
|---|---:|---:|---:|---:|---:|
| Persistence | 170.38 | 207.50 | 399.50 | -1.04 | 32.47 |
| RandomForest | 127.34 | 146.02 | 306.54 | -0.01 | 50.97 |
| XGBoost | 128.22 | 147.94 | 316.32 | -0.04 | 52.16 |
| LightGBM | 128.76 | 147.93 | 314.16 | -0.04 | 51.84 |
| Ensemble | 127.84 | 146.74 | 312.23 | -0.02 | 51.73 |
| **Tuned_XGBoost** | **127.04** | **145.30** | **308.71** | **-0.00** | **50.11** |

- **Optuna Best Trial:** CV RMSE `144.58`, Test RMSE `145.30` (`n_estimators=106`, `max_depth=3`, `learning_rate=0.0101`).
- **Residuals (`Tuned_XGBoost`):** Mean `3.99`, std `145.24`, range `[-252.31, 248.24]`; `18.1%` within `±50` AQI and `37.4%` within `±100` AQI.
- **Top Features:** `City_Delhi` (`0.048`), `IsWeekend` (`0.045`), `Year` (`0.040`), `Season_Winter` (`0.039`), `Month` (`0.039`), `AQI_roll_mean_7` (`0.038`), `City_Mumbai` (`0.038`), `CO` (`0.038`), `AQI_lag1` (`0.037`), `Season_NE-Monsoon` (`0.037`).

### Key Observations
- **Why $R^2$ is near zero:** `city_day.csv` is uniformly distributed (`AQI` mean `251.11`, std `144.50`) with near-zero pollutant correlations (max `r = 0.0111`) and near-zero lag-1 autocorrelation (`-0.036` to `+0.013`). Optuna minimized RMSE by pushing trees toward predicting the global mean (`~251`), which beats persistence (`207.50` → `145.30`) only because persistence doubles variance on uncorrelated data.
- **Forecast Variability:** Because point predictions hover near `~251`, `AQIForecastSystem.forecast()` applies seasonal pollutant sampling, day-of-week ratios, and daily Gaussian noise (`0.3 * change_std`) to generate multi-day trajectories (`212–270` city averages across 30/60/90-day runs).

---

## How to Use

Load the exported deployment package (`aqi_forecast_deployment.joblib`) in Python:

```python
import joblib

pkg = joblib.load("aqi_forecast_deployment.joblib")
model = pkg["model"]                 # Trained Tuned_XGBoost Pipeline
feature_cols = pkg["feature_cols"]   # 23 input feature names
preprocessor = pkg["preprocessor"]   # Fitted ColumnTransformer
historical_data = pkg["historical_data"]
cities = pkg["cities"]
aqi_categories = pkg["aqi_categories"]
```

Inside `aqi-forecasting-code.ipynb` (after initializing `forecast_system`):

```python
forecast_df = predict_aqi("Mumbai", start_date="2025-01-01", num_days=30, display=True)
summary = forecast_system.get_forecast_summary(forecast_df)
```

---

## Honest Future Improvements

- **Use real CPCB/OpenAQ station data:** Near-zero $R^2$ (`-0.00`), `50.11%` direction accuracy, and `18.1%` within-50-AQI accuracy reflect uncorrelated synthetic inputs rather than real air quality dynamics.
- **Add meteorological features:** Incorporate wind speed, temperature, humidity, rainfall, and boundary-layer height.
- **Direct multi-horizon forecasting:** Predict $y_{t+h}$ using only features known at time $t$ instead of relying on same-day pollutants at test time and sampling them during recursive rollouts.
- **Global date-sorted panel CV:** Sort by `Date` (or use date-cutoff rolling-origin folds across all cities simultaneously) before `TimeSeriesSplit`.
- **Calibrated uncertainty intervals:** Replace heuristic Gaussian noise injection with quantile regression or conformal prediction.

---

## Setup & Run Instructions

1. Install dependencies (Python 3.8+):
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm optuna statsmodels scipy joblib
   ```
2. In Cell 4 of `aqi-forecasting-code.ipynb`, change `/kaggle/input/datasets/sia3136/city-day/city_day.csv` to `'city_day.csv'` if running locally.
3. Open `aqi-forecasting-code.ipynb` in Jupyter or Kaggle and run all cells sequentially.

---

## Contributing

Contributions and improvements are welcome! You can use or extend this repository as follows:

1. **Fork** the repository and clone your fork.
2. **Create a branch** (`git checkout -b feature/your-improvement`).
3. **Make your changes** (for example, integrating real CPCB/OpenAQ data, adding weather features, or implementing direct multi-horizon forecasting).
4. **Commit and push** (`git commit -m "Describe your improvement"` && `git push origin feature/your-improvement`).
5. **Open a Pull Request** describing what changed and how it affected evaluation metrics.
