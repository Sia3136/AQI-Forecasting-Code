# Air Quality Index (AQI) Forecasting System

Time-series regression and multi-horizon AQI forecasting pipeline for five Indian cities (**Delhi, Mumbai, Chennai, Kolkata, and Bangalore**), built with scikit-learn, XGBoost, LightGBM, Optuna, and statsmodels.

---

## Repository Contents

| File | Description |
|---|---|
| `aqi-forecasting-code.ipynb` | Complete 51-cell notebook covering EDA, stationarity testing, feature engineering, model benchmarking, Optuna tuning, diagnostics, recursive forecasting prototype, and the consolidated `AQIForecastSystem` class. |
| `city_day.csv` | Daily air quality dataset (`18,265` rows, `16` columns) across 5 cities from `2015-01-01` to `2024-12-31`. |
| `aqi_forecast_deployment.joblib` | Serialized deployment dictionary containing the trained `Tuned_XGBoost` pipeline, `feature_cols`, `preprocessor`, `historical_data`, `cities`, `aqi_categories`, and metadata. |

---

## Dataset & Input Schema

- **Rows:** `18,265` (`3,653` daily records per city, `0` missing values); `18,260` rows after computing lag features and dropping the first day per city.
- **Date Range:** `2015-01-01` to `2024-12-31`.
- **Train/Test Split:** Chronological 6-month holdout (`split_date = 2024-06-30`):
  - **Train:** `2015-01-02` to `2024-06-29` (`17,335` rows)
  - **Test:** `2024-06-30` to `2024-12-31` (`925` rows)

### Raw Columns (`16`)
| Column | Type | Role | Description |
|---|---|---|---|
| `City` | `object` | Categorical feature | `Delhi`, `Mumbai`, `Chennai`, `Kolkata`, `Bangalore` |
| `Datetime` | `object` | Index (`Date`) | Observation date (`YYYY-MM-DD`) |
| `PM2.5`, `PM10` | `float64` | Numeric features | Particulate matter concentrations |
| `NO`, `NO2`, `NOx`, `NH3`, `CO`, `SO2`, `O3` | `float64` | Numeric features | Gaseous pollutant concentrations |
| `Benzene`, `Toluene`, `Xylene` | `float64` | Numeric features | Volatile organic compound (VOC) concentrations |
| `AQI` | `float64` | Target (`y`) | Daily Air Quality Index (`0.0` to `500.0`) |
| `AQI_Bucket` | `object` | Dropped | Categorical bucket (excluded from `X` to prevent target leakage) |

### Engineered Features (`23` Total Input Features)
- **Temporal & Cyclical:** `Month`, `Year`, `DayOfWeek`, `IsWeekend`, `Month_sin`, `Month_cos`
- **City-Specific Season:** `Season` via `get_season(city, month)` (`Winter`, `Summer`, `Monsoon`, `Post-Monsoon`, plus `SW-Monsoon` / `NE-Monsoon` for Chennai)
- **Minimal Lags:** `AQI_lag1` (previous day's AQI per city), `AQI_roll_mean_7` (shifted 7-day rolling mean per city)
- **Pollutant Ratio:** `PM_ratio = PM2.5 / (PM10 + 1e-8)`

---

## Code Architecture, Functions & Models

### 1. Preprocessing & Diagnostics
- **`adfuller` / `kpss` & `plot_acf` / `plot_pacf`:** Verified stationarity across all 5 cities (`ADF p = 0.0000`, `KPSS p = 0.1000`) and inspected 30-lag autocorrelation on Delhi, which motivated keeping lag features minimal.
- **`ColumnTransformer` (`preprocessor`):**
  - Numeric pipeline (`21` features): `SimpleImputer(strategy="median")`
  - Categorical pipeline (`City`, `Season`): `SimpleImputer(strategy="most_frequent")` + `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`

### 2. Models & Hyperparameter Tuning
| Model | Configuration / Parameters |
|---|---|
| **Persistence** | Naive baseline using `AQI_lag1` (`y_pred = X_test['AQI_lag1']`) |
| **RandomForest** | `RandomForestRegressor(n_estimators=300, random_state=41, n_jobs=-1)` |
| **XGBoost** | `XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=41)` |
| **LightGBM** | `LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=41, verbose=-1)` |
| **Ensemble** | Unweighted mean `(y_pred_rf + y_pred_xgb + y_pred_lgbm) / 3` |
| **Tuned_XGBoost** | Optuna `TPESampler(seed=41)` over `50` trials with `TimeSeriesSplit(n_splits=5)` optimizing 9 parameters: `n_estimators` `[100, 500]`, `max_depth` `[3, 10]`, `learning_rate` `[0.01, 0.2]` (log), `subsample` `[0.6, 1.0]`, `colsample_bytree` `[0.6, 1.0]`, `min_child_weight` `[1, 10]`, `gamma` `[0, 0.5]`, `reg_alpha` `[1e-8, 10]` (log), `reg_lambda` `[1e-8, 10]` (log). |

### 3. Evaluation & Diagnostic Functions
- `evaluate_model(y_true, y_pred, model_name)` — Computes MAE, RMSE, MAPE, $R^2$, and day-to-day direction accuracy.
- `plot_error_over_time_simple(y_true, y_pred, dates, model_name)` — Plots raw errors, 7-day rolling error, absolute error, monthly MAE, and error histogram.
- `plot_feature_importance(pipeline, top_n=15)` — Extracts and ranks feature importances from the fitted pipeline.
- `plot_actual_vs_predicted(y_true, y_pred, model_name)` — Plots scatter vs. diagonal, 200-sample time-series overlay, and error distribution, followed by city-level mean/std error aggregation.
- `plot_residual_analysis(y_true, y_pred, model_name)` — Plots residuals vs. predicted, residuals over time, normal-overlay histogram, and Q-Q normality plot.

### 4. Forecasting Prototype & Final System
- **Prototype (`forecast_aqi_with_variability`):** Standalone recursive forecaster (`30`, `60`, `90` days) with seasonal pollutant sampling, day-of-week normalization, daily volatility noise, and expanding 95% confidence intervals, visualized via `plot_city_forecast`, `plot_all_cities_grid`, and `plot_forecast_comparison_line`.
- **Final Class (`AQIForecastSystem`):**
  - `forecast(city, start_date, num_days=30)` — Recursive forecasting up to 90 days with India CPCB category classification (`Good` to `Severe`) and weighted dominant pollutant attribution (`PM2.5`, `PM10`, `NO2`, `CO`, `SO2`, `O3`).
  - `display_forecast(forecast_df)` & `get_forecast_summary(forecast_df)` — Formatted daily tables and structured summary dictionaries.
- **Helper Functions:** `predict_aqi(city, start_date, num_days=30, display=True)`, `compare_cities_forecast(start_date, num_days=30)`, and `plot_forecast(forecast_df, figsize=(14, 8))`.

---

## Outputs, Observations & Conclusions

### Test Set Model Comparison (`2024-06-30` to `2024-12-31`)
| Model | MAE | RMSE | MAPE (%) | R² | Direction Acc (%) |
|---|---:|---:|---:|---:|---:|
| Persistence | 170.38 | 207.50 | 399.50 | -1.04 | 32.47 |
| RandomForest | 127.34 | 146.02 | 306.54 | -0.01 | 50.97 |
| XGBoost | 128.22 | 147.94 | 316.32 | -0.04 | 52.16 |
| LightGBM | 128.76 | 147.93 | 314.16 | -0.04 | 51.84 |
| Ensemble | 127.84 | 146.74 | 312.23 | -0.02 | 51.73 |
| **Tuned_XGBoost** | **127.04** | **145.30** | **308.71** | **-0.00** | **50.11** |

- **Optuna Result:** Best CV RMSE `144.58`, Test RMSE `145.30` with shallow, heavily regularized trees (`n_estimators=106`, `max_depth=3`, `learning_rate=0.0101`).
- **Residual Statistics (`Tuned_XGBoost`):** Mean residual `3.99`, std `145.24`, min `-252.31`, max `248.24`; `18.1%` of predictions within `±50` AQI and `37.4%` within `±100` AQI.
- **City-Level Test Errors:** Mean errors were `-2.56` (Bangalore), `15.91` (Chennai), `1.96` (Delhi), `9.24` (Kolkata), and `-4.60` (Mumbai), with error stds around `140–148` across all cities.
- **Top 10 Features:** `City_Delhi` (`0.048`), `IsWeekend` (`0.045`), `Year` (`0.040`), `Season_Winter` (`0.039`), `Month` (`0.039`), `AQI_roll_mean_7` (`0.038`), `City_Mumbai` (`0.038`), `CO` (`0.038`), `AQI_lag1` (`0.037`), `Season_NE-Monsoon` (`0.037`).

### Key Observations from the Outputs
1. **Why $R^2$ is `0.00` despite beating Persistence by 30%:** `city_day.csv` behaves like uniform random noise (`AQI` mean `251.11`, std `144.50`, near-zero pollutant correlations and near-zero lag-1 autocorrelation). On uncorrelated noise, persistence (`y_t = y_{t-1}`) doubles error variance (`RMSE 207.50`), whereas Optuna regularized XGBoost toward predicting the global mean (`~251`), yielding `RMSE 145.30` and `R² ≈ 0`.
2. **Flat Feature Importances:** Top features cluster tightly between `0.037` and `0.048`, confirming no single pollutant or lag carries strong predictive signal in this dataset.
3. **Recursive Forecast Behavior:** Because the base model outputs near the mean (`~251`), `forecast_aqi_with_variability` and `AQIForecastSystem.forecast` rely on seasonal pollutant sampling and daily noise injection to produce realistic-looking multi-day trajectories (`212–270` city averages across 30/60/90-day runs).

---

## Using the Saved Output (`aqi_forecast_deployment.joblib`)

The notebook exports a self-contained deployment dictionary to `aqi_forecast_deployment.joblib`:

```python
import joblib

pkg = joblib.load("aqi_forecast_deployment.joblib")
model = pkg["model"]                 # Trained Tuned_XGBoost sklearn Pipeline
feature_cols = pkg["feature_cols"]   # 23 expected input columns
preprocessor = pkg["preprocessor"]   # Fitted ColumnTransformer
historical_data = pkg["historical_data"]
cities = pkg["cities"]               # ['Delhi', 'Mumbai', 'Chennai', 'Kolkata', 'Bangalore']
aqi_categories = pkg["aqi_categories"]
```

To generate a forecast DataFrame or structured JSON-ready summary inside the notebook environment after running the `AQIForecastSystem` cell:

```python
forecast_df = predict_aqi("Mumbai", start_date="2025-01-01", num_days=30, display=True)
summary = forecast_system.get_forecast_summary(forecast_df)
```

---

## Honest Future Improvements (Based on Observed Outputs)

1. **Replace synthetic data with verified CPCB/OpenAQ station data:** The near-zero $R^2$ (`-0.00`), `50.11%` directional accuracy, and `18.1%` within-50-AQI hit rate stem from uncorrelated input data. Real station data is required for meaningful predictive signal.
2. **Add exogenous meteorological drivers:** Include wind speed, temperature, relative humidity, rainfall, and boundary-layer height, which drive real-world pollutant dispersion.
3. **Eliminate same-day feature dependency:** Train direct multi-horizon models ($y_{t+h}$) using only features known at forecast time $t$ instead of feeding same-day pollutants at test time and sampling them at inference.
4. **Fix panel `TimeSeriesSplit` ordering:** Sort globally by `Date` (or use date-cutoff rolling-origin folds across all cities simultaneously) instead of sorting by `['City', 'Date']` before `TimeSeriesSplit`.
5. **Calibrated uncertainty intervals:** Replace ad-hoc Gaussian noise injection with quantile regression or conformal prediction evaluated on empirical coverage.

---

## Setup & Run Instructions

1. **Install dependencies** (Python 3.8+):
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm optuna statsmodels scipy joblib
   ```
2. **Configure dataset path:** In Cell 4 of `aqi-forecasting-code.ipynb`, update `pd.read_csv(...)` from `/kaggle/input/datasets/sia3136/city-day/city_day.csv` to `'city_day.csv'` when running locally.
3. **Run:** Open `aqi-forecasting-code.ipynb` in Jupyter or Kaggle and execute all cells sequentially.

---

## Contributing & Forking

Contributions, forks, and experiments (e.g., swapping in real CPCB/OpenAQ datasets or testing direct multi-horizon models) are welcome:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-improvement`).
3. Commit your changes (`git commit -m "Describe your change"`).
4. Push to your branch and open a Pull Request.
