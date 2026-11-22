# Air Quality Index (AQI) Forecasting — Learning Project & Retrospective

A time-series machine learning project built in Python to explore daily Air Quality Index (AQI) forecasting across **5 major Indian cities** (**Delhi, Mumbai, Chennai, Kolkata, and Bangalore**) over a 10-year span (`2015-01-01` to `2024-12-31`).

---

## Motivation

This project started out of a practical concern: AQI levels in **Mumbai** were spiking, and I wanted to explore whether historical air pollution data could be used to forecast AQI days or weeks ahead of time. More importantly, I built this as a hands-on learning project to practice the full ML lifecycle—from stationarity testing and cyclical feature engineering to tree-based ensembles, Bayesian hyperparameter tuning with Optuna, residual diagnostics, and packaging a forecasting class.

---

## Repository Structure

| File | Description |
|---|---|
| `aqi_forecasting_code.ipynb` | Full 51-cell Jupyter notebook covering EDA, feature engineering, model training, Optuna tuning, diagnostics, recursive forecasting prototype, and the consolidated `AQIForecastSystem` class. |
| `city_day.csv` | Daily air quality dataset (`18,265` rows, `16` columns) across 5 cities from 2015 to 2024. |

---

## Dataset & Honest Data Audit

- **Observations:** `18,265` daily records (`3,653` days per city, `0` missing values).
- **Features:** `City`, `Datetime`, and 12 pollutants (`PM2.5`, `PM10`, `NO`, `NO2`, `NOx`, `NH3`, `CO`, `SO2`, `O3`, `Benzene`, `Toluene`, `Xylene`).
- **Target:** `AQI` (and `AQI_Bucket`, dropped before training to avoid target leakage).

> **Important Note on Data Quality:**  
> Statistical inspection of `city_day.csv` after building the pipeline revealed that this specific CSV is **synthetic uniform data** rather than real CPCB station readings:
> - `AQI` is distributed nearly uniformly on `[0, 500]` (`mean = 251.11`, `std = 144.50`), and all 5 cities share nearly identical 10-year averages (`249.8` to `253.3`).
> - Linear correlations between individual pollutants and `AQI` are near zero (highest is `NO2` at `r = 0.0111`; `PM2.5` is `r = 0.0007`).
> - Lag-1 autocorrelation of `AQI` per city is near zero (`-0.0358` for Delhi to `+0.0131` for Kolkata).
> - `AQI_Bucket` labels match India CPCB AQI thresholds only `16.17%` of the time (essentially random across 6 classes).
>
> Keeping this documented is intentional: it explains why model $R^2$ stayed near `0` regardless of tuning and highlights why **data quality audits** should happen before modeling.

---

## Pipeline & Methodology

### 1. Preprocessing & EDA
- Parsed `Datetime` to a `Date` index, sorted by `['City', 'Date']`, and removed rows with missing lag values (`18,265` → `18,260` rows).
- Plotted 7-day moving averages and monthly resampled AQI trends per city.
- Ran **ADF** (`p = 0.0000` for all cities) and **KPSS** (`p = 0.1000` for all cities) stationarity tests, plus **ACF/PACF** plots (30 lags on Delhi) which showed weak autocorrelation.

### 2. Feature Engineering (`23` features)
- **Calendar features:** `Month`, `Year`, `DayOfWeek`, `IsWeekend`.
- **Cyclical encoding:** `Month_sin` and `Month_cos` so December and January remain adjacent.
- **City-specific Indian seasons (`get_season`):** Custom mapping per city (e.g., `Winter`, `Summer`, `Monsoon`, `Post-Monsoon`, plus `SW-Monsoon` and `NE-Monsoon` for Chennai).
- **Minimal lag features:** `AQI_lag1` and shifted `AQI_roll_mean_7` (kept minimal due to weak ACF).
- **Pollutant interaction:** `PM_ratio = PM2.5 / (PM10 + 1e-8)`.

### 3. Train/Test Split & Preprocessing Pipeline
- **Chronological holdout (last 6 months):**
  - **Train:** `2015-01-02` to `2024-06-29` (`17,335` rows)
  - **Test:** `2024-06-30` to `2024-12-31` (`925` rows)
- **`ColumnTransformer`:**
  - Numeric features (`21`): `SimpleImputer(strategy="median")`
  - Categorical features (`City`, `Season`): `SimpleImputer(strategy="most_frequent")` + `OneHotEncoder(handle_unknown="ignore")`

### 4. Models Trained & Hyperparameter Tuning
- **Persistence Baseline:** Predicts tomorrow's AQI equals yesterday's (`AQI_lag1`).
- **Random Forest:** `RandomForestRegressor(n_estimators=300, random_state=41)`
- **XGBoost:** `XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6)`
- **LightGBM:** `LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=6)`
- **Ensemble:** Simple unweighted average of Random Forest, XGBoost, and LightGBM predictions.
- **Optuna-Tuned XGBoost:** Bayesian optimization (`TPESampler`, `50` trials, 5-fold `TimeSeriesSplit` CV) tuning `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight`, `gamma`, `reg_alpha`, and `reg_lambda`.

### 5. Iterative Forecasting System
- **Initial Prototype (`forecast_aqi_with_variability`):** Standalone recursive multi-horizon forecaster (`30`, `60`, `90` days) with seasonal pollutant sampling, day-of-week multipliers, noise injection, and expanding 95% confidence bounds.
- **Final Consolidated Class (`AQIForecastSystem`):** Combines recursive forecasting, India CPCB AQI categorization (`Good` through `Severe`), dominant pollutant attribution, formatted reporting (`predict_aqi`, `compare_cities_forecast`), 4-panel dashboard plotting (`plot_forecast`), and `joblib` export (`aqi_forecast_deployment.joblib`).

---

## Model Performance & Outputs

### Test Set Results (Last 6 Months: `2024-06-30` to `2024-12-31`)

| Model | MAE | RMSE | MAPE (%) | R² | Direction Acc (%) |
|---|---:|---:|---:|---:|---:|
| Persistence (`AQI_lag1`) | 170.38 | 207.50 | 399.50 | -1.04 | 32.47 |
| LightGBM | 128.76 | 147.93 | 314.16 | -0.04 | 51.84 |
| XGBoost | 127.95 | 147.68 | 310.52 | -0.04 | 52.38 |
| Ensemble (RF + XGB + LGBM) | 127.73 | 146.67 | 310.29 | -0.02 | 51.30 |
| Random Forest | 127.34 | 146.02 | 306.54 | -0.01 | 50.97 |
| **Tuned XGBoost** | **126.98** | **145.29** | **308.66** | **-0.002** | **52.92** |

- **Optuna Search (50 trials, 5-fold `TimeSeriesSplit`):** Best CV RMSE `144.60`; Test RMSE `145.39` (best parameters converged to shallow, heavily regularized trees: `n_estimators=100`, `max_depth=3`, `learning_rate≈0.0116`).
- **Residual Analysis (Tuned XGBoost):** Mean residual `3.87`, std `145.24`, min `-257.61`, max `248.39`; `17.9%` of test predictions fell within `±50` AQI and `37.4%` within `±100` AQI.
- **Top Features by Importance:** `City_Bangalore` (`0.041`), `Season_Summer` (`0.038`), `City_Kolkata` (`0.036`), `AQI_roll_mean_7` (`0.036`), `Season_Winter` (`0.035`), `PM_ratio` (`0.035`), `AQI_lag1` (`0.035`), `Toluene` (`0.035`), `Month_cos` (`0.034`), `Benzene` (`0.034`).

---

## Key Lessons Learned & Limitations

1. **Beating Persistence isn't enough:** On uncorrelated noise, day-over-day persistence doubles error variance ($\text{RMSE} \approx 207.5$). Predicting a constant training mean (`250.89`) achieves `RMSE ≈ 145.22`, meaning the `30%` RMSE drop over persistence reflects baseline choice rather than learned signal.
2. **Stationarity $\neq$ Predictability:** White noise passes both ADF and KPSS tests; correlation, distribution, and target-label consistency checks are essential before training.
3. **Panel `TimeSeriesSplit` Ordering:** Sorting by `['City', 'Date']` before passing `X_train` to `TimeSeriesSplit` splits folds across city blocks rather than strictly by global date.
4. **Same-Day Feature Leakage & Recursive Noise:** Using same-day pollutant readings during test evaluation does not reflect real multi-day forecasting (where future pollutants are unknown), and injecting Gaussian noise during recursive rollouts fabricates variance rather than modeling calibrated uncertainty.

---

## Running the Notebook

1. **Prerequisites:** Python 3.8+ with `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`, `lightgbm`, `optuna`, `statsmodels`, `scipy`, and `joblib`.
2. **Dataset path:** In Cell 4, the notebook reads `/kaggle/input/cityday/city_day.csv` for Kaggle execution. If running locally in this repository, change that path to `'city_day.csv'`.
3. **Execute:** Open `aqi_forecasting_code.ipynb` in Jupyter or Kaggle and run cells top-to-bottom.

---

## Future Improvements

- Replace `city_day.csv` with verified **real-world CPCB / OpenAQ** monitoring data and validate `PM2.5`-to-`AQI` consistency first.
- Add **meteorological features** (wind speed, temperature, humidity, rainfall) and seasonal event indicators (e.g., Diwali, crop burning).
- Switch to **direct multi-horizon forecasting** ($y_{t+h}$) using only features known at forecast time $t$.
- Use **chronological panel rolling-origin CV** (splitting by date across all cities simultaneously).
- Replace ad-hoc noise injection with **quantile regression** or **conformal prediction** for calibrated uncertainty intervals.
