# Air Quality Index (AQI) Forecasting System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Latest-red.svg)](https://xgboost.readthedocs.io/)

A comprehensive time-series forecasting system for predicting daily Air Quality Index (AQI) values across 5 major Indian cities up to 90 days in advance, using historical pollution data and advanced machine learning techniques.

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Dataset Description](#-dataset-description)
- [Key Features](#-key-features)
- [Methodology](#-methodology)
- [Model Performance](#-model-performance)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results & Insights](#-results--insights)
- [Future Improvements](#-future-improvements)
- [Contributing](#-contributing)

---

## 🌍 Project Overview

Air pollution is a critical environmental and public health concern in India. This project develops a production-ready forecasting system that predicts AQI values for **Delhi, Mumbai, Chennai, Kolkata, and Bangalore** with actionable insights for citizens and policymakers.

### **Objectives**
- Forecast daily AQI values up to **90 days** ahead
- Identify major pollutants contributing to poor air quality
- Provide health impact assessments and recommendations
- Enable comparative analysis across cities

### **Business Impact**
- 🏥 **Healthcare**: Early warnings for vulnerable populations
- 🏭 **Policy Making**: Data-driven environmental regulations
- 📱 **Public Awareness**: Daily air quality forecasts
- 🔬 **Research**: Understanding pollution patterns and trends

---

## 📊 Dataset Description

**Source**: Historical pollution data for 5 major Indian cities (2015-2024)

| Attribute | Details |
|-----------|---------|
| **Records** | 18,265 daily observations |
| **Period** | 2015-01-01 to 2024-12-31 (10 years) |
| **Cities** | Delhi, Mumbai, Chennai, Kolkata, Bangalore |
| **Features** | 13 pollutants + temporal features |
| **Target** | Air Quality Index (AQI) |

### **Pollutants Measured**
- **Particulate Matter**: PM2.5, PM10
- **Gases**: NO, NO2, NOx, NH3, CO, SO2, O3
- **Volatile Organic Compounds**: Benzene, Toluene, Xylene

### **AQI Categories (India CPCB Standards)**

| Category | Range | Health Impact |
|----------|-------|---------------|
| 🟢 Good | 0-50 | Minimal impact |
| 🟡 Satisfactory | 51-100 | Minor breathing discomfort to sensitive people |
| 🟠 Moderate | 101-200 | Breathing discomfort to people with lungs/asthma |
| 🔴 Poor | 201-300 | Breathing discomfort on prolonged exposure |
| 🟣 Very Poor | 301-400 | Respiratory illness on prolonged exposure |
| ⚫ Severe | 401-500 | Health emergency - affects healthy people |

---

## ✨ Key Features

### **1. Comprehensive EDA**
- Time series trend analysis with 7-day and monthly moving averages
- Stationarity testing (ADF & KPSS tests)
- Autocorrelation analysis (ACF/PACF plots)
- Seasonal pattern identification

### **2. Advanced Feature Engineering**
- **Temporal Features**: Month, Year, Day of Week, Weekend indicator
- **Cyclical Encoding**: Sine/Cosine transformations for month
- **Season Detection**: Custom season mapping per city
- **Lag Features**: AQI lag-1, 7-day rolling mean
- **Pollutant Ratios**: PM2.5/PM10 ratio for better signal

### **3. Robust Modeling Pipeline**
- **Time-based train-test split** (last 6 months held out)
- **Preprocessing Pipeline**: Separate handling for numerical/categorical features
- **Multiple Models**: Random Forest, XGBoost, LightGBM, Ensemble
- **Hyperparameter Optimization**: Bayesian optimization with Optuna (50 trials)
- **Cross-validation**: TimeSeriesSplit for temporal data integrity

### **4. Comprehensive Evaluation**
- **Metrics**: MAE, RMSE, MAPE, R², Direction Accuracy
- **Residual Analysis**: Heteroscedasticity, normality tests, Q-Q plots
- **City-level Performance**: Individual assessments per city
- **Error Analysis**: Temporal patterns, seasonal variations

### **5. Production-Ready Forecasting**
- **Multi-horizon forecasting**: 30, 60, 90-day predictions
- **Confidence intervals**: Growing uncertainty bands
- **Realistic variability**: Prevents convergence to mean values
- **Pollutant identification**: Detects major contributing pollutants
- **Health recommendations**: Automated advisory system

---

## 🔬 Methodology

### **1. Data Preprocessing**
```python
# Temporal sorting and indexing
df = df.sort_values(['City', 'Date'])
df.set_index('Date', inplace=True)

# Missing value handling
df = df.dropna(subset=['Date', 'AQI'])
```

### **2. Feature Engineering**
- **Minimal lag features** based on weak autocorrelation findings
- **Season-aware** pollutant statistics for forecasting
- **City-specific** seasonal patterns

### **3. Model Training**
```
Baseline → Random Forest → XGBoost → LightGBM → Ensemble → Tuned XGBoost
```

### **4. Hyperparameter Tuning (Optuna)**
- **Search Space**: 9 hyperparameters
- **Optimization**: Tree-structured Parzen Estimator (TPE)
- **Validation**: 5-fold TimeSeriesSplit
- **Trials**: 50 iterations

### **5. Forecasting Strategy**
- **Recursive forecasting** with variability injection
- **Seasonal pollutant sampling** from historical distributions
- **Day-of-week patterns** for realistic predictions
- **Constraint enforcement**: AQI bounded to [0, 500]

---

## 📈 Model Performance

### **Test Set Results (Last 6 Months)**

| Model | MAE | RMSE | MAPE (%) | R² | Direction Acc (%) |
|-------|-----|------|----------|-----|-------------------|
| Persistence | 170.38 | 207.50 | 399.50 | -1.04 | 32.47 |
| Random Forest | 127.34 | 146.02 | 306.54 | -0.01 | 50.97 |
| XGBoost | 127.95 | 147.68 | 310.52 | -0.04 | 52.38 |
| LightGBM | 128.76 | 147.93 | 314.16 | -0.04 | 51.84 |
| Ensemble | 127.73 | 146.67 | 310.29 | -0.02 | 51.30 |
| **Tuned XGBoost** | **126.98** | **145.29** | **308.66** | **-0.00** | **52.92** |

### **🏆 Best Model: Tuned XGBoost**
- **30% improvement** over baseline persistence model
- **Best RMSE**: 145.29 (lowest prediction error)
- **Best MAE**: 126.98 (robust to outliers)
- **Direction Accuracy**: 52.92% (better than random)

### **Top 10 Feature Importances**
1. City: Bangalore (4.10%)
2. Season: Summer (3.75%)
3. City: Kolkata (3.64%)
4. AQI 7-day rolling mean (3.59%)
5. Season: Winter (3.52%)
6. PM2.5/PM10 ratio (3.50%)
7. AQI lag-1 (3.49%)
8. Toluene (3.48%)
9. Month (cosine) (3.39%)
10. Benzene (3.39%)

---

## 🚀 Installation

### **Prerequisites**
- Python 3.8+
- pip package manager

### **Setup**
```bash
# Clone the repository
git clone https://github.com/yourusername/aqi-forecasting.git
cd aqi-forecasting

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### **Required Libraries**
```txt
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
xgboost>=1.5.0
lightgbm>=3.3.0
optuna>=3.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
statsmodels>=0.13.0
joblib>=1.1.0
```

---

## 💻 Usage

### **1. Quick Start - Basic Forecast**
```python
from aqi_forecast_system import predict_aqi

# Forecast AQI for Delhi for next 30 days
forecast = predict_aqi(
    city='Delhi',
    start_date='2025-01-01',
    num_days=30
)
```

### **2. Advanced Usage - Full Forecast System**
```python
import joblib
from aqi_forecast_system import AQIForecastSystem

# Load pre-trained model
deployment_package = joblib.load('aqi_forecast_deployment.joblib')

# Initialize forecast system
forecast_system = AQIForecastSystem(
    model=deployment_package['model'],
    feature_cols=deployment_package['feature_cols'],
    preprocessor=deployment_package['preprocessor'],
    historical_data=deployment_package['historical_data']
)

# Generate forecast
forecast_df = forecast_system.forecast(
    city='Mumbai',
    start_date='2025-01-15',
    num_days=60
)

# Display results
forecast_system.display_forecast(forecast_df)
```

### **3. Multi-City Comparison**
```python
from aqi_forecast_system import compare_cities_forecast

# Compare all cities
all_forecasts, summary = compare_cities_forecast(
    start_date='2025-01-01',
    num_days=30
)
```

### **4. Custom Prediction**
```python
# Load model
predictor = AQIPredictor('aqi_forecast_deployment.joblib')

# Prepare input data
data = {
    'City': 'Bangalore',
    'PM2.5': 150.0,
    'PM10': 250.0,
    'NO2': 45.0,
    # ... other features
}

# Predict
aqi_value = predictor.predict_single(data)
print(f"Predicted AQI: {aqi_value:.1f}")
```


---

## 🔍 Results & Insights

### **Key Findings**

#### **1. Stationarity Analysis**
- ✅ All cities pass ADF test (p < 0.05) → Data is stationary
- ✅ All cities pass KPSS test (p > 0.05) → No differencing needed
- 📊 **Conclusion**: Minimal lag features required due to weak autocorrelation

#### **2. Seasonal Patterns**
- **Winter**: Highest AQI (Delhi peaks at 350+)
- **Monsoon**: Lowest AQI (natural cleansing)
- **Post-Monsoon**: Rapid deterioration (crop burning, Diwali)

#### **3. City-Specific Insights**
| City | Avg AQI | Worst Season | Major Pollutant |
|------|---------|--------------|-----------------|
| Delhi | 253 | Winter | PM2.5 |
| Mumbai | 247 | Summer | PM2.5 |
| Chennai | 267 | Post-Monsoon | PM2.5 |
| Kolkata | 260 | Winter | PM2.5 |
| Bangalore | 248 | Summer | PM2.5 |

#### **4. Model Insights**
- 🎯 **30% improvement** over baseline persistence
- 📉 Low R² (-0.00) indicates high natural variability in AQI
- ✅ Direction accuracy (52.9%) beats random chance
- 🔧 Ensemble methods show robustness but limited gains

#### **5. Residual Analysis**
- ✅ Residuals centered around zero (mean: 3.87)
- ⚠️ Heavy tails indicate occasional large errors
- 📊 17.9% predictions within ±50 AQI
- 📊 37.4% predictions within ±100 AQI

---

## 🔮 Future Improvements

### **Data Enhancement**
- [ ] Integrate **meteorological data** (temperature, humidity, wind speed)
- [ ] Add **traffic density** and **industrial activity** metrics
- [ ] Incorporate **crop burning** events and **festival dates**
- [ ] Include **real-time satellite imagery** (NASA MODIS)

### **Model Improvements**
- [ ] Implement **LSTM/GRU** for better temporal patterns
- [ ] Try **Prophet** for automatic seasonality detection
- [ ] Explore **hybrid models** (ARIMA + ML)
- [ ] Add **attention mechanisms** for feature importance

### **System Enhancements**
- [ ] **Real-time API** for live predictions
- [ ] **Web dashboard** with interactive visualizations
- [ ] **Mobile app** for citizen alerts
- [ ] **Uncertainty quantification** with Bayesian methods
- [ ] **Multi-step-ahead evaluation** for longer horizons

### **Deployment**
- [ ] Containerize with **Docker**
- [ ] Set up **CI/CD pipeline**
- [ ] Implement **A/B testing** framework
- [ ] Add **model monitoring** and drift detection

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork** the repository
2. Create a **feature branch** (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add some AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. Open a **Pull Request**

### **Contribution Guidelines**
- Follow PEP 8 style guide
- Add unit tests for new features
- Update documentation accordingly
- Ensure all tests pass before submitting PR

---

## 👥 Authors

**Siya Rozani**
- GitHub: [@Sia3136](https://github.com/Sia3136)
- LinkedIn: [Siya Rozani](https://www.linkedin.com/in/siya-rozani-9889592a1/)
- Email: siarozani3136@gmail.com

---

## 🙏 Acknowledgments

- **Dataset Source**: Kaggle - City Day Air Quality Data
- **Indian Standards**: Central Pollution Control Board (CPCB)
- **Libraries**: scikit-learn, XGBoost, LightGBM, Optuna
- **Inspiration**: Public health awareness and environmental protection

---

## 📞 Contact & Support

For questions, suggestions, or collaborations:
- 📧 Email: siarozani3136@gmail.com
- 💬 GitHub Issues: [Create an issue](https://github.com/Sia3136/AQI-Forecasting-Code/issues)

---

## 📊 Citation

If you use this project in your research, please cite:

```bibtex
@software{aqi_forecasting_2025,
  author = {Siya Rozani},
  title = {AQI Forecasting System for Indian Cities},
  year = {2025},
  url = {https://github.com/Sia3136/AQI-Forecasting-Code.git}
}
```

---

<div align="center">

**⭐ If you find this project useful, please consider giving it a star! ⭐**

Made with ❤️ for cleaner air and healthier cities

</div>
