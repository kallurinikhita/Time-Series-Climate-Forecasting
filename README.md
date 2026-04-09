# Climate Time Series Forecasting (SARIMA)

## Overview

This project analyzes long-term global temperature trends and forecasts future climate patterns using time series modeling. Using historical land and ocean temperature anomaly data, we model deviations from pre-industrial baselines to better understand the trajectory of global warming.

---

## Dataset

* Source: NASA GISS Surface Temperature Analysis & NOAA National Climatic Data Center
* Time range: 1905–2016
* Variable: Global land and ocean temperature anomalies

The dataset captures annual mean temperature deviations relative to pre-industrial levels, providing a standardized measure of climate change over time.

---

## Objective

* Analyze long-term temperature trends
* Transform and prepare time series data for modeling
* Build and evaluate forecasting models
* Predict future global temperature anomalies

---

## Methodology

### 1. Data Preprocessing

* Checked for stationarity
* Applied transformations:

  * Log transformation
  * Box-Cox transformation
* Performed:

  * Trend differencing
  * Seasonal differencing

---

### 2. Model Selection

* Evaluated multiple **SARIMA models**
* Used **Akaike Information Criterion (AICc)** for model comparison
* Selected best-performing model based on:

  * Fit quality
  * Residual diagnostics

---

### 3. Final Model

SARIMA(2,1,2)(1,1,1)_{12}

* Satisfies stationarity and invertibility conditions
* Residuals pass normality checks
* Suitable for forecasting

---

## Results & Insights

* The model captures overall upward temperature trends consistent with global warming
* Forecasts align closely with observed values
* Confidence intervals successfully capture true observations

However:

* Some deviations occur due to:

  * Natural climate variability
  * Volcanic activity
  * External shocks (e.g., COVID-19, emissions changes)

---

## Key Takeaways

* SARIMA models can effectively model long-term climate trends
* Climate data contains inherent variability that limits perfect prediction
* Forecasting must account for both statistical patterns and real-world uncertainties

---

## Limitations

* Does not explicitly model external drivers (e.g., CO₂ emissions, policy changes)
* Sensitive to structural breaks in climate patterns
* Assumes historical trends continue into the future

---

## Future Work

* Incorporate exogenous variables (e.g., emissions, ENSO indices)
* Explore machine learning models (LSTM, Transformers)
* Extend dataset beyond 2016
* Compare with climate simulation models

---

## Tech Stack

* Python
* statsmodels
* pandas, numpy
* matplotlib
