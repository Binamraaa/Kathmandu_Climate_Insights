# Kathmandu Climate Insights: A Multi-Horizon Temperature Forecasting Project

**Project Status (as of July 2025):** Phase 1 Complete. A robust baseline model has been established and analyzed. Phase 2 (integration of external climate variables) is underway.

This repository documents the end-to-end development of a machine learning model to forecast daily maximum and minimum temperatures for Kathmandu, Nepal, across a 1- to 5-day horizon.

### Key Visual: Actual vs. Predicted 5-Day Ahead Forecast (2020)

![Actual vs Predicted Plot t+5](/images/Actual%20vs%20Predicted%20Max%20and%20Min%20temp.png)
*This plot visualizes the core challenge: while the model captures the overall seasonal trend (climatology), it struggles to predict the sharp, day-to-day fluctuations (weather).*

---

### Project Goal

The objective is to apply professional ML engineering practices to build an accurate and reliable multi-horizon temperature forecasting model. This involves a rigorous process of data cleaning, exploratory data analysis, feature engineering, modeling, and iterative improvement based on detailed error analysis.

---

### Methodology & Workflow

This project follows a structured, iterative workflow.

#### 1. Data Ingestion & Cleaning
*   **Source:** 25+ years of historical daily temperature and precipitation data for Kathmandu (`Kathmandu_temp_minmax.csv`).
*   **Process:** Handled multi-line headers, consolidated twice-daily readings into single daily records, and performed data type conversions. Isolated missing values were handled using interpolation for temperature and fill-with-zero for precipitation.

#### 2. Exploratory Data Analysis (EDA)
EDA revealed key patterns in the data, including:
*   Strong annual seasonality for both maximum and minimum temperatures.
*   A clear monsoon season (June-September) with a dominant impact on precipitation and Diurnal Temperature Range (DTR).
*   A bimodal distribution for minimum temperature, indicating distinct winter and monsoon thermal regimes.

#### 3. Feature Engineering
To prepare the data for modeling, a rich feature set was created:
*   **Cyclical Time Features:** `Month_sin/cos`, `Day_of_Year_sin/cos` to capture seasonality.
*   **Lagged Features:** Past values of temperature and precipitation (`lag_1, lag_2, lag_3, lag_7`) to capture auto-correlation and weekly patterns.
*   **Rolling Window Features:** 7-day and 30-day rolling `mean` and `std` for temperature, and `sum` for precipitation, to provide context on recent trends and volatility.
*   **Derived Features:** `Daily_Temp_Range` to capture the difference between daily max and min temperatures.

#### 4. Modeling & Evaluation

A **Linear Regression** model was established as a robust baseline. An **XGBoost** model was then implemented and tuned using `RandomizedSearchCV` with time-series-aware cross-validation to find the optimal hyperparameters.

**Results on Test Set (MAE in °C):**

| Horizon | Temp. | Baseline (LR) | Tuned (XGBoost) | **Improvement** |
|:-------:|:-----:|:---------------:|:-----------------:|:-----------------:|
| **t+1** | Max | 1.35 °C | 1.35 °C | 0.00 °C |
| | Min | 0.96 °C | 0.95 °C | **+0.01 °C** |
| **t+3** | Max | 1.65 °C | 1.58 °C | **+0.07 °C** |
| | Min | 1.13 °C | 1.11 °C | **+0.02 °C** |
| **t+5** | Max | 1.74 °C | 1.64 °C | **+0.10 °C** |
| | Min | 1.20 °C | 1.17 °C | **+0.03 °C** |

The tuned XGBoost model consistently outperforms the baseline, with the performance gap widening on longer, more difficult forecast horizons.

---

### Core Findings & Error Analysis

While the metrics show a solid performance, a deep dive into the prediction errors revealed a clear, systematic pattern.

![Residuals Plot t+5](/images/residuals_5day.PNG)

1.  **The Smoothing Effect:** The model is excellent at predicting the long-term seasonal average but struggles to capture the magnitude of short-term weather volatility. It consistently **underestimates peak high temperatures** and **overestimates extreme low temperatures**.

2.  **Systematic Seasonal Errors:** The residual plot clearly shows that forecast errors are much larger and more volatile during the pre-monsoon and monsoon seasons (April-September). This indicates the model lacks features to understand the chaotic weather patterns of this period.

**Conclusion:** The current feature set, while powerful, is "blind" to the drivers of abrupt weather changes.

---

### Next Steps

The insights from the error analysis provide a clear path for Phase 2 of the project:

1.  **Integrate External Variables:** The top priority is to merge new datasets for **Relative Humidity (RH)** and **Wind Speed/Direction**. These variables are direct drivers of temperature change and are expected to significantly improve the model's ability to predict volatility.
2.  **Advanced Feature Engineering:** Create new features based on the new data, including lagged RH values and cyclical wind direction components (`Wind_Dir_sin`, `Wind_Dir_cos`).
3.  **Re-evaluate Models:** Re-train and re-evaluate the entire modeling pipeline with this enriched feature set to quantify the performance gain.

---

### How to Run

1.  Set up the conda environment.
2.  Ensure key libraries are installed: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `matplotlib`.
3.  Run the notebooks in the `/notebooks` directory in numerical order.