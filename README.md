# Kathmandu Climate Insights: A Multi-Horizon Temperature Forecasting Project

This repository documents the end-to-end development of a machine learning model to forecast daily maximum and minimum temperatures for Kathmandu, Nepal, across a 1- to 5-day horizon. The project emphasizes a rigorous, iterative MLOps workflow from data ingestion to error analysis.

**Project Status (as of July 2025):** Phase 1 Complete. A robust baseline model has been established and analyzed. Phase 2 (integration of external climate variables) is underway.

---

### Project Goal

The primary objective is to apply ML practices to build an accurate and reliable multi-horizon temperature forecasting model. This project serves as a practical demonstration of time-series analysis, feature engineering, and model evaluation techniques in a real-world climate science context.

---

### Key Findings & TL;DR

*   **Successful Baseline:** A tuned XGBoost model was successfully built, outperforming a Linear Regression baseline on 8 out of 10 forecasting tasks.
*   **Performance vs. Horizon:** The model's advantage is most significant on longer horizons (`t+3` to `t+5`), where it better captures non-linear trends. For the `t+5` forecast, it reduced the Mean Absolute Error (MAE) by **0.10°C** compared to the baseline.
*   **Insight from Error Analysis:** Visual analysis revealed the model excels at predicting long-term seasonal trends (**climatology**) but struggles with sharp, day-to-day volatility (**weather**). This is the key limitation to address.
*   **Data-Driven Next Steps:** The error analysis points directly to the need for more descriptive features. The next phase will focus on integrating **Relative Humidity (RH)** and **Wind Speed/Direction** data to better model the drivers of abrupt weather changes.


---

### Methodology & Workflow

#### 1. Data Ingestion & Cleaning
*   **Source:** 25+ years of historical daily temperature and precipitation data for Kathmandu.
*   **Process:** Handled multi-line headers, consolidated twice-daily readings into single daily records, and performed data type conversions. Isolated missing values were handled using interpolation for temperature and fill-with-zero for precipitation.

#### 2. Exploratory Data Analysis (EDA)
EDA was critical for understanding the underlying patterns in the data.

| Daily max/min temperature profile (1999-2024) | 
| :--------------------: |
| ![Daily Temp](images/temperature.png) | 

7-day rolling sum of Precipitation (1999-2024) |
| :------------------------------: |
|![Distributions](images/precipitation.png) |

| Average Monthly Temperature Cycle |
| :--------------------: |
| ![Monthly Avg Temp](images/average_monthly_temp.png) | 

 Data Distributions |
| :------------------------------: |
|![Distributions](images/distribution.png) |

*   **Key Insight 1:** The thermal cycle clearly defines the seasons, with the coldest months in Jan/Dec and the warmest from June-August.
*   **Key Insight 2:** The bimodal distribution of Minimum Temperature reveals two distinct thermal regimes: a cold, dry winter and a warm, humid monsoon season.

#### 3. Feature Engineering
A rich feature set was created to capture the temporal dynamics of the data:
*   **Cyclical Time Features:** `Month_sin/cos`, `Day_of_Year_sin/cos`.
*   **Lagged Features:** Past values of temperature and precipitation (`lag_1, lag_2, lag_3, lag_7`).
*   **Rolling Window Features:** 7-day and 30-day rolling `mean` and `std` for temperature and `sum` for precipitation.
*   **Derived Features:** `Daily_Temp_Range`.

#### 4. Modeling & Evaluation
A **Linear Regression** model was established as a robust baseline. An **XGBoost** model was then implemented and tuned using `RandomizedSearchCV` with time-series-aware cross-validation.

**Results on Test Set (MAE in °C):**

| Horizon | Temp. | Baseline (LR) | Tuned (XGBoost) | **Improvement** |
|:-------:|:-----:|:---------------:|:-----------------:|:-----------------:|
| **t+1** | Max | 1.35 °C | 1.35 °C | 0.00 °C |
| | Min | 0.96 °C | 0.95 °C | **+0.01 °C** |
| **t+3** | Max | 1.65 °C | 1.58 °C | **+0.07 °C** |
| | Min | 1.13 °C | 1.11 °C | **+0.02 °C** |
| **t+5** | Max | 1.74 °C | 1.64 °C | **+0.10 °C** |
| | Min | 1.20 °C | 1.17 °C | **+0.03 °C** |

---

### Deep Dive: Error Analysis & Model Limitations

While the metrics are strong, visual inspection provides deeper insights into the model's behavior.

#### Actual vs. Predicted Performance by Horizon

| 1-Day Ahead Forecast | 
| :---: |
| ![t+1 Plot](images/day1_actualvspredicted.PNG) | 

| 3-Day Ahead Forecast |
| :---: |
|![t+3 Plot](images/day3_actualvspred.PNG) | 

| 5-Day Ahead Forecast |
| :---: |
|![t+5 Plot](images/Actual%20vs%20Predicted%20Max%20and%20Min%20temp.png) |

As the forecast horizon increases, the model's predictions become noticeably smoother. It successfully captures the overall seasonal curve but increasingly fails to predict the amplitude of daily weather events.

#### Analysis of Residuals (Actual - Predicted)

![Residuals Plot t+5](images/residuals_5day.PNG)

This plot of the 5-day ahead prediction errors confirms several key limitations:
1.  **Heteroscedasticity:** The errors are not random. Their variance is much higher during the volatile pre-monsoon and monsoon seasons (center of the plot), indicating the model is less certain during chaotic weather.
2.  **Systematic Bias:** The model consistently **under-predicts** peak hot days (points > 0) and **over-predicts** sharp cold snaps or cool, rainy days (points < 0).

**Conclusion:** The current feature set, while powerful, is "blind" to the external drivers of abrupt weather changes, forcing the model to be overly conservative.

---

### Future Work & Next Steps
The insights from the error analysis provide a clear, data-driven path for Phase 2:

1.  **Integrate External Variables:** The top priority is to merge new datasets for **Relative Humidity (RH)** and **Wind Speed/Direction**. These variables are direct drivers of temperature change and are expected to significantly improve the model's ability to predict volatility.
2.  **Advanced Feature Engineering:** Create new features based on this new data, including lagged RH values and cyclical wind direction components (`Wind_Dir_sin`, `Wind_Dir_cos`).
3.  **Re-evaluate Models:** Re-train and re-evaluate the entire modeling pipeline with this enriched feature set to quantify the performance gain.

---

### Technologies Used
*   **Languages:** Python
*   **Libraries:** Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn

### How to Run
1.  Set up the conda environment.
2.  Install all required libraries.
3.  Place data in the `/data` directory.
4.  Run the notebooks in `/notebooks` in numerical order, to make sure all file paths are relative.

---
*Developed by Binamra, July 2025.*