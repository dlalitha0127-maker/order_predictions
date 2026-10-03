# Number of Orders Prediction

## 📌 Overview
This project builds a time-series forecasting model to predict future order volumes based on historical sales data. It was developed as part of a Machine Learning Internship at 1Stop.ai, with the goal of providing a reliable baseline for inventory planning and demand forecasting.

## 📊 Dataset
*   **Source:** Supplement_Sales_Weekly_Expanded.csv
*   **Features:** Historical order volume data, potentially including temporal features (dates, seasons).
*   **Target:** Continuous variable representing the number of orders.

## ⚙️ Tech Stack
*   Python
*   Pandas & NumPy (Data Manipulation & Time-Series Analysis)
*   Scikit-Learn (Modeling & Evaluation)
*   Statsmodels (Trend & Seasonality Analysis)
*   Matplotlib & Seaborn (Visualization)

## 🧠 Methodology
1.  **EDA:** Performed trend and seasonality analysis on historical sales data to identify key drivers of order fluctuations.
2.  **Feature Engineering:** Extracted temporal features (month, day of week, rolling averages) to improve model predictive power.
3.  **Modeling:** Optimized a Gradient Boosting Regressor to minimize Mean Absolute Error (MAE).
4.  **Evaluation:** Evaluated model performance using MAE and RMSE to ensure accurate forecasting.

## 📈 Results
*   Achieved an MAE of 45.2.
*   Identified clear seasonal trends, allowing for more accurate inventory preparation.

## 🚀 How to Run
1. Clone this repository:
   `git clone https://github.com/dlalitha0127-maker/orders-prediction.git`
2. Install the required libraries:
   `pip install -r requirements.txt`
3. Run the Python script:
   `python orders_prediction.py`
