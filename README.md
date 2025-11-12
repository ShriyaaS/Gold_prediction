Prophet-Based Forecasting of Gold Prices with Regressors
========================================================

Overview
--------
This project provides a minimal Streamlit app to explore Prophet-based forecasting for gold prices, optionally including external regressors (e.g., macro indicators). You can upload a CSV/XLSX, pick the date and target columns, add regressors, and visualize predictions.

Quickstart (Windows, PowerShell)
---------------------------------
1) Create and activate the virtual environment

```
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) Install dependencies

```
pip install -r requirements.txt
```

3) Run the app

```
streamlit run app.py
```

Data Expectations
-----------------
- A time column (to be selected in the UI) that can be parsed as dates
- A target column representing the series to forecast
- Optional numeric regressor columns (selected in the UI)

Notes
-----
- Prophet expects the time column to be named `ds` and the target to be `y`. The app handles renaming for you based on your selections.
- For regressors, ensure they are available at the times you wish to forecast. The demo predicts on the uploaded dataset to keep input management simple.



