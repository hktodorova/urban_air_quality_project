# Urban Air Quality Project

|            |                              |
| ---------- | ---------------------------- |
| **Author** | Hristina Todorova            |
| **Course** | Data Science — Final Project |
| **Date**   | April 2026                   |
| **Python** | 3.10+                        |

This project explores whether traffic and weather data can help explain and predict PM2.5 air pollution levels.

Two datasets are used:

- `data/weather_air_quality.csv` — weather variables and PM2.5 values
- `data/traffic_counts.csv` — traffic volume, average speed and congestion

Both datasets include a `timestamp` column and are merged by hour.

> **Note:** The datasets are synthetic and created for educational purposes.

---

## 🎯 Main Question

Can traffic and weather variables be used to predict PM2.5 levels and identify periods of higher pollution risk?

---

## 📦 What is included

The notebook covers:

- loading and inspecting the datasets
- handling missing values and duplicates
- merging data sources
- creating time-based features (hour, weekday, weekend)
- exploratory data analysis (plots and correlations)
- regression models for PM2.5 prediction
- classification models for high pollution risk
- model comparison and interpretation
- discussion of limitations

---

## 📊 Results

- **Regression:** Linear Regression (OLS) achieved the best performance on the chronological test set (RMSE = 3.212 µg/m³, R² = 0.715), with `pm25_lag_1h` included as a feature
- **Classification:** Tuned Random Forest reached F1 = 0.674 and ROC-AUC = 0.951 for identifying high-risk hours (threshold = 25 µg/m³, an educational hourly proxy; 10.9 % positive share in the test set)
- Traffic, weather, and the lag feature together explain ~71 % of PM2.5 variance
- The majority-class baseline scores F1 = 0.00, confirming the classifier adds genuine value
- Adding `pm25_lag_1h` improved classifier F1 from 0.667 to 0.674 vs the no-lag baseline

---

## 📈 Example Visualizations

Visualizations are generated inside the notebook and include:

- Hourly PM2.5 time series
- PM2.5 distribution with risk threshold
- Scatter plots: traffic volume and congestion index vs PM2.5
- Boxplots by hour of day and day of week
- Correlation heatmap
- Predicted vs actual PM2.5
- ROC and Precision-Recall curves
- Confusion matrix

Run the notebook to reproduce all plots.

---

## 🧠 Key Takeaways

- Air pollution is influenced by both traffic and weather conditions
- Feature engineering (especially time-based features) improves model performance
- Tree-based models (e.g., Random Forest) handle nonlinear relationships effectively
- Data preprocessing is essential for reliable results
- Results should be interpreted cautiously due to synthetic data

---

## 🚀 How to Run

Create and activate a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # macOS / Linux
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the notebook:

```bash
jupyter notebook notebooks/urban_air_quality_analysis.ipynb
```

Run all cells in order.

---

## 📁 Project Structure

```
urban_air_quality_project/
├── data/
│   ├── weather_air_quality.csv
│   └── traffic_counts.csv
│   └── data_dictionary.md
├── notebooks/
│   └── urban_air_quality_analysis.ipynb
├── src/
│   └── analysis_utils.py
├── .gitignore
├── README.md
└── requirements.txt
```

---

## ⚠️ Notes

This project is intended for educational purposes. The dataset is synthetic, and results should not be interpreted as real-world air quality predictions.
