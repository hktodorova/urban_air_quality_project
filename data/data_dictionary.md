# Data Dictionary

## Key column

- `timestamp` - hourly time value used for joining the two datasets.

## traffic_counts.csv

- `traffic_volume` - number of vehicles counted in the hour.
- `avg_speed_kmh` - average speed of vehicles.
- `congestion_index` - value from 0 to 1. Higher means more congestion.

## weather_air_quality.csv

- `temperature_c` - temperature in Celsius.
- `humidity_pct` - relative humidity percentage.
- `wind_speed_ms` - wind speed in meters per second.
- `pm25_ugm3` - PM2.5 concentration. This is the main value I predict.

## Columns made in the notebook

- `hour` - hour from the timestamp.
- `day_of_week` - weekday number, where Monday is 0.
- `is_weekend` - 1 for Saturday/Sunday, otherwise 0.
- `high_pm25_risk` - 1 when PM2.5 is above the chosen limit, otherwise 0.
- `pm25_lag_1h` - PM2.5 value from the previous hour. Captures autocorrelation: a high pollution hour is likely to be followed by another high pollution hour. The first row is dropped after adding this column.
