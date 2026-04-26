"""Generate synthetic CSV files for the Urban Air Quality project.

Both files cover 90 days (Jan–Mar 2026) at hourly resolution (2160 rows each).
They share the same timestamp column so they can be merged in the notebook.

Design choices
--------------
- Temperature rises linearly from winter to early spring with daily noise.
- Traffic volume and PM2.5 follow a two-peak weekday pattern
  (morning rush ~07:00, evening rush ~16:00) that is dampened on weekends.
- PM2.5 is driven by traffic (congestion), wind dilution, temperature
  inversion, and autocorrelation noise, so the notebook's regression and
  classification results are meaningful but not trivially perfect.
- congestion_index is derived from traffic volume to keep it consistent.
- avg_speed_kmh is inversely related to congestion.
- Random seed is fixed so the output is reproducible.

Usage
-----
    python src/generate_data.py               # writes to data/
    python src/generate_data.py --output-dir /tmp/test_data
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# ── Parameters ────────────────────────────────────────────────────────────────
SEED = 42
START = "2026-01-01"
PERIODS = 90 * 24          # 90 days × 24 hours = 2160 rows
MAX_TRAFFIC_WEEKDAY = 750   # peak rush-hour vehicles (weekday)
MAX_TRAFFIC_WEEKEND = 480   # peak vehicles (weekend)
PM25_RISK_THRESHOLD = 25    # µg/m³ – must match notebook constant

# Hour-of-day multipliers (index = hour 0–23): two rush peaks
_HOUR_TRAFFIC_PROFILE = np.array([
    0.52, 0.57, 0.59, 0.63, 0.63, 0.65,   # 00–05  night/early
    0.65, 1.00, 0.99, 0.97,                # 06–09  morning rush
    0.59, 0.56, 0.54, 0.50, 0.48, 0.47,   # 10–15  midday lull
    0.80, 0.79, 0.78,                      # 16–18  evening rush
    0.43, 0.44, 0.47, 0.49, 0.49,         # 19–23  evening/night
])


def _hour_profile(hour: np.ndarray, weekend: np.ndarray) -> np.ndarray:
    """Return per-row traffic volume fraction (0–1) based on hour and day type."""
    profile_wd = _HOUR_TRAFFIC_PROFILE[hour]
    profile_we = _HOUR_TRAFFIC_PROFILE[hour] * 0.55 + 0.20   # flatter on weekends
    return np.where(weekend, profile_we, profile_wd)


def generate(output_dir: Path, seed: int = SEED) -> None:
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range(start=START, periods=PERIODS, freq="h")
    n = len(timestamps)

    hour       = timestamps.hour.to_numpy()
    day_of_week = timestamps.dayofweek.to_numpy()
    is_weekend = (day_of_week >= 5).astype(int)
    day_index  = (timestamps - timestamps[0]).days.to_numpy()  # 0 … 89

    # ── Weather ───────────────────────────────────────────────────────────────
    # Temperature: linear winter→spring trend + daily cycle + noise
    temp_trend = -5 + day_index * (15 / 89)          # -5 °C → +10 °C over 90 days
    temp_daily = 3 * np.sin(2 * np.pi * (hour - 14) / 24)
    temp_noise = rng.normal(0, 1.0, n)
    temperature_c = np.round(temp_trend + temp_daily + temp_noise, 1)

    # Humidity: inversely correlated with temperature, range ~40–90 %
    humidity_pct = np.round(
        75 - 0.8 * temperature_c + rng.normal(0, 5, n), 1
    ).clip(30, 95)

    # Wind speed: log-normal, slightly higher in daytime
    wind_base = 3.0 + 0.5 * np.sin(2 * np.pi * (hour - 12) / 24)
    wind_speed_ms = np.round(
        rng.lognormal(np.log(wind_base), 0.3, n), 2
    ).clip(0.5, 12)

    # ── Traffic ───────────────────────────────────────────────────────────────
    profile = _hour_profile(hour, is_weekend)
    max_vol  = np.where(is_weekend, MAX_TRAFFIC_WEEKEND, MAX_TRAFFIC_WEEKDAY)
    vol_base = (max_vol * profile).astype(float)
    traffic_volume = np.round(
        vol_base * rng.lognormal(0, 0.15, n)
    ).astype(int).clip(60, 1000)

    # congestion_index: 0–1, driven by volume and inversely by speed headroom
    raw_congestion = (traffic_volume / 1000) ** 1.3 + rng.normal(0, 0.04, n)
    congestion_index = np.round(raw_congestion.clip(0, 1), 3)

    avg_speed_kmh = np.round(
        55 * (1 - 0.6 * congestion_index) + rng.normal(0, 2, n), 1
    ).clip(10, 65)

    # ── PM2.5 ─────────────────────────────────────────────────────────────────
    # Base driven by congestion (traffic emissions), moderated by wind,
    # slightly elevated in cold/inversion conditions, plus autocorrelated noise.
    pm25_base = (
        8
        + 22 * congestion_index           # traffic contribution
        - 1.5 * wind_speed_ms             # dilution by wind
        + np.where(temperature_c < 5, 3, 0)  # cold inversion effect
    )
    # Autocorrelated noise via AR(1) with φ = 0.55
    phi = 0.55
    noise = rng.normal(0, 2.5, n)
    ar_noise = np.zeros(n)
    ar_noise[0] = noise[0]
    for i in range(1, n):
        ar_noise[i] = phi * ar_noise[i - 1] + noise[i]

    pm25_ugm3 = np.round((pm25_base + ar_noise).clip(3, 60), 2)

    # ── Write CSVs ────────────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)

    weather = pd.DataFrame({
        "timestamp":     timestamps,
        "temperature_c": temperature_c,
        "humidity_pct":  humidity_pct,
        "wind_speed_ms": wind_speed_ms,
        "pm25_ugm3":     pm25_ugm3,
    })
    weather.to_csv(output_dir / "weather_air_quality.csv", index=False)

    traffic = pd.DataFrame({
        "timestamp":        timestamps,
        "traffic_volume":   traffic_volume,
        "avg_speed_kmh":    avg_speed_kmh,
        "congestion_index": congestion_index,
    })
    traffic.to_csv(output_dir / "traffic_counts.csv", index=False)

    print(f"Generated {n} rows → {output_dir}/weather_air_quality.csv")
    print(f"Generated {n} rows → {output_dir}/traffic_counts.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic air quality data.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data",
        help="Directory to write the CSV files (default: project data/ folder).",
    )
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed.")
    args = parser.parse_args()
    generate(args.output_dir, seed=args.seed)
