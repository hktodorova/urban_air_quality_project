"""Unit tests for src/generate_data.py."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from generate_data import generate


# ── Helpers ───────────────────────────────────────────────────────────────────

def _run(tmp_path, seed=42):
    """Generate data into tmp_path and return (weather_df, traffic_df)."""
    generate(tmp_path, seed=seed)
    weather = pd.read_csv(tmp_path / "weather_air_quality.csv", parse_dates=["timestamp"])
    traffic = pd.read_csv(tmp_path / "traffic_counts.csv", parse_dates=["timestamp"])
    return weather, traffic


# ── Output files ──────────────────────────────────────────────────────────────

class TestOutputFiles:
    def test_both_files_created(self, tmp_path):
        generate(tmp_path)
        assert (tmp_path / "weather_air_quality.csv").exists()
        assert (tmp_path / "traffic_counts.csv").exists()

    def test_output_dir_created_if_missing(self, tmp_path):
        new_dir = tmp_path / "sub" / "dir"
        generate(new_dir)
        assert new_dir.exists()


# ── Shape and columns ─────────────────────────────────────────────────────────

class TestShapeAndColumns:
    def test_weather_shape(self, tmp_path):
        weather, _ = _run(tmp_path)
        assert weather.shape == (2160, 5)  # 90 days × 24 h

    def test_traffic_shape(self, tmp_path):
        _, traffic = _run(tmp_path)
        assert traffic.shape == (2160, 4)

    def test_weather_columns(self, tmp_path):
        weather, _ = _run(tmp_path)
        assert set(weather.columns) == {
            "timestamp", "temperature_c", "humidity_pct", "wind_speed_ms", "pm25_ugm3"
        }

    def test_traffic_columns(self, tmp_path):
        _, traffic = _run(tmp_path)
        assert set(traffic.columns) == {
            "timestamp", "traffic_volume", "avg_speed_kmh", "congestion_index"
        }


# ── Timestamps ────────────────────────────────────────────────────────────────

class TestTimestamps:
    def test_timestamps_are_datetime(self, tmp_path):
        weather, traffic = _run(tmp_path)
        assert pd.api.types.is_datetime64_any_dtype(weather["timestamp"])
        assert pd.api.types.is_datetime64_any_dtype(traffic["timestamp"])

    def test_shared_timestamps(self, tmp_path):
        weather, traffic = _run(tmp_path)
        pd.testing.assert_series_equal(
            weather["timestamp"].reset_index(drop=True),
            traffic["timestamp"].reset_index(drop=True),
            check_names=False,
        )

    def test_hourly_frequency(self, tmp_path):
        weather, _ = _run(tmp_path)
        diffs = weather["timestamp"].diff().dropna()
        assert (diffs == pd.Timedelta("1h")).all()

    def test_starts_2026_01_01(self, tmp_path):
        weather, _ = _run(tmp_path)
        assert weather["timestamp"].iloc[0] == pd.Timestamp("2026-01-01 00:00:00")


# ── Value ranges ──────────────────────────────────────────────────────────────

class TestValueRanges:
    def test_pm25_range(self, tmp_path):
        weather, _ = _run(tmp_path)
        assert weather["pm25_ugm3"].between(3, 60).all()

    def test_humidity_range(self, tmp_path):
        weather, _ = _run(tmp_path)
        assert weather["humidity_pct"].between(30, 95).all()

    def test_wind_speed_range(self, tmp_path):
        weather, _ = _run(tmp_path)
        assert weather["wind_speed_ms"].between(0.5, 12).all()

    def test_traffic_volume_range(self, tmp_path):
        _, traffic = _run(tmp_path)
        assert traffic["traffic_volume"].between(60, 1000).all()

    def test_congestion_index_range(self, tmp_path):
        _, traffic = _run(tmp_path)
        assert traffic["congestion_index"].between(0, 1).all()

    def test_avg_speed_range(self, tmp_path):
        _, traffic = _run(tmp_path)
        assert traffic["avg_speed_kmh"].between(10, 65).all()


# ── No missing values ─────────────────────────────────────────────────────────

class TestNoMissingValues:
    def test_no_nulls_weather(self, tmp_path):
        weather, _ = _run(tmp_path)
        assert weather.isna().sum().sum() == 0

    def test_no_nulls_traffic(self, tmp_path):
        _, traffic = _run(tmp_path)
        assert traffic.isna().sum().sum() == 0


# ── Reproducibility ───────────────────────────────────────────────────────────

class TestReproducibility:
    def test_same_seed_same_output(self, tmp_path):
        w1, t1 = _run(tmp_path / "run1", seed=0)
        w2, t2 = _run(tmp_path / "run2", seed=0)
        pd.testing.assert_frame_equal(w1, w2)
        pd.testing.assert_frame_equal(t1, t2)

    def test_different_seeds_differ(self, tmp_path):
        w1, _ = _run(tmp_path / "a", seed=1)
        w2, _ = _run(tmp_path / "b", seed=2)
        assert not w1["pm25_ugm3"].equals(w2["pm25_ugm3"])
