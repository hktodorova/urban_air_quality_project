"""Unit tests for src/analysis_utils.py."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from analysis_utils import (
    add_time_features,
    check_hourly_continuity,
    make_pm25_risk_label,
    sorted_target_correlations,
    validate_timestamp_key,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _hourly_df(start="2026-01-01", periods=24):
    """Return a small DataFrame with a clean hourly timestamp column."""
    ts = pd.date_range(start=start, periods=periods, freq="h")
    return pd.DataFrame({"timestamp": ts, "value": range(periods)})


# ── validate_timestamp_key ────────────────────────────────────────────────────

class TestValidateTimestampKey:
    def test_clean_df(self):
        df = _hourly_df(periods=10)
        result = validate_timestamp_key(df)
        assert result["rows"] == 10
        assert result["missing_timestamps"] == 0
        assert result["duplicate_timestamps"] == 0

    def test_missing_timestamp(self):
        df = _hourly_df(periods=5)
        df.loc[2, "timestamp"] = pd.NaT
        result = validate_timestamp_key(df)
        assert result["missing_timestamps"] == 1

    def test_duplicate_timestamp(self):
        df = _hourly_df(periods=5)
        df.loc[3, "timestamp"] = df.loc[0, "timestamp"]
        result = validate_timestamp_key(df)
        assert result["duplicate_timestamps"] == 1

    def test_custom_column_name(self):
        df = _hourly_df(periods=3).rename(columns={"timestamp": "ts"})
        result = validate_timestamp_key(df, timestamp_col="ts")
        assert result["rows"] == 3


# ── add_time_features ─────────────────────────────────────────────────────────

class TestAddTimeFeatures:
    def test_columns_added(self):
        df = _hourly_df()
        out = add_time_features(df)
        assert {"hour", "day_of_week", "is_weekend"}.issubset(out.columns)

    def test_hour_values(self):
        df = _hourly_df(start="2026-01-01 00:00", periods=24)
        out = add_time_features(df)
        assert list(out["hour"]) == list(range(24))

    def test_is_weekend_saturday(self):
        # 2026-01-03 is a Saturday (dayofweek == 5)
        df = pd.DataFrame({"timestamp": pd.date_range("2026-01-03", periods=1, freq="h")})
        out = add_time_features(df)
        assert out.loc[0, "is_weekend"] == 1

    def test_is_weekend_monday(self):
        # 2026-01-05 is a Monday (dayofweek == 0)
        df = pd.DataFrame({"timestamp": pd.date_range("2026-01-05", periods=1, freq="h")})
        out = add_time_features(df)
        assert out.loc[0, "is_weekend"] == 0

    def test_does_not_modify_original(self):
        df = _hourly_df(periods=5)
        original_cols = set(df.columns)
        _ = add_time_features(df)
        assert set(df.columns) == original_cols


# ── make_pm25_risk_label ──────────────────────────────────────────────────────

class TestMakePm25RiskLabel:
    def _pm25_df(self):
        return pd.DataFrame({"pm25_ugm3": [10.0, 24.9, 25.0, 30.0, 50.0]})

    def test_default_threshold(self):
        out = make_pm25_risk_label(self._pm25_df())
        assert list(out["high_pm25_risk"]) == [0, 0, 1, 1, 1]

    def test_custom_threshold(self):
        out = make_pm25_risk_label(self._pm25_df(), threshold=30.0)
        assert list(out["high_pm25_risk"]) == [0, 0, 0, 1, 1]

    def test_label_column_only_0_and_1(self):
        out = make_pm25_risk_label(self._pm25_df())
        assert set(out["high_pm25_risk"].unique()).issubset({0, 1})

    def test_custom_label_col_name(self):
        out = make_pm25_risk_label(self._pm25_df(), label_col="risk")
        assert "risk" in out.columns

    def test_does_not_modify_original(self):
        df = self._pm25_df()
        _ = make_pm25_risk_label(df)
        assert "high_pm25_risk" not in df.columns


# ── check_hourly_continuity ───────────────────────────────────────────────────

class TestCheckHourlyContinuity:
    def test_no_gaps(self):
        df = _hourly_df(periods=48)
        result = check_hourly_continuity(df)
        assert result["missing_hours"] == 0
        assert result["gap_timestamps"] == []
        assert result["rows"] == 48

    def test_one_missing_hour(self):
        ts = pd.date_range("2026-01-01", periods=10, freq="h").tolist()
        ts.pop(5)  # remove one hour → one 2-hour gap = 1 missing hour
        df = pd.DataFrame({"timestamp": ts})
        result = check_hourly_continuity(df)
        assert result["missing_hours"] == 1
        assert len(result["gap_timestamps"]) == 1

    def test_two_consecutive_missing_hours(self):
        # Removing 2 consecutive hours creates a 3-hour gap = 2 missing hours
        ts = pd.date_range("2026-01-01", periods=10, freq="h").tolist()
        ts.pop(5)
        ts.pop(5)
        df = pd.DataFrame({"timestamp": ts})
        result = check_hourly_continuity(df)
        assert result["missing_hours"] == 2

    def test_unsorted_input_still_works(self):
        df = _hourly_df(periods=24).sample(frac=1, random_state=0).reset_index(drop=True)
        result = check_hourly_continuity(df)
        assert result["missing_hours"] == 0


# ── sorted_target_correlations ────────────────────────────────────────────────

class TestSortedTargetCorrelations:
    def _corr_df(self):
        import numpy as np
        rng = pd.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": [5, 4, 3, 2, 1],  # perfect negative correlation with a
            "target": [1, 2, 3, 4, 5],  # perfect positive correlation with a
        })
        return rng

    def test_excludes_self_correlation(self):
        df = self._corr_df()
        result = sorted_target_correlations(df, df.columns, "target")
        assert "target" not in result.index

    def test_sorted_descending(self):
        df = self._corr_df()
        result = sorted_target_correlations(df, df.columns, "target")
        assert list(result) == sorted(result, reverse=True)

    def test_correct_values(self):
        df = self._corr_df()
        result = sorted_target_correlations(df, df.columns, "target")
        # a has correlation +1 with target; b has correlation -1
        assert pytest.approx(result["a"], abs=1e-6) == 1.0
        assert pytest.approx(result["b"], abs=1e-6) == -1.0
