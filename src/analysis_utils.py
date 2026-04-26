"""Small helper functions used in the notebook.

I kept these outside the notebook so the main file is a bit cleaner.
"""

import pandas as pd


def validate_timestamp_key(df, timestamp_col="timestamp"):
    """Check the timestamp column before merging data."""
    return {
        "rows": len(df),
        "missing_timestamps": int(df[timestamp_col].isna().sum()),
        "duplicate_timestamps": int(df[timestamp_col].duplicated().sum()),
    }


def add_time_features(df, timestamp_col="timestamp"):
    """Add a few simple time columns."""
    data = df.copy()
    data["hour"] = data[timestamp_col].dt.hour
    data["day_of_week"] = data[timestamp_col].dt.dayofweek
    data["is_weekend"] = (data["day_of_week"] >= 5).astype(int)
    return data


def make_pm25_risk_label(df, pm25_col="pm25_ugm3", threshold=25.0, label_col="high_pm25_risk"):
    """Make a 0/1 column for higher PM2.5 values."""
    data = df.copy()
    data[label_col] = (data[pm25_col] >= threshold).astype(int)
    return data


def check_hourly_continuity(df, timestamp_col="timestamp"):
    """Check for gaps in the expected hourly timestamp sequence after sorting.

    Returns a dict with:
        rows            – total number of rows
        missing_hours   – number of consecutive-hour gaps found
        gap_timestamps  – list of timestamps where a gap starts (empty if none)
    """
    sorted_ts = df[timestamp_col].sort_values().reset_index(drop=True)
    expected = pd.Timedelta("1h")
    diffs = sorted_ts.diff().dropna()
    gaps = diffs[diffs != expected]
    # Count actual missing hours (a 3-hour gap means 2 missing hours, not 1)
    missing_hours = int(((gaps - expected) / expected).sum()) if len(gaps) > 0 else 0
    return {
        "rows": len(df),
        "missing_hours": missing_hours,
        "gap_timestamps": sorted_ts[gaps.index].tolist() if len(gaps) > 0 else [],
    }


def sorted_target_correlations(df, columns, target_col):
    """Show correlations with the target column, excluding self-correlation."""
    corr = df[list(columns)].corr()[target_col].sort_values(ascending=False)
    return corr.drop(index=target_col, errors="ignore")
