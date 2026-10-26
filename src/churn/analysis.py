"""Descriptive churn analysis: rates by segment, in one place.

Every helper takes the raw frame, applies :func:`add_features`, and
returns a tidy DataFrame the dashboard (or a notebook) can plot as-is.
"""

from __future__ import annotations

import pandas as pd

from .features import add_features


def churn_rate_by(df: pd.DataFrame, column: str) -> pd.DataFrame:
    feats = add_features(df)
    grouped = feats.groupby(column).agg(
        customers=("churn_flag", "size"),
        churned=("churn_flag", "sum"),
    )
    grouped["churn_rate"] = grouped["churned"] / grouped["customers"]
    return grouped.reset_index().sort_values("churn_rate", ascending=False)


def charges_by_churn(df: pd.DataFrame) -> pd.DataFrame:
    feats = add_features(df)
    return (
        feats.groupby("churn")[["tenure", "monthly_charges", "total_charges"]]
        .mean()
        .round(2)
        .reset_index()
    )


def summary(df: pd.DataFrame) -> dict:
    feats = add_features(df)
    return {
        "customers": int(len(feats)),
        "churn_rate": float(feats["churn_flag"].mean()),
        "avg_tenure": float(feats["tenure"].mean()),
        "avg_monthly_charges": float(feats["monthly_charges"].mean()),
        "month_to_month_churn": float(
            feats.loc[feats["contract"] == "Month-to-month", "churn_flag"].mean()
        ),
        "two_year_churn": float(
            feats.loc[feats["contract"] == "Two year", "churn_flag"].mean()
        ),
    }
