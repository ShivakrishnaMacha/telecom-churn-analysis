"""Feature engineering for the telecom churn dataset.

All features are derived from columns that exist in the raw CSV, so the
same function serves training, evaluation and the dashboard's live
risk calculator. No feature uses the target, and no feature leaks
post-churn information (``total_charges`` is tenure x monthly charges,
both known while the customer is active).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SERVICE_COLS = [
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
]

NUMERIC_FEATURES = [
    "tenure",
    "monthly_charges",
    "total_charges",
    "senior_citizen",
    "num_addons",
    "has_no_tech_support",
    "is_new_customer",
    "is_long_term",
]

CATEGORICAL_FEATURES = ["contract", "payment_method", "internet_service"]


def tenure_band(tenure: int) -> str:
    if tenure <= 12:
        return "0–12 months"
    if tenure <= 24:
        return "13–24 months"
    if tenure <= 36:
        return "25–36 months"
    if tenure <= 48:
        return "37–48 months"
    return "49+ months"


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with engineered columns added (raw columns kept)."""
    out = df.copy()
    out["tenure_band"] = out["tenure"].map(tenure_band)
    out["num_addons"] = sum(
        (out[c] == "Yes").astype(int) for c in SERVICE_COLS
    )
    out["has_no_tech_support"] = (
        (out["tech_support"] == "No") & (out["internet_service"] != "No")
    ).astype(int)
    out["is_month_to_month"] = (out["contract"] == "Month-to-month").astype(int)
    out["is_new_customer"] = (out["tenure"] <= 12).astype(int)
    out["is_long_term"] = (out["tenure"] >= 48).astype(int)
    out["avg_monthly_spend"] = out["total_charges"] / np.maximum(out["tenure"], 1)
    out["churn_flag"] = (out["churn"] == "Yes").astype(int) if "churn" in out else 0
    return out
