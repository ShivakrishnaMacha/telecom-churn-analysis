import pandas as pd

from churn.features import add_features, tenure_band


def test_tenure_band_boundaries():
    assert tenure_band(0) == "0–12 months"
    assert tenure_band(12) == "0–12 months"
    assert tenure_band(13) == "13–24 months"
    assert tenure_band(72) == "49+ months"


def test_add_features():
    df = pd.DataFrame([{
        "tenure": 6, "monthly_charges": 90.0, "total_charges": 540.0,
        "senior_citizen": 0, "internet_service": "Fiber optic",
        "contract": "Month-to-month", "churn": "Yes",
        "online_security": "Yes", "online_backup": "No",
        "device_protection": "No", "tech_support": "No",
        "streaming_tv": "Yes", "streaming_movies": "No",
    }])
    out = add_features(df)
    assert out.loc[0, "num_addons"] == 2
    assert out.loc[0, "has_no_tech_support"] == 1
    assert out.loc[0, "is_new_customer"] == 1
    assert out.loc[0, "churn_flag"] == 1
