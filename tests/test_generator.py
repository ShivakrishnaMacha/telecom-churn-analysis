import pandas as pd

from churn.generator import generate_customers


def test_deterministic():
    a = generate_customers(n=500, seed=42)
    b = generate_customers(n=500, seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_shape_and_churn_rate():
    df = generate_customers(n=2000, seed=42)
    assert len(df) == 2000
    assert set(df["churn"].unique()) <= {"Yes", "No"}
    rate = (df["churn"] == "Yes").mean()
    assert 0.15 < rate < 0.40
    assert df.isna().sum().sum() == 0
    assert df["customer_id"].is_unique


def test_month_to_month_churns_more():
    df = generate_customers(n=4000, seed=7)
    m2m = (df.loc[df["contract"] == "Month-to-month", "churn"] == "Yes").mean()
    two = (df.loc[df["contract"] == "Two year", "churn"] == "Yes").mean()
    assert m2m > two
