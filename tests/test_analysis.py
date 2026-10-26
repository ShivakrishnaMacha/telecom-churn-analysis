from churn.analysis import churn_rate_by, summary
from churn.generator import generate_customers


def test_churn_rate_by_sums_to_customers():
    df = generate_customers(n=1000, seed=42)
    by_contract = churn_rate_by(df, "contract")
    assert by_contract["customers"].sum() == 1000
    assert ((by_contract["churn_rate"] >= 0)
            & (by_contract["churn_rate"] <= 1)).all()


def test_summary():
    df = generate_customers(n=1000, seed=42)
    s = summary(df)
    assert s["customers"] == 1000
    assert 0 < s["churn_rate"] < 1
    assert s["month_to_month_churn"] > s["two_year_churn"]
