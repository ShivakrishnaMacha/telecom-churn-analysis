from churn.generator import generate_customers
from churn.model import train_churn_model


def test_model_beats_chance_and_flags_contract():
    df = generate_customers(n=1500, seed=42)
    result = train_churn_model(df)
    assert result["test"]["auc"] > 0.60
    assert 0.0 <= result["test"]["accuracy"] <= 1.0
    drivers = result["model"].top_drivers(15)
    names = drivers["feature"].tolist()
    assert any("Month-to-month" in n for n in names)
    m2m = drivers.loc[
        drivers["feature"] == "contract=Month-to-month", "coefficient"
    ].iloc[0]
    assert m2m > 0


def test_predict_proba_range():
    df = generate_customers(n=800, seed=3)
    result = train_churn_model(df)
    probs = result["model"].predict_proba(df.head(10))
    assert ((probs >= 0) & (probs <= 1)).all()
