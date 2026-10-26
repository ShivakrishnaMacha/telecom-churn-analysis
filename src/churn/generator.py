"""Seeded synthetic telecom-customer generator.

The data is synthetic and deliberately realistic: churn is *not* random.
It is driven by a known logit (month-to-month contract, fibre, electronic
cheque, short tenure and high monthly charges raise churn; a long tenure
and a partner lower it), so the EDA and the model in this project recover
real, checkable structure instead of noise.

Determinism: the same ``seed`` always produces the same CSV, byte for byte
(apart from the header, which is fixed).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CONTRACTS = ["Month-to-month", "One year", "Two year"]
PAYMENTS = [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
]
INTERNET = ["DSL", "Fiber optic", "No"]
SERVICE_COLS = [
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
]


def _tenure(rng: np.random.Generator, n: int) -> np.ndarray:
    """Mixture tenure: many new customers, a long tail of loyal ones."""
    segment = rng.random(n)
    tenure = np.empty(n, dtype=int)
    low = segment < 0.30
    mid = (segment >= 0.30) & (segment < 0.60)
    tenure[low] = rng.integers(0, 13, size=int(low.sum()))
    tenure[mid] = rng.integers(13, 37, size=int(mid.sum()))
    tenure[~low & ~mid] = rng.integers(37, 73, size=int((~low & ~mid).sum()))
    return tenure


def _contract(rng: np.random.Generator, tenure: np.ndarray) -> np.ndarray:
    """Contract depends on tenure: new customers rarely commit long."""
    out = []
    for t in tenure:
        if t < 12:
            p = [0.75, 0.15, 0.10]
        elif t < 36:
            p = [0.55, 0.25, 0.20]
        else:
            p = [0.25, 0.30, 0.45]
        out.append(rng.choice(CONTRACTS, p=p))
    return np.array(out)


def generate_customers(n: int = 6000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tenure = _tenure(rng, n)
    contract = _contract(rng, tenure)
    internet = rng.choice(
        INTERNET, size=n, p=[0.35, 0.44, 0.21]
    )
    senior = (rng.random(n) < 0.16).astype(int)
    partner = rng.random(n) < 0.48
    dependents = (rng.random(n) < 0.30) & partner
    payment = rng.choice(
        PAYMENTS, size=n, p=[0.34, 0.22, 0.22, 0.22]
    )
    paperless = rng.random(n) < 0.59
    base = np.where(
        internet == "Fiber optic",
        rng.normal(88, 10, n),
        np.where(internet == "DSL", rng.normal(56, 9, n), rng.normal(24, 5, n)),
    )
    services = {}
    for col in SERVICE_COLS:
        has = (internet != "No") & (rng.random(n) < 0.45)
        services[col] = np.where(has, "Yes", "No")
    n_addons = sum((v == "Yes").astype(int) for v in services.values())
    monthly = np.clip(base + 3.5 * n_addons + rng.normal(0, 4, n), 18, 120)
    total = np.clip(tenure * monthly + rng.normal(0, 60, n), 0, None)
    total = np.where(tenure == 0, monthly, total)
    logit = (
        -1.90
        + 1.40 * (contract == "Month-to-month")
        + 0.90 * (internet == "Fiber optic")
        + 0.70 * (payment == "Electronic check")
        - 0.030 * tenure
        + 0.014 * (monthly - 70)
        + 0.50 * (services["tech_support"] == "No")
        + 0.40 * senior
        - 0.30 * partner
    )
    prob = 1 / (1 + np.exp(-logit))
    churn = rng.random(n) < prob
    df = pd.DataFrame(
        {
            "customer_id": [f"TEL-{i + 1:06d}" for i in range(n)],
            "gender": rng.choice(["Female", "Male"], size=n),
            "senior_citizen": senior,
            "partner": np.where(partner, "Yes", "No"),
            "dependents": np.where(dependents, "Yes", "No"),
            "tenure": tenure,
            "phone_service": np.where(rng.random(n) < 0.90, "Yes", "No"),
            "multiple_lines": np.where(rng.random(n) < 0.42, "Yes", "No"),
            "internet_service": internet,
            "contract": contract,
            "paperless_billing": np.where(paperless, "Yes", "No"),
            "payment_method": payment,
            "monthly_charges": np.round(monthly, 2),
            "total_charges": np.round(total, 2),
            "churn": np.where(churn, "Yes", "No"),
        }
    )
    for col, vals in services.items():
        df[col] = vals
    order = [
        "customer_id", "gender", "senior_citizen", "partner", "dependents",
        "tenure", "phone_service", "multiple_lines", "internet_service",
        "online_security", "online_backup", "device_protection",
        "tech_support", "streaming_tv", "streaming_movies",
        "contract", "paperless_billing", "payment_method",
        "monthly_charges", "total_charges", "churn",
    ]
    return df[order]


if __name__ == "__main__":
    import argparse
    import pathlib

    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="data/telecom_customers.csv")
    args = parser.parse_args()
    df = generate_customers(n=args.n, seed=args.seed)
    path = pathlib.Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"wrote {len(df):,} customers to {path} "
          f"(churn rate {float((df['churn'] == 'Yes').mean()):.1%})")
