"""Regenerate the committed demo dataset (seeded, deterministic)."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

from churn.generator import generate_customers  # noqa: E402

if __name__ == "__main__":
    df = generate_customers(n=6000, seed=42)
    out = pathlib.Path(__file__).parent / "telecom_customers.csv"
    df.to_csv(out, index=False)
    print(f"wrote {len(df):,} customers to {out} "
          f"(churn rate {float((df['churn'] == 'Yes').mean()):.1%})")
