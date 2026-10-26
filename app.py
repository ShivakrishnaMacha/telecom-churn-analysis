"""Streamlit dashboard: telecom churn EDA + churn-risk calculator.

Loads the committed seeded CSV, so the deployed app works with no build
step. The model is trained on demand (NumPy logistic regression, a few
seconds on 6,000 rows) and cached for the session.
"""

from __future__ import annotations

import pathlib
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))
from churn.analysis import charges_by_churn, churn_rate_by, summary  # noqa: E402
from churn.features import SERVICE_COLS, add_features  # noqa: E402
from churn.model import train_churn_model  # noqa: E402

ROOT = pathlib.Path(__file__).parent
DATA = ROOT / "data" / "telecom_customers.csv"

st.set_page_config(page_title="Telecom Churn Analysis", layout="wide")
st.title("Telecom Churn Analysis")
st.caption("6,000 seeded synthetic customers · EDA by segment · "
           "NumPy logistic-regression churn model · live risk calculator")


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA)


@st.cache_resource(show_spinner=False)
def load_model(df: pd.DataFrame):
    return train_churn_model(df)


df = load_data()
feats = add_features(df)

# ------------------------------------------------------------------ filters
st.sidebar.header("Filters")
contracts = st.sidebar.multiselect(
    "Contract", sorted(df["contract"].unique()),
    default=sorted(df["contract"].unique()),
)
internet = st.sidebar.multiselect(
    "Internet service", sorted(df["internet_service"].unique()),
    default=sorted(df["internet_service"].unique()),
)
tenure_max = st.sidebar.slider("Max tenure (months)", 0, 72, 72)

view = feats[
    feats["contract"].isin(contracts)
    & feats["internet_service"].isin(internet)
    & (feats["tenure"] <= tenure_max)
]
if view.empty:
    st.warning("No customers match those filters.")
    st.stop()

# ------------------------------------------------------------------ KPIs
s = summary(view)
k1, k2, k3, k4 = st.columns(4)
k1.metric("Customers", f"{s['customers']:,}")
k2.metric("Churn rate", f"{s['churn_rate']:.1%}")
k3.metric("Avg tenure", f"{s['avg_tenure']:.1f} months")
k4.metric("Avg monthly charges", f"${s['avg_monthly_charges']:.2f}")

st.subheader("Where churn concentrates")
c1, c2 = st.columns(2)
with c1:
    by_contract = churn_rate_by(view, "contract")
    st.plotly_chart(
        px.bar(by_contract, x="contract", y="churn_rate", text="customers",
               title="Churn rate by contract"),
        use_container_width=True,
    )
with c2:
    by_band = churn_rate_by(view, "tenure_band")
    order = ["0–12 months", "13–24 months", "25–36 months",
             "37–48 months", "49+ months"]
    by_band["tenure_band"] = pd.Categorical(
        by_band["tenure_band"], categories=order, ordered=True)
    by_band = by_band.sort_values("tenure_band")
    st.plotly_chart(
        px.bar(by_band, x="tenure_band", y="churn_rate",
               title="Churn rate by tenure band"),
        use_container_width=True,
    )

c3, c4 = st.columns(2)
with c3:
    by_internet = churn_rate_by(view, "internet_service")
    st.plotly_chart(
        px.bar(by_internet, x="internet_service", y="churn_rate",
               title="Churn rate by internet service"),
        use_container_width=True,
    )
with c4:
    by_payment = churn_rate_by(view, "payment_method")
    st.plotly_chart(
        px.bar(by_payment, x="payment_method", y="churn_rate",
               title="Churn rate by payment method"),
        use_container_width=True,
    )

st.subheader("Charges: churned vs retained")
st.dataframe(charges_by_churn(view), use_container_width=True, hide_index=True)
st.plotly_chart(
    px.histogram(view, x="monthly_charges", color="churn", barmode="overlay",
                 nbins=30, title="Monthly charges distribution by churn"),
    use_container_width=True,
)

# ------------------------------------------------------------------ model
st.subheader("Predictive model (logistic regression, NumPy)")
with st.spinner("Training the churn model…"):
    result = load_model(df)
model = result["model"]
m1, m2, m3 = st.columns(3)
m1.metric("Test AUC", f"{result['test']['auc']:.3f}")
m2.metric("Test accuracy", f"{result['test']['accuracy']:.1%}")
m3.metric("Test F1", f"{result['test']['f1']:.3f}")
st.caption(f"Trained on {result['n_train']:,} customers, "
           f"tested on {result['n_test']:,}.")
drivers = model.top_drivers(10)
st.plotly_chart(
    px.bar(drivers, x="coefficient", y="feature", orientation="h",
           title="Top churn drivers (coefficient: + raises churn risk)"),
    use_container_width=True,
)
with st.expander("Full test metrics"):
    st.json(result["test"])

# ------------------------------------------------------------------ calculator
st.subheader("Churn-risk calculator")
col_a, col_b = st.columns(2)
with col_a:
    tenure = st.slider("Tenure (months)", 0, 72, 6)
    monthly = st.slider("Monthly charges ($)", 18, 120, 85)
    contract = st.selectbox("Contract",
                            ["Month-to-month", "One year", "Two year"])
    internet_service = st.selectbox("Internet service",
                                    ["DSL", "Fiber optic", "No"])
with col_b:
    payment = st.selectbox("Payment method",
                           ["Electronic check", "Mailed check",
                            "Bank transfer (automatic)",
                            "Credit card (automatic)"])
    senior = st.checkbox("Senior citizen")
    tech = st.selectbox("Tech support", ["Yes", "No"])
    partner = st.checkbox("Partner")

row = {
    "customer_id": "CALC-000001", "gender": "Female",
    "senior_citizen": int(senior), "partner": "Yes" if partner else "No",
    "dependents": "No", "tenure": tenure, "phone_service": "Yes",
    "multiple_lines": "No", "internet_service": internet_service,
    "contract": contract, "paperless_billing": "Yes",
    "payment_method": payment, "monthly_charges": float(monthly),
    "total_charges": float(tenure * monthly), "churn": "No",
}
for col in SERVICE_COLS:
    row[col] = "Yes" if (col == "tech_support" and tech == "Yes") else "No"
prob = float(model.predict_proba(pd.DataFrame([row]))[0])
st.metric("Estimated churn probability", f"{prob:.1%}")
if prob >= 0.5:
    st.warning("High risk — month-to-month terms, fibre pricing and missing "
               "tech support are the levers this model flags most often.")
elif prob >= 0.3:
    st.info("Moderate risk — a longer contract or support bundle would "
            "lower the modelled risk most.")
else:
    st.success("Low modelled risk for this profile.")
