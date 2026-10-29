from pathlib import Path
import gzip
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Smart Chain AI | Delivery Risk", page_icon="📦", layout="wide")

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
PINK, BLUE, ORANGE = "#e653b4", "#1688ff", "#ff7138"

st.markdown("""
<style>
.stApp {background:#11151d;color:#f5f7fb}
[data-testid="stSidebar"] {background:#202633}
h1,h2,h3,p,label {color:#f5f7fb}
div[data-testid="stMetric"] {background:#242d3c;padding:14px;border-radius:12px;border-left:4px solid #e653b4}
.insight {background:#242d3c;padding:12px 15px;border-radius:10px;margin:7px 0}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    pred_file = DATA / "ML_Risk_Predictions.csv.xlsx"
    order_file = DATA / "APL_Logistics.csv.gz"
    if not pred_file.exists() or not order_file.exists():
        missing = [str(p.relative_to(ROOT)) for p in (pred_file, order_file) if not p.exists()]
        raise FileNotFoundError("Missing required data file(s): " + ", ".join(missing))
    predictions = pd.read_excel(pred_file)
    orders = pd.read_csv(order_file, compression="gzip", encoding="latin-1", low_memory=False)
    return predictions, orders

def find_col(df, names):
    normalized = {" ".join(str(c).lower().replace("_", " ").split()): c for c in df.columns}
    for name in names:
        key = " ".join(name.lower().replace("_", " ").split())
        if key in normalized:
            return normalized[key]
    for name in names:
        key = name.lower().replace("_", " ")
        for normalized_name, original in normalized.items():
            if key in normalized_name:
                return original
    return None

try:
    predictions, orders = load_data()
except Exception as exc:
    st.error(f"Dashboard data could not be loaded: {exc}")
    st.info("Keep the `data` folder from the project ZIP beside app.py when uploading the project.")
    st.stop()

late_col = find_col(orders, ["Late_delivery_risk", "Late Delivery Risk"])
shipping_col = find_col(orders, ["Shipping Mode"])
region_col = find_col(orders, ["Order Region"])
market_col = find_col(orders, ["Market"])
sales_col = find_col(orders, ["Sales"])
category_col = find_col(orders, ["Category Name"])
risk_col = find_col(predictions, ["Risk_Category"])
prob_col = find_col(predictions, ["Late_Delivery_Probability"])

late = pd.to_numeric(orders[late_col], errors="coerce") if late_col else pd.Series(dtype=float)
valid_late = late[late.isin([0, 1])]
late_rate = valid_late.mean() * 100 if not valid_late.empty else None
late_count = int((valid_late == 1).sum()) if not valid_late.empty else None
high_risk = int(predictions[risk_col].astype(str).str.lower().str.contains("high|critical").sum()) if risk_col else None

st.title("📦 Smart Chain AI — Late Delivery Risk")
st.caption("Executive overview • Machine-learning risk predictions • Operational performance")

st.subheader("Executive overview")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total orders", f"{len(orders):,}")
k2.metric("High-risk predictions", f"{high_risk:,}" if high_risk is not None else "N/A")
k3.metric("Late delivery rate", f"{late_rate:.2f}%" if late_rate is not None else "N/A")
k4.metric("Late orders", f"{late_count:,}" if late_count is not None else "N/A")

st.subheader("Main insights")
insights = []
if late_rate is not None:
    insights.append(f"**{late_rate:.2f}%** of original orders with a valid late-risk label are marked as late.")
if high_risk is not None:
    insights.append(f"The model prediction file contains **{high_risk:,} high-risk predictions** from {len(predictions):,} prediction records.")
if shipping_col and late_col:
    ship = pd.DataFrame({"Shipping mode": orders[shipping_col].astype(str), "Late": late}).dropna()
    ship = ship[ship["Late"].isin([0, 1])]
    if not ship.empty:
        sr = ship.groupby("Shipping mode")["Late"].mean().mul(100).sort_values(ascending=False)
        insights.append(f"**{sr.index[0]}** has the highest observed late-delivery rate by shipping mode (**{sr.iloc[0]:.1f}%**).")
if region_col and late_col:
    reg = pd.DataFrame({"Region": orders[region_col].astype(str), "Late": late}).dropna()
    reg = reg[reg["Late"].isin([0, 1])]
    if not reg.empty:
        rr = reg.groupby("Region")["Late"].mean().mul(100).sort_values(ascending=False)
        insights.append(f"**{rr.index[0]}** has the highest observed late-delivery rate by region (**{rr.iloc[0]:.1f}%**).")
for item in insights:
    st.markdown(f'<div class="insight">{item}</div>', unsafe_allow_html=True)

left, right = st.columns(2)
with left:
    st.subheader("Predicted risk distribution")
    if risk_col:
        risk_counts = predictions[risk_col].fillna("Unknown").astype(str).value_counts().rename_axis("Risk category").reset_index(name="Records")
        fig = px.pie(risk_counts, names="Risk category", values="Records", hole=.48,
                     color_discrete_sequence=[BLUE, PINK, ORANGE, "#7c83fd"])
        st.plotly_chart(fig, width="stretch")
    elif prob_col:
        probs = pd.to_numeric(predictions[prob_col], errors="coerce").dropna()
        if not probs.empty and probs.max() > 1:
            probs = probs / 100
        fig = px.histogram(x=probs, nbins=25, labels={"x":"Late-delivery probability", "y":"Predictions"},
                           color_discrete_sequence=[PINK])
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("No recognized risk category or probability field found.")

with right:
    st.subheader("Late delivery by shipping mode")
    if shipping_col and late_col:
        frame = pd.DataFrame({"Shipping mode": orders[shipping_col].astype(str), "Late": late}).dropna()
        frame = frame[frame["Late"].isin([0, 1])]
        chart = frame.groupby("Shipping mode", as_index=False)["Late"].mean()
        chart["Late rate (%)"] = chart["Late"] * 100
        fig = px.bar(chart.sort_values("Late rate (%)", ascending=False), x="Shipping mode",
                     y="Late rate (%)", text_auto=".1f", color_discrete_sequence=[PINK])
        st.plotly_chart(fig, width="stretch")

left, right = st.columns(2)
with left:
    st.subheader("Late delivery by region")
    if region_col and late_col:
        frame = pd.DataFrame({"Region": orders[region_col].astype(str), "Late": late}).dropna()
        frame = frame[frame["Late"].isin([0, 1])]
        chart = frame.groupby("Region", as_index=False)["Late"].mean()
        chart["Late rate (%)"] = chart["Late"] * 100
        chart = chart.sort_values("Late rate (%)", ascending=False).head(12)
        fig = px.bar(chart, x="Late rate (%)", y="Region", orientation="h", color_discrete_sequence=[BLUE])
        st.plotly_chart(fig, width="stretch")

with right:
    st.subheader("Sales by market")
    if market_col and sales_col:
        frame = orders[[market_col, sales_col]].copy()
        frame[sales_col] = pd.to_numeric(frame[sales_col], errors="coerce")
        chart = frame.groupby(market_col, dropna=False)[sales_col].sum().reset_index()
        chart.columns = ["Market", "Sales"]
        fig = px.bar(chart.sort_values("Sales", ascending=False), x="Market", y="Sales",
                     color_discrete_sequence=[ORANGE])
        st.plotly_chart(fig, width="stretch")

with st.expander("View prediction sample"):
    st.dataframe(predictions.head(300), width="stretch", hide_index=True)
    st.download_button("Download predictions CSV", predictions.to_csv(index=False).encode("utf-8"),
                       file_name="predictions.csv", mime="text/csv")

st.caption("Prediction results and original orders are different populations; their counts should not be treated as the same denominator.")
