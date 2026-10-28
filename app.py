import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="Smart Chain AI | Delivery Risk", page_icon="📦", layout="wide")

# ---------- Theme ----------
st.markdown("""
<style>
.stApp {background:#10141c;color:#f4f6fb}
[data-testid="stSidebar"] {background:#1b2230}
h1,h2,h3,p,label {color:#f4f6fb}
div[data-testid="stMetric"] {background:#202938;padding:16px;border-radius:12px;border-left:4px solid #e653b4}
.insight {background:#202938;border-radius:10px;padding:14px;margin-bottom:8px}
</style>
""", unsafe_allow_html=True)

PINK, BLUE, ORANGE = "#e653b4", "#1688ff", "#ff7138"

def load_file(file):
    if file is None:
        return None
    name = getattr(file, "name", str(file)).lower()
    try:
        if name.endswith(".csv"):
            return pd.read_csv(file, low_memory=False)
        return pd.read_excel(file)
    except Exception:
        try:
            file.seek(0)
            return pd.read_csv(file, low_memory=False)
        except Exception:
            st.error(f"Could not read {name}. Check the file format.")
            return None

def default_file(names):
    root = Path(__file__).resolve().parent
    for folder in (root / "data", root):
        for name in names:
            path = folder / name
            if path.exists():
                return path
    return None

def column(df, names):
    if df is None:
        return None
    norm = {str(c).strip().lower().replace("_", " "): c for c in df.columns}
    for n in names:
        key = n.lower().replace("_", " ")
        if key in norm:
            return norm[key]
    for n in names:
        key = n.lower().replace("_", " ")
        for k, original in norm.items():
            if key in k:
                return original
    return None

def binary(series):
    nums = pd.to_numeric(series, errors="coerce")
    if nums.notna().mean() > 0.8:
        return nums.where(nums.isin([0, 1]))
    mapping = {"yes":1, "no":0, "true":1, "false":0, "late":1,
               "on time":0, "on-time":0, "high risk":1, "low risk":0}
    return series.astype(str).str.strip().str.lower().map(mapping)

def read_default(path):
    if path is None:
        return None
    try:
        return pd.read_csv(path, low_memory=False) if path.suffix.lower() == ".csv" else pd.read_excel(path)
    except Exception:
        return None

# ---------- Data loading: bundled data first, uploads as fallback ----------
st.sidebar.title("Dashboard data")
pred_default = default_file(["ML_Risk_Predictions.csv", "ML_Risk_Predictions.xlsx", "ML_Risk_Predictions.csv.xlsx"])
orders_default = default_file(["orders.xlsx", "orders.csv", "APL_Lo...ed.csv.xlsx", "DataCoSupplyChainDataset.csv", "DataCoSupplyChainDataset.xlsx"])

pred_upload = st.sidebar.file_uploader("Prediction results (CSV/XLSX)", type=["csv","xlsx","xls"])
orders_upload = st.sidebar.file_uploader("Original orders (CSV/XLSX)", type=["csv","xlsx","xls"])

pred = load_file(pred_upload) if pred_upload else read_default(pred_default)
orders = load_file(orders_upload) if orders_upload else read_default(orders_default)

st.title("📦 Smart Chain AI — Late Delivery Risk")
st.caption("Machine-learning risk review • Operational performance • Actionable insights")

if pred is None and orders is None:
    st.warning("The dashboard code is deployed, but no default datasets were found. Add approved files to a `data/` folder or upload them in the sidebar.")
    st.markdown("**Expected files:** `data/ML_Risk_Predictions.csv` and `data/orders.xlsx`.")
    st.stop()
    

# ---------- Identify fields ----------
late_col = column(orders, ["Late_delivery_risk", "Late Delivery Risk", "Late Risk", "Is Late"])
shipping_col = column(orders, ["Shipping Mode", "Ship Mode"])
region_col = column(orders, ["Order Region", "Region"])
country_col = column(orders, ["Order Country", "Country"])
market_col = column(orders, ["Market"])
sales_col = column(orders, ["Sales", "Sales Amount", "Revenue"])
category_col = column(orders, ["Category Name", "Product Category"])
status_col = column(orders, ["Order Status", "Delivery Status"])
prob_col = column(pred, ["Late_Delivery_Probability", "Late Delivery Probability", "Probability"])
risk_col = column(pred, ["Risk_Category", "Risk Category", "Risk Level"])
pred_col = column(pred, ["Predicted_Late_Risk", "Predicted Late Risk", "Prediction"])

late = binary(orders[late_col]) if orders is not None and late_col else pd.Series(dtype=float)
rate = late.dropna().mean() * 100 if len(late.dropna()) else None
late_count = int((late == 1).sum()) if len(late.dropna()) else None
total_orders = len(orders) if orders is not None else (len(pred) if pred is not None else 0)

if pred is not None and risk_col:
    risk_labels = pred[risk_col].fillna("Unknown").astype(str)
    high_risk = int(risk_labels.str.lower().str.contains("high|critical").sum())
elif pred is not None and prob_col:
    probability = pd.to_numeric(pred[prob_col], errors="coerce")
    if probability.max() > 1:
        probability = probability / 100
    high_risk = int((probability >= 0.7).sum())
elif pred is not None and pred_col:
    high_risk = int((binary(pred[pred_col]) == 1).sum())
else:
    high_risk = None

# ---------- KPI overview ----------
a,b,c,d = st.columns(4)
a.metric("Total orders", f"{total_orders:,}")
b.metric("High-risk predictions", f"{high_risk:,}" if high_risk is not None else "N/A")
c.metric("Late delivery rate", f"{rate:.2f}%" if rate is not None else "N/A")
d.metric("Late orders", f"{late_count:,}" if late_count is not None else "N/A")

# ---------- Evidence-based insight summary ----------
st.subheader("Key insights")
insights = []
if rate is not None:
    insights.append(f"Late delivery rate is **{rate:.2f}%** across {len(late.dropna()):,} orders with a valid late-risk label.")
if high_risk is not None and pred is not None:
    insights.append(f"The prediction file contains **{high_risk:,} high-risk records** out of {len(pred):,} predictions.")
if orders is not None and shipping_col and len(late.dropna()):
    tmp = pd.DataFrame({"Shipping mode": orders[shipping_col].astype(str), "Late": late}).dropna()
    if not tmp.empty:
        rates = tmp.groupby("Shipping mode")["Late"].mean().mul(100).sort_values(ascending=False)
        insights.append(f"Highest late rate by shipping mode: **{rates.index[0]} ({rates.iloc[0]:.1f}%)**.")
if orders is not None and region_col and len(late.dropna()):
    tmp = pd.DataFrame({"Region": orders[region_col].astype(str), "Late": late}).dropna()
    if not tmp.empty:
        rates = tmp.groupby("Region")["Late"].mean().mul(100).sort_values(ascending=False)
        insights.append(f"Highest late rate by region: **{rates.index[0]} ({rates.iloc[0]:.1f}%)**.")
if not insights:
    insights.append("Upload datasets containing recognizable prediction and late-delivery fields to generate evidence-based insights.")
for insight in insights:
    st.markdown(f'<div class="insight">{insight}</div>', unsafe_allow_html=True)

# ---------- Charts ----------
left, right = st.columns(2)
with left:
    st.subheader("Predicted risk distribution")
    if pred is not None and risk_col:
        counts = pred[risk_col].fillna("Unknown").astype(str).value_counts().rename_axis("Risk").reset_index(name="Orders")
        st.plotly_chart(px.pie(counts, names="Risk", values="Orders", hole=.48,
                               color_discrete_sequence=[BLUE, PINK, ORANGE, "#7c83fd"]), use_container_width=True)
    elif pred is not None and prob_col:
        p = pd.to_numeric(pred[prob_col], errors="coerce").dropna()
        if p.max() > 1: p = p / 100
        st.plotly_chart(px.histogram(x=p, nbins=25, labels={"x":"Late-delivery probability","y":"Records"},
                                     color_discrete_sequence=[PINK]), use_container_width=True)
    else:
        st.info("Prediction risk category/probability column not found.")

with right:
    st.subheader("Late delivery by shipping mode")
    if orders is not None and shipping_col and len(late.dropna()):
        chart_df = pd.DataFrame({"Shipping mode": orders[shipping_col].astype(str), "Late": late}).dropna()
        chart_df = chart_df.groupby("Shipping mode", as_index=False)["Late"].mean()
        chart_df["Late rate (%)"] = chart_df["Late"] * 100
        st.plotly_chart(px.bar(chart_df.sort_values("Late rate (%)", ascending=False),
                               x="Shipping mode", y="Late rate (%)", text_auto=".1f",
                               color_discrete_sequence=[PINK]), use_container_width=True)
    else:
        st.info("Shipping mode or late-delivery label is unavailable.")

left, right = st.columns(2)
with left:
    st.subheader("Late delivery by region")
    if orders is not None and region_col and len(late.dropna()):
        chart_df = pd.DataFrame({"Region": orders[region_col].astype(str), "Late": late}).dropna()
        chart_df = chart_df.groupby("Region", as_index=False)["Late"].mean()
        chart_df["Late rate (%)"] = chart_df["Late"] * 100
        chart_df = chart_df.sort_values("Late rate (%)", ascending=False).head(15)
        st.plotly_chart(px.bar(chart_df, x="Late rate (%)", y="Region", orientation="h",
                               color_discrete_sequence=[BLUE]), use_container_width=True)
    else:
        st.info("Region or late-delivery label is unavailable.")

with right:
    st.subheader("Sales by market")
    if orders is not None and market_col and sales_col:
        sales = orders.copy()
        sales[sales_col] = pd.to_numeric(sales[sales_col], errors="coerce")
        chart_df = sales.groupby(market_col, dropna=False)[sales_col].sum().reset_index()
        chart_df.columns = ["Market", "Sales"]
        st.plotly_chart(px.bar(chart_df.sort_values("Sales", ascending=False), x="Market", y="Sales",
                               color_discrete_sequence=[ORANGE]), use_container_width=True)
    else:
        st.info("Market or sales columns are unavailable.")

with st.expander("View prediction records"):
    if pred is not None:
        st.dataframe(pred.head(500), width="stretch", hide_index=True)
        st.download_button("Download predictions as CSV", pred.to_csv(index=False).encode("utf-8"),
                           "predictions.csv", "text/csv")
    else:
        st.info("No prediction file loaded.")

st.caption("Insights are calculated from the loaded data. Prediction records and original orders may represent different populations.")
