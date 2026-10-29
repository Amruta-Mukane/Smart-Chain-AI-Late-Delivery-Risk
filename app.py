from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(
    page_title="Smart Chain AI | Delivery Risk",
    page_icon="📦",
    layout="wide",
)

# ---------- Styling ----------
PINK = "#e653b4"
BLUE = "#1688ff"
ORANGE = "#ff7138"

st.markdown(
    """
    <style>
    .stApp {background:#11151d;color:#f5f7fb}
    [data-testid="stSidebar"] {background:#202633}
    h1,h2,h3,p,label {color:#f5f7fb}
    div[data-testid="stMetric"] {
        background:#242d3c;padding:15px;border-radius:12px;
        border-left:4px solid #e653b4
    }
    .insight {
        background:#242d3c;padding:12px 15px;border-radius:10px;
        margin:7px 0
    }
    </style>
    """,
    unsafe_allow_html=True,
)

ROOT = Path(__file__).resolve().parent


def find_file(candidates):
    """Search the repository root first, then the optional data/ folder."""
    for folder in (ROOT, ROOT / "data"):
        for name in candidates:
            candidate = folder / name
            if candidate.is_file():
                return candidate
    return None


@st.cache_data(show_spinner="Loading project datasets...")
def load_predictions(path_string, modified_time):
    path = Path(path_string)
    return pd.read_excel(path)


@st.cache_data(show_spinner="Loading order dataset...")
def load_orders(path_string, modified_time):
    path = Path(path_string)
    if path.name.lower().endswith(".gz"):
        return pd.read_csv(
            path, compression="gzip", encoding="latin-1", low_memory=False
        )
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, encoding="latin-1", low_memory=False)
    return pd.read_excel(path)


def find_col(df, candidates):
    if df is None:
        return None
    columns = {
        " ".join(str(col).lower().replace("_", " ").split()): col
        for col in df.columns
    }
    for candidate in candidates:
        key = " ".join(candidate.lower().replace("_", " ").split())
        if key in columns:
            return columns[key]
    for candidate in candidates:
        key = candidate.lower().replace("_", " ")
        for normalized, original in columns.items():
            if key in normalized:
                return original
    return None


def to_binary(series):
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().mean() > 0.8:
        return numeric.where(numeric.isin([0, 1]))
    mapping = {
        "1": 1, "0": 0, "yes": 1, "no": 0, "true": 1, "false": 0,
        "late": 1, "late delivery": 1, "on time": 0, "on-time": 0,
    }
    return series.astype(str).str.strip().str.lower().map(mapping)


# ---------- Load datasets automatically from the repository ----------
prediction_path = find_file([
    "ML_Risk_Predictions.csv.xlsx",
    "ML_Risk_Predictions.xlsx",
    "ML_Risk_Predictions.csv",
])
orders_path = find_file([
    "APL_Logistics.csv.gz",
    "APL_Logistics.csv",
    "APL_Logistics(1).csv",
    "orders.csv.gz",
    "orders.csv",
    "orders.xlsx",
    "DataCoSupplyChainDataset.csv",
])

st.title("📦 Smart Chain AI — Late Delivery Risk")
st.caption("Machine-learning risk review • Delivery performance • Main business insights")

missing = []
if prediction_path is None:
    missing.append("ML_Risk_Predictions.csv.xlsx")
if orders_path is None:
    missing.append("APL_Logistics.csv.gz")

if missing:
    st.error("The deployed app cannot find: " + ", ".join(missing))
    st.info(
        "Put the dataset files in the same GitHub folder as app.py, "
        "or in a folder named data/. Keep the filenames listed in the code."
    )
    st.stop()

try:
    predictions = load_predictions(
        str(prediction_path), prediction_path.stat().st_mtime
    )
    orders = load_orders(str(orders_path), orders_path.stat().st_mtime)
except Exception as exc:
    st.error(f"Could not load project data: {exc}")
    st.stop()

# ---------- Find required columns ----------
late_col = find_col(orders, ["Late_delivery_risk", "Late Delivery Risk"])
shipping_col = find_col(orders, ["Shipping Mode", "Ship Mode"])
region_col = find_col(orders, ["Order Region", "Region"])
market_col = find_col(orders, ["Market"])
sales_col = find_col(orders, ["Sales", "Sales Amount", "Revenue"])
status_col = find_col(orders, ["Order Status", "Delivery Status"])
category_col = find_col(orders, ["Category Name", "Product Category"])

risk_col = find_col(predictions, ["Risk_Category", "Risk Category"])
prob_col = find_col(predictions, ["Late_Delivery_Probability", "Probability"])

late = (
    to_binary(orders[late_col])
    if late_col is not None
    else pd.Series(index=orders.index, dtype="float64")
)
valid_late = late.dropna()
late_rate = valid_late.mean() * 100 if not valid_late.empty else None
late_orders = int((valid_late == 1).sum()) if not valid_late.empty else None

if risk_col is not None:
    risk_values = predictions[risk_col].fillna("Unknown").astype(str)
    high_risk = int(risk_values.str.lower().str.contains("high|critical").sum())
elif prob_col is not None:
    probability = pd.to_numeric(predictions[prob_col], errors="coerce")
    if probability.notna().any() and probability.max() > 1:
        probability = probability / 100
    high_risk = int((probability >= 0.7).sum())
else:
    high_risk = None

# ---------- KPI cards ----------
st.subheader("Executive overview")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total orders", f"{len(orders):,}")
k2.metric("High-risk predictions", f"{high_risk:,}" if high_risk is not None else "N/A")
k3.metric("Late delivery rate", f"{late_rate:.2f}%" if late_rate is not None else "N/A")
k4.metric("Late orders", f"{late_orders:,}" if late_orders is not None else "N/A")

      
# ---------- Main charts only ----------
left, right = st.columns(2)

with left:
    st.subheader("Predicted risk distribution")
    if risk_col is not None:
        counts = risk_values.value_counts().rename_axis("Risk category").reset_index(name="Records")
        fig = px.pie(
            counts, names="Risk category", values="Records", hole=0.48,
            color_discrete_sequence=[BLUE, PINK, ORANGE, "#7c83fd"],
        )
        st.plotly_chart(fig, width="stretch")
    elif prob_col is not None:
        prob = pd.to_numeric(predictions[prob_col], errors="coerce").dropna()
        if not prob.empty and prob.max() > 1:
            prob = prob / 100
        fig = px.histogram(
            x=prob, nbins=25,
            labels={"x": "Late-delivery probability", "y": "Predictions"},
            color_discrete_sequence=[PINK],
        )
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("Risk category/probability column was not found in the prediction file.")

with right:
    st.subheader("Late delivery by shipping mode")
    if shipping_col is not None and not valid_late.empty:
        temp = pd.DataFrame({
            "Shipping mode": orders[shipping_col].astype(str),
            "Late": late,
        }).dropna()
        chart = temp.groupby("Shipping mode", as_index=False)["Late"].mean()
        chart["Late rate (%)"] = chart["Late"] * 100
        fig = px.bar(
            chart.sort_values("Late rate (%)", ascending=False),
            x="Shipping mode", y="Late rate (%)", text_auto=".1f",
            color_discrete_sequence=[PINK],
        )
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("Shipping mode or late-delivery label was not found.")

left, right = st.columns(2)

with left:
    st.subheader("Late delivery by region")
    if region_col is not None and not valid_late.empty:
        temp = pd.DataFrame({"Region": orders[region_col].astype(str), "Late": late}).dropna()
        chart = temp.groupby("Region", as_index=False)["Late"].mean()
        chart["Late rate (%)"] = chart["Late"] * 100
        chart = chart.sort_values("Late rate (%)", ascending=False).head(12)
        fig = px.bar(
            chart, x="Late rate (%)", y="Region", orientation="h",
            color_discrete_sequence=[BLUE],
        )
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("Region or late-delivery label was not found.")

with right:
    st.subheader("Sales by market")
    if market_col is not None and sales_col is not None:
        temp = orders[[market_col, sales_col]].copy()
        temp[sales_col] = pd.to_numeric(temp[sales_col], errors="coerce")
        chart = temp.groupby(market_col, dropna=False)[sales_col].sum().reset_index()
        chart.columns = ["Market", "Sales"]
        fig = px.bar(
            chart.sort_values("Sales", ascending=False),
            x="Market", y="Sales", color_discrete_sequence=[ORANGE],
        )
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("Market or sales column was not found.")

with st.expander("Preview prediction records"):
    st.dataframe(predictions.head(300), width="stretch", hide_index=True)
    st.download_button(
        "Download predictions CSV",
        predictions.to_csv(index=False).encode("utf-8"),
        file_name="predictions.csv",
        mime="text/csv",
    )

st.caption(
    "Metrics are calculated from the available datasets. "
    "Prediction records and original orders may represent different populations."
)
