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
