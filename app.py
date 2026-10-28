
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import html
from io import BytesIO
from pathlib import Path

# =========================================================
# PAGE SETUP
# =========================================================
st.set_page_config(
    page_title="Smart Chain AI Late Delivery Risk",
    page_icon="📦",
    layout="wide"
)

PINK = "#e653b4"
BLUE = "#1688ff"
ORANGE = "#ff7138"

st.markdown("""
<style>
.stApp {
    background: #171717;
}
[data-testid="stSidebar"] {
    background: #24242c;
}
h1, h2, h3, p, label {
    color: #ffffff;
}
.kpi-card {
    background: #ffffff;
    border-left: 6px solid #e653b4;
    border-radius: 12px;
    min-height: 112px;
    padding: 17px 10px;
    text-align: center;
    box-sizing: border-box;
}
.kpi-label {
    color: #333333;
    font-size: 15px;
    font-weight: 600;
    margin-bottom: 12px;
}
.kpi-value {
    color: #111111;
    font-size: 27px;
    font-weight: 700;
    overflow-wrap: anywhere;
}
.chart-heading {
    color: #ffffff;
    font-size: 18px;
    font-weight: 600;
    margin: 8px 0;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# DATA HELPERS
# =========================================================
def read_data(file_source):
    """Read CSV/Excel from a Streamlit upload or a local project file."""
    if file_source is None:
        return None

    if hasattr(file_source, "getvalue"):
        raw = file_source.getvalue()
        filename = file_source.name.lower()
        display_name = file_source.name
    else:
        path = Path(file_source)
        if not path.exists() or not path.is_file():
            return None
        raw = path.read_bytes()
        filename = path.name.lower()
        display_name = path.name

    if filename.endswith(".csv"):
        try:
            return pd.read_csv(BytesIO(raw), low_memory=False)
        except UnicodeDecodeError:
            return pd.read_csv(BytesIO(raw), encoding="latin-1", low_memory=False)

    if filename.endswith((".xlsx", ".xls")):
        try:
            return pd.read_excel(BytesIO(raw))
        except Exception as excel_error:
            # Some files contain CSV data despite having an Excel-like extension.
            try:
