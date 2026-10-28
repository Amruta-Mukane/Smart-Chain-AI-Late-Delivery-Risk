
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
                
    if numeric.notna().mean() >= 0.8:
        return numeric.where(numeric.isin([0, 1]))

    mapping = {
        "1": 1, "0": 0,
        "yes": 1, "no": 0,
        "true": 1, "false": 0,
        "late": 1,
        "on time": 0,
        "on-time": 0,
        "ontime": 0,
        "high risk": 1,
        "low risk": 0
    }

    return series.astype(str).str.strip().str.lower().map(mapping)


def probabilities(series):
    values = pd.to_numeric(series, errors="coerce")
    if values.notna().sum() == 0:
        return values

    if values.max() > 1:
        values = values / 100

    return values.clip(0, 1)


def labels(series):
    return series.fillna("Unknown").astype(str).replace("", "Unknown")


def kpi_card(label, value):
    safe_label = html.escape(str(label))
    safe_value = html.escape(str(value))

    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{safe_label}</div>
            <div class="kpi-value">{safe_value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def heading(text):
    st.markdown(
        f'<div class="chart-heading">{html.escape(text)}</div>',
        unsafe_allow_html=True
    )


def show_chart(fig, height=320):
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#333333", size=11),
        margin=dict(l=15, r=15, t=40, b=25),
        height=height
    )
    st.plotly_chart(fig, width="stretch")


def rate_table(df, group_col):
    temp = df[[group_col, "_late"]].copy()
    temp["_late"] = pd.to_numeric(temp["_late"], errors="coerce")
    temp = temp.dropna(subset=["_late"])

    if temp.empty:
        return pd.DataFrame()

    temp[group_col] = labels(temp[group_col])

    result = temp.groupby(group_col, as_index=False).agg(
        Orders=("_late", "size"),
        Late_Rate=("_late", "mean")
    )
    result["Late Rate (%)"] = result["Late_Rate"] * 100
    return result


def money_or_number(value):
    return f"{value:,.2f}"
