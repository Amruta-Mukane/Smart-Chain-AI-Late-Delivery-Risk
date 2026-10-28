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
                            show_chart(fig)
        else:
            st.info("The order date column could not be parsed.")
    else:
        st.info("An order date column was not found.")


# =========================================================
# RISK DRIVER ASSOCIATIONS
# =========================================================
st.divider()
st.markdown("## Top Late Delivery Risk Drivers")

st.caption(
    "These are associations in the uploaded dataset, not trained-model "
    "feature importance scores."
)

if data is not None and data["_late"].notna().any():
    factor_cols = [
        shipping_col, market_col, region_col, category_col, status_col
    ]
    factor_cols = list(dict.fromkeys(
        col for col in factor_cols if col is not None
    ))

    factors = []

    for col in factor_cols:
        temp = data[[col, "_late"]].copy()
        temp["_late"] = pd.to_numeric(temp["_late"], errors="coerce")
        temp = temp.dropna(subset=["_late"])

        if temp.empty:
            continue

        temp["_group"] = labels(temp[col])
        rates = temp.groupby("_group")["_late"].mean()

        if len(rates) > 1:
            factors.append({
                "Factor": str(col),
                "Association": float((rates.max() - rates.min()) * 100)
            })

    if factors:
        factor_df = pd.DataFrame(factors).sort_values(
            "Association", ascending=True
        )

        fig = px.bar(
            factor_df,
            x="Association",
            y="Factor",
            orientation="h",
            text="Association",
            color_discrete_sequence=[PINK]
        )
        fig.update_layout(
            xaxis_title="Difference in late rate (percentage points)",
            yaxis_title="",
            showlegend=False
        )
        show_chart(fig)
    else:
        st.info("Not enough category variation was found to compare factors.")
else:
    st.info(
        "Upload original orders with a recognizable late-risk column "
        "to calculate risk-driver associations."
    )


# =========================================================
# DATA TABLE AND DOWNLOADS
# =========================================================
st.divider()
st.markdown("## Prediction Results Explorer")

search_text = st.text_input(
    "Search prediction results",
    placeholder="Type a risk label or value"
)

result = pred.copy()

if search_text.strip():
    matches = result.astype(str).apply(
        lambda col: col.str.contains(
            search_text, case=False, na=False
        )
    ).any(axis=1)
    result = result[matches]

st.caption(f"{len(result):,} matching prediction rows.")

st.dataframe(
    result.head(1000),
    width="stretch",
    hide_index=True
)

st.download_button(
    "Download filtered predictions",
    data=result.to_csv(index=False).encode("utf-8"),
    file_name="filtered_predictions.csv",
    mime="text/csv"
)

if data is not None:
    with st.expander("Inspect original order data"):
        st.write(f"Rows: {len(data):,}")
        st.write(f"Columns: {len(data.columns)}")
        st.dataframe(data.head(100), width="stretch")

st.caption(
    "Smart Chain AI Late Delivery Risk | Built with Streamlit, "
    "Pandas and Plotly"
)

            show_chart(fig)
        else:
            st.info("The order date column could not be parsed.")
    else:
        st.info("An order date column was not found.")


# =========================================================
# RISK DRIVER ASSOCIATIONS
# =========================================================
st.divider()
st.markdown("## Top Late Delivery Risk Drivers")

st.caption(
    "These are associations in the uploaded dataset, not trained-model "
    "feature importance scores."
)

if data is not None and data["_late"].notna().any():
    factor_cols = [
        shipping_col, market_col, region_col, category_col, status_col
    ]
    factor_cols = list(dict.fromkeys(
        col for col in factor_cols if col is not None
    ))

    factors = []

    for col in factor_cols:
        temp = data[[col, "_late"]].copy()
        temp["_late"] = pd.to_numeric(temp["_late"], errors="coerce")
        temp = temp.dropna(subset=["_late"])

        if temp.empty:
            continue

        temp["_group"] = labels(temp[col])
        rates = temp.groupby("_group")["_late"].mean()

        if len(rates) > 1:
            factors.append({
                "Factor": str(col),
                "Association": float((rates.max() - rates.min()) * 100)
            })

    if factors:
        factor_df = pd.DataFrame(factors).sort_values(
            "Association", ascending=True
        )

        fig = px.bar(
            factor_df,
            x="Association",
            y="Factor",
            orientation="h",
            text="Association",
            color_discrete_sequence=[PINK]
        )
        fig.update_layout(
            xaxis_title="Difference in late rate (percentage points)",
            yaxis_title="",
            showlegend=False
        )
        show_chart(fig)
    else:
        st.info("Not enough category variation was found to compare factors.")
else:
    st.info(
        "Upload original orders with a recognizable late-risk column "
        "to calculate risk-driver associations."
    )


# =========================================================
# DATA TABLE AND DOWNLOADS
# =========================================================
st.divider()
st.markdown("## Prediction Results Explorer")

search_text = st.text_input(
    "Search prediction results",
    placeholder="Type a risk label or value"
)

result = pred.copy()

if search_text.strip():
    matches = result.astype(str).apply(
        lambda col: col.str.contains(
            search_text, case=False, na=False
        )
    ).any(axis=1)
    result = result[matches]

st.caption(f"{len(result):,} matching prediction rows.")

st.dataframe(
    result.head(1000),
    width="stretch",
    hide_index=True
)

st.download_button(
    "Download filtered predictions",
    data=result.to_csv(index=False).encode("utf-8"),
    file_name="filtered_predictions.csv",
    mime="text/csv"
)

if data is not None:
    with st.expander("Inspect original order data"):
        st.write(f"Rows: {len(data):,}")
        st.write(f"Columns: {len(data.columns)}")
        st.dataframe(data.head(100), width="stretch")

st.caption(
    "Smart Chain AI Late Delivery Risk | Built with Streamlit, "
    "Pandas and Plotly"
)
            show_chart(fig)
        else:
            st.info("The order date column could not be parsed.")
    else:
        st.info("An order date column was not found.")


# =========================================================
# RISK DRIVER ASSOCIATIONS
# =========================================================
st.divider()
st.markdown("## Top Late Delivery Risk Drivers")

st.caption(
    "These are associations in the uploaded dataset, not trained-model "
    "feature importance scores."
)

if data is not None and data["_late"].notna().any():
    factor_cols = [
        shipping_col, market_col, region_col, category_col, status_col
    ]
    factor_cols = list(dict.fromkeys(
        col for col in factor_cols if col is not None
    ))

    factors = []

    for col in factor_cols:
        temp = data[[col, "_late"]].copy()
        temp["_late"] = pd.to_numeric(temp["_late"], errors="coerce")
        temp = temp.dropna(subset=["_late"])

        if temp.empty:
            continue

        temp["_group"] = labels(temp[col])
        rates = temp.groupby("_group")["_late"].mean()

        if len(rates) > 1:
            factors.append({
                "Factor": str(col),
                "Association": float((rates.max() - rates.min()) * 100)
            })

    if factors:
        factor_df = pd.DataFrame(factors).sort_values(
            "Association", ascending=True
        )

        fig = px.bar(
            factor_df,
            x="Association",
            y="Factor",
            orientation="h",
            text="Association",
            color_discrete_sequence=[PINK]
        )
        fig.update_layout(
            xaxis_title="Difference in late rate (percentage points)",
            yaxis_title="",
            showlegend=False
        )
        show_chart(fig)
    else:
        st.info("Not enough category variation was found to compare factors.")
else:
    st.info(
        "Upload original orders with a recognizable late-risk column "
        "to calculate risk-driver associations."
    )


# =========================================================
# DATA TABLE AND DOWNLOADS
# =========================================================
st.divider()
st.markdown("## Prediction Results Explorer")

search_text = st.text_input(
    "Search prediction results",
    placeholder="Type a risk label or value"
)

result = pred.copy()

if search_text.strip():
    matches = result.astype(str).apply(
        lambda col: col.str.contains(
            search_text, case=False, na=False
        )
    ).any(axis=1)
    result = result[matches]

st.caption(f"{len(result):,} matching prediction rows.")

st.dataframe(
    result.head(1000),
    width="stretch",
    hide_index=True
)

st.download_button(
    "Download filtered predictions",
    data=result.to_csv(index=False).encode("utf-8"),
    file_name="filtered_predictions.csv",
    mime="text/csv"
)

if data is not None:
    with st.expander("Inspect original order data"):
        st.write(f"Rows: {len(data):,}")
        st.write(f"Columns: {len(data.columns)}")
        st.dataframe(data.head(100), width="stretch")

st.caption(
    "Smart Chain AI Late Delivery Risk | Built with Streamlit, "
    "Pandas and Plotly"
)

risk_counts = (
    labels(pred["_risk_category"])
    .value_counts()
    .rename_axis("Risk Category")
    .reset_index(name="Orders")
)

left, right = st.columns(2)

with left:
    heading("Predicted Risk Distribution")

    fig = px.pie(
        risk_counts,
        names="Risk Category",
        values="Orders",
        hole=0.55,
        color_discrete_sequence=[BLUE, "#2333ad", ORANGE, PINK, "#8b5cf6"]
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.add_annotation(
        text=f"{len(pred):,}<br>Orders",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=20, color="#333333")
    )
    show_chart(fig)

with right:
    heading("Late Delivery Probability")

    valid_prob = pred["_prob"].dropna()

    if len(valid_prob):
        fig = px.histogram(
            x=valid_prob * 100,
            nbins=25,
            labels={"x": "Probability (%)", "y": "Orders"},
            color_discrete_sequence=[PINK]
        )
        show_chart(fig)
    else:
        st.info("No probability column was found in the prediction file.")

left, right = st.columns(2)

with left:
    heading("Risk Level Breakdown")

    fig = px.bar(
        risk_counts,
        x="Risk Category",
        y="Orders",
        color_discrete_sequence=[PINK],
        text="Orders"
    )
    show_chart(fig)

with right:
    heading("Average Risk Probability")

    if len(valid_prob):
        avg_prob = float(valid_prob.mean() * 100)

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=avg_prob,
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": PINK},
                "steps": [
                    {"range": [0, 40], "color": "#d1fae5"},
                    {"range": [40, 70], "color": "#fef3c7"},
                    {"range": [70, 100], "color": "#fee2e2"}
                ]
            }
        ))
        fig.update_layout(
            template="plotly_white",
            height=320,
            paper_bgcolor="white",
            font_color="#333333"
        )
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("Upload a probability column to show the gauge.")

            show_chart(fig)
        else:
            st.info("The order date column could not be parsed.")
    else:
        st.info("An order date column was not found.")


# =========================================================
# RISK DRIVER ASSOCIATIONS
# =========================================================
st.divider()
st.markdown("## Top Late Delivery Risk Drivers")

st.caption(
    "These are associations in the uploaded dataset, not trained-model "
    "feature importance scores."
)

if data is not None and data["_late"].notna().any():
    factor_cols = [
        shipping_col, market_col, region_col, category_col, status_col
    ]
    factor_cols = list(dict.fromkeys(
        col for col in factor_cols if col is not None
    ))

    factors = []

    for col in factor_cols:
        temp = data[[col, "_late"]].copy()
        temp["_late"] = pd.to_numeric(temp["_late"], errors="coerce")
        temp = temp.dropna(subset=["_late"])

        if temp.empty:
            continue

        temp["_group"] = labels(temp[col])
        rates = temp.groupby("_group")["_late"].mean()

        if len(rates) > 1:
            factors.append({
                "Factor": str(col),
                "Association": float((rates.max() - rates.min()) * 100)
            })

    if factors:
        factor_df = pd.DataFrame(factors).sort_values(
            "Association", ascending=True
        )

        fig = px.bar(
            factor_df,
            x="Association",
            y="Factor",
            orientation="h",
            text="Association",
            color_discrete_sequence=[PINK]
        )
        fig.update_layout(
            xaxis_title="Difference in late rate (percentage points)",
            yaxis_title="",
            showlegend=False
        )
        show_chart(fig)
    else:
        st.info("Not enough category variation was found to compare factors.")
else:
    st.info(
        "Upload original orders with a recognizable late-risk column "
        "to calculate risk-driver associations."
    )


# =========================================================
# DATA TABLE AND DOWNLOADS
# =========================================================
st.divider()
st.markdown("## Prediction Results Explorer")

search_text = st.text_input(
    "Search prediction results",
    placeholder="Type a risk label or value"
)

result = pred.copy()

if search_text.strip():
    matches = result.astype(str).apply(
        lambda col: col.str.contains(
            search_text, case=False, na=False
        )
    ).any(axis=1)
    result = result[matches]

st.caption(f"{len(result):,} matching prediction rows.")

st.dataframe(
    result.head(1000),
    width="stretch",
    hide_index=True
)

st.download_button(
    "Download filtered predictions",
    data=result.to_csv(index=False).encode("utf-8"),
    file_name="filtered_predictions.csv",
    mime="text/csv"
)

if data is not None:
    with st.expander("Inspect original order data"):
        st.write(f"Rows: {len(data):,}")
        st.write(f"Columns: {len(data.columns)}")
        st.dataframe(data.head(100), width="stretch")

st.caption(
    "Smart Chain AI Late Delivery Risk | Built with Streamlit, "
    "Pandas and Plotly"
)
