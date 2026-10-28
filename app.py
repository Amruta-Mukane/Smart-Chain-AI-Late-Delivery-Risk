
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import html
from io import BytesIO

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
def read_data(uploaded_file):
    """Read CSV or Excel, including files with a misleading extension."""
    if uploaded_file is None:
        return None

    raw = uploaded_file.getvalue()
    filename = uploaded_file.name.lower()

    if filename.endswith(".csv"):
        try:
            return pd.read_csv(BytesIO(raw), low_memory=False)
        except UnicodeDecodeError:
            return pd.read_csv(
                BytesIO(raw), encoding="latin-1", low_memory=False
            )

    if filename.endswith((".xlsx", ".xls")):
        try:
            return pd.read_excel(BytesIO(raw))
        except Exception as excel_error:
            # Some files have a CSV extension added before .xlsx.
            try:
                return pd.read_csv(BytesIO(raw), low_memory=False)
            except Exception:
                try:
                    return pd.read_csv(
                        BytesIO(raw), encoding="latin-1", low_memory=False
                    )
                except Exception:
                    st.error(
                        f"Could not read {uploaded_file.name}. "
                        f"Excel error: {excel_error}"
                    )
                    return None

    # Try CSV first, then Excel.
    try:
        return pd.read_csv(BytesIO(raw), low_memory=False)
    except Exception:
        try:
            return pd.read_excel(BytesIO(raw))
        except Exception as error:
            st.error(f"Unable to read file: {error}")
            return None


def find_col(df, candidates):
    """Find a column by normalized exact name, then partial name."""
    if df is None:
        return None

    def normalize(value):
        return " ".join(
            str(value).strip().lower().replace("_", " ").split()
        )

    actual = {normalize(c): c for c in df.columns}

    for candidate in candidates:
        key = normalize(candidate)
        if key in actual:
            return actual[key]

    for candidate in candidates:
        key = normalize(candidate)
        for normalized_name, original_name in actual.items():
            if key and key in normalized_name:
                return original_name

    return None


def binary_values(series):
    """Convert common late-risk labels to 0 or 1."""
    numeric = pd.to_numeric(series, errors="coerce")

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
    st.plotly_chart(fig, use_container_width=True)


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


# =========================================================
# SIDEBAR UPLOADS
# =========================================================
st.sidebar.title("Dashboard Filters")

prediction_file = st.sidebar.file_uploader(
    "Upload prediction results",
    type=["csv", "xlsx", "xls"],
    key="prediction_file"
)

orders_file = st.sidebar.file_uploader(
    "Upload original order dataset",
    type=["csv", "xlsx", "xls"],
    key="orders_file"
)

st.sidebar.divider()
st.sidebar.caption(
    "Prediction results drive AI risk charts. "
    "The original dataset drives order, sales, region and shipping charts."
)


# =========================================================
# HEADER
# =========================================================
st.title("📦 Smart Chain AI Late Delivery Risk")
st.caption("Machine Learning-Based Late Delivery Risk Prediction")


# =========================================================
# LOAD PREDICTION FILE
# =========================================================
pred = read_data(prediction_file)

if pred is None:
    st.info(
        "Upload ML_Risk_Predictions.csv in the sidebar to start the dashboard."
    )
    st.markdown("""
    ### Dashboard features
    - Total orders and late-delivery KPIs
    - Predicted risk distribution
    - Risk probability analysis
    - Shipping and regional analysis
    - Sales and product-category charts
    - Country-level risk map
    - Risk-driver associations
    """)
    st.stop()

if pred.empty:
    st.error("The prediction file has no rows.")
    st.stop()

pred.columns = [str(c).strip() for c in pred.columns]

# Prediction columns
prob_col = find_col(pred, [
    "Late_Delivery_Probability",
    "Late Delivery Probability",
    "Probability"
])

predicted_col = find_col(pred, [
    "Predicted_Late_Risk",
    "Predicted Late Risk",
    "Predicted",
    "Prediction"
])

actual_col = find_col(pred, [
    "Actual_Late_Risk",
    "Actual Late Risk",
    "Actual"
])

risk_category_col = find_col(pred, [
    "Risk_Category",
    "Risk Category",
    "Risk Level"
])

pred["_prob"] = (
    probabilities(pred[prob_col])
    if prob_col else float("nan")
)

pred["_predicted"] = (
    binary_values(pred[predicted_col])
    if predicted_col else float("nan")
)

pred["_actual"] = (
    binary_values(pred[actual_col])
    if actual_col else float("nan")
)

if risk_category_col:
    pred["_risk_category"] = labels(pred[risk_category_col])
elif pred["_prob"].notna().any():
    pred["_risk_category"] = pd.cut(
        pred["_prob"],
        bins=[-0.001, 0.4, 0.7, 1.001],
        labels=["Low Risk", "Medium Risk", "High Risk"]
    ).astype("object").fillna("Unknown")
elif pred["_predicted"].notna().any():
    pred["_risk_category"] = pred["_predicted"].map({
        0: "Low Risk",
        1: "High Risk"
    }).fillna("Unknown")
else:
    pred["_risk_category"] = "Unknown"


# =========================================================
# LOAD ORIGINAL ORDER DATA
# =========================================================
orders = read_data(orders_file)

if orders is not None:
    orders.columns = [str(c).strip() for c in orders.columns]

    shipping_col = find_col(orders, ["Shipping Mode", "Ship Mode"])
    market_col = find_col(orders, ["Market"])
    region_col = find_col(orders, ["Order Region", "Region"])
    country_col = find_col(orders, ["Order Country", "Country"])
    category_col = find_col(orders, ["Category Name", "Product Category"])
    sales_col = find_col(orders, ["Sales", "Sales Amount", "Revenue"])
    status_col = find_col(orders, ["Order Status", "Delivery Status"])
    late_col = find_col(
        orders,
        ["Late_delivery_risk", "Late Delivery Risk", "Late Risk", "Is Late"]
    )
    date_col = find_col(orders, ["order date", "Order Date", "Date"])

    if late_col:
        orders["_late"] = binary_values(orders[late_col])
    else:
        orders["_late"] = float("nan")

else:
    shipping_col = market_col = region_col = country_col = None
    category_col = sales_col = status_col = late_col = date_col = None


# =========================================================
# OPTIONAL FILTERS
# =========================================================
data = orders.copy() if orders is not None else None

if data is not None:
    with st.sidebar.expander("Filter order data", expanded=False):
        if market_col:
            options = sorted(labels(data[market_col]).unique())
            selected = st.multiselect("Market", options, default=options)
            data = data[labels(data[market_col]).isin(selected)]

        if shipping_col:
            options = sorted(labels(data[shipping_col]).unique())
            selected = st.multiselect(
                "Shipping mode", options, default=options
            )
            data = data[labels(data[shipping_col]).isin(selected)]

        if region_col:
            options = sorted(labels(data[region_col]).unique())
            selected = st.multiselect(
                "Region", options, default=options
            )
            data = data[labels(data[region_col]).isin(selected)]


# =========================================================
# KPI CARDS - EXPLICIT HTML TEXT
# =========================================================
st.markdown("## Executive Overview")

total_orders = len(data) if data is not None else len(pred)

high_categories = labels(pred["_risk_category"]).str.lower()
high_mask = high_categories.str.contains(
    r"high|critical", regex=True, na=False
)

if high_mask.any():
    high_risk_count = int(high_mask.sum())
elif pred["_prob"].notna().any():
    high_risk_count = int((pred["_prob"] >= 0.7).sum())
elif pred["_predicted"].notna().any():
    high_risk_count = int((pred["_predicted"] == 1).sum())
else:
    high_risk_count = None

late_rate = None
late_orders = None

if data is not None and data["_late"].notna().any():
    valid_late = data["_late"].dropna()
    late_rate = valid_late.mean() * 100
    late_orders = int((valid_late == 1).sum())

c1, c2, c3, c4 = st.columns(4)

with c1:
    kpi_card("Total Orders", f"{total_orders:,}")

with c2:
    kpi_card(
        "High-Risk Orders",
        f"{high_risk_count:,}" if high_risk_count is not None else "N/A"
    )

with c3:
    kpi_card(
        "Late Delivery Rate",
        f"{late_rate:.2f}%" if late_rate is not None else "N/A"
    )

with c4:
    kpi_card(
        "Late Orders",
        f"{late_orders:,}" if late_orders is not None else "N/A"
    )

st.caption(
    "KPIs are calculated from the uploaded data. If a required column "
    "is missing, its metric displays N/A rather than inventing a value."
)


# =========================================================
# PREDICTION CHARTS
# =========================================================
st.divider()
st.markdown("## Predicted Delivery Risk Distribution")

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
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Upload a probability column to show the gauge.")


# =========================================================
# ACTUAL VS PREDICTED
# =========================================================
st.markdown("## Model Output Review")

if pred["_actual"].notna().any() and pred["_predicted"].notna().any():
    compare = pred[["_actual", "_predicted"]].dropna()

    if not compare.empty:
        accuracy = (
            compare["_actual"] == compare["_predicted"]
        ).mean() * 100

        st.metric("Actual vs Predicted Accuracy", f"{accuracy:.2f}%")

        matrix = pd.crosstab(
            compare["_actual"].map({0: "Not Late", 1: "Late"}),
            compare["_predicted"].map({0: "Not Late", 1: "Late"})
        )

        fig = px.imshow(
            matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="Blues",
            labels={
                "x": "Predicted",
                "y": "Actual",
                "color": "Orders"
            }
        )
        show_chart(fig)
else:
    st.info(
        "Actual-versus-predicted comparison requires both actual and "
        "predicted label columns."
    )


# =========================================================
# ORIGINAL ORDER ANALYTICS
# =========================================================
st.divider()
st.markdown("## Operational Supply Chain Analytics")

if data is None:
    st.warning(
        "Upload the original order dataset to display shipping, sales, "
        "region, country, status and category charts."
    )
else:
    st.success(
        f"Original order data loaded: {len(data):,} rows and "
        f"{len(data.columns)} columns."
    )

    if not data["_late"].notna().any():
        st.warning(
            "A late-risk column was not recognized. Expected a column such "
            "as Late_delivery_risk. Check the dataset's column names."
        )

    # Shipping mode and sales by market
    left, right = st.columns(2)

    with left:
        heading("Late Delivery Risk by Shipping Mode")

        if shipping_col and data["_late"].notna().any():
            grouped = rate_table(data, shipping_col)

            fig = px.bar(
                grouped.sort_values("Orders", ascending=False),
                x=shipping_col,
                y="Orders",
                color_discrete_sequence=[PINK],
                text="Orders",
                hover_data=["Late Rate (%)"]
            )
            fig.update_layout(
                xaxis_title="Shipping Mode",
                yaxis_title="Orders",
                showlegend=False
            )
            show_chart(fig)
        else:
            st.info("Shipping mode or late-risk column is missing.")

    with right:
        heading("Sales Performance by Market")

        if market_col and sales_col:
            temp = data.copy()
            temp["_sales"] = pd.to_numeric(
                temp[sales_col], errors="coerce"
            )
            temp["_market"] = labels(temp[market_col])

            grouped = (
                temp.groupby("_market", as_index=False)["_sales"]
                .sum()
                .sort_values("_sales", ascending=False)
            )

            fig = px.bar(
                grouped,
                x="_sales",
                y="_market",
                orientation="h",
                text_auto=".3s",
                color_discrete_sequence=[PINK]
            )
            fig.update_layout(
                xaxis_title="Sales",
                yaxis_title="Market",
                showlegend=False
            )
            show_chart(fig)
        else:
            st.info("Market or sales column is missing.")

    # Regional rate and country map
    left, right = st.columns(2)

    with left:
        heading("Late Delivery Rate by Region (%)")

        if region_col and data["_late"].notna().any():
            grouped = rate_table(data, region_col).sort_values(
                "Late Rate (%)", ascending=True
            )

            fig = px.bar(
                grouped,
                x="Late Rate (%)",
                y=region_col,
                orientation="h",
                color_discrete_sequence=[PINK],
                hover_data=["Orders"]
            )
            fig.update_layout(
                xaxis_title="Late Rate (%)",
                yaxis_title="Region",
                showlegend=False
            )
            show_chart(fig)
        else:
            st.info("Region or late-risk column is missing.")

    with right:
        heading("Late Delivery Risk by Country")

        if country_col and data["_late"].notna().any():
            grouped = rate_table(data, country_col)

            fig = px.choropleth(
                grouped,
                locations=country_col,
                locationmode="country names",
                color="Late Rate (%)",
                hover_name=country_col,
                hover_data=["Orders"],
                color_continuous_scale="RdYlBu_r",
                projection="natural earth"
            )
            fig.update_layout(
                geo=dict(showframe=False, showcoastlines=True)
            )
            show_chart(fig)
        else:
            st.info("Country or late-risk column is missing.")

    # Region and shipping mode table, plus status distribution
    left, right = st.columns(2)

    with left:
        heading("Orders by Region and Shipping Mode")

        if region_col and shipping_col:
            temp = data.copy()
            temp["_region"] = labels(temp[region_col])
            temp["_shipping"] = labels(temp[shipping_col])

            pivot = pd.crosstab(
                temp["_region"], temp["_shipping"], margins=True
            )
            st.dataframe(
                pivot,
                use_container_width=True,
                height=320
            )
        else:
            st.info("Region or shipping mode column is missing.")

    with right:
        heading("Order Status Distribution")

        if status_col:
            counts = (
                labels(data[status_col])
                .value_counts()
                .rename_axis("Order Status")
                .reset_index(name="Orders")
            )

            fig = px.pie(
                counts,
                names="Order Status",
                values="Orders",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig.update_traces(textposition="outside", textinfo="value+percent")
            show_chart(fig)
        else:
            st.info("Order status column is missing.")

    # Product category and gauge
    left, right = st.columns(2)

    with left:
        heading("Orders by Product Category")

        if category_col:
            counts = (
                labels(data[category_col])
                .value_counts()
                .rename_axis("Product Category")
                .reset_index(name="Orders")
            )

            fig = px.treemap(
                counts,
                path=["Product Category"],
                values="Orders",
                color="Orders",
                color_continuous_scale="RdPu"
            )
            show_chart(fig)
        else:
            st.info("Product category column is missing.")

    with right:
        heading("Late Delivery Rate Gauge")

        if data["_late"].notna().any():
            gauge_rate = float(data["_late"].dropna().mean() * 100)

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=gauge_rate,
                number={"suffix": "%"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": PINK},
                    "steps": [
                        {"range": [0, 40], "color": "#eeeeee"},
                        {"range": [40, 70], "color": "#d7d7d7"},
                        {"range": [70, 100], "color": "#bbbbbb"}
                    ]
                }
            ))
            fig.update_layout(
                template="plotly_white",
                height=320,
                paper_bgcolor="white",
                font_color="#333333"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Late-risk data is required for this gauge.")

    # Monthly order trend
    st.markdown("### Order Trend Over Time")

    if date_col:
        temp = data.copy()
        temp["_date"] = pd.to_datetime(
            temp[date_col], errors="coerce", dayfirst=True
        )
        temp = temp.dropna(subset=["_date"])

        if not temp.empty:
            temp["_month"] = temp["_date"].dt.to_period("M").astype(str)
            monthly = temp.groupby("_month").size().reset_index(name="Orders")

            fig = px.line(
                monthly,
                x="_month",
                y="Orders",
                markers=True,
                color_discrete_sequence=[PINK]
            )
            fig.update_layout(
                xaxis_title="Month",
                yaxis_title="Orders"
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
    use_container_width=True,
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
        st.dataframe(data.head(100), use_container_width=True)

st.caption(
    "Smart Chain AI Late Delivery Risk | Built with Streamlit, "
    "Pandas and Plotly"
)
