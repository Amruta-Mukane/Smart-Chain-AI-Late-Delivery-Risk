
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Smart Chain AI Late Delivery Risk",
    page_icon="📦",
    layout="wide"
)

PINK = "#e653b4"
BLUE = "#1688ff"
ORANGE = "#ff7138"
PURPLE = "#6516a8"
DARK = "#171717"

# =========================================================
# 2. DASHBOARD STYLING
# =========================================================
st.markdown("""
<style>
    .stApp {
        background-color: #171717;
        color: #ffffff;
    }
    [data-testid="stSidebar"] {
        background-color: #24242c;
    }
    h1, h2, h3, p, label {
        color: #ffffff;
    }
    div[data-testid="stMetric"] {
        background-color: white;
        color: #222222;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #e653b4;
    }
    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] [data-testid="stMetricValue"],
    div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
        color: #222222 !important;
    }
    .dashboard-note {
        color: #d1d5db;
        font-size: 14px;
    }
    div[data-testid="stDataFrame"] {
        background-color: white;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. HELPER FUNCTIONS
# =========================================================
def read_data(uploaded_file):
    """Read a CSV or Excel file."""
    if uploaded_file is None:
        return None

    try:
        uploaded_file.seek(0)

        if uploaded_file.name.lower().endswith(".csv"):
            try:
                return pd.read_csv(uploaded_file, low_memory=False)
            except UnicodeDecodeError:
                uploaded_file.seek(0)
                return pd.read_csv(
                    uploaded_file,
                    encoding="latin-1",
                    low_memory=False
                )

        return pd.read_excel(uploaded_file)

    except Exception as error:
        st.error(f"Unable to read {uploaded_file.name}: {error}")
        return None


def find_col(df, names):
    """Find a column regardless of capitalization or spaces."""
    if df is None:
        return None

    normalized = {
        str(c).strip().lower().replace("_", " "): c
        for c in df.columns
    }

    for name in names:
        key = name.strip().lower().replace("_", " ")
        if key in normalized:
            return normalized[key]

    for name in names:
        key = name.strip().lower().replace("_", " ")
        for normalized_name, original_name in normalized.items():
            if key in normalized_name:
                return original_name

    return None


def binary_values(series):
    """Convert common yes/no, late/on-time or numeric labels to 0/1."""
    numeric = pd.to_numeric(series, errors="coerce")

    if numeric.notna().mean() >= 0.8:
        return numeric.where(numeric.isin([0, 1]))

    text = series.astype(str).str.strip().str.lower()

    mapping = {
        "1": 1,
        "0": 0,
        "yes": 1,
        "no": 0,
        "true": 1,
        "false": 0,
        "late": 1,
        "on time": 0,
        "on-time": 0,
        "ontime": 0,
        "high risk": 1,
        "low risk": 0
    }

    return text.map(mapping)


def make_probability(series):
    """Normalize probabilities to the 0–1 range."""
    values = pd.to_numeric(series, errors="coerce")

    if values.notna().sum() == 0:
        return values

    if values.max() > 1:
        values = values / 100

    return values.clip(0, 1)


def clean_labels(series):
    return series.fillna("Unknown").astype(str).replace("", "Unknown")


def styled_chart(fig, height=320):
    """Apply consistent styling to Plotly charts."""
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#333333", size=11),
        margin=dict(l=15, r=15, t=48, b=15),
        height=height,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            xanchor="left",
            x=0
        )
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eeeeee")
    fig.update_yaxes(showgrid=True, gridcolor="#eeeeee")
    st.plotly_chart(fig, use_container_width=True)


def chart_title(title):
    st.markdown(
        f"<h4 style='color:#222222; text-align:center; "
        f"font-weight:500'>{title}</h4>",
        unsafe_allow_html=True
    )


def safe_rate(df, group_col, risk_col):
    """Calculate late rate and order count by a category."""
    temp = df[[group_col, risk_col]].copy()
    temp[risk_col] = pd.to_numeric(temp[risk_col], errors="coerce")
    temp = temp.dropna(subset=[risk_col])
    temp[group_col] = clean_labels(temp[group_col])

    if temp.empty:
        return pd.DataFrame()

    result = temp.groupby(group_col, as_index=False).agg(
        Orders=(risk_col, "size"),
        Late_Rate=(risk_col, "mean")
    )
    result["Late Rate (%)"] = result["Late_Rate"] * 100
    return result


# =========================================================
# 4. SIDEBAR: UPLOADS AND FILTERS
# =========================================================
st.sidebar.title("Dashboard Filters")

prediction_file = st.sidebar.file_uploader(
    "Upload prediction results",
    type=["csv", "xlsx"],
    help="Upload ML_Risk_Predictions.csv"
)

orders_file = st.sidebar.file_uploader(
    "Upload original order dataset",
    type=["csv", "xlsx"],
    help="Upload the original dataset used for operational analysis."
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Use the original order dataset for shipping, market, region, "
    "country, sales and product-category charts."
)

# =========================================================
# 5. TITLE
# =========================================================
st.title("📦 Smart Chain AI Late Delivery Risk")
st.caption("Machine Learning-Based Late Delivery Risk Prediction")


# =========================================================
# 6. LOAD DATA
# =========================================================
pred = read_data(prediction_file)
orders = read_data(orders_file)

if pred is None:
    st.info(
        "Upload your ML_Risk_Predictions.csv file using the sidebar "
        "to display the prediction dashboard."
    )
    st.markdown("""
    ### Dashboard features
    - Order and late-delivery KPIs
    - Predicted delivery-risk distribution
    - Shipping-mode and regional analysis
    - Sales and product-category analysis
    - Country-level risk map
    - Risk factors associated with late deliveries
    """)
    st.stop()

if pred.empty:
    st.error("The prediction dataset is empty.")
    st.stop()

pred.columns = [str(c).strip() for c in pred.columns]

# Prediction columns
prob_col = find_col(pred, [
    "Late_Delivery_Probability",
    "Late Delivery Probability",
    "Probability",
    "Predicted Probability"
])

pred_col = find_col(pred, [
    "Predicted_Late_Risk",
    "Predicted Late Risk",
    "Prediction",
    "Predicted"
])

actual_pred_col = find_col(pred, [
    "Actual_Late_Risk",
    "Actual Late Risk",
    "Actual"
])

category_pred_col = find_col(pred, [
    "Risk_Category",
    "Risk Category",
    "Risk Level"
])

if prob_col:
    pred["_prob"] = make_probability(pred[prob_col])
else:
    pred["_prob"] = float("nan")

if pred_col:
    pred["_predicted"] = binary_values(pred[pred_col])
else:
    pred["_predicted"] = float("nan")

if actual_pred_col:
    pred["_actual"] = binary_values(pred[actual_pred_col])
else:
    pred["_actual"] = float("nan")

if category_pred_col:
    pred["_risk_category"] = clean_labels(pred[category_pred_col])
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
# 7. PREPARE ORIGINAL ORDER DATA
# =========================================================
if orders is not None:
    orders.columns = [str(c).strip() for c in orders.columns]

    shipping_col = find_col(orders, [
        "Shipping Mode", "Ship Mode"
    ])
    market_col = find_col(orders, [
        "Market"
    ])
    region_col = find_col(orders, [
        "Order Region", "Region"
    ])
    country_col = find_col(orders, [
        "Order Country", "Country"
    ])
    category_col = find_col(orders, [
        "Category Name", "Product Category"
    ])
    sales_col = find_col(orders, [
        "Sales", "Sales Amount", "Revenue"
    ])
    status_col = find_col(orders, [
        "Order Status", "Delivery Status", "Status"
    ])
    late_col = find_col(orders, [
        "Late_delivery_risk", "Late Delivery Risk",
        "Late Risk", "Is Late"
    ])
    date_col = find_col(orders, [
        "order date", "Order Date", "Date"
    ])
    scheduled_days_col = find_col(orders, [
        "Days for shipment (scheduled)",
        "Days for shipment scheduled",
        "Scheduled Shipping Days"
    ])
    actual_days_col = find_col(orders, [
        "Days for shipping (real)",
        "Days for shipping real",
        "Actual Shipping Days"
    ])

    if late_col:
        orders["_late"] = binary_values(orders[late_col])
    else:
        orders["_late"] = float("nan")

else:
    shipping_col = market_col = region_col = country_col = None
    category_col = sales_col = status_col = late_col = None
    date_col = scheduled_days_col = actual_days_col = None


# =========================================================
# 8. FILTERS
# =========================================================
filtered_orders = orders.copy() if orders is not None else None

if filtered_orders is not None:
    with st.sidebar.expander("Filter original order data", expanded=False):
        if market_col:
            market_options = sorted(
                clean_labels(filtered_orders[market_col]).unique().tolist()
            )
            selected_markets = st.multiselect(
                "Market",
                market_options,
                default=market_options
            )
            filtered_orders = filtered_orders[
                clean_labels(filtered_orders[market_col]).isin(selected_markets)
            ]

        if shipping_col:
            shipping_options = sorted(
                clean_labels(filtered_orders[shipping_col]).unique().tolist()
            )
            selected_shipping = st.multiselect(
                "Shipping mode",
                shipping_options,
                default=shipping_options
            )
            filtered_orders = filtered_orders[
                clean_labels(filtered_orders[shipping_col]).isin(selected_shipping)
            ]

        if region_col:
            region_options = sorted(
                clean_labels(filtered_orders[region_col]).unique().tolist()
            )
            selected_regions = st.multiselect(
                "Order region",
                region_options,
                default=region_options
            )
            filtered_orders = filtered_orders[
                clean_labels(filtered_orders[region_col]).isin(selected_regions)
            ]


# =========================================================
# 9. KPI CARDS
# =========================================================
st.markdown("## Executive Overview")

total_orders = (
    len(filtered_orders)
    if filtered_orders is not None
    else len(pred)
)

# Identify high risk from category, probability, or predicted label.
risk_text = pred["_risk_category"].astype(str).str.lower()
high_mask = risk_text.str.contains(
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

if filtered_orders is not None and filtered_orders["_late"].notna().any():
    valid_late = filtered_orders["_late"].dropna()
    late_rate = valid_late.mean() * 100
    late_orders = int((valid_late == 1).sum())

k1, k2, k3, k4 = st.columns(4)

k1.metric("Total Orders", f"{total_orders:,}")
k2.metric(
    "High-Risk Orders",
    f"{high_risk_count:,}" if high_risk_count is not None else "N/A"
)
k3.metric(
    "Late Delivery Rate",
    f"{late_rate:.2f}%" if late_rate is not None else "N/A"
)
k4.metric(
    "Late Orders",
    f"{late_orders:,}" if late_orders is not None else "N/A"
)

st.caption(
    "Total orders, late orders and late-delivery rate come from the "
    "original order dataset when uploaded. High-risk orders come from "
    "the prediction results. Counts may represent different datasets."
)


# =========================================================
# 10. MAIN DASHBOARD CHARTS
# =========================================================
st.markdown("---")

if filtered_orders is None:
    st.warning(
        "Upload the original order dataset to show shipping, market, "
        "region, country, status, category and risk-driver charts."
    )
else:
    # Row 1: Shipping mode, market, regional late rate, country map
    col1, col2 = st.columns(2)

    with col1:
        chart_title("Late Delivery Risk by Shipping Mode")

        if shipping_col and filtered_orders["_late"].notna().any():
            shipping_data = safe_rate(
                filtered_orders, shipping_col, "_late"
            )
            shipping_data = shipping_data.sort_values(
                "Orders", ascending=False
            )

            fig = px.bar(
                shipping_data,
                x=shipping_col,
                y="Orders",
                color_discrete_sequence=[PINK],
                text="Orders"
            )
            fig.update_layout(
                showlegend=False,
                xaxis_title="Shipping Mode",
                yaxis_title="Number of Orders"
            )
            styled_chart(fig, 290)
        else:
            st.info("Shipping mode and late-risk columns are required.")

    with col2:
        chart_title("Sales Performance by Market")

        if market_col and sales_col:
            sales_data = filtered_orders.copy()
            sales_data["_sales_value"] = pd.to_numeric(
                sales_data[sales_col], errors="coerce"
            )
            sales_data["_market_label"] = clean_labels(
                sales_data[market_col]
            )
            market_sales = (
                sales_data.groupby("_market_label", as_index=False)
                ["_sales_value"].sum()
                .sort_values("_sales_value", ascending=False)
            )

            fig = px.bar(
                market_sales,
                x="_sales_value",
                y="_market_label",
                orientation="h",
                text_auto=".3s",
                color_discrete_sequence=[PINK]
            )
            fig.update_layout(
                showlegend=False,
                xaxis_title="Sales",
                yaxis_title=""
            )
            styled_chart(fig, 290)
        else:
            st.info("Market and sales columns are required.")

    col1, col2 = st.columns(2)

    with col1:
        chart_title("Late Delivery Rate by Region (%)")

        if region_col and filtered_orders["_late"].notna().any():
            region_data = safe_rate(
                filtered_orders, region_col, "_late"
            ).sort_values("Late Rate (%)", ascending=False)

            fig = px.bar(
                region_data,
                x="Late Rate (%)",
                y=region_col,
                orientation="h",
                color_discrete_sequence=[PINK],
                hover_data=["Orders"]
            )
            fig.update_layout(
                showlegend=False,
                xaxis_title="Late Delivery Rate (%)",
                yaxis_title=""
            )
            styled_chart(fig, 320)
        else:
            st.info("Region and late-risk columns are required.")

    with col2:
        chart_title("Late Delivery Risk by Country & Region")

        if country_col and filtered_orders["_late"].notna().any():
            country_data = safe_rate(
                filtered_orders, country_col, "_late"
            )
            country_data = country_data.sort_values(
                "Orders", ascending=False
            ).head(100)

            # Use country names on Plotly's world map.
            fig = px.choropleth(
                country_data,
                locations=country_col,
                locationmode="country names",
                color="Late Rate (%)",
                hover_name=country_col,
                hover_data=["Orders", "Late Rate (%)"],
                color_continuous_scale="RdYlBu_r",
                projection="natural earth"
            )
            fig.update_layout(
                geo=dict(
                    showframe=False,
                    showcoastlines=True,
                    bgcolor="white"
                ),
                coloraxis_colorbar_title="Late Rate (%)"
            )
            styled_chart(fig, 320)
        else:
            st.info("Country and late-risk columns are required.")

    # Row 2: Region/shipping table, status pie, category treemap
    col1, col2 = st.columns(2)

    with col1:
        chart_title("Orders by Region and Shipping Mode")

        if region_col and shipping_col:
            pivot_data = pd.pivot_table(
                filtered_orders,
                index=region_col,
                columns=shipping_col,
                values=filtered_orders.columns[0],
                aggfunc="count",
                fill_value=0
            )
            pivot_data["Total"] = pivot_data.sum(axis=1)
            pivot_data = pivot_data.sort_values(
                "Total", ascending=False
            )
            st.dataframe(
                pivot_data,
                use_container_width=True,
                height=330
            )
        else:
            st.info("Region and shipping mode columns are required.")

    with col2:
        chart_title("Order Status Distribution")

        if status_col:
            status_counts = (
                clean_labels(filtered_orders[status_col])
                .value_counts()
                .rename_axis("Order Status")
                .reset_index(name="Orders")
            )

            fig = px.pie(
                status_counts,
                names="Order Status",
                values="Orders",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig.update_traces(
                textposition="outside",
                textinfo="value+percent"
            )
            styled_chart(fig, 330)
        else:
            st.info("An order status column was not found.")

    col1, col2 = st.columns(2)

    with col1:
        chart_title("Late Orders by Product Category")

        if category_col:
            category_counts = (
                clean_labels(filtered_orders[category_col])
                .value_counts()
                .rename_axis("Product Category")
                .reset_index(name="Orders")
            )

            fig = px.treemap(
                category_counts,
                path=["Product Category"],
                values="Orders",
                color="Orders",
                color_continuous_scale="RdPu"
            )
            styled_chart(fig, 330)
        else:
            st.info("A product category column was not found.")

    with col2:
        chart_title("Late Delivery Rate Gauge")

        if filtered_orders["_late"].notna().any():
            gauge_value = filtered_orders["_late"].mean() * 100

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=gauge_value,
                number={"suffix": "%", "font": {"size": 34}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": PINK},
                    "steps": [
                        {"range": [0, 40], "color": "#eeeeee"},
                        {"range": [40, 70], "color": "#d6d6d6"},
                        {"range": [70, 100], "color": "#bdbdbd"}
                    ]
                }
            ))
            fig.update_layout(
                height=330,
                paper_bgcolor="white",
                font_color="#333333",
                margin=dict(t=45, b=15, l=20, r=20)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("A valid late-risk column is required.")


# =========================================================
# 11. TOP 10 RISK FACTORS ASSOCIATED WITH LATE DELIVERIES
# =========================================================
st.markdown("---")
st.markdown("## Top 10 Late Delivery Risk Drivers")

st.caption(
    "This section identifies associations in the uploaded data. "
    "These are not model feature importances unless you provide "
    "a trained model's feature-importance output."
)

if filtered_orders is not None and late_col is not None:
    candidate_cols = [
        shipping_col,
        market_col,
        region_col,
        category_col,
        status_col
    ]

    # Add common DataCo columns when present.
    for possible in [
        "Days for shipment (scheduled)",
        "Days for shipping (real)",
        "Customer Segment",
        "Department Name",
        "Order Item Quantity",
        "Order Item Discount Rate",
        "Benefit per order",
        "Latitude",
        "Longitude"
    ]:
        found = find_col(filtered_orders, [possible])
        if found:
            candidate_cols.append(found)

    candidate_cols = list(dict.fromkeys(
        c for c in candidate_cols
        if c is not None and c != late_col
    ))

    factors = []

    for col in candidate_cols:
        temp = filtered_orders[[col, "_late"]].copy()
        temp["_late"] = pd.to_numeric(temp["_late"], errors="coerce")
        temp = temp.dropna(subset=["_late"])

        if temp.empty:
            continue

        if (
            pd.api.types.is_numeric_dtype(temp[col])
            and temp[col].nunique() > 10
        ):
            x = pd.to_numeric(temp[col], errors="coerce")
            valid = x.notna() & temp["_late"].notna()

            if valid.sum() > 2 and x[valid].nunique() > 1:
                correlation = x[valid].corr(temp.loc[valid, "_late"])

                if pd.notna(correlation):
                    factors.append({
                        "Factor": str(col),
                        "Association": abs(float(correlation)) * 100
                    })
        else:
            temp["_group"] = clean_labels(temp[col])

            group_rates = temp.groupby("_group")["_late"].mean()

            if len(group_rates) >= 2:
                spread = (
                    group_rates.max() - group_rates.min()
                ) * 100

                factors.append({
                    "Factor": str(col),
                    "Association": float(spread)
                })

    if factors:
        factor_df = pd.DataFrame(factors)
        factor_df = factor_df.sort_values(
            "Association", ascending=False
        ).head(10).sort_values("Association")

        fig = px.bar(
            factor_df,
            x="Association",
            y="Factor",
            orientation="h",
            text="Association",
            color_discrete_sequence=[PINK]
        )
        fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        fig.update_layout(
            xaxis_title="Association score (percentage points or |correlation| × 100)",
            yaxis_title="",
            showlegend=False,
            height=360
        )
        styled_chart(fig, 390)
    else:
        st.info("Not enough suitable columns were found to compare risk factors.")
else:
    st.info(
        "Upload original order data with a recognizable late-risk column "
        "to calculate associated risk factors."
    )


# =========================================================
# 12. PREDICTED DELIVERY RISK DISTRIBUTION
# =========================================================
st.markdown("---")
st.markdown("## Predicted Delivery Risk Distribution")

risk_counts = (
    clean_labels(pred["_risk_category"])
    .value_counts()
    .rename_axis("Risk Category")
    .reset_index(name="Orders")
)

col1, col2 = st.columns([1, 1])

with col1:
    fig = px.pie(
        risk_counts,
        names="Risk Category",
        values="Orders",
        hole=0.62,
        color_discrete_sequence=[BLUE, "#1c2eae", ORANGE, PINK, PURPLE]
    )
    fig.update_traces(
        textposition="outside",
        textinfo="value+percent"
    )
    fig.add_annotation(
        text=f"{len(pred):,}<br>Orders",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=20, color="#333333")
    )
    styled_chart(fig, 350)

with col2:
    st.markdown("### Prediction summary")

    st.metric("Prediction Rows", f"{len(pred):,}")

    if pred["_prob"].notna().any():
        st.metric(
            "Average Predicted Probability",
            f"{pred['_prob'].mean() * 100:.2f}%"
        )

    st.dataframe(
        risk_counts,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# 13. PREDICTION DATA EXPLORER
# =========================================================
st.markdown("---")
st.markdown("## Prediction Results Explorer")

search = st.text_input(
    "Search prediction results",
    placeholder="Search for a risk label or value"
)

result_view = pred.copy()

if search.strip():
    mask = result_view.astype(str).apply(
        lambda column: column.str.contains(
            search, case=False, na=False
        )
    ).any(axis=1)
    result_view = result_view[mask]

st.caption(f"{len(result_view):,} rows match your search.")

st.dataframe(
    result_view.head(1000),
    use_container_width=True,
    hide_index=True
)

st.download_button(
    "Download filtered predictions",
    data=result_view.to_csv(index=False).encode("utf-8"),
    file_name="filtered_predictions.csv",
    mime="text/csv"
)

if filtered_orders is not None:
    with st.expander("Inspect original order dataset"):
        st.write("Rows:", len(filtered_orders))
        st.write("Columns:", len(filtered_orders.columns))
        st.dataframe(
            filtered_orders.head(100),
            use_container_width=True,
            hide_index=True
        )

st.markdown("---")
st.caption(
    "Smart Chain AI Late Delivery Risk | Built with Streamlit, "
    "Pandas and Plotly"
)
