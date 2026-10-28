
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Smart Chain AI Late Delivery Risk",
    page_icon="📦",
    layout="wide"
)

PINK = "#E65DB1"
BLUE = "#1689FF"
NAVY = "#2439A6"

st.markdown("""
<style>
.stApp {background-color: #f5f6fa;}
[data-testid="stMetric"] {
    background: white;
    border-left: 5px solid #E65DB1;
    padding: 14px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

st.title("📦 Smart Chain AI Late Delivery Risk")
st.caption("Machine Learning-Based Late Delivery Risk Prediction")

with st.sidebar:
    st.header("Dashboard Filters")
    pred_file = st.file_uploader(
        "Upload prediction results",
        type=["csv", "xlsx"]
    )
    order_file = st.file_uploader(
        "Upload original order dataset",
        type=["csv", "xlsx"]
    )

def read_file(file):
    if file is None:
        return None
    file.seek(0)
    if file.name.lower().endswith(".csv"):
        return pd.read_csv(file)
    return pd.read_excel(file)

def find_col(df, choices):
    if df is None:
        return None
    normalized = {
        str(c).lower().replace("_", "").replace(" ", ""): c
        for c in df.columns
    }
    for choice in choices:
        key = choice.lower().replace("_", "").replace(" ", "")
        if key in normalized:
            return normalized[key]
    return None

def binary(series):
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    mapping = {
        "1": 1, "true": 1, "yes": 1, "late": 1,
        "0": 0, "false": 0, "no": 0, "on time": 0
    }
    return series.astype(str).str.lower().str.strip().map(mapping)

def chart(fig, height=350):
    fig.update_layout(
        height=height,
        margin=dict(l=15, r=15, t=50, b=15),
        paper_bgcolor="white",
        plot_bgcolor="white"
    )
    st.plotly_chart(fig, use_container_width=True)

pred = read_file(pred_file)
orders = read_file(order_file)

if pred is None:
    st.info(
        "Upload ML_Risk_Predictions.csv.xlsx in the sidebar "
        "to begin. Upload the original order dataset too "
        "to display the operational charts."
    )
    st.stop()

prob_col = find_col(pred, ["Late_Delivery_Probability"])
actual_col = find_col(pred, ["Actual_Late_Risk"])
pred_col = find_col(pred, ["Predicted_Late_Risk"])
category_col = find_col(pred, ["Risk_Category"])

prob = (
    pd.to_numeric(pred[prob_col], errors="coerce")
    if prob_col else pd.Series(dtype=float)
)
actual = binary(pred[actual_col]) if actual_col else None

if category_col:
    high_risk = pred[category_col].astype(str).str.contains(
        "high", case=False, na=False
    ).sum()
elif len(prob):
    high_risk = (prob >= 0.7).sum()
else:
    high_risk = None

late_rate = (
    actual.mean() * 100
    if actual is not None and actual.notna().any()
    else None
)
late_count = (
    int(actual.sum())
    if actual is not None and actual.notna().any()
    else None
)

if orders is not None:
    risk_col = find_col(
        orders, ["Late_delivery_risk", "Actual_Late_Risk"]
    )
    if risk_col:
        order_risk = binary(orders[risk_col])
        if order_risk.notna().any():
            late_rate = order_risk.mean() * 100
            late_count = int(order_risk.sum())

total_orders = len(orders) if orders is not None else len(pred)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Orders / Rows", f"{total_orders:,}")
k2.metric(
    "High-Risk Orders",
    f"{int(high_risk):,}" if high_risk is not None else "N/A"
)
k3.metric(
    "Late Delivery Rate",
    f"{late_rate:.2f}%" if late_rate is not None else "N/A"
)
k4.metric(
    "Late Orders",
    f"{late_count:,}" if late_count is not None else "N/A"
)

if category_col:
    st.subheader("Predicted Delivery Risk Distribution")
    risk_counts = pred[category_col].fillna("Unknown").value_counts()
    chart(px.pie(
        names=risk_counts.index,
        values=risk_counts.values,
        hole=0.55,
        color_discrete_sequence=[BLUE, NAVY, "#F06435", PINK]
    ))

elif prob_col:
    st.subheader("Predicted Delivery Risk Distribution")
    probability = prob.dropna()
    if not probability.empty:
        if probability.max() <= 1:
            probability = probability * 100
        risk_group = pd.cut(
            probability,
            bins=[-0.01, 33.33, 66.67, 100.01],
            labels=["Low Risk", "Medium Risk", "High Risk"]
        )
        counts = risk_group.value_counts().reindex(
            ["Low Risk", "Medium Risk", "High Risk"],
            fill_value=0
        )
        chart(px.pie(
            names=counts.index,
            values=counts.values,
            hole=0.55,
            color_discrete_sequence=[NAVY, BLUE, "#F06435"]
        ))

if prob_col:
    st.subheader("Late Delivery Probability")
    probability = prob.dropna()
    if not probability.empty:
        probability = probability * (
            100 if probability.max() <= 1 else 1
        )
        chart(px.histogram(
            x=probability,
            nbins=20,
            labels={"x": "Estimated probability (%)"},
            color_discrete_sequence=[PINK]
        ))

if late_rate is not None:
    st.subheader("Late Delivery Rate Gauge")
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=max(0, min(100, late_rate)),
        number={"suffix": "%"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": PINK},
            "steps": [
                {"range": [0, 50], "color": "#eeeeee"},
                {"range": [50, 100], "color": "#fce7f3"}
            ]
        }
    ))
    chart(gauge, 280)

if orders is not None:
    st.divider()
    st.header("Operational Insights")

    # Interactive filters
    filtered = orders.copy()
    with st.sidebar:
        st.subheader("Order Data Filters")
        for label, choices in [
            ("Market", ["Market"]),
            ("Region", ["Order Region", "Region"]),
            ("Shipping Mode", ["Shipping Mode"])
        ]:
            col = find_col(filtered, choices)
            if col:
                values = sorted(
                    filtered[col].dropna().astype(str).unique()
                )
                selected = st.multiselect(
                    label, values, default=values
                )
                filtered = filtered[
                    filtered[col].astype(str).isin(selected)
                ]

    ship = find_col(filtered, ["Shipping Mode"])
    risk = find_col(
        filtered, ["Late_delivery_risk", "Actual_Late_Risk"]
    )
    market = find_col(filtered, ["Market"])
    sales = find_col(filtered, ["Sales"])
    region = find_col(filtered, ["Order Region", "Region"])
    country = find_col(filtered, ["Order Country", "Country"])
    status = find_col(filtered, ["Order Status"])
    category = find_col(
        filtered, ["Category Name", "Product Category"]
    )

    a, b = st.columns(2)

    with a:
        st.subheader("Late Delivery Risk by Shipping Mode")
        if ship and risk:
            temp = filtered[[ship, risk]].copy()
            temp["Late"] = binary(temp[risk])
            summary = temp.groupby(ship)["Late"].sum().reset_index()
            chart(px.bar(
                summary, x=ship, y="Late",
                color_discrete_sequence=[PINK]
            ))
        else:
            st.info("Requires Shipping Mode and late-risk columns.")

    with b:
        st.subheader("Sales Performance by Market")
        if market and sales:
            temp = filtered.copy()
            temp[sales] = pd.to_numeric(temp[sales], errors="coerce")
            summary = temp.groupby(market)[sales].sum().reset_index()
            chart(px.bar(
                summary, x=sales, y=market, orientation="h",
                color_discrete_sequence=[PINK]
            ))
        else:
            st.info("Requires Market and Sales columns.")

    a, b = st.columns(2)

    with a:
        st.subheader("Late Delivery Rate by Region (%)")
        if region and risk:
            temp = filtered[[region, risk]].copy()
            temp["Late"] = binary(temp[risk])
            summary = (
                temp.groupby(region)["Late"].mean()
                .mul(100).reset_index(name="Late Rate (%)")
            )
            chart(px.line(
                summary, x=region, y="Late Rate (%)",
                markers=True, color_discrete_sequence=[PINK]
            ))
        else:
            st.info("Requires Region and late-risk columns.")

    with b:
        st.subheader("Order Status Distribution")
        if status:
            counts = filtered[status].fillna("Unknown").value_counts()
            chart(px.pie(
                names=counts.index, values=counts.values,
                hole=0.4
            ))
        else:
            st.info("Order Status column not found.")

    a, b = st.columns(2)

    with a:
        st.subheader("Late Orders by Product Category")
        if category and risk:
            temp = filtered[[category, risk]].copy()
            temp["Late"] = binary(temp[risk])
            summary = temp.groupby(category)["Late"].sum().reset_index()
            summary = summary.sort_values("Late", ascending=False).head(15)
            chart(px.treemap(
                summary, path=[category], values="Late",
                color="Late", color_continuous_scale="RdPu"
            ))
        else:
            st.info("Requires category and late-risk columns.")

    with b:
        st.subheader("Late Delivery Risk by Country")
        if country and risk:
            temp = filtered[[country, risk]].copy()
            temp["Late"] = binary(temp[risk])
            summary = temp.groupby(country)["Late"].mean().mul(100)
            summary = summary.sort_values(ascending=False).head(20)
            chart(px.bar(
                x=summary.values, y=summary.index,
                orientation="h",
                labels={"x": "Late risk (%)", "y": "Country"},
                color_discrete_sequence=[PINK]
            ))
        else:
            st.info("Requires Country and late-risk columns.")

    st.subheader("Top Late Delivery Risk Drivers")
    if risk:
        numeric = filtered.select_dtypes(include="number").columns
        target = binary(filtered[risk])
        correlations = []
        for col in numeric:
            if col == risk or filtered[col].nunique() < 2:
                continue
            pair = pd.concat(
                [pd.to_numeric(filtered[col], errors="coerce"), target],
                axis=1
            ).dropna()
            if len(pair) > 5 and pair.iloc[:, 0].nunique() > 1:
                corr = pair.iloc[:, 0].corr(pair.iloc[:, 1])
                if pd.notna(corr):
                    correlations.append((col, corr))
        if correlations:
            drivers = pd.DataFrame(
                correlations, columns=["Feature", "Correlation"]
            )
            drivers["Strength"] = drivers["Correlation"].abs()
            drivers = drivers.nlargest(10, "Strength")
            chart(px.bar(
                drivers, x="Correlation", y="Feature",
                orientation="h", color_discrete_sequence=[PINK]
            ))
            st.caption(
                "These are correlations, not model feature importance."
            )
        else:
            st.info("Not enough numeric data to estimate drivers.")

    st.subheader("Filtered Order Data")
    st.dataframe(filtered.head(100), use_container_width=True)

else:
    st.warning(
        "Upload your original order dataset to unlock market, "
        "shipping, region, country, status, category and driver insights."
    )

with st.expander("Prediction data preview"):
    st.write("Columns:", list(pred.columns))
    st.dataframe(pred.head(100), use_container_width=True)

st.caption(
    "Smart Chain AI • Metrics are calculated from uploaded data. "
    "Missing data is not fabricated."
)
