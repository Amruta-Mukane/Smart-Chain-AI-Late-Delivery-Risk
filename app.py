            fig = px.treemap(grouped, path=[category], values=value, color=value, color_continuous_scale="RdPu")
            show_chart(fig)
        else:
            st.info("Requires a product category column.")
    else:
        st.info("Requires the original order dataset.")

with r2c:
    st.subheader("Predicted Delivery Risk Distribution")
    if risk_cat_col:
        grouped = pred[risk_cat_col].fillna("Unknown").astype(str).value_counts().reset_index()
        grouped.columns = ["Risk Category", "Orders"]
        fig = px.pie(grouped, names="Risk Category", values="Orders", hole=0.58,
                     color="Risk Category",
                     color_discrete_map={"Low Risk": "#2439A6", "Medium Risk": "#1689FF", "High Risk": "#F06435"})
        fig.update_traces(textinfo="percent+label")
        show_chart(fig)
    elif prob_col:
        risk_cats = pd.cut(prob, bins=[-0.001, 0.33, 0.66, 1.001], labels=["Low Risk", "Medium Risk", "High Risk"])
        grouped = risk_cats.value_counts().reset_index()
        grouped.columns = ["Risk Category", "Orders"]
        fig = px.pie(grouped, names="Risk Category", values="Orders", hole=0.58,
                     color_discrete_sequence=["#2439A6", "#1689FF", "#F06435"])
        show_chart(fig)
    else:
        st.info("No risk category or probability column found.")

# Row 3: gauge, drivers, prediction probability distribution
r3a, r3b, r3c = st.columns(3)
with r3a:
    st.subheader("Late Delivery Rate Gauge")
    if actual_late_rate is not None:
        value = max(0, min(100, actual_late_rate))
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=value,
            number={"suffix": "%", "font": {"size": 34}},
            gauge={
                "axis": {"range": [0, 100], "ticksuffix": "%"},
                "bar": {"color": PINK},
                "steps": [{"range": [0, 50], "color": "#e5e7eb"}, {"range": [50, 100], "color": "#fce7f3"}],
                "threshold": {"line": {"color": NAVY, "width": 3}, "value": 50},
            },
        ))
        show_chart(fig, 290)
    else:
        st.info("Requires actual late-risk labels in the prediction or order data.")

with r3b:
    st.subheader("Top Late Delivery Risk Drivers")
    if filtered_orders is not None:
        risk = find_col(filtered_orders, ["Late_delivery_risk", "Late Delivery Risk", "Actual_Late_Risk"])
        numeric_cols = filtered_orders.select_dtypes(include="number").columns.tolist()
        excluded = {risk, find_col(filtered_orders, ["Order Id", "Order ID", "Customer Id", "Customer ID"])}
        candidates = [c for c in numeric_cols if c not in excluded and filtered_orders[c].nunique(dropna=True) > 1]
        if risk and candidates:
            target = as_binary(filtered_orders[risk])
            correlations = []
            for col in candidates:
                vals = pd.to_numeric(filtered_orders[col], errors="coerce")
                aligned = pd.concat([vals.rename("x"), target.rename("y")], axis=1).dropna()
                if len(aligned) > 5 and aligned["x"].nunique() > 1 and aligned["y"].nunique() > 1:
                    corr = aligned["x"].corr(aligned["y"])
                    if pd.notna(corr):
                        correlations.append((col, abs(float(corr)), float(corr)))
            if correlations:
                driver_df = pd.DataFrame(correlations, columns=["Feature", "Absolute correlation", "Correlation"])
                driver_df = driver_df.sort_values("Absolute correlation", ascending=True).tail(10)
                fig = px.bar(driver_df, x="Correlation", y="Feature", orientation="h",
                             color="Correlation", color_continuous_scale="RdPu",
                             hover_data=["Absolute correlation"])
                fig.update_layout(coloraxis_showscale=False)
                show_chart(fig, 330)
                st.caption("These are correlations with the late-risk label, not model feature importance.")
            else:
                st.info("Not enough numeric variation to estimate correlations.")
        else:
            st.info("Requires a late-risk label and numeric features.")
    else:
        st.info("Upload the original order dataset to calculate candidate risk drivers.")

with r3c:
    st.subheader("Late Delivery Probability")
    if prob_col and prob.notna().any():
        prob_pct = prob.dropna() * (100 if prob.dropna().max() <= 1 else 1)
        fig = px.histogram(
            pd.DataFrame({"Probability (%)": prob_pct}),
            x="Probability (%)", nbins=20, color_discrete_sequence=[PINK],
        )
        fig.update_layout(xaxis_title="Estimated probability (%)", yaxis_title="Prediction rows")
        show_chart(fig, 290)
    else:
        st.info("Probability column not found.")

with st.expander("View data and detected columns"):
    st.write("**Prediction data columns**")
    st.write(list(pred.columns))
    st.dataframe(pred.head(100), width="stretch")
    if orders is not None:
        st.write("**Order dataset columns**")
        st.write(list(orders.columns))
        st.caption(f"Showing up to 100 rows of {len(orders):,} original order rows.")
        st.dataframe(filtered_orders.head(100), width="stretch")

st.caption("Smart Chain AI • Late Delivery Risk Analytics | Metrics are calculated from the uploaded data. Missing source columns are not fabricated.")
