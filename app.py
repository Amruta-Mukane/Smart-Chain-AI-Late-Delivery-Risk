
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Smart Chain AI Late Delivery Risk",
    page_icon="📦",
    layout="wide"
)

st.title("📦 Smart Chain AI Late Delivery Risk")
st.markdown(
    "### Machine Learning-Based Late Delivery Risk Prediction"
)
st.write(
    "Analyze delivery performance, explore logistics trends, "
    "and understand late-delivery risk."
)

st.info(
    "Welcome! Upload your logistics dataset to begin exploring "
    "delivery performance."
)

uploaded_file = st.file_uploader(
    "Upload your cleaned dataset (CSV or Excel)",
    type=["csv", "xlsx"]
)

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.success("Dataset loaded successfully!")
        st.subheader("Dataset Preview")
        st.dataframe(df.head(100), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        with col2:
            st.metric("Total Columns", len(df.columns))

        st.subheader("Explore Your Data")
        st.write("Column names:", list(df.columns))

        numeric_columns = df.select_dtypes(
            include="number"
        ).columns.tolist()

        if numeric_columns:
            selected_column = st.selectbox(
                "Choose a numeric column to visualize",
                numeric_columns
            )
            st.bar_chart(df[selected_column].dropna())

    except Exception as e:
        st.error(f"Could not read the dataset: {e}")
else:
    st.caption(
        "Your data stays unprocessed until you upload a file. "
        "Use a dataset you are permitted to share."
    )
