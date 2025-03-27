import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession

# Initialize Spark session
spark = SparkSession.builder.appName("BigData_EDA").getOrCreate()

# Streamlit App Configuration
st.set_page_config(page_title="Big Data EDA", layout="wide")

# Sidebar for Navigation
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Overview", "📊 Visualizations", "📈 Correlation Analysis"])

# Hardcoded file path
file_path = "bigdata.csv"

try:
    # Load CSV into PySpark DataFrame
    df = spark.read.csv(file_path, header=True, inferSchema=True)

    # Convert PySpark DataFrame to Pandas for visualization
    pandas_df = df.toPandas()

    if page == "🏠 Overview":
        st.title("🏠 Dataset Overview")
        
        with st.container():
            st.subheader("🔍 Preview of Data")
            st.write(pandas_df.head())

        with st.container():
            st.subheader("📏 Dataset Shape")
            st.write(f"Rows: {pandas_df.shape[0]}, Columns: {pandas_df.shape[1]}")

        with st.container():
            st.subheader("📌 Data Summary")
            st.write(pandas_df.describe())

        with st.container():
            st.subheader("⚠️ Missing Values")
            st.write(pandas_df.isnull().sum())

    elif page == "📊 Visualizations":
        st.title("📊 Exploratory Data Analysis")

        # Numeric & Categorical Columns
        numeric_cols = pandas_df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = pandas_df.select_dtypes(exclude=['number']).columns.tolist()

        with st.container():
            st.subheader("📊 Feature Distributions")
            selected_num = st.sidebar.selectbox("Select numeric feature", numeric_cols)
            
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.histplot(pandas_df[selected_num], kde=True, bins=30, ax=ax)
            ax.set_title(f"Distribution of {selected_num}")
            st.pyplot(fig)

        with st.container():
            st.subheader("🛠 Categorical Feature Analysis")
            selected_cat = st.sidebar.selectbox("Select categorical feature", categorical_cols)

            fig, ax = plt.subplots(figsize=(8, 4))
            sns.countplot(data=pandas_df, x=selected_cat, order=pandas_df[selected_cat].value_counts().index, ax=ax)
            ax.set_title(f"Count Plot of {selected_cat}")
            plt.xticks(rotation=45)
            st.pyplot(fig)

    elif page == "📈 Correlation Analysis":
        st.title("📈 Correlation Matrix")

        with st.container():
            selected_corr_cols = st.multiselect("Select columns for correlation matrix", numeric_cols, default=numeric_cols[:5])

            if len(selected_corr_cols) > 1:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(pandas_df[selected_corr_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
                st.pyplot(fig)
            else:
                st.warning("⚠️ Please select at least two numeric columns.")

    st.success("✅ EDA Completed Successfully!")

except Exception as e:
    st.error(f"❌ Error loading dataset: {e}")
    st.info("Make sure 'bigdata.csv' is present in the working directory.")

# Run with: streamlit run lab9.py
