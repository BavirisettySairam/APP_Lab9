import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession

# Initialize Spark session
spark = SparkSession.builder.appName("BigData_EDA").getOrCreate()

# Streamlit App UI Configuration
st.set_page_config(page_title="Big Data EDA", layout="wide")

# App Title
st.title("📊 Big Data EDA with PySpark & Streamlit")

# Hardcoded file path
file_path = "bigdata.csv"

try:
    # Load CSV into PySpark DataFrame
    df = spark.read.csv(file_path, header=True, inferSchema=True)
    
    # Convert PySpark DataFrame to Pandas for visualization
    pandas_df = df.toPandas()

    # Display dataset overview
    st.subheader("🔍 Dataset Preview")
    st.write(pandas_df.head())

    # Show dataset information
    st.subheader("📌 Data Summary")
    st.write(pandas_df.describe())

    # Column Selection for Correlation Heatmap
    st.subheader("📈 Correlation Matrix")

    # Select only numeric columns
    numeric_df = pandas_df.select_dtypes(include=['number'])

    if numeric_df.shape[1] > 0:
        selected_columns = st.multiselect("Select columns for correlation matrix", numeric_df.columns, default=numeric_df.columns.tolist())

        if selected_columns:
            # Interactive Heatmap
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(numeric_df[selected_columns].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
            st.pyplot(fig)
        else:
            st.warning("⚠️ Please select at least one numeric column.")

    else:
        st.warning("❌ No numeric columns found for correlation analysis.")

    # Display column data types
    st.subheader("🔢 Column Data Types")
    st.write(pandas_df.dtypes)

    # Missing Values Check
    st.subheader("⚠️ Missing Values")
    st.write(pandas_df.isnull().sum())

    # Dataset Shape
    st.subheader("📏 Dataset Shape")
    st.write(f"Rows: {pandas_df.shape[0]}, Columns: {pandas_df.shape[1]}")

    st.success("✅ EDA Completed Successfully!")

except Exception as e:
    st.error(f"❌ Error loading dataset: {e}")
    st.info("Make sure 'bigdata.csv' is present in the working directory.")

# Run with: streamlit run lab9.py
