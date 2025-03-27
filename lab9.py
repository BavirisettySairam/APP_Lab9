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

    # Show dataset summary
    st.subheader("📌 Data Summary")
    st.write(pandas_df.describe())

    # Display column data types
    st.subheader("🔢 Column Data Types")
    st.write(pandas_df.dtypes)

    # Dataset Shape
    st.subheader("📏 Dataset Shape")
    st.write(f"Rows: {pandas_df.shape[0]}, Columns: {pandas_df.shape[1]}")

    # Check for missing values
    st.subheader("⚠️ Missing Values")
    st.write(pandas_df.isnull().sum())

    # Correlation Heatmap
    st.subheader("📈 Correlation Matrix")
    numeric_df = pandas_df.select_dtypes(include=['number'])

    if numeric_df.shape[1] > 0:
        selected_columns = st.multiselect("Select columns for correlation matrix", numeric_df.columns, default=numeric_df.columns.tolist())

        if selected_columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(numeric_df[selected_columns].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
            st.pyplot(fig)
        else:
            st.warning("⚠️ Please select at least one numeric column.")
    else:
        st.warning("❌ No numeric columns found for correlation analysis.")

    # Distribution Plots for Numeric Features
    st.subheader("📊 Feature Distributions")
    num_cols = st.multiselect("Select numeric columns for distribution plots", numeric_df.columns, default=numeric_df.columns.tolist())

    if num_cols:
        for col in num_cols:
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.histplot(pandas_df[col], kde=True, bins=30, ax=ax)
            ax.set_title(f"Distribution of {col}")
            st.pyplot(fig)

    # Count Plots for Categorical Features
    st.subheader("🛠 Categorical Feature Analysis")
    cat_cols = pandas_df.select_dtypes(exclude=['number']).columns.tolist()

    if cat_cols:
        selected_cat_cols = st.multiselect("Select categorical columns for count plots", cat_cols, default=cat_cols[:1])

        for col in selected_cat_cols:
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.countplot(data=pandas_df, x=col, order=pandas_df[col].value_counts().index, ax=ax)
            ax.set_title(f"Count Plot of {col}")
            plt.xticks(rotation=45)
            st.pyplot(fig)

    # Pairplot for Relationships
    st.subheader("🔍 Pairplot Analysis (Sampled Data)")
    sample_df = pandas_df.sample(frac=0.1, random_state=42) if len(pandas_df) > 5000 else pandas_df
    pairplot_cols = st.multiselect("Select columns for pairplot (max 4)", numeric_df.columns, default=numeric_df.columns[:2])

    if len(pairplot_cols) > 1:
        fig = sns.pairplot(sample_df[pairplot_cols])
        st.pyplot(fig)

    st.success("✅ EDA Completed Successfully!")

except Exception as e:
    st.error(f"❌ Error loading dataset: {e}")
    st.info("Make sure 'bigdata.csv' is present in the working directory.")

# Run with: streamlit run lab9.py
