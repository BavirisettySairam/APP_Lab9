import streamlit as st
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, mean, count, when
from pyspark.mllib.stat import Statistics
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Initialize Spark session
spark = SparkSession.builder.appName("BigData_EDA").getOrCreate()

# Load dataset
rdd = spark.sparkContext.textFile("bigdata.csv")

# Remove header
header = rdd.first()
rdd = rdd.filter(lambda row: row != header)

def parse_row(row):
    parts = row.split(",")
    try:
        return (int(parts[0]), parts[1], int(parts[2]) if parts[2] else None, 
                float(parts[3]) if parts[3] else None, int(parts[4]), parts[5])
    except:
        return None

rdd = rdd.map(parse_row).filter(lambda x: x is not None)
columns = ["ID", "Name", "Age", "Salary", "Experience", "Department"]
df = spark.createDataFrame(rdd, columns)

# Handling missing values
age_mean = df.select(mean(col("Age"))).collect()[0][0]
salary_mean = df.select(mean(col("Salary"))).collect()[0][0]
df = df.fillna({"Age": age_mean, "Salary": salary_mean})

# Convert to Pandas for Streamlit visualization
pandas_df = df.toPandas()

# Streamlit App
st.title("Big Data EDA - Employee Dataset 🎉")
st.write("Lab Program 9 - Done by Bavirisetty Sairam (2447115)")

# Interactive Effects
if st.button("Celebrate with Balloons! 🎈"):
    st.balloons()
if st.button("Let it Snow! ❄️"):
    st.snow()

st.subheader("Dataset Overview")
st.write(pandas_df.head())

st.subheader("Missing Values")
missing_data = pandas_df.isnull().sum()
st.write(missing_data)

st.subheader("Summary Statistics")
st.write(pandas_df.describe())

# Correlation Heatmap
st.subheader("Correlation Heatmap")
plt.figure(figsize=(6, 4))
sns.heatmap(pandas_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
st.pyplot(plt)

# Salary Distribution
st.subheader("Salary Distribution")
plt.figure(figsize=(6, 4))
sns.histplot(pandas_df['Salary'], kde=True, bins=20)
st.pyplot(plt)

# Department-wise Salary Analysis
st.subheader("Department-wise Salary")
plt.figure(figsize=(6, 4))
sns.boxplot(x='Department', y='Salary', data=pandas_df)
st.pyplot(plt)

# Stop Spark
spark.stop()