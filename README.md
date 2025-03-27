# Big Data EDA with PySpark and Streamlit

## Overview
This project performs Exploratory Data Analysis (EDA) on an employee dataset using PySpark and visualizes the results with Streamlit. It includes:
- Data loading and transformation using PySpark RDDs.
- Handling missing values.
- Statistical summary and correlation analysis.
- Interactive data visualizations (heatmaps, distributions, and boxplots).
- Streamlit UI enhancements with balloons and snowflakes for an engaging experience.

## Features
✅ Load and clean big data using PySpark RDDs.
✅ Handle missing values efficiently.
✅ Perform statistical analysis on employee data.
✅ Generate interactive visualizations (salary distribution, correlation heatmaps, etc.).
✅ Streamlit UI with balloons and snow effects for interactivity.

## Installation
Ensure you have Python 3 installed, then run the following:
```sh
pip install -r requirements.txt
```

## Running the Application
To start the Streamlit application, run:
```sh
streamlit run pyspark_eda_streamlit.py
```

## Dataset Format
The dataset `bigdata.csv` should have the following columns:
| ID  | Name  | Age  | Salary  | Experience | Department |
|-----|-------|------|---------|-----------|------------|
| 101 | John  | 25   | 50000.0 | 3         | IT         |
| 102 | Alice | 30   | 60000.0 | 5         | HR         |

## Author
Lab Program 9 - Done by Bavirisetty Sairam (2447115).

