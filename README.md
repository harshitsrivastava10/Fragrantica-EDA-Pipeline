# Fragrantica Data Engineering & Exploratory Analysis

## Project Objective
This project establishes a standard data-cleaning pipeline and performs Exploratory Data Analysis (EDA) using community-sourced data from Fragrantica. The workflow processes structural issues, treats missing data points, limits the variance of extreme outliers, and visualizes underlying patterns within public perception data.

## Dataset Profile
* **Dataset Name:** Fragrantica.com Fragrance Dataset
* **Observations:** 500+ records
* **Core Dimensions:** Categorical attributes (Brand, Gender Marketing, Scent Profiles across Top/Middle/Base layers) and continuous performance metrics (Community Rating, Total User Votes).

## Engineering & Insights Pipeline
1. **Imputation Strategy:** Numerical gaps in community metrics were resolved using median distribution values. Text elements representing missing scent notes were normalized to a generic 'Unknown' string field to retain structural integrity.
2. **Outlier Boundaries:** Outliers within user review counts (`Votes`) were effectively capped at the $Q3 + 1.5 \times IQR$ threshold to avoid massive distortion from viral blockbusters during predictive scaling phases.
3. **Key Finding:** The correlation layout suggests a distinctive curve relating user review volume and average ratings, showing a stabilization pattern among high-volume releases.

## Local Configuration
Execute the analysis steps using the command below:
```bash
pip install -r requirements.txt
jupyter notebook data_cleaning_eda.ipynb