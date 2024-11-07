# Adidas Sales Analysis and Forecasting

## Project Overview

This project analyzes Adidas sales data to gain insights into sales trends, customer preferences, geographical performance, and product profitability. Through statistical analysis and forecasting, the project aims to predict future sales and help identify high-performing products, key markets, and optimize inventory and marketing efforts. The primary tools used include Python for data processing and statistical analysis and models like SARIMA, Prophet, and Random Forest for sales forecasting.

## Data Summary and Initial Insights

The dataset consists of 9,648 entries with information on retailer, region, product category, units sold, total sales, and profit metrics.

### Key Columns

- **Retailer**: The name of the retailer selling Adidas products.
- **Invoice Date**: The date of the transaction.
- **Product**: Product category (e.g., Men's Street Footwear, Women's Apparel).
- **Units Sold**: Quantity of units sold.
- **Total Sales**: Total sales amount.
- **Operating Profit** and **Operating Margin**: Profit metrics for performance analysis.

### Initial Insights

- **Total Revenue**: $899 million
- **Top 5 Products by Revenue:**
  - Men's Street Footwear: $208 million
  - Women's Apparel: $179 million
  - Men's Athletic Footwear: $153 million
  - Women's Street Footwear: $128 million
  - Men's Apparel: $123 million
- **Average Daily Sales by Retailer:**
  - Walmart: $119,102
  - Amazon: $81,874
  - Foot Locker: $83,464

## Statistical Analysis

### 1. Basic Statistics

Key statistics were computed for the main numeric columns (Price per Unit, Units Sold, Total Sales, Operating Profit, and Operating Margin):

| Metric            |     Min     |     Mean     |     Max     | Std. Dev   |
|-------------------|-------------|--------------|-------------|------------|
| Price per Unit    | $7.00       | $45.22       | $110.00     | $14.71     |
| Units Sold        | 0           | 256.93       | 1,275       | 214.25     |
| Total Sales       | $0.00       | $45,216.63   | $390,000    | $54,193.11 |
| Operating Profit  | $0.00       | $34,425.24   | $390,000    | $54,193.11 |
| Operating Margin  | 0.1         | 0.42         | 0.8         | 0.10       |

### 2. Covariance and Correlation

- **Positive Correlation**: Total Sales and Units Sold had a moderate positive correlation (0.44), indicating higher unit sales generally correlate with higher revenue.
- **Negative Correlation**: Operating Margin was negatively correlated with Units Sold and Total Sales, likely due to discounts on high-volume sales affecting overall margins.

### 3. Geographical Analysis

Top regions, states, and cities were identified for revenue:

- **Top Regions**: Northeast, West, and Southeast.
- **Top Cities**: New York ($39.8 million), San Francisco ($34.5 million), and Miami ($31.6 million).

## Data Visualization

### 1. Distribution of Total Sales

The histogram displays the distribution of Total Sales, with a skew towards lower sales values.

![Total Sales Distribution by Product](https://github.com/user-attachments/assets/e98bacda-8819-43ae-9f79-4d6860b92cb5)


### 2. Top Regions by Revenue

Top regions by revenue with New York, San Francisco, and Miami leading in sales.

### 3. Profit Margins by Product

Men’s Street Footwear had the highest profit margin, followed by Women’s Apparel.

## Forecasting Models

### 1. SARIMA Model

The SARIMA model (Seasonal ARIMA) was implemented to capture seasonality in monthly sales.

**Summary:**

- **Order**: (1, 1, 1) with a seasonal order of (1, 1, 1, 12)
- **AIC**: 823.92

**Insights:**

- The model displayed instability warnings due to a high condition number, suggesting possible overfitting or data limitations.
- A high variance of approximately 6.97×10<sup>13</sup> suggests significant noise, which may have impacted model accuracy.

*SARIMA model forecast for Adidas sales.*

### 2. Prophet Model

Prophet, developed by Facebook, was used due to its ability to handle seasonality and trend components with less tuning.

**Insights:**

- The model automatically detected seasonal trends and produced a forecast based on these patterns.
- **Deprecation Note**: The 'M' frequency should be updated to 'ME' in future versions.

*Prophet forecast for Adidas sales.*

### 3. Random Forest Model

A Random Forest model was applied as a non-linear regression approach. However, it encountered challenges with time-dependent data:

- **MAE (Mean Absolute Error)**: ~9 million, indicating high prediction error likely due to lack of time-series dependencies in Random Forest.
- **Feature Engineering**: Additional temporal features (e.g., lagged sales) were used but were insufficient for high accuracy.

*Random Forest model forecast for Adidas sales.*

## Conclusion and Recommendations

### Key Findings

- **Top Performers**: Men’s Street Footwear and Women’s Apparel lead both in sales volume and profit margin.
- **Geographical Insights**: The Northeast and West regions, particularly New York and San Francisco, are highly profitable markets.

### Model Performance

- **SARIMA and Prophet**: Captured seasonality, though SARIMA showed instability due to data limitations.
- **Random Forest**: Encountered challenges in predicting time-series data due to lack of temporal dependencies.

### Future Work

- **Additional Data**: Collecting more years of data will enhance model accuracy and seasonality analysis.
- **Advanced Models**: LSTM or GRU neural networks could better capture temporal patterns in sales data.
- **Feature Engineering**: Adding external data (e.g., economic indicators) may improve forecasting performance.

## How to Use the Repository

Clone the repository, install required packages, and run the analysis:

```bash
git clone https://github.com/username/adidas-sales-analysis.git
cd adidas-sales-analysis
python3 Adidas_Sales.py

Ensure necessary libraries are installed:
pip install pandas numpy statsmodels prophet matplotlib seaborn

Acknowledgments

This project uses a dataset from Kaggle and relies on Python libraries such as pandas, statsmodels, prophet, and sklearn for analysis. Visualizations were created with matplotlib and seaborn.
