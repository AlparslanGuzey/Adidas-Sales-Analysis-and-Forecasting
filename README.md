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

Description: A box plot showing the distribution of total sales across different product categories.
Insights:Categories such as "Footwear" and "Apparel" show varying distributions, with "Footwear" generally having a higher sales range. Outliers in each category could indicate products that performed exceptionally well or poorly.
This distribution analysis helps in identifying top-performing products and those that may need reevaluation.
The histogram displays the distribution of Total Sales, with a skew towards lower sales values.

![Total Sales Distribution by Product](https://github.com/user-attachments/assets/e98bacda-8819-43ae-9f79-4d6860b92cb5)


### 2. Monthly Total Sales
Description: This line chart shows the trend in total sales on a monthly basis.

Insights:
- The chart reveals a pattern of sales fluctuating over the period, with a notable dip in early 2021, followed by a sharp rise.
- Peaks in sales may indicate seasonal demand or successful sales campaigns.
- This trend helps identify periods of high and low sales, which is essential for inventory and supply chain management.
![Monthly Total Sales](https://github.com/user-attachments/assets/44a1fcc8-0283-4af2-8c58-d58b56e9ea76)

### 3. Units Sold vs Total Sales
Description: A scatter plot illustrating the relationship between units sold and total sales.

Insights:
- There is a positive correlation between units sold and total sales, which is expected since more units sold usually lead to higher revenue.
- This plot helps verify that sales increase in proportion to the number of units sold.
- Clusters or patterns in this chart can reveal different tiers or segments in pricing, product types, or other factors affecting sales volume.
![Units Sold vs Total Sales](https://github.com/user-attachments/assets/2d3aeec2-f533-4768-91c6-0a24fe1404e6)


### 4. Prophet Model Decomposition (Trend and Yearly Seasonality)

Description: Prophet’s decomposition of the time series, showing the overall trend and yearly seasonality.

Insights:
- The trend component shows a steady increase in sales over time, indicating growth.
- The yearly seasonality plot suggests regular fluctuations, which could be tied to annual trends such as holiday seasons or specific shopping events.
- The seasonal insight is valuable for planning marketing and inventory for high-sales periods.
![Trend ](https://github.com/user-attachments/assets/8fdff7ad-7880-4e0b-b642-369b9c83e115)


## Forecasting Models

### 1. SARIMA Model

The SARIMA model (Seasonal ARIMA) was implemented to capture seasonality in monthly sales. The SARIMA model forecast, showing historical sales and forecasted sales with confidence intervals.

**Summary:**

- **Order**: (1, 1, 1) with a seasonal order of (1, 1, 1, 12)
- **AIC**: 823.92

**Insights:**

- The model displayed instability warnings due to a high condition number, suggesting possible overfitting or data limitations.
- A high variance of approximately 6.97×10<sup>13</sup> suggests significant noise, which may have impacted model accuracy.
- The SARIMA model captures seasonality and provides a detailed confidence interval. However, the wide intervals indicate a degree of uncertainty in predictions, especially beyond the immediate future.
- SARIMA performs better than random forest for time series with clear seasonality but may still have limitations due to the data's short history.
![SARIMA Sales Forecast](https://github.com/user-attachments/assets/11a651fc-9c71-40d0-abb7-d280c02e0663)



### 2. Prophet Model

Prophet, developed by Facebook, was used due to its ability to handle seasonality and trend components with less tuning. Prophet’s forecast model, which includes historical data and a forecast for future sales with confidence intervals.

**Insights:**

- The model automatically detected seasonal trends and produced a forecast based on these patterns.
- The forecast (blue line) projects a steady increase in total sales over time, suggesting a positive growth trend.
- The shaded area represents the confidence interval, showing where the actual values are likely to fall.
- The broader confidence interval toward the forecast's end suggests higher uncertainty in long-term predictions.
- Prophet models seasonality and trend well, making it suitable for business forecasting, especially where seasonality patterns are present.
- **Deprecation Note**: The 'M' frequency should be updated to 'ME' in future versions.

![Prophet Sales Forecast](https://github.com/user-attachments/assets/ba3ef3e8-265b-4c5f-8de9-dcdfb7f9d093)


### 3. Random Forest Model

A Random Forest model was applied as a non-linear regression approach. A random forest-based sales forecast, visualized alongside historical data.However, it encountered challenges with time-dependent data:

- **MAE (Mean Absolute Error)**: ~9 million, indicating high prediction error likely due to lack of time-series dependencies in Random Forest.
- **Feature Engineering**: Additional temporal features (e.g., lagged sales) were used but were insufficient for high accuracy.
![Random Forest Sales Forecast](https://github.com/user-attachments/assets/45f5de53-b073-4c92-98f8-08854e2e4e95)



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
