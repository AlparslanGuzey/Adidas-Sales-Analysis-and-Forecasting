# Adidas Sales Analysis and Forecasting

This project involves the analysis and forecasting of Adidas sales data using various statistical and machine learning methods. The visualizations were created using Tableau to provide a clear, interactive view of key metrics, sales trends, and insights.

## Project Overview

- **Data Size**: 9,648 records
- **Metrics Analyzed**:
  - **Total Sales**: $899,122,125
  - **Units Sold**: 2,478,861
  - **Operating Profit**: $332,134,761
  - **Average Daily Sales**: $1,242,959

## Visualizations

### 1. Dashboard Overview

The main dashboard provides a snapshot of the sales performance across different metrics. It includes visuals on operating margin, sales over time, sales by retailer, product popularity, and sales distribution. Each section of the dashboard provides essential insights into the sales dynamics and profitability.
<img width="1506" alt="Dashboard" src="https://github.com/user-attachments/assets/cbcb982a-dc38-4df2-973d-c6233fe50452">


### 2. Operating Margin by Retailer

This bar chart visualizes the Operating Margin for each retailer. Key observations:

- **Foot Locker** and **West Gear** have the highest operating margins, indicating higher profitability.
- **Walmart** has a comparatively lower operating margin, suggesting lower profitability per unit sold.
<img width="1095" alt="Operating Margin by Retailer" src="https://github.com/user-attachments/assets/8213f041-f9f0-49ec-9f22-915e6d2a9a20">


### 3. Sales Over Time

The time series bar chart shows Weekly Total Sales trends. Notable points:

- Sales volumes fluctuate significantly, with some weeks showing spikes, potentially due to seasonal sales or promotions.
- A significant sales increase is observed during certain months, which may align with seasonal demand.
<img width="1084" alt="Sales Over Time" src="https://github.com/user-attachments/assets/d35498c0-3892-4187-9fa3-c531af2d3d7e">


### 4. Sales by Retailer

This treemap represents Total Sales by Retailer. Key insights:

- **West Gear** generates the highest sales volume, followed by **Sports Direct** and **Kohl's**.
- This visualization helps understand which retailers contribute the most to total revenue.
<img width="1105" alt="Sales by Retailer" src="https://github.com/user-attachments/assets/41c7f78a-4ead-4f08-9632-3528d51db836">


### 5. Product Popularity

The bubble chart displays Popular Products based on total sales. Insights include:

- **Men's Street Footwear** and **Women's Street Footwear** are the most popular products, indicating high demand.
- Apparel categories, such as **Men’s Apparel** and **Women's Apparel**, also show strong sales figures, highlighting significant market interest.
<img width="1108" alt="Product Popularity" src="https://github.com/user-attachments/assets/42233c6c-5fab-42b2-833d-1bb595856be3">


### 6. Sales Distribution

This geographic heatmap illustrates Sales Distribution across the United States. Key takeaways:

- States like **California**, **New York**, and **Texas** have higher sales volumes, indicating these regions as prime markets.
- The map provides insight into geographic sales concentration, which could guide regional marketing and inventory decisions.
<img width="1102" alt="Sales Distribution" src="https://github.com/user-attachments/assets/9665a9e6-88ed-4232-99d0-a66556f62b9a">


## Methods Used for Forecasting

The following methods were used for sales forecasting:

- **SARIMA**: Seasonal ARIMA model to account for trends and seasonality in sales data.
- **Prophet**: Facebook's Prophet model for flexible forecasting with holidays and seasonality effects.
- **Random Forest**: Ensemble learning method for more complex, non-linear patterns in sales data.

## Forecast Results

Forecasting models were applied to predict future sales patterns. Detailed analysis of each method is documented in the code.

## Installation and Usage

To reproduce the analysis and forecasting models:

1. **Clone this repository**:

   ```bash
   git clone https://github.com/username/adidas-sales-analysis.git
   cd adidas-sales-analysis

2. **Install required packages**:
   pip install pandas numpy statsmodels fbprophet scikit-learn matplotlib
   
4. **Run the analysis script**
   python Adidas_Sales_Analysis.py
   
5. View Tableau Dashboard by opening the .twbx file in Tableau Desktop.

## Insights and Recommendations:

- Retailer Profitability: Focus on partnerships with high-margin retailers like Foot Locker and West Gear for higher profitability.
- Seasonal Promotions: Capitalize on peak sales periods by aligning promotions with observed sales trends.
- Regional Marketing: Increase marketing efforts in high-sales regions (California, New York, Texas) to maximize revenue.
