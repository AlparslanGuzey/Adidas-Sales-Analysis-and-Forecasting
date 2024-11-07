# Import necessary libraries
import pandas as pd
from pandasql import sqldf  # For SQL-like queries with pandas DataFrames
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error



# Function to use SQL syntax directly on DataFrames
pysqldf = lambda q: sqldf(q, globals())

# Load the dataset
file_path = '/Users/alparslanguzey/Desktop/archive/Adidas US Sales Datasets.xlsx'
df_adidas_sales = pd.read_excel(file_path, sheet_name='Data Sales Adidas', skiprows=3, usecols="B:N")

# Rename columns for clarity
df_adidas_sales.columns = [
    "Retailer", "Retailer ID", "Invoice Date", "Region", "State", "City",
    "Product", "Price per Unit", "Units Sold", "Total Sales", "Operating Profit",
    "Operating Margin", "Sales Method"
]

# Remove any rows where the header information is repeated within the data
df_adidas_sales = df_adidas_sales[df_adidas_sales["Retailer"] != "Retailer"]

# Convert 'Invoice Date' to datetime format
df_adidas_sales["Invoice Date"] = pd.to_datetime(df_adidas_sales["Invoice Date"], errors='coerce')

# Convert numeric columns to appropriate types for analysis
numeric_columns = ["Price per Unit", "Units Sold", "Total Sales", "Operating Profit", "Operating Margin"]
df_adidas_sales[numeric_columns] = df_adidas_sales[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Check the data structure to confirm setup
print(df_adidas_sales.info())
print(df_adidas_sales.head())

# Total revenue generated over the years
query_total_revenue = "SELECT SUM([Total Sales]) AS Total_Revenue FROM df_adidas_sales"
total_revenue = pysqldf(query_total_revenue)
print("Total Revenue:", total_revenue)

# Top 5 products over the years
query_top_products = """
SELECT Product, SUM([Total Sales]) AS Total_Revenue
FROM df_adidas_sales
GROUP BY Product
ORDER BY Total_Revenue DESC
LIMIT 5
"""
top_products = pysqldf(query_top_products)
print("Top 5 Products by Revenue:")
print(top_products)

# Average daily sales value by retailer
query_avg_daily_sales = """
SELECT Retailer, AVG([Total Sales]) AS Average_Daily_Sales
FROM df_adidas_sales
GROUP BY Retailer
"""
avg_daily_sales = pysqldf(query_avg_daily_sales)
print("Average Daily Sales by Retailer:")
print(avg_daily_sales)

# Top regions, states, and cities by sales
query_top_regions = """
SELECT Region, State, City, SUM([Total Sales]) AS Total_Revenue
FROM df_adidas_sales
GROUP BY Region, State, City
ORDER BY Total_Revenue DESC
LIMIT 10
"""
top_regions = pysqldf(query_top_regions)
print("Top Regions, States, and Cities by Revenue:")
print(top_regions)

# Profit margin by product
query_profit_margin = """
SELECT Product, AVG([Operating Margin]) AS Average_Profit_Margin
FROM df_adidas_sales
GROUP BY Product
ORDER BY Average_Profit_Margin DESC
"""
profit_margin_by_product = pysqldf(query_profit_margin)
print("Profit Margin by Product:")
print(profit_margin_by_product)

# Export cleaned data to CSV for Tableau
df_adidas_sales.to_csv('/Users/alparslanguzey/Desktop/archive/Adidas_Sales_Cleaned.csv', index=False)


# Calculate basic statistics
basic_stats = df_adidas_sales[['Price per Unit', 'Units Sold', 'Total Sales', 'Operating Profit', 'Operating Margin']].describe()
print("Basic Statistics:\n", basic_stats)

# Calculate covariance and correlation
cov_matrix = df_adidas_sales[['Price per Unit', 'Units Sold', 'Total Sales', 'Operating Profit', 'Operating Margin']].cov()
cor_matrix = df_adidas_sales[['Price per Unit', 'Units Sold', 'Total Sales', 'Operating Profit', 'Operating Margin']].corr()

print("\nCovariance Matrix:\n", cov_matrix)
print("\nCorrelation Matrix:\n", cor_matrix)


# Ensure 'Invoice Date' is set as datetime
df_adidas_sales['Invoice Date'] = pd.to_datetime(df_adidas_sales['Invoice Date'])

# Set 'Invoice Date' as index for time series and use 'Total Sales' column
df_sales_ts = df_adidas_sales.set_index('Invoice Date')['Total Sales'].resample('M').sum()

# Fit ARIMA model (simple model for testing)
arima_model = ARIMA(df_sales_ts, order=(1, 1, 1))
arima_result = arima_model.fit()

print("\nARIMA Model Summary:\n", arima_result.summary())


# Ensure 'Invoice Date' is set as datetime
df_adidas_sales['Invoice Date'] = pd.to_datetime(df_adidas_sales['Invoice Date'])

# Set 'Invoice Date' as index for time series
df_sales_ts = df_adidas_sales.set_index('Invoice Date')['Total Sales'].resample('M').sum()

# Fit ARIMA model (simple model for testing)
arima_model = ARIMA(df_sales_ts, order=(1, 1, 1))
arima_result = arima_model.fit()

print("\nARIMA Model Summary:\n", arima_result.summary())


# Convert 'Invoice Date' to datetime and set it as the index
df_adidas_sales["Invoice Date"] = pd.to_datetime(df_adidas_sales["Invoice Date"])
df_sales_ts = df_adidas_sales.set_index("Invoice Date")['Total Sales'].resample('M').sum()

# Plot the monthly sales to visually inspect for trends and seasonality
df_sales_ts.plot(title='Monthly Total Sales', figsize=(10, 6))
plt.show()

# Define the SARIMA model - initial parameters
# SARIMA(p, d, q)(P, D, Q, s)
# Here, we assume s=12 for annual seasonality; initial (p, d, q) and (P, D, Q) can be (1, 1, 1)
model = SARIMAX(df_sales_ts,
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 12),
                enforce_stationarity=False,
                enforce_invertibility=False)

# Fit the model
sarima_result = model.fit()

# Model Summary
print(sarima_result.summary())

# Forecasting for the next 12 months
forecast = sarima_result.get_forecast(steps=12)
forecast_index = pd.date_range(df_sales_ts.index[-1] + pd.offsets.MonthEnd(1), periods=12, freq='M')
forecast_series = forecast.predicted_mean
confidence_intervals = forecast.conf_int()

# Plotting the forecast
plt.figure(figsize=(10, 6))
plt.plot(df_sales_ts, label='Historical Sales')
plt.plot(forecast_index, forecast_series, label='Forecasted Sales', color='orange')
plt.fill_between(forecast_index,
                 confidence_intervals.iloc[:, 0],
                 confidence_intervals.iloc[:, 1],
                 color='orange', alpha=0.3)
plt.title('SARIMA Sales Forecast')
plt.legend()
plt.show()


# Prepare the data in Prophet's required format
df_prophet = df_sales_ts.reset_index()
df_prophet.columns = ['ds', 'y']

# Initialize the Prophet model
prophet_model = Prophet(yearly_seasonality=True)
prophet_model.fit(df_prophet)

# Create a DataFrame to hold future dates for forecasting
future_dates = prophet_model.make_future_dataframe(periods=12, freq='M')

# Generate forecast
forecast = prophet_model.predict(future_dates)

# Plot forecast
fig = prophet_model.plot(forecast)
plt.title('Prophet Sales Forecast')
plt.show()

# Plot seasonality components
fig = prophet_model.plot_components(forecast)
plt.show()



#Random Forrest
# Create features based on date
df_rf = df_sales_ts.reset_index()
df_rf['year'] = df_rf['Invoice Date'].dt.year
df_rf['month'] = df_rf['Invoice Date'].dt.month

# Lagged features
df_rf['lag1'] = df_rf['Total Sales'].shift(1)
df_rf['lag12'] = df_rf['Total Sales'].shift(12)

# Drop missing values from lags
df_rf = df_rf.dropna()

# Define features (X) and target (y)
X = df_rf[['year', 'month', 'lag1', 'lag12']]
y = df_rf['Total Sales']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Initialize and fit the Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=0)
rf_model.fit(X_train, y_train)

# Predict and evaluate
y_pred = rf_model.predict(X_test)
print("Random Forest MAE:", mean_absolute_error(y_test, y_pred))

# Simulate forecast for the next 12 months
future_predictions = []
last_known = df_rf.iloc[-1].copy()

for i in range(12):
    # Prepare the next row based on last known data
    new_row = {
        'year': last_known['year'] + (last_known['month'] == 12),
        'month': 1 if last_known['month'] == 12 else last_known['month'] + 1,
        'lag1': last_known['Total Sales'],
        'lag12': df_sales_ts[-12 + i] if i < 12 else future_predictions[i - 12]
    }

    # Predict
    pred = rf_model.predict(pd.DataFrame([new_row]))
    future_predictions.append(pred[0])

    # Update last_known for the next iteration
    last_known['Total Sales'] = pred[0]
    last_known['month'] = new_row['month']
    last_known['year'] = new_row['year']

# Plot Random Forest Forecast
plt.figure(figsize=(10, 6))
plt.plot(df_sales_ts, label='Historical Sales')
plt.plot(pd.date_range(df_sales_ts.index[-1] + pd.offsets.MonthEnd(1), periods=12, freq='ME'),
         future_predictions, label='Forecasted Sales (RF)', color='green')
plt.title('Random Forest Sales Forecast')
plt.legend()
plt.show()

# Histogram for Total Sales
plt.figure(figsize=(10, 6))
sns.histplot(df_adidas_sales['Total Sales'], kde=True)
plt.title('Distribution of Total Sales')
plt.xlabel('Total Sales')
plt.ylabel('Frequency')
plt.show()

# Boxplot for Sales across different products
plt.figure(figsize=(10, 5))
sns.boxplot(x='Product', y='Total Sales', data=df_adidas_sales)
plt.title('Total Sales Distribution by Product')
plt.xticks(rotation=40)
plt.show()

# Scatter plot to observe correlation between Units Sold and Total Sales
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Units Sold', y='Total Sales', data=df_adidas_sales)
plt.title('Units Sold vs Total Sales')
plt.xlabel('Units Sold')
plt.ylabel('Total Sales')
plt.show()