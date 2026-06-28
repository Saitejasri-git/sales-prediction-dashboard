import pandas as pd

df = pd.read_csv("data/orders.csv")

print("First 5 Rows:")
print(df.head())

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

print("\nMissing Values:")
print(df.isnull().sum())

print(df["Customer Status"].value_counts())

#data cleaning 

df["Customer Status"]=df["Customer Status"].str.upper()
print(df["Customer Status"].value_counts())

# KPIs
total_revenue = df["Total Retail Price for This Order"].sum()
total_orders = df["Order ID"].nunique()
total_customers = df["Customer ID"].nunique()

print("\nTotal Revenue:", total_revenue)
print("Total Orders:", total_orders)
print("Total Customers:", total_customers)

#plotting for customer status
import matplotlib.pyplot as plt

status_counts = df["Customer Status"].value_counts()

plt.figure(figsize=(6,4))
status_counts.plot(kind="bar")

plt.title("Customer Status Distribution")
plt.xlabel("Customer Status")
plt.ylabel("Count")

plt.show()

#collecting revenue based on customer status
revenue_by_status = df.groupby("Customer Status")["Total Retail Price for This Order"].sum()

print("\nRevenue By Status:")
print(revenue_by_status.round(2))

# Convert date column to datetime

df["Date Order was placed"] = pd.to_datetime(
    df["Date Order was placed"]
)

# Revenue by date

revenue_by_date = df.groupby("Date Order was placed")[
    "Total Retail Price for This Order"
].sum()

print(revenue_by_date.head())

plt.figure(figsize=(12,5))

revenue_by_date.plot()

plt.title("Revenue Over Time")
plt.xlabel("Date")
plt.ylabel("Revenue")

plt.show()

#converting data from daily revenue to monthly revenue

df["Month"]=df["Date Order was placed"].dt.to_period("M")
monthly_revenue = df.groupby("Month")[ "Total Retail Price for This Order"].sum()
print(monthly_revenue.head())

plt.figure(figsize=(12,5))
monthly_revenue.plot()
plt.title("Monthly Revenue Trend")
plt.xlabel("Month")
plt.ylabel("Revenue")

plt.show()

#Top 10 best-selling products
top_products = df.groupby("Product ID")["Quantity Ordered"].sum()
top_products = top_products.sort_values(ascending=False)
print(top_products.head(10))

top_products.head(10).plot(kind="bar", figsize=(10,5))

plt.title("Top 10 Best-Selling Products")
plt.xlabel("Product ID")
plt.ylabel("Quantity Ordered")

plt.show()

#Top best selling products

customer_spending = df.groupby("Customer ID")["Total Retail Price for This Order"].sum()
customer_spending = customer_spending.sort_values(ascending=False)
print(customer_spending.head(10))

customer_spending.head(10).plot( kind="bar",figsize=(10,5))

plt.title("Top 10 Customers by Spending")
plt.xlabel("Customer ID")
plt.ylabel("Total Spending")

plt.show()

#profit calculation 

df["Total Cost"] = (df["Cost Price Per Unit"] * df["Quantity Ordered"])
df["Profit"] = (df["Total Retail Price for This Order"]- df["Total Cost"])
total_profit = df["Profit"].sum()
print("Total profit: ", total_profit)
