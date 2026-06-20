import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np

# ==================================
# STEP 1: CREATE DATASET
# ==================================

data = {
    "Month": ["Jan", "Feb", "Mar", "Apr", "May",
              "Jun", "Jul", "Aug", "Sep", "Oct"],
    "Sales": [15000, 18000, 17000, 22000, 25000,
              27000, 26000, 30000, 32000, 35000],
    "Customers": [120, 140, 135, 180, 200,
                  220, 210, 250, 270, 300],
    "Orders": [100, 120, 115, 150, 170,
               190, 185, 220, 240, 260]
}

df = pd.DataFrame(data)

print("=" * 50)
print("RETAIL SALES DATASET")
print("=" * 50)
print(df)

# ==================================
# STEP 2: STATISTICAL SUMMARY
# ==================================

print("\nSTATISTICAL SUMMARY")
print(df.describe())

# ==================================
# STEP 3: MONTHLY SALES TREND
# ==================================

plt.figure(figsize=(8,5))
plt.plot(df["Month"], df["Sales"], marker="o")
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.grid(True)
plt.show()

# ==================================
# STEP 4: CUSTOMERS VS SALES
# ==================================

plt.figure(figsize=(7,5))
sns.scatterplot(
    data=df,
    x="Customers",
    y="Sales"
)
plt.title("Customers vs Sales")
plt.show()

# ==================================
# STEP 5: ORDERS PER MONTH
# ==================================

plt.figure(figsize=(8,5))
plt.bar(df["Month"], df["Orders"])
plt.title("Orders Per Month")
plt.xlabel("Month")
plt.ylabel("Orders")
plt.show()

# ==================================
# STEP 6: SALES DISTRIBUTION
# ==================================

plt.figure(figsize=(7,5))
sns.histplot(df["Sales"], bins=5, kde=True)
plt.title("Sales Distribution")
plt.show()

# ==================================
# STEP 7: CORRELATION HEATMAP
# ==================================

numeric_df = df[["Sales", "Customers", "Orders"]]

plt.figure(figsize=(6,5))
sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm"
)
plt.title("Correlation Heatmap")
plt.show()

# ==================================
# STEP 8: SALES PREDICTION
# ==================================

df["Month_Number"] = np.arange(1, len(df)+1)

X = df[["Month_Number"]]
y = df["Sales"]

model = LinearRegression()
model.fit(X, y)

future_months = np.array([[11], [12], [13]])
predicted_sales = model.predict(future_months)

print("\nPREDICTED SALES")
print("Month 11:", round(predicted_sales[0]))
print("Month 12:", round(predicted_sales[1]))
print("Month 13:", round(predicted_sales[2]))

# ==================================
# STEP 9: PREDICTION GRAPH
# ==================================

future_df = pd.DataFrame({
    "Month_Number": [11, 12, 13],
    "Sales": predicted_sales
})

plt.figure(figsize=(8,5))

plt.plot(
    df["Month_Number"],
    df["Sales"],
    marker="o",
    label="Actual Sales"
)

plt.plot(
    future_df["Month_Number"],
    future_df["Sales"],
    marker="o",
    linestyle="--",
    label="Predicted Sales"
)

plt.title("Sales Prediction")
plt.xlabel("Month Number")
plt.ylabel("Sales")
plt.legend()
plt.grid(True)
plt.show()

# ==================================
# STEP 10: BUSINESS INSIGHTS
# ==================================

print("\nBUSINESS INSIGHTS")
print("- Average Monthly Sales:", round(df["Sales"].mean(), 2))
print("- Average Customers:", round(df["Customers"].mean(), 2))
print("- Average Orders:", round(df["Orders"].mean(), 2))

print("\nOBSERVATIONS")
print("1. Sales show continuous growth over months.")
print("2. More customers lead to higher sales.")
print("3. Orders and sales have strong positive correlation.")
print("4. Future sales are expected to increase.")