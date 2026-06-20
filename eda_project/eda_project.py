import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================
# DATASET
# ==========================

data = {
    "Customer_Age":[18,22,25,30,35,40,28,23,32,27],
    "Order_Value":[250,400,150,600,550,700,300,200,500,350],
    "Delivery_Time":[20,30,15,45,40,50,25,18,35,28],
    "Customer_Rating":[5,4,5,3,4,2,5,5,4,4],
    "Orders_Per_Month":[12,10,15,6,8,4,11,14,7,10]
}

df = pd.DataFrame(data)

# ==========================
# BASIC INFO
# ==========================

print("DATASET")
print(df)

print("\nSTATISTICAL SUMMARY")
print(df.describe())

print("\nCORRELATION MATRIX")
print(df.corr())

# ==========================
# GRAPH 1
# Order Value Distribution
# ==========================

plt.figure(figsize=(7,5))
sns.histplot(df["Order_Value"], bins=5, kde=True)
plt.title("Order Value Distribution")
plt.xlabel("Order Value")
plt.ylabel("Frequency")
plt.show()

# ==========================
# GRAPH 2
# Delivery Time vs Rating
# ==========================

plt.figure(figsize=(7,5))
sns.scatterplot(
    data=df,
    x="Delivery_Time",
    y="Customer_Rating"
)

plt.title("Delivery Time vs Customer Rating")
plt.xlabel("Delivery Time")
plt.ylabel("Customer Rating")
plt.show()

# ==========================
# GRAPH 3
# Customer Age vs Order Value
# ==========================

plt.figure(figsize=(7,5))
sns.scatterplot(
    data=df,
    x="Customer_Age",
    y="Order_Value"
)

plt.title("Customer Age vs Order Value")
plt.xlabel("Customer Age")
plt.ylabel("Order Value")
plt.show()

# ==========================
# GRAPH 4
# Orders Per Month
# ==========================

plt.figure(figsize=(8,5))
plt.bar(
    df["Customer_Age"],
    df["Orders_Per_Month"]
)

plt.title("Orders Per Month by Customer Age")
plt.xlabel("Customer Age")
plt.ylabel("Orders Per Month")
plt.show()

# ==========================
# GRAPH 5
# Correlation Heatmap
# ==========================

plt.figure(figsize=(8,6))
sns.heatmap(
    df.corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")
plt.show()

# ==========================
# GRAPH 6
# Customer Rating Distribution
# ==========================

plt.figure(figsize=(6,6))
rating_count = df["Customer_Rating"].value_counts()

plt.pie(
    rating_count,
    labels=rating_count.index,
    autopct="%1.1f%%"
)

plt.title("Customer Rating Distribution")
plt.show()

# ==========================
# INSIGHTS
# ==========================

print("\nINSIGHTS")
print("- Average Order Value:", round(df["Order_Value"].mean(),2))
print("- Average Delivery Time:", round(df["Delivery_Time"].mean(),2))
print("- Average Customer Rating:", round(df["Customer_Rating"].mean(),2))

print("\nObservation:")
print("1. Faster delivery usually gets better ratings.")
print("2. Young customers place more frequent orders.")
print("3. High order values are common among adults.")
print("4. Delivery time has an impact on customer satisfaction.")