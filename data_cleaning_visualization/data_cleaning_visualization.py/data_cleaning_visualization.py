import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================
# STEP 1: CREATE RAW DATASET
# =====================================

data = {
    "Employee_ID": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Age": [25, 28, np.nan, 35, 29, 42, 31, 27, 45, 25],
    "Salary": [30000, 35000, 40000, 50000, 35000, np.nan, 45000, 32000, 60000, 30000],
    "Experience": [2, 4, 5, 8, 4, 12, 6, 3, 15, 2],
    "Department": ["IT", "HR", "Sales", "IT", "HR", "Finance", "Sales", "IT", "Finance", "IT"],
    "Performance_Score": [78, 82, 85, 90, 82, 88, 86, 80, 92, 78]
}

df = pd.DataFrame(data)

print("=" * 50)
print("ORIGINAL DATASET")
print("=" * 50)
print(df)

# =====================================
# STEP 2: DATA PREPROCESSING
# =====================================

print("\nDataset Shape Before Cleaning:", df.shape)

print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())

# Fill Missing Values
df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Salary"] = df["Salary"].fillna(df["Salary"].mean())

# Remove Duplicates
before_rows = len(df)
df = df.drop_duplicates()
after_rows = len(df)

print("\nDuplicates Removed:", before_rows - after_rows)

print("\nDataset Shape After Cleaning:", df.shape)

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# =====================================
# STEP 3: DATA VISUALIZATION
# =====================================

# Graph 1 - Department Count

plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="Department")
plt.title("Department Wise Employee Count")
plt.xlabel("Department")
plt.ylabel("Number of Employees")
plt.show()

# Graph 2 - Salary Distribution

plt.figure(figsize=(7, 5))
plt.hist(df["Salary"], bins=5)
plt.title("Salary Distribution")
plt.xlabel("Salary")
plt.ylabel("Frequency")
plt.show()

# Graph 3 - Experience vs Salary

plt.figure(figsize=(7, 5))
sns.scatterplot(
    data=df,
    x="Experience",
    y="Salary"
)
plt.title("Experience vs Salary")
plt.show()

# Graph 4 - Correlation Heatmap

numeric_df = df.select_dtypes(include=np.number)

plt.figure(figsize=(8, 6))
sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm"
)
plt.title("Correlation Heatmap")
plt.show()

# Graph 5 - Salary Outlier Detection

plt.figure(figsize=(7, 5))
sns.boxplot(y=df["Salary"])
plt.title("Salary Outlier Detection")
plt.show()

# Graph 6 - Performance Score Analysis

plt.figure(figsize=(7, 5))
sns.barplot(
    data=df,
    x="Department",
    y="Performance_Score"
)
plt.title("Average Performance Score by Department")
plt.show()

# =====================================
# STEP 4: FINAL INSIGHTS
# =====================================

print("\n" + "=" * 50)
print("FINAL INSIGHTS")
print("=" * 50)

print("Average Age:", round(df["Age"].mean(), 2))
print("Average Salary:", round(df["Salary"].mean(), 2))
print("Average Experience:", round(df["Experience"].mean(), 2))
print("Average Performance Score:", round(df["Performance_Score"].mean(), 2))

print("\nData Cleaning Completed Successfully")
print("Data Visualization Completed Successfully")