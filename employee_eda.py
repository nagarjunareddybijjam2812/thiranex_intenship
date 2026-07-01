import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
print("EDA Started")
df = pd.read_csv(
    "employee_turnover.csv"
)
print(df.head())
df.shape
df.info()
df.describe()
df.isnull().sum()
plt.figure(figsize=(10,5))
sns.heatmap(
    df.isnull(),
    cbar=False
)
plt.title(
"Missing Values Analysis"
)
plt.show()
df.Employee_Turnover.value_counts()
sns.countplot(
x="Employee_Turnover",
data=df
)
plt.title(
"Employee Turnover Distribution"
)
plt.show()
numeric_columns = [

"Age",
"Monthly_Income",
"Years_At_Company",
"Distance_From_Home",
"Job_Satisfaction"
]
sns.histplot(
df["Age"],
kde=True
)
plt.title(
"Age Distribution"
)
plt.show()
sns.histplot(
df["Monthly_Income"],
kde=True
)
plt.title(
"Income Distribution"
)
plt.show()
sns.boxplot(
x="Employee_Turnover",
y="Job_Satisfaction",
data=df
)
plt.title(
"Satisfaction vs Turnover"
)
plt.show()
sns.boxplot(
x="Employee_Turnover",
y="Years_At_Company",
data=df
)
plt.title(
"Experience vs Turnover"
)
plt.show()
department = (
df.groupby(
"Department"
)
.Employee_Turnover
.mean()
)
department.plot(
kind="bar",
figsize=(8,5)
)
plt.title(
"Turnover Rate by Department"
)
plt.ylabel(
"Turnover Percentage"
)
plt.show()
role = (
df.groupby(
"Employee_Role"
)
.Employee_Turnover
.mean()
)
role.plot(
kind="bar"
)
plt.title(
"Turnover by Role"
)
plt.show()
corr = df.corr(
numeric_only=True
)
plt.figure(figsize=(12,8))
sns.heatmap(
corr,
annot=True
)
plt.title(
"Feature Correlation Matrix"
)
plt.show()
turnover_corr = (
corr[
"Employee_Turnover"
]
.sort_values(
ascending=False
)
)
print(turnover_corr)
sns.pairplot(
df[[
"Age",
"Monthly_Income",
"Years_At_Company",
"Employee_Turnover"
]]
)
plt.show()
df.describe().to_csv(
"eda_summary.csv"
)