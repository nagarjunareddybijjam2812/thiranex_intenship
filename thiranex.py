import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
print("Project Started")
df = pd.read_csv(
    "employee_turnover.csv"
)
print(df.head())
print(df.shape)
df.info()
df.describe()
print(df.isnull().sum())
plt.figure(figsize=(10,5))
sns.heatmap(
    df.isnull(),
    cbar=False
)
plt.title(
    "Missing Values"
)
plt.show()
df.fillna(
df.mean(numeric_only=True),
inplace=True
)
df.fillna(
"Unknown",
inplace=True
)
print(
df.duplicated().sum()
)
df.drop_duplicates(
inplace=True
)

print(df.shape)
plt.figure(figsize=(8,4))
sns.boxplot(
x=df["Monthly_Income"]
)

plt.title(
"Income Outliers"
)
plt.show()
Q1=df["Monthly_Income"].quantile(0.25)

Q3=df["Monthly_Income"].quantile(0.75)

IQR=Q3-Q1
df=df[
(df["Monthly_Income"] >= Q1-1.5*IQR)
&
(df["Monthly_Income"] <= Q3+1.5*IQR)
]
sns.countplot(
x="Employee_Turnover",
data=df
)
plt.title(
"Employee Turnover Distribution"
)
plt.show()
rate = (
df["Employee_Turnover"]
.mean()*100
)
print(
"Turnover Rate:",
round(rate,2),
"%"
)
sns.boxplot(
x="Employee_Turnover",
y="Job_Satisfaction",
data=df
)

plt.title(
"Job Satisfaction vs Turnover"
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
plt.figure(figsize=(12,8))


sns.heatmap(
df.corr(numeric_only=True),
annot=True
)


plt.title(
"Feature Correlation"
)


plt.show()
df.to_csv(
"clean_employee_turnover.csv",
index=False
)

print(
"Clean file saved"
)