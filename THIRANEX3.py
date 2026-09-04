import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# STEP 1: LOAD TITANIC DATASET
# ============================================================

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

df = pd.read_csv(url)

print("=" * 60)
print("EXPLORATORY DATA ANALYSIS - TITANIC DATASET")
print("=" * 60)

# ============================================================
# STEP 2: BASIC DATA EXPLORATION
# ============================================================

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())


# ============================================================
# STEP 3: STATISTICAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("STATISTICAL SUMMARY")
print("=" * 60)

print(df.describe())


# ============================================================
# STEP 4: HANDLE MISSING VALUES
# ============================================================

df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

print("\nMissing values after cleaning:")
print(df.isnull().sum())


# ============================================================
# STEP 5: SURVIVAL ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("SURVIVAL ANALYSIS")
print("=" * 60)

survival_count = df["Survived"].value_counts()

print("\nSurvival Count:")
print(survival_count)

survival_rate = df["Survived"].mean() * 100

print(f"\nOverall Survival Rate: {survival_rate:.2f}%")


# ============================================================
# STEP 6: SURVIVAL BY GENDER
# ============================================================

print("\n" + "=" * 60)
print("SURVIVAL BY GENDER")
print("=" * 60)

gender_survival = df.groupby("Sex")["Survived"].mean() * 100

print(gender_survival)


# ============================================================
# STEP 7: SURVIVAL BY PASSENGER CLASS
# ============================================================

print("\n" + "=" * 60)
print("SURVIVAL BY PASSENGER CLASS")
print("=" * 60)

class_survival = df.groupby("Pclass")["Survived"].mean() * 100

print(class_survival)


# ============================================================
# STEP 8: CORRELATION ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("CORRELATION MATRIX")
print("=" * 60)

numeric_data = df.select_dtypes(include=np.number)

correlation = numeric_data.corr()

print(correlation)


# ============================================================
# STEP 9: VISUALIZATION - SURVIVAL
# ============================================================

plt.figure(figsize=(8, 5))

df["Survived"].value_counts().sort_index().plot(kind="bar")

plt.title("Titanic Survival Count")
plt.xlabel("Survival Status")
plt.ylabel("Number of Passengers")

plt.xticks(
    [0, 1],
    ["Did Not Survive", "Survived"],
    rotation=0
)

plt.tight_layout()
plt.show()


# ============================================================
# STEP 10: SURVIVAL BY GENDER
# ============================================================

gender_table = pd.crosstab(df["Sex"], df["Survived"])

gender_table.plot(kind="bar", figsize=(8, 5))

plt.title("Survival by Gender")
plt.xlabel("Gender")
plt.ylabel("Number of Passengers")

plt.xticks(rotation=0)

plt.legend(
    ["Did Not Survive", "Survived"]
)

plt.tight_layout()
plt.show()


# ============================================================
# STEP 11: SURVIVAL BY CLASS
# ============================================================

class_table = pd.crosstab(df["Pclass"], df["Survived"])

class_table.plot(kind="bar", figsize=(8, 5))

plt.title("Survival by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Number of Passengers")

plt.xticks(rotation=0)

plt.legend(
    ["Did Not Survive", "Survived"]
)

plt.tight_layout()
plt.show()


# ============================================================
# STEP 12: AGE DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 5))

plt.hist(
    df["Age"],
    bins=20,
    edgecolor="black"
)

plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Passengers")

plt.tight_layout()
plt.show()


# ============================================================
# STEP 13: FARE DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 5))

plt.hist(
    df["Fare"],
    bins=20,
    edgecolor="black"
)

plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Number of Passengers")

plt.tight_layout()
plt.show()


# ============================================================
# STEP 14: AGE VS FARE
# ============================================================

plt.figure(figsize=(8, 5))

plt.scatter(
    df["Age"],
    df["Fare"],
    alpha=0.5
)

plt.title("Age vs Fare")
plt.xlabel("Age")
plt.ylabel("Fare")

plt.tight_layout()
plt.show()


# ============================================================
# STEP 15: CORRELATION HEATMAP
# ============================================================

plt.figure(figsize=(9, 7))

plt.imshow(
    correlation,
    interpolation="nearest"
)

plt.colorbar()

plt.xticks(
    range(len(correlation.columns)),
    correlation.columns,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(correlation.columns)),
    correlation.columns
)

plt.title("Correlation Matrix")

plt.tight_layout()
plt.show()


# ============================================================
# STEP 16: KEY INSIGHTS
# ============================================================

print("\n" + "=" * 60)
print("KEY INSIGHTS")
print("=" * 60)

female_rate = df[df["Sex"] == "female"]["Survived"].mean() * 100
male_rate = df[df["Sex"] == "male"]["Survived"].mean() * 100

first_class_rate = df[df["Pclass"] == 1]["Survived"].mean() * 100
third_class_rate = df[df["Pclass"] == 3]["Survived"].mean() * 100

print(f"\nFemale Survival Rate: {female_rate:.2f}%")
print(f"Male Survival Rate: {male_rate:.2f}%")

print(f"\n1st Class Survival Rate: {first_class_rate:.2f}%")
print(f"3rd Class Survival Rate: {third_class_rate:.2f}%")

print(f"\nAverage Age: {df['Age'].mean():.2f}")

print(f"Average Fare: {df['Fare'].mean():.2f}")

print("\nEDA completed successfully!")
