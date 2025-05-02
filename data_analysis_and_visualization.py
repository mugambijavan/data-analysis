# Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Task 1: Load and Explore the Dataset
# Using the Iris dataset
iris = load_iris(as_frame=True)
df = iris.frame

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(df.head())

# Explore the structure of the dataset
print("\nDataset Info:")
print(df.info())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Clean the dataset (No missing values in Iris dataset, so no cleaning required)

# Task 2: Basic Data Analysis
# Compute basic statistics
print("\nBasic Statistics:")
print(df.describe())

# Perform groupings: Mean of numerical columns grouped by species
grouped = df.groupby('target').mean()
print("\nMean of numerical columns grouped by species:")
print(grouped)

# Identify patterns or findings
print("\nInteresting Findings:")
print("Setosa species tend to have smaller sepal and petal lengths compared to Versicolor and Virginica.")

# Task 3: Data Visualization
# 1. Line Chart - Trends over "sepal length" for the first 30 samples
plt.figure(figsize=(10, 6))
plt.plot(df.index[:30], df['sepal length (cm)'][:30], marker='o', label='Sepal Length')
plt.title("Line Chart: Sepal Length Trend for First 30 Samples")
plt.xlabel("Sample Index")
plt.ylabel("Sepal Length (cm)")
plt.legend()
plt.grid()
plt.show()

# 2. Bar Chart - Average petal length per species
plt.figure(figsize=(10, 6))
sns.barplot(x=grouped.index, y='petal length (cm)', data=grouped, palette='viridis')
plt.title("Bar Chart: Average Petal Length per Species")
plt.xlabel("Species")
plt.ylabel("Average Petal Length (cm)")
plt.show()

# 3. Histogram - Distribution of sepal width
plt.figure(figsize=(10, 6))
plt.hist(df['sepal width (cm)'], bins=15, color='skyblue', edgecolor='black')
plt.title("Histogram: Distribution of Sepal Width")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Frequency")
plt.show()

# 4. Scatter Plot - Sepal length vs. Petal length
plt.figure(figsize=(10, 6))
sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='target', data=df, palette='coolwarm')
plt.title("Scatter Plot: Sepal Length vs. Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend(title='Species')
plt.show()

