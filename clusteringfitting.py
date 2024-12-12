#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


# Load the dataset
file_path = 'winequality-red.csv' 
data = pd.read_csv(file_path, delimiter=';')
data.head()
data.info()
data.decribe()
# Numeric data for clustering
numeric_data = data.select_dtypes(include=['float64', 'int64']).dropna()

# numeric data for clustering
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_data)


def plot_histogram(data, column):
    """Plots a histogram."""
    plt.figure(figsize=(9, 5))
    plt.hist(data[column], bins=10, color='pink', edgecolor='black')
    plt.title(f"Histogram of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.show()


def plot_scatter(data, col1, col2):
    """Plots a scatter plot"""
    plt.figure(figsize=(9, 5))
    plt.scatter(data[col1], data[col2], c='red', alpha=0.5)
    plt.title(f"Scatter Plot: {col1} vs {col2}")
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.show()


def plot_heatmap(data):
    """Plots a heatmap for the correlation matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Heatmap of Correlation Matrix")
    plt.show()


def kmeans_elbow_plot(data, max_clusters=10):
    """Plots an elbow plot for K-Means clustering."""
    distortions = []
    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=40)
        kmeans.fit(data)
        distortions.append(kmeans.inertia_)

    plt.figure(figsize=(9, 5))
    plt.plot(range(1, max_clusters + 1), distortions, marker='o', color='blue')
    plt.title("Elbow Plot for K-Means Clustering")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia")
    plt.show()


def line_fitting(data, col1, col2):
    """Fits a linear regression line"""
    x = data[[col1]].values
    y = data[col2].values
    model = LinearRegression()
    model.fit(x, y)
    predictions = model.predict(x)

    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, color='yellow', label='Data Points')
    plt.plot(x, predictions, color='purple', label='Fitted Line')
    plt.title(f"Line Fitting: {col1} vs {col2}")
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.legend()
    plt.show()



# In[ ]:




