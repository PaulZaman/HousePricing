import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Importing the dataset
data = pd.read_csv("data/cleaned_data.csv")

# remove the highest 10% values for "Valeur fonciere"
data = data[data["Valeur fonciere"] < data["Valeur fonciere"].quantile(0.90)]

# Extract the "valeur foncière" column
target_column = "Valeur fonciere"
y = data[target_column]

# Iterate over each column (except the target column)
for column in data.columns:
    if column != target_column:
        # Extract the column
        X = data[column]
        
        # Reshape the data for KMeans input
        X = X.values.reshape(-1, 1)
        
        # Instantiate the KMeans model with the desired number of clusters
        kmeans = KMeans(n_clusters=3, random_state=42)
        
        # Fit the model to your data
        kmeans.fit(X)
        
        # Get the cluster labels for each data point
        cluster_labels = kmeans.labels_
        # Get the coordinates of the cluster centers
        cluster_centers = kmeans.cluster_centers_
        
        # Visualize the clusters
        plt.scatter(X, y, c=cluster_labels)
        plt.xlabel(column)
        plt.ylabel(target_column)
        plt.title(f"Clustering: {column} vs {target_column}")
        plt.scatter(cluster_centers, cluster_centers, c="red", s=200, alpha=0.5)
        plt.show()
        
        # Print cluster centers and labels
        print("Cluster Centers:")
        print(cluster_centers)
        print("Cluster Labels:")
        print(cluster_labels)
