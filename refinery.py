import pandas as pd
from sklearn_extra.cluster import KMedoids
import matplotlib.pyplot as plt

# Load the datasets
distance_matrix = pd.read_csv('/content/drive/MyDrive/supplychain/Final_data/Depotanalysis/dm_depots.csv', index_col=0)
locations_data = pd.read_csv('/content/drive/MyDrive/supplychain/Final_data/Depotanalysis/locations.csv', index_col='Index')  # Set 'Index' as the index

# Limit the maximum number of clusters
max_clusters = 5

# Get the number of locations and clusters
num_locations = locations_data.shape[0]
num_clusters = min(max_clusters, num_locations)

# Perform K-Medoids clustering
kmedoids = KMedoids(n_clusters=num_clusters, random_state=42, metric='precomputed')
cluster_labels = kmedoids.fit_predict(distance_matrix)

# Add cluster labels to the locations data
locations_data["Cluster"] = cluster_labels

# Select a point as the centroid for each cluster
centroid_indices = []
for cluster_idx in range(num_clusters):
    cluster_sites = locations_data[locations_data["Cluster"] == cluster_idx]
    centroid_index = cluster_sites.sample(1).index[0]  # Select a random point as the centroid
    centroid_indices.append(centroid_index)

    # Calculate and print total load of the cluster
    total_load = cluster_sites["Load"].sum()
    print(f"Cluster {cluster_idx} - Total Load: {total_load}, Centroid Index: {centroid_index}")

# Plot clusters and centroids using Matplotlib
plt.figure(figsize=(10, 8))
for cluster_idx in range(num_clusters):
    cluster_sites = locations_data[locations_data["Cluster"] == cluster_idx]
    plt.scatter(cluster_sites["Longitude"], cluster_sites["Latitude"], label=f"Cluster {cluster_idx}", alpha=0.7)
plt.scatter(locations_data.loc[centroid_indices, "Longitude"],
            locations_data.loc[centroid_indices, "Latitude"],
            marker="x", color="red", label="Centroids")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Clustered Locations with Centroids")
plt.legend()
plt.show()
