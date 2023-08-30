import pandas as pd
import numpy as np
from sklearn_extra.cluster import KMedoids
import matplotlib.pyplot as plt

# Load the datasets
biomass_history = pd.read_csv('/content/drive/MyDrive/supplychain/Final_data/Biomass_predictions.csv')
distance_matrix = pd.read_csv('/content/drive/MyDrive/supplychain/Final_data/Distance_Matrix.csv')

# Remove the index column from the distance matrix
distance_matrix = distance_matrix.iloc[:, 1:]

# Use the biomass generation data for the years 2010 to 2019 for clustering
biomass_data = biomass_history.iloc[:, 11:12].values

# Determine the number of clusters (K) based on the constraint
max_clusters = 25
num_harvesting_sites = biomass_data.shape[0]
num_clusters = min(max_clusters, num_harvesting_sites)

# Perform K-Medoids clustering
kmedoids = KMedoids(n_clusters=num_clusters, random_state=42, metric='precomputed')
distances = distance_matrix.values
cluster_labels = kmedoids.fit_predict(distances)

# Add cluster labels to the biomass history data
biomass_history["Cluster"] = cluster_labels

# Initialize a list to store depot locations along with indices and loads
depot_locations = []

# Find the optimal depot location within each cluster and calculate load
for cluster_idx in range(num_clusters):
    cluster_sites = biomass_history[biomass_history["Cluster"] == cluster_idx]
    cluster_avg_biomass_values = cluster_sites["2019"].mean()
    distances_to_avg = np.abs(cluster_sites["2019"] - cluster_avg_biomass_values)
    optimal_depot_idx = cluster_sites.index[np.argmin(distances_to_avg)]
    cluster_load = cluster_sites["2018"].sum()  # Calculate load for the cluster
    if cluster_load <= 20000:
        depot_info = cluster_sites.loc[optimal_depot_idx, ["Latitude", "Longitude", "Index"]]
        depot_locations.append((depot_info, cluster_load))

# Print depot locations and cluster loads
for depot_info, cluster_load in depot_locations:
    print("Depot Index:", depot_info["Index"])
    print("Cluster Load:", cluster_load)
    print("------------------------------")

# Visualize the clusters on the map using Latitude and Longitude
plt.figure(figsize=(12, 8))
plt.scatter(biomass_history["Longitude"], biomass_history["Latitude"], c=biomass_history["Cluster"], cmap="viridis", marker="o", alpha=0.7)
plt.scatter(pd.DataFrame(depot_locations)[0]["Longitude"], pd.DataFrame(depot_locations)[0]["Latitude"], c="red", marker="x", label="Depot Locations")
for idx, (depot_info, cluster_load) in enumerate(depot_locations):
    plt.text(depot_info["Longitude"] + 0.01, depot_info["Latitude"] + 0.01, f"Depot {int(depot_info['Index'])} (Load: {cluster_load})", fontsize=10, color='black')
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Biomass Generation Clusters with Depot Locations")
plt.legend()
plt.colorbar(label="Cluster")
plt.show()
