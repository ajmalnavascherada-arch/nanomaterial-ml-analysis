# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load dataset
data = pd.read_csv("materials_dataset.csv")

# Display first rows
print("Dataset preview:")
print(data.head())

# Feature scaling (important for ML)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

from sklearn.decomposition import PCA

# Apply PCA
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

# Elbow method to find best number of clusters
inertia = []
k_range = range(1, 6)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

# Plot elbow curve
plt.figure()
plt.plot(k_range, inertia, marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal Clusters")
plt.show()

# Apply KMeans (choose 3 clusters)
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(scaled_data)

# Add cluster labels to dataset
data["Cluster"] = clusters

print("\nClustered Data:")
print(data)

# Visualize clusters
plt.figure()
plt.scatter(pca_data[:, 0], pca_data[:, 1], c=data["Cluster"])
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Visualization of Clusters")
plt.show()
plt.savefig("results.png")