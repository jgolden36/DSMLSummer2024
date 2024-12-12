import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn_extra.cluster import KMedoids as SKLearnKMedoids
#K-means from scratch
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
class KMeans:
    def __init__(self, k=3, max_iters=100):
        self.k = k
        self.max_iters = max_iters
        self.centroids = None

    def fit(self, X):
        # Randomly initialize centroids
        self.centroids = X[np.random.choice(X.shape[0], self.k, replace=False)]
        
        for _ in range(self.max_iters):
            # Assign labels based on closest centroid
            labels = self._assign_labels(X)
            # Calculate new centroids from the means of the points
            new_centroids = self._calculate_centroids(X, labels)
            
            # Check for convergence (if centroids do not change)
            if np.all(self.centroids == new_centroids):
                break
                
            self.centroids = new_centroids

    def _assign_labels(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def _calculate_centroids(self, X, labels):
        return np.array([X[labels == i].mean(axis=0) for i in range(self.k)])
    
    def predict(self, X):
        return self._assign_labels(X)

kmeans = KMeans(k=4)
kmeans.fit(X)
labels = kmeans.predict(X)

# Plot results
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], s=300, c='red', marker='X')
plt.title('K-Means Clustering')
plt.show()

#K-means from Sklearn
from sklearn.cluster import KMeans as SKLearnKMeans
sklearn_kmeans = SKLearnKMeans(n_clusters=4)
sklearn_kmeans.fit(X)
labels = sklearn_kmeans.predict(X)

# Plot results
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(sklearn_kmeans.cluster_centers_[:, 0], sklearn_kmeans.cluster_centers_[:, 1], s=300, c='red', marker='X')
plt.title('K-Means Clustering with Scikit-Learn')
plt.show()

#K-mediods from scratch
class KMedoids:
    def __init__(self, k=3, max_iters=100):
        self.k = k
        self.max_iters = max_iters
        self.medoids = None

    def fit(self, X):
        # Randomly initialize medoids
        self.medoids = X[np.random.choice(X.shape[0], self.k, replace=False)]
        
        for _ in range(self.max_iters):
            # Assign labels based on closest medoid
            labels = self._assign_labels(X)
            # Calculate new medoids
            new_medoids = self._calculate_medoids(X, labels)
            
            # Check for convergence (if medoids do not change)
            if np.all(self.medoids == new_medoids):
                break
                
            self.medoids = new_medoids

    def _assign_labels(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.medoids, axis=2)
        return np.argmin(distances, axis=1)

    def _calculate_medoids(self, X, labels):
        new_medoids = np.zeros((self.k, X.shape[1]))
        for i in range(self.k):
            cluster_points = X[labels == i]
            medoid = cluster_points[np.argmin(np.sum(np.linalg.norm(cluster_points[:, np.newaxis] - cluster_points, axis=2), axis=1))]
            new_medoids[i] = medoid
        return new_medoids
    
    def predict(self, X):
        return self._assign_labels(X)

kmedoids = KMedoids(k=4)
kmedoids.fit(X)
labels = kmedoids.predict(X)

# Plot results
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(kmedoids.medoids[:, 0], kmedoids.medoids[:, 1], s=300, c='red', marker='X')
plt.title('K-Medoids Clustering')
plt.show()

#K-mediods from Sklearn
sklearn_kmedoids = SKLearnKMedoids(n_clusters=4)
sklearn_kmedoids.fit(X)
labels = sklearn_kmedoids.predict(X)

# Plot results
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(sklearn_kmedoids.cluster_centers_[:, 0], sklearn_kmedoids.cluster_centers_[:, 1], s=300, c='red', marker='X')
plt.title('K-Medoids Clustering with Scikit-Learn')
plt.show()

#Example with Real World Data