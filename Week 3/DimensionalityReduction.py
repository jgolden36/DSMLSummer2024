import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE

# Example matrix
A = np.array([[3, 1, 1], [-1, 3, 1]])

# Compute SVD
U, Sigma, VT = np.linalg.svd(A)

print("U matrix:")
print(U)
print("Sigma values:")
print(Sigma)
print("V^T matrix:")
print(VT)
# PCA with Numpy
X = np.array([[2.5, 2.4], [0.5, 0.7], [2.2, 2.9], [1.9, 2.2], [3.1, 3.0], [2.3, 2.7], [2, 1.6], [1, 1.1], [1.5, 1.6], [1.1, 0.9]])

# Standardize the data
X_mean = np.mean(X, axis=0)
X_centered = X - X_mean

# Compute the covariance matrix
cov_matrix = np.cov(X_centered.T)

# Perform eigen decomposition
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Select the top k eigenvectors (k=1 in this case)
k = 1
principal_components = eigenvectors[:, :k]

# Project the data onto the principal components
X_reduced = np.dot(X_centered, principal_components)

print("Principal components:")
print(principal_components)
print("Reduced dataset:")
print(X_reduced)

# Visualize the results
plt.scatter(X_centered[:, 0], X_centered[:, 1], color='blue', label='Original Data')
plt.scatter(X_reduced, np.zeros_like(X_reduced), color='red', label='Projected Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()

#PCA with SkLearn

pca = PCA(n_components=1)
X_reduced = pca.fit_transform(X)

print("Principal components:")
print(pca.components_)
print("Explained variance ratio:")
print(pca.explained_variance_ratio_)
print("Reduced dataset:")
print(X_reduced)

#LDA with Sklearn

y = np.array([0, 1, 0, 0, 1, 0, 1, 1, 1, 1])

# Perform LDA
lda = LDA(n_components=1)
X_reduced = lda.fit_transform(X, y)

print("LDA components:")
print(lda.scalings_)
print("Reduced dataset:")
print(X_reduced)

#t-Distributed Stochastic Neighbor Embedding (t-SNE), a manifold embedding technique

tsne = TSNE(n_components=2, perplexity=30.0, n_iter=1000)
X_reduced = tsne.fit_transform(X)

print("Reduced dataset:")
print(X_reduced)

# Visualize the results
plt.scatter(X_reduced[:, 0], X_reduced[:, 1])
plt.title('t-SNE')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.show()