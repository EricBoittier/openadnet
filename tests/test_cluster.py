from test_data import data
from test_fingerprints import fingerprints_func
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
print(f"{len(data)} rows loaded")

fingerprints = [fingerprints_func(row["smiles"]) for row in data]

fingerprints = np.array(fingerprints)
print(fingerprints.shape)
kmeans = KMeans(n_clusters=30, random_state=42)
kmeans.fit(fingerprints)

print(kmeans.labels_)
print("--------------------------------")
tsne = TSNE(n_components=2, random_state=42)
fingerprints_tsne = tsne.fit_transform(fingerprints)
print(fingerprints_tsne.shape)
print("--------------------------------")
pca = PCA(n_components=2, random_state=42)
fingerprints_pca = pca.fit_transform(fingerprints)
print(fingerprints_pca.shape)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_)

print("--------------------------------")
fig, axes = plt.subplots(2, 2, figsize=(10, 5))
axes[0, 0].scatter(fingerprints_tsne[:, 0], fingerprints_tsne[:, 1], c=kmeans.labels_, cmap='viridis')
axes[0, 0].set_title('TSNE')
axes[0, 1].scatter(fingerprints_pca[:, 0], fingerprints_pca[:, 1], c=kmeans.labels_, cmap='viridis')
axes[0, 1].set_title('PCA')
axes[1, 0].scatter(fingerprints[:, 0], fingerprints[:, 1], c=kmeans.labels_, cmap='viridis')
axes[1, 0].set_title('Original')
axes[1, 1].scatter(fingerprints[:, 0], fingerprints[:, 1], c=kmeans.labels_, cmap='viridis')
axes[1, 1].set_title('Original')
plt.scatter(fingerprints_tsne[:, 0], fingerprints_tsne[:, 1], c=kmeans.labels_, cmap='viridis')
plt.scatter(fingerprints_pca[:, 0], fingerprints_pca[:, 1], c=kmeans.labels_, cmap='viridis')
plt.show()
