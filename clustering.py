import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE


def get_dbscan_clusters(embeddings, eps, min_samples):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(embeddings)
    print(f"Unique labels found: {np.unique(cluster_labels)}")
    return cluster_labels

def get_tsne_reduction(embeddings, perplexity, random_state):
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
    tsne_results = tsne.fit_transform(embeddings)
    print(f"t-SNE results shape: {tsne_results.shape}")
    return tsne_results
