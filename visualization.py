import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.neighbors import NearestNeighbors


def plot_scaled_features(df, scaled_data):
    scaled_df = pd.DataFrame(scaled_data, columns=["heart_rate_scaled", "flex_pca_scaled"])
    scaled_df["Time"] = df["Time"]

    plt.figure(figsize=(10,6))
    plt.plot(scaled_df["Time"], scaled_df["heart_rate_scaled"], label="Heart Rate (scaled)")
    plt.plot(scaled_df["Time"], scaled_df["flex_pca_scaled"], label="Flex PCA (scaled)")
    plt.xlabel("Time (Minute)")
    plt.ylabel("Scaled Value")
    plt.legend()

    step_plot = 60
    plt.xticks(scaled_df["Time"].iloc[::step_plot], rotation=45)
    plt.tight_layout()

    if not os.path.exists('plot'):
        os.makedirs('plot')
    plt.savefig(os.path.join("plot", 'scaled_features.png'))
    print("plot/scaled_features.png")
    plt.close()

def plot_k_distance_graph(X, k):
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(X)

    distances, _ = neigh.kneighbors(X)
    distances = np.sort(distances[:, k-1])

    plt.figure(figsize=(10, 6))
    plt.plot(distances)
    plt.xlabel('Points')
    plt.ylabel(f'{k}-th nearest neighbor distance')
    plt.grid()

    if not os.path.exists('plot'):
        os.makedirs('plot')
    plt.savefig(os.path.join("plot", 'k_distance_graph.png'))
    print("plot/k_distance_graph.png")
    plt.close()

def plot_tsne_clusters(tsne_results, cluster_labels):
    _, ax = plt.subplots(figsize=(10, 8))
    
    scatter = ax.scatter(
        tsne_results[:, 0],
        tsne_results[:, 1],
        c=cluster_labels,
        cmap="viridis",
        alpha=0.8,
    )

    uniq, counts = np.unique(cluster_labels, return_counts=True)
    handles = []
    for lab, cnt in zip(uniq, counts):
        color = scatter.cmap(scatter.norm(lab))
        name = "Noise" if lab == -1 else f"Cluster {lab}"
        handles.append(
            Line2D([0], [0],
                   marker='o', linestyle='none',
                   markerfacecolor=color, markeredgecolor='none',
                   markersize=8, label=f"{name} (n={cnt})")
        )

    ax.legend(handles=handles, frameon=False)
    ax.set_title("t-SNE of TS2Vec Embeddings (DBSCAN Clusters)")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    
    if not os.path.exists('plot'):
        os.makedirs('plot')
    plt.savefig(os.path.join("plot", 'tsne_clusters.png'))
    print("plot/tsne_clusters.png")
    plt.close()
