import numpy as np
import pandas as pd

import config
from clustering import get_dbscan_clusters, get_tsne_reduction
from data_processing import load_preprocess, sliding_windows
from model import train_ts2vec
from visualization import (plot_k_distance_graph, plot_scaled_features,
                           plot_tsne_clusters)


def get_cluster_statistics(df, cluster_labels, window_size, step):
    signals = ["heart_rate", "flex_right", "flex_left"]
    rows = []
    n_windows = len(cluster_labels)
    
    for idx in range(n_windows):
        start = idx * step
        end = start + window_size
        if end > len(df):
            continue
            
        raw_window = df.iloc[start:end][signals]
        
        row = {"cluster": int(cluster_labels[idx])}
        for s in signals:
            row[f"{s}_mean"] = raw_window[s].mean()
            row[f"{s}_var"]  = raw_window[s].var()
        rows.append(row)

    window_stats = pd.DataFrame(rows)
    
    cluster_stats = (
        window_stats.groupby("cluster")
        .agg("mean")
        .join(window_stats.groupby("cluster").size().rename("n_windows"))
        .sort_index()
        .round(3)
    )
    
    return cluster_stats

def main():
    # Load the data
    df, X_scaled = load_preprocess(config.DATA_FILE)
   
    # Slide the data
    windows = sliding_windows(X_scaled, config.WINDOW_SIZE, config.STEP)
         
    embeddings = train_ts2vec(
        windows,
        input_dims=X_scaled.shape[1],
        output_dims=config.TS2VEC_OUTPUT_DIMS,
        n_epochs=config.TS2VEC_EPOCHS
    )
   
    # Cluster embedded data
    cluster_labels = get_dbscan_clusters(
        embeddings,
        config.DBSCAN_EPS,
        config.DBSCAN_MIN_SAMPLES
    )
   
    # t-SNE reduction
    tsne_results = get_tsne_reduction(
        embeddings,
        config.TSNE_PERPLEXITY,
        config.TSNE_RANDOM_STATE
    )
   
    # Visualize the result
    plot_scaled_features(df, X_scaled)
    plot_k_distance_graph(embeddings, k=config.K_NEIGHBORS)
    plot_tsne_clusters(tsne_results, cluster_labels)
   
    # Print cluster stats
    cluster_stats = get_cluster_statistics(
        df,
        cluster_labels,
        config.WINDOW_SIZE,
        config.STEP
    )
    
    print("Cluster Statistics")
    print(cluster_stats)

if __name__ == "__main__":
    main()
