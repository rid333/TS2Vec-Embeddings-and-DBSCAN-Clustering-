import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def load_preprocess(filepath):
    df = pd.read_csv(filepath)
    scaler_flex = StandardScaler()
    flex_scaled = scaler_flex.fit_transform(df[["flex_right", "flex_left"]])
    pca = PCA(n_components=1)
    df["flex_pca"] = pca.fit_transform(flex_scaled)
    scaler_all = StandardScaler()
    X_scaled = scaler_all.fit_transform(df[["heart_rate", "flex_pca"]])
    return df, X_scaled

def sliding_windows(data, window_size, step):
    T, D = data.shape
    windows = []
    for start in range(0, T - window_size + 1, step):
        end = start + window_size
        windows.append(data[start:end])
    return np.array(windows)
