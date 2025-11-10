from ts2vec import TS2Vec


def train_ts2vec(windows, input_dims, output_dims, n_epochs):
    model = TS2Vec(input_dims=input_dims, output_dims=output_dims)
    model.fit(windows, n_epochs=n_epochs, verbose=True)
    embeddings = model.encode(windows, encoding_window="full_series")
    print(f"Embeddings shape: {embeddings.shape}")
    return embeddings
