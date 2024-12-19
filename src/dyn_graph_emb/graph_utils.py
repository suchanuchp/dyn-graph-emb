import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from stellargraph import StellarGraph
import networkx as nx
from sklearn.decomposition import PCA


def euclidean_similarity_matrix(A):
    distances = pdist(A, 'euclidean')
    distance_matrix = squareform(distances)
    similarity_matrix = 1 / (1 + distance_matrix)
    return similarity_matrix


def keep_top_k(matrix, k):
    # Initialize a zero matrix of the same shape as the input
    result = np.zeros_like(matrix)

    # For each row, find the indices of the top k values
    top_k_indices = np.argpartition(matrix, -k, axis=1)[:, -k:]

    # Use advanced indexing to set the top k values in the result matrix
    row_indices = np.arange(matrix.shape[0])[:, None]
    result[row_indices, top_k_indices] = matrix[row_indices, top_k_indices]

    return result


def get_structural_sim_network(dgraphlet_path, nodes_st, k):
    from sklearn.preprocessing import StandardScaler

    dgdv, zeros_indices = read_dgdvs(dgraphlet_path, total_rows=116)
    scaler = StandardScaler()
    scaled_dgdv = scaler.fit_transform(dgdv)

    pca = PCA(n_components=0.9, random_state=0)
    scaled_dgdv = pca.fit_transform(scaled_dgdv)
    sim = euclidean_similarity_matrix(scaled_dgdv)

    np.fill_diagonal(sim, 0)
    sim[zeros_indices] = 0
    sim[:, zeros_indices] = 0
    # if k == -1:
    #     k = int(np.mean(list(graph.node_degrees().values())))

    sim = keep_top_k(sim, k)
    df_sim = pd.DataFrame(sim, index=nodes_st, columns=nodes_st)
    g_sim = nx.from_pandas_adjacency(df_sim)
    g_sim = StellarGraph.from_networkx(g_sim)
    return g_sim


def read_dgdvs(file_path, total_rows=116):
    data = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.split()
            index = int(parts[0])
            vals = [int(num) for num in parts[1:]]
            data[index] = vals
            dim = len(vals)

    zeros = [0] * dim

    max_index = total_rows - 1
    full_data = []
    zeros_indices = []
    for i in range(max_index):  # TODO: check if this should be max_index or total_rows
        if i in data:
            full_data.append(data[i])
        else:
            zeros_indices.append(i)
            full_data.append(zeros)

    full_data_array = np.array(full_data)

    return full_data_array, zeros_indices
