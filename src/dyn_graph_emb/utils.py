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


def get_structural_sim_network(graph, nodes_st, k, data_path):
    from sklearn.preprocessing import StandardScaler

    dgdv = np.loadtxt(os.path.join(data_path, 'dynamic_graphlets/sorted_output_dgdv_6_4_1.txt'))
    scaler = StandardScaler()
    scaled_dgdv = scaler.fit_transform(dgdv)

    pca = PCA(n_components=0.9, random_state=0)
    scaled_dgdv = pca.fit_transform(scaled_dgdv)
    sim = euclidean_similarity_matrix(scaled_dgdv)

    np.fill_diagonal(sim, 0)
    if k == -1:
        k = int(np.mean(list(graph.node_degrees().values())))

    sim = keep_top_k(sim, k)
    df_sim = pd.DataFrame(sim, index=nodes_st, columns=nodes_st)
    g_sim = nx.from_pandas_adjacency(df_sim)
    g_sim = StellarGraph.from_networkx(g_sim)
    return g_sim
