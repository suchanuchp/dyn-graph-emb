import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import networkx as nx
from stellargraph import StellarGraph

from dyn_graph_emb.ts_model import DynConnectomeEmbed
from dyn_graph_emb.evaluation import train_multiclass, train_multiclass_v2
from dyn_graph_emb.tdgraphembed import TdGraphEmbed
from dyn_graph_emb.utils import save_list_to_file
from dyn_graph_emb.graph_utils import get_structural_sim_network
# nohup python -u src/dyn_graph_emb/run_embedding.py --datadir data/prep_w50_s5_aal_all --savedir output/embeddings/emb_b0_ts1_a010_r100_l20_w5 -r 100 -l 20 -w 5 --num_nodes 116 --include_same_timestep_neighbors 1 --run_baseline 0 --start 0 --end 10 > logs/emb_b0_ts1_a010_r100_l20_w5.txt
# emb_b1_ts1_a1520_r10_l20_w5.txt
# nohup python -u src/dyn_graph_emb/run_embedding.py --datadir data/prep_w50_s5_aal_all --savedir output/embeddings/emb_b1_a1015_r20_l20_w5 -r 20 -l 20 -w 5 --num_nodes 116 --include_same_timestep_neighbors 0 --run_baseline 1 --run_tswalk 0 --start 0 --end 10 > logs/emb_b1_a1015_r20_l20_w5.txt &


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--datadir', type=str, default='data/prep_test')
    parser.add_argument('-s', '--savedir', type=str, default='data/test')
    parser.add_argument('-d', '--embedding_dimension', type=int, default=32)
    parser.add_argument('-r', '--random_walks_per_node', type=int, default=20)
    parser.add_argument('--label_path', type=str, default='data/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv')
    parser.add_argument('--dgraphlet_path', type=str, default='')
    parser.add_argument('-l', '--maximum_walk_length', type=int, default=20)
    parser.add_argument('-w', '--context_window_size', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('-k', '--k', type=int, default=-1)
    parser.add_argument('-z', '--save_embeddings', type=int, default=0)
    parser.add_argument('--alpha', type=float, default=0.)
    parser.add_argument('--num_nodes', type=int, default=116)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=10)
    parser.add_argument('--run_baseline', type=int, default=1)
    parser.add_argument('--run_tswalk', type=int, default=1)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--include_same_timestep_neighbors', type=int, default=0)

    args = parser.parse_args()
    opt = vars(args)

    data_dir = opt['datadir']
    save_dir = opt['savedir']
    label_path = opt['label_path']
    n_nodes = opt['num_nodes']
    start = opt['start']
    end = opt['end']

    if not os.path.exists(data_dir):
        raise FileExistsError("data dir does not exist")

    if not os.path.exists(label_path):
        raise FileExistsError("label path does not exist")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(os.path.join(save_dir, 'params.txt'), "w") as f:
        for key, value in opt.items():
            print("{}: {}\n".format(key, value))
            f.write("{}: {}\n".format(key, value))

    df_info = pd.read_csv(label_path)
    filenames = os.listdir(data_dir)
    filenames = sorted([filename for filename in filenames if filename.endswith('.csv')])
    # end = end if end != -1 else len(filenames)

    df_info = df_info[(df_info['AGE_AT_SCAN'] >= start) & (df_info['AGE_AT_SCAN'] < end)]
    poi_files = df_info.FILE_ID.tolist()
    filtered_filenames = []
    labels = []
    graph_indices = []
    for filename in filenames:
        file_id = filename[:filename.find('_func_preproc.csv')]
        if file_id in poi_files:
            filtered_filenames.append(filename)
            group = df_info[df_info.FILE_ID == file_id].iloc[0].DX_GROUP
            label = 0 if group == 2 else 1  # 0: control, 1: autism
            labels.append(label)
            graph_indices.append(file_id)

    labels = np.array(labels)
    print(f'------- filtered files: {len(filtered_filenames)}')
    save_list_to_file(graph_indices, os.path.join(save_dir, 'graph_indices.txt'))
    np.savetxt(os.path.join(save_dir, 'labels.txt'), labels)

    if opt['run_baseline']:
        run_tdgraphembed(filtered_filenames, labels, opt)

    if opt['run_tswalk']:
        run_tswalk(filtered_filenames, labels, opt)


def run_tswalk(filtered_filenames, labels, opt):
    data_dir = opt['datadir']
    n_nodes = opt['num_nodes']
    dgraphlet_path = opt['dgraphlet_path']
    k = opt['k']
    nodes = np.arange(n_nodes)
    nodes = [str(node) for node in nodes]
    graphs = []
    dgraphlet_graphs = []
    for filename in tqdm(filtered_filenames):
        filepath = os.path.join(data_dir, filename)
        df_graph = pd.read_csv(filepath, index_col=False, names=['src', 'dst', 't'])
        df_graph.src = df_graph.src.astype(str)
        df_graph.dst = df_graph.dst.astype(str)
        dynamic_graph = StellarGraph(
            nodes=pd.DataFrame(index=nodes),
            edges=df_graph,
            source_column='src',
            target_column='dst',
            edge_weight_column='t',
        )

        graphs.append(dynamic_graph)

        if opt['alpha'] != 0:
            file_id = filename[:filename.find('_func_preproc.csv')]
            dgdv_path = os.path.join(dgraphlet_path, f'{file_id}_dgdv_5_3_1.txt')  # TODO: check this
            dgraphlet_sim_graph = get_structural_sim_network(dgraphlet_path=dgdv_path, nodes_st=nodes, k=k)
            dgraphlet_graphs.append(dgraphlet_sim_graph)

    model = DynConnectomeEmbed(graphs=graphs,
                               structural_graphs=dgraphlet_graphs if opt['alpha'] != 0 else None,
                               labels=labels,
                               config=opt)
    walk_sequences = model.get_random_walk_sequences()
    model.run_doc2vec(walk_sequences)
    emb = model.get_embeddings()
    print('------ts model logistic---------')
    train_multiclass(emb, labels)
    print('------ts model svm: linear---------')
    train_multiclass_v2(emb, labels, kernel='linear')
    print('------ts model svm: rbf---------')
    train_multiclass_v2(emb, labels, kernel='rbf')


def run_tdgraphembed(filtered_filenames, labels, opt):
    data_dir = opt['datadir']
    n_nodes = opt['num_nodes']
    nodes = np.arange(n_nodes)
    nodes_st = [str(node) for node in nodes]
    graphs = []

    for filename in tqdm(filtered_filenames):
        filepath = os.path.join(data_dir, filename)
        df_graph = pd.read_csv(filepath, index_col=False, names=['src', 'dst', 't'])
        df_graph.src = df_graph.src.astype(str)
        df_graph.dst = df_graph.dst.astype(str)
        g, max_t = get_temporal_graphs_dict(df_graph, nodes_st)
        graphs.append(g)

    model = TdGraphEmbed(graphs=graphs,
                         labels=labels,
                         config=opt)
    walk_sequences, max_ts = model.get_documents_from_graph()
    model.run_doc2vec(walk_sequences, max_ts)
    emb = model.aggregate_embedding_snapshots(max_ts)
    print('----logistic tdgraphembed----')
    train_multiclass(emb, labels)
    print('----svm tdgraphembed: linear----')
    train_multiclass_v2(emb, labels, kernel='linear')
    print('----svm tdgraphembed: rbf----')
    train_multiclass_v2(emb, labels, kernel='rbf')


def get_temporal_graphs_dict(df, nodes_st):
    G = {}
    for t, time_group in df.groupby(df.t):
        g = nx.from_pandas_edgelist(time_group, source='src', target='dst',
                                    create_using=nx.Graph())
        g.add_nodes_from(nodes_st)
        G[t] = g

    return G, np.max(df.t)


if __name__ == "__main__":
    main()
