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
# python -u src/dyn_graph_emb/run_embedding.py --datadir data/prep_w50_s8_aal_batch2 --savedir data/emb_w50_s8_aal_batch2_l10_w4 -r 10 --maximum_walk_length 10 --context_window_size 4 --num_nodes 116


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--datadir', type=str, default='data/prep_test')
    parser.add_argument('-s', '--savedir', type=str, default='data/test')
    parser.add_argument('-d', '--embedding_dimension', type=int, default=32)
    parser.add_argument('-r', '--random_walks_per_node', type=int, default=20)
    parser.add_argument('--label_path', type=str, default='data/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv')
    parser.add_argument('-l', '--maximum_walk_length', type=int, default=20)
    parser.add_argument('-w', '--context_window_size', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('-k', '--k', type=int, default=-1)
    parser.add_argument('-z', '--save_embeddings', type=int, default=0)
    parser.add_argument('--alpha', type=float, default=0.)
    parser.add_argument('--num_nodes', type=int, default=200)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=10)
    parser.add_argument('--run_baseline', type=int, default=1)

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
    nodes = np.arange(n_nodes)
    nodes = [str(node) for node in nodes]
    filenames = os.listdir(data_dir)
    filenames = sorted([filename for filename in filenames if filename.endswith('.csv')])
    # end = end if end != -1 else len(filenames)

    df_info = df_info[(df_info['AGE_AT_SCAN'] >= start) & (df_info['AGE_AT_SCAN'] < end)]
    poi_files = df_info.FILE_ID.tolist()
    filtered_filenames = []
    for filename in filenames:
        file_id = filename[:filename.find('_func_preproc.csv')]
        if file_id in poi_files:
            filtered_filenames.append(filename)

    print(f'------- filtered files: {len(filtered_filenames)}')

    if opt['run_baseline']:
        run_tdgraphembed(filtered_filenames, opt)

    graphs = []
    labels = []
    for filename in tqdm(filtered_filenames):
        file_id = filename[:filename.find('_func_preproc.csv')]
        group = df_info[df_info.FILE_ID == file_id].iloc[0].DX_GROUP
        label = 0 if group == 2 else 1  # 0: control, 1: autism
        labels.append(label)

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

        file_id = filename[:filename.find('_func_preproc.csv')]
        group = df_info[df_info.FILE_ID == file_id].iloc[0].DX_GROUP
        label = 0 if group == 2 else 1  # 0: control, 1: autism
        labels.append(label)

    labels = np.array(labels)
    model = DynConnectomeEmbed(graphs=graphs,
                               labels=labels,
                               config=opt)
    walk_sequences = model.get_random_walk_sequences()
    model.run_doc2vec(walk_sequences)
    emb = model.get_embeddings()
    train_multiclass(emb, labels)


def run_tdgraphembed(filtered_filenames, opt):
    data_dir = opt['datadir']
    label_path = opt['label_path']
    n_nodes = opt['num_nodes']
    nodes = np.arange(n_nodes)
    nodes_st = [str(node) for node in nodes]
    df_info = pd.read_csv(label_path)

    graphs = []
    labels = []
    max_ts = []

    for filename in tqdm(filtered_filenames):
        filepath = os.path.join(data_dir, filename)
        file_id = filename[:filename.find('_func_preproc.csv')]
        group = df_info[df_info.FILE_ID == file_id].iloc[0].DX_GROUP
        label = 0 if group == 2 else 1  # 0: control, 1: autism
        labels.append(label)
        df_graph = pd.read_csv(filepath, index_col=False, names=['src', 'dst', 't'])
        df_graph.src = df_graph.src.astype(str)
        df_graph.dst = df_graph.dst.astype(str)
        g, max_t = get_temporal_graphs_dict(df_graph, nodes_st)
        graphs.append(g)
        max_ts.append(max_t)

    labels = np.array(labels)
    model = TdGraphEmbed(graphs=graphs,
                         labels=labels,
                         config=opt)
    walk_sequences = model.get_documents_from_graph()
    model.run_doc2vec(walk_sequences)
    emb = aggregate_graph_snapshots(model.model, len(graphs), max_ts)
    print('----logistic----')
    train_multiclass(emb, labels)
    print('----svm----')
    train_multiclass_v2(emb, labels)


def aggregate_graph_snapshots(model, n_graphs, max_ts):
    aggregated_emb = []
    for max_t, gi in zip(max_ts, np.arange(n_graphs)):
        emb = np.mean([model.dv[str((gi, ti))] for ti in range(1, max_t+1)], axis=0)
        print(emb.shape)
        aggregated_emb.append(emb)
    return np.array(aggregated_emb)


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
