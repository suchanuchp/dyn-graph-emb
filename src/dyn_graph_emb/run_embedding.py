import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

from stellargraph import StellarGraph
from dyn_graph_emb.ts_model import DynConnectomeEmbed
from dyn_graph_emb.evaluation import train_multiclass
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
    parser.add_argument('-k', '--k', type=int, default=-1)
    parser.add_argument('-z', '--save_embeddings', type=int, default=0)
    parser.add_argument('--alpha', type=float, default=0.)
    parser.add_argument('--num_nodes', type=int, default=200)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=-1)

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
    end = end if end != -1 else len(filenames)

    graphs = []
    labels = []
    for filename in tqdm(filenames[start:end]):
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

    model = DynConnectomeEmbed(graphs=graphs,
                               labels=np.array(labels),
                               config=opt)
    walk_sequences = model.get_random_walk_sequences()
    model.run_doc2vec(walk_sequences)
    emb = model.get_embeddings()
    train_multiclass(emb, labels)


if __name__ == "__main__":
    main()
