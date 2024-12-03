import os
from tqdm import tqdm
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from node2vec import Node2Vec
import numpy as np

from dyn_graph_emb.utils import save_list_to_file


class TdGraphEmbed:
    def __init__(self, graphs, labels, config):
        self.num_walks = config["random_walks_per_node"]
        self.embedding_dim = config["embedding_dimension"]
        self.window_size = config["context_window_size"]
        self.walk_length = config["maximum_walk_length"]
        self.epochs = config["epochs"]
        self.p = 1
        self.q = 0.5
        self.graphs = graphs
        self.labels = labels
        self.n_graphs = len(self.graphs)
        self.config = config
        self.save_dir = config["savedir"]
        self.model = None
        self.workers = config["workers"]
        # self.alpha = config["alpha"]

    def get_documents_from_graph(self):
        '''

                :param graphs: dictionary of key- time, value- nx.Graph
                :return: list of documents
                '''
        print("running random walk...")
        documents = []
        max_ts = []
        for gi, graphs_dict in tqdm(enumerate(self.graphs)):
            max_t = 0
            for t in graphs_dict.keys():
                node2vec = Node2Vec(graphs_dict[t], walk_length=self.walk_length, num_walks=self.num_walks,
                                    p=self.p, q=self.q, quiet=True, workers=self.workers)#, weight_key='weight')
                walks = node2vec.walks
                # walks = [[str(word) for word in walk] for walk in walks]
                len_walks = np.mean([len(walk) for walk in walks])
                print(f'average walk length: {len_walks}')
                documents.append([TaggedDocument(doc, [str((gi, t))]) for doc in walks])
                max_t = t if t > max_t else max_t
            max_ts.append(max_t)

        documents = sum(documents, [])
        return documents

    def run_doc2vec(self, documents, max_ts):
        model = Doc2Vec(vector_size=self.embedding_dim, window=self.window_size, workers=self.workers)# alpha=self.alpha, , min_alpha=self.alpha)
        model.build_vocab(documents)
        print('---- training doc2vec ----')
        model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)
        save_path = os.path.join(self.save_dir, 'tdgraphembed.model')
        self.model = model
        model.save(save_path)
        print("Model Saved")
        emb = self.aggregate_embedding_snapshots(max_ts)
        np.savetxt(os.path.join(self.save_dir, 'tdgraphembed.txt'), emb)

    def aggregate_embedding_snapshots(self, max_ts):
        n_graphs = len(self.labels)
        aggregated_emb = []
        for max_t, gi in zip(max_ts, np.arange(n_graphs)):
            emb = np.mean([self.model.dv[str((gi, ti))] for ti in range(1, max_t + 1)], axis=0)
            aggregated_emb.append(emb)
        return np.array(aggregated_emb)
