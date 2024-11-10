import json
import os
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import numpy as np

from dyn_graph_emb.random_walk import TemporalStructuralRandomWalk
from dyn_graph_emb.utils import get_structural_sim_network


class DynConnectomeEmbed():
    def __init__(self, graphs, labels, config):
        # INPUT: directories of dynamic graph files
        self.num_walks = config["random_walks_per_node"]
        self.embedding_dim = config["embedding_dimension"]
        self.window_size = config["context_window_size"]
        self.walk_length = config["maximum_walk_length"]
        self.k = config["k"]
        self.alpha = config["alpha"]
        self.num_walks = config["random_walks_per_node"]
        self.graphs = graphs
        self.labels = labels
        self.n_graphs = len(self.graphs)
        self.nodes_st = None
        self.config = config
        self.save_dir = config["savedir"]

    def get_random_walk_sequences(self):
        '''

        :param graphs: dictionary of key- time, value- nx.Graph
        :return: list of documents
        '''
        documents = []
        for gi, graph in enumerate(self.graphs):
            if self.alpha > 0:
                structural_graph = get_structural_sim_network(graph, nodes_st=self.nodes_st, k=self.k,
                                                              data_path='')  # TODO: change data path
            else:
                structural_graph = None
            cross_temporal_rw = TemporalStructuralRandomWalk(graph, structural_graph=structural_graph)
            cross_walks = cross_temporal_rw.run(
                num_cw=self.num_walks,
                cw_size=self.window_size,
                max_walk_length=self.walk_length,
                walk_bias="exponential",
                seed=0,
                alpha=self.alpha,
            )
            # walks = [[str(word) for word in walk] for walk in walks]
            documents.append([TaggedDocument(doc, [gi]) for doc in cross_walks])

        documents = sum(documents, [])

        return documents

    def run_doc2vec(self, documents):
        model = Doc2Vec(vector_size=self.embedding_dim, window=self.window_size)  # epochs=self.epochs)
        model.build_vocab(documents)
        model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)
        save_path = os.path.join(self.save_dir, 'model.model')
        model.save(save_path)
        np.savetxt(os.path.join(self.save_dir, 'labels.txt'), self.labels)
        print("Model Saved")

    def get_embeddings(self):
        '''

        :return: temporal graph vectors for each time step.
        numpy array of shape (number of time steps, graph vector dimension size)
        '''
        model_path = f"trained_models/{self.dataset_name}.model"
        model = Doc2Vec.load(model_path)
        graph_vecs = model.dv
        graph_vecs = graph_vecs[np.argsort([model.docvecs.index_to_doctag(i) for i in range(0, graph_vecs.shape[0])])]
        return graph_vecs
