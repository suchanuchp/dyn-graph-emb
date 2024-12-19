import os
from tqdm import tqdm
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import numpy as np

from dyn_graph_emb.random_walk import TemporalStructuralRandomWalk
from dyn_graph_emb.graph_utils import get_structural_sim_network


class DynConnectomeEmbed:
    def __init__(self, graphs, structural_graphs, labels, config):
        self.num_walks = config["random_walks_per_node"]
        self.embedding_dim = config["embedding_dimension"]
        self.window_size = config["context_window_size"]
        self.walk_length = config["maximum_walk_length"]
        self.k = config["k"]
        self.alpha = config["alpha"]
        self.num_walks = config["random_walks_per_node"]
        self.epochs = config["epochs"]
        self.graphs = graphs
        self.structural_graphs = structural_graphs
        self.labels = labels
        self.n_graphs = len(self.graphs)
        self.nodes_st = None
        self.config = config
        self.save_dir = config["savedir"]
        self.model = None
        self.workers = config["workers"]
        self.include_same_timestep_neighbors = config["include_same_timestep_neighbors"]

    def get_random_walk_sequences(self):
        '''

        :param graphs: dictionary of key- time, value- nx.Graph
        :return: list of documents
        '''
        print("running random walk...")
        documents = []
        for gi, graph in tqdm(enumerate(self.graphs)):
            if self.alpha > 0:
                structural_graph = self.structural_graphs[gi]
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
                include_same_timestep_neighbors=self.include_same_timestep_neighbors,
            )
            len_walks = np.mean([len(walk) for walk in cross_walks])
            print(f'average walk length: {len_walks}')
            documents.append([TaggedDocument(doc, [gi]) for doc in cross_walks])

        documents = sum(documents, [])

        return documents

    def run_doc2vec(self, documents):
        model = Doc2Vec(vector_size=self.embedding_dim, window=self.window_size, epochs=self.epochs, workers=self.workers)
        model.build_vocab(documents)
        model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)
        save_path = os.path.join(self.save_dir, 'model.model')
        model.save(save_path)
        self.model = model
        print("Model Saved")
        emb = self.get_embeddings()
        np.savetxt(os.path.join(self.save_dir, 'tsembed.txt'), emb)

    def get_embeddings(self):
        '''

        :return: temporal graph vectors for each time step.
        numpy array of shape (number of time steps, graph vector dimension size)
        '''
        if self.model is None:
            raise Exception

        return np.array([self.model.dv.get_vector(i) for i in np.arange(len(self.labels))])
