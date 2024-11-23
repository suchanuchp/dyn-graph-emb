import os
from tqdm import tqdm
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from node2vec import Node2Vec
import numpy as np


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
        # self.alpha = config["alpha"]

    def get_documents_from_graph(self):
        '''

                :param graphs: dictionary of key- time, value- nx.Graph
                :return: list of documents
                '''
        print("running random walk...")
        documents = []
        for gi, graphs_dict in tqdm(enumerate(self.graphs)):
            for t in graphs_dict.keys():
                node2vec = Node2Vec(graphs_dict[t], walk_length=self.walk_length, num_walks=self.num_walks,
                                    p=self.p, q=self.q, quiet=True)#, weight_key='weight')
                walks = node2vec.walks
                # walks = [[str(word) for word in walk] for walk in walks]
                len_walks = np.mean([len(walk) for walk in walks])
                print(f'average walk length: {len_walks}')
                documents.append([TaggedDocument(doc, [str((gi, t))]) for doc in walks])

        documents = sum(documents, [])
        return documents

    def run_doc2vec(self, documents):
        model = Doc2Vec(vector_size=self.embedding_dim, window=self.window_size)# alpha=self.alpha, , min_alpha=self.alpha)
        model.build_vocab(documents)
        print('---- training doc2vec ----')
        model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)
        save_path = os.path.join(self.save_dir, 'tdgraphembed.model')
        self.model = model
        model.save(save_path)
        np.savetxt(os.path.join(self.save_dir, 'labels.txt'), self.labels)
        print("Model Saved")

    # def get_embeddings(self):
    #     '''
    #
    #     :return: temporal graph vectors for each time step.
    #     numpy array of shape (number of time steps, graph vector dimension size)
    #     '''
    #     model_path = f"trained_models/{self.dataset_name}.model"
    #     model = Doc2Vec.load(model_path)
    #     graph_vecs = model.docvecs.doctag_syn0
    #     graph_vecs = graph_vecs[np.argsort([model.docvecs.index_to_doctag(i) for i in range(0, graph_vecs.shape[0])])]
    #     return graph_vecs
