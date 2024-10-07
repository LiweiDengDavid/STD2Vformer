from torch_utils import node2vec
from data.data_process import get_adjacency_matrix,load_adjacency_matrix
import networkx as nx
from gensim.models import Word2Vec


class Generate_SE():
    def __init__(self,Adj_file,SE_file,node_num):
        self.is_directed = True
        self.p = 2
        self.q = 1
        self.num_walks = 100
        self.walk_length = 80
        self.dimensions = 64
        self.window_size = 10
        self.iter = 1000 # Maximum number of loops
        self.Adj_file = Adj_file
        self.SE_file = SE_file
        self.node_num = node_num
        self.generate_SE() # call the main function to generate the corresponding SE file

    def read_graph(self,adj_matrix_file):
        if adj_matrix_file.split('.')[-1] == 'csv':
            adj = get_adjacency_matrix(adj_matrix_file, num_of_vertices=self.node_num)
        elif adj_matrix_file.split('.')[-1] == 'pkl':
            adj = load_adjacency_matrix(adj_matrix_file, num_of_vertices=self.node_num)
        else:
            raise print('Path error in adjacency matrix. No path with pathname {0}'.format(adj_matrix_file))
        G = nx.from_numpy_array(adj)
        return G

    def learn_embeddings(self,walks, dimensions, output_file):
        walks = [list(map(str, walk)) for walk in walks]
        model = Word2Vec(
            walks, vector_size=dimensions, window=10, min_count=0, sg=1,
            workers=8,epochs=self.iter)
        model.wv.save_word2vec_format(output_file)

        return

    '''main function'''
    def generate_SE(self):
        nx_G = self.read_graph(self.Adj_file)
        G = node2vec.Graph(nx_G, self.is_directed, self.p, self.q)
        G.preprocess_transition_probs()
        walks = G.simulate_walks(self.num_walks, self.walk_length) # randomize walks to get pairs of co-occurrence cases
        self.learn_embeddings(walks, self.dimensions, self.SE_file) # Word2Vec Embedding based on co-occurrence --> get Embedding of nodes

