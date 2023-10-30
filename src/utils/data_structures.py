import jax.numpy as jnp
from abc import ABC, abstractmethod
import ast
import marisa_trie
import networkx as nx
#
#
#
# class samples(ABC):
#     '''
#     samples class was written in the spirit to organize the training data better for running gflow-extrapolate algorithms
#     Our implementation of samples are based upon strings
#     '''
#     def __init__(self, data, sample_structure):
#         '''
#         :param data: Preprocessed data where each data point is a word. Characters are indexes where elements that composed the object mapped to.
#         :param sample_structure:
#         notes: Based on sample structure, the data can be transformed to desired format before storing in the suitable data structures
#         '''
#         self.samples_seq = jpn.array(array_like)
#         self.num_samples = array_like[0]
#         self.samples_structure = self.organize_samples(self.samples_seq)
#     def organize_samples(self,samples):
#         samples_structure = marisa_trie.Trie(samples)
#         return samples_structure
#
#     def samplesWithPrefix(self, prefix):
#         return self.samples_structure.keys(prefix)
#     def samplesPrefixQuery(self, query):
#         return self.samples_structure.prefixes(query)



class Trie:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.root = 0
        self.graph.add_node(self.root)

    def insert(self, sentence):
        '''
        :param word: a series of characters which are indexes of elements of an object
        :param weight:
        :return:
        q: how can we make this function more adaptable to other use cases such as including weights?
        '''
        cursor = self.root
        for word in sentence:
            if not self.graph.has_edge(cursor, word):
                # When initializing the Trie, the weight are used to represent edge flows
                # Assumption of sample structure and a medium to represent the sample structure should be not be part of the gflow network.
                self.graph.add_edge(cursor, word)
            cursor = word
     #   self.graph.nodes[current_node]['is_end_of_word'] = True
        # nx.set_node_attributes(G, {0: "red", 1: "blue"}, name="color")
    def search(self, sentence):
        cursor = self.root
        for word in sentence:
            if not self.graph.has_edge(cursor, word):
                return False
            cursor = word
        return self.graph.nodes[current_node].get('is_end_of_word', False)

    # Visualization helper using matplotlib
    def visualize(self):
        pos = nx.spring_layout(self.graph)
        nx.draw_networkx_nodes(self.graph, pos)
        nx.draw_networkx_edges(self.graph, pos)
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels={(u, v): self.graph[u][v]['label'] for u, v in self.graph.edges()})
        nx.draw_networkx_labels(self.graph, pos)

trie = Trie()
trie.insert("hello", weight=2)
trie.insert("hell", weight=3)
trie.insert("help", weight=1)
trie.visualize()
