import jax.numpy as jnp
from abc import ABC, abstractmethod
import ast
import marisa_trie
import networkx as nx

class sample(ABC):
    def __init__(self, array_like):
        self.sample_seq = jnp.array(array_like)
        self.len = len(array_like)
    def back_track(self, back_steps):
        back_steps +=1
        return self.sample_seq[self.len-back_steps],




class samples(sample):
    '''
    samples class was written in the spirit to organize the training data better for running gflow-extrapolate algorithms
    Our implementation of samples are based upon strings
    '''
    def __init__(self,array_like):
        self.samples_seq = jpn.array(array_like)
        self.num_samples = array_like[0]
        self.samples_structure = self.organize_samples(self.samples_seq)
    def organize_samples(self,samples):
        samples_structure = marisa_trie.Trie(samples)
        return samples_structure

    def samplesWithPrefix(self, prefix):
        return self.samples_structure.keys(prefix)
    def samplesPrefixQuery(self, query):
        return self.samples_structure.prefixes(query)



class Trie:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.root = 0
        self.graph.add_node(self.root)

    def insert(self, word, weight=1):
        '''

        :param word: a series of indexes for vocabularies
        :param weight:
        :return:
        '''
        current_node = self.root
        for char in word:
            next_node = current_node + char
            if not self.graph.has_edge(current_node, next_node):
                self.graph.add_edge(current_node, next_node, weight=weight, label=char)
            else:
                self.graph[current_node][next_node]['weight'] += weight
            current_node = next_node
        self.graph.nodes[current_node]['is_end_of_word'] = True

    def search(self, word):
        current_node = self.root
        for char in word:
            next_node = current_node + char
            if not self.graph.has_edge(current_node, next_node):
                return False
            current_node = next_node
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
