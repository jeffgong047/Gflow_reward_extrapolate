from abc import ABC, abstractmethod

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


class Trie_node():
    def __init__(self,vocab_size):
        assert isinstance(vocab_size,int)
        self.children = [None ]*vocab_size
        self.end_of_Sentence = False
    def get_child(self):
        return self.child

    def add_child(self,element):
        self.child.append(element)




class TrieNode:

    # Trie node class
    def __init__(self):
        self.children = [None]*26

        # isEndOfWord is True if node represent the end of the word
        self.isEndOfWord = False

class Trie:

    # Trie data structure class
    def __init__(self):
        self.root = self.getNode()

    def getNode(self):

        # Returns new trie node (initialized to NULLs)
        return TrieNode()

    def _charToIndex(self,ch):

        # private helper function
        # Converts key current character into index
        # use only 'a' through 'z' and lower case

        return ord(ch)-ord('a')


    def insert(self,key):

        # If not present, inserts key into trie
        # If the key is prefix of trie node,
        # just marks leaf node
        pCrawl = self.root
        length = len(key)
        for level in range(length):
            index = self._charToIndex(key[level])

            # if current character is not present
            if not pCrawl.children[index]:
                pCrawl.children[index] = self.getNode()
            pCrawl = pCrawl.children[index]

        # mark last node as leaf
        pCrawl.isEndOfWord = True

    def search(self, key):

        # Search key in the trie
        # Returns true if key presents
        # in trie, else false
        pCrawl = self.root
        length = len(key)
        for level in range(length):
            index = self._charToIndex(key[level])
            if not pCrawl.children[index]:
                return False
            pCrawl = pCrawl.children[index]

        return pCrawl.isEndOfWord

            # for word in sentence:
            #     if word not in cursor.get_child():
            #         cursor.add_child(word)
            #         if index ==length-1:
            #             self.graph.add_node(sentence[:index+1], children =[word] ,is_end=True, flow = reward)
            #         else:
            #             self.graph.add_node(word,is_end=False, flow = None)
            #     else:
            #         #revise
            #         self.graph.nodes[sentence[:index+1]]['children']=self.graph.nodes[sentence[:index+1]]['children'].append(word)
            # cursor = sentence[:index+1]

     #   self.graph.nodes[current_node]['is_end_of_word'] = True
        # nx.set_node_attributes(G, {0: "red", 1: "blue"}, name="color")
    # def search(self, sentence):
    #     cursor = self.root
    #     for word in sentence:
    #         if not self.graph.has_edge(cursor, word):
    #             return False
    #         cursor = word
    #     return self.graph.nodes[current_node].get('is_end_of_word', False)
    #
    #

    # # Visualization helper using matplotlib
    # def visualize(self):
    #     pos = nx.spring_layout(self.graph)
    #     nx.draw_networkx_nodes(self.graph, pos)
    #     nx.draw_networkx_edges(self.graph, pos)
    #     nx.draw_networkx_edge_labels(self.graph, pos, edge_labels={(u, v): self.graph[u][v]['label'] for u, v in self.graph.edges()})
    #     nx.draw_networkx_labels(self.graph, pos)


