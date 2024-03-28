from abc import ABC, abstractmethod

import networkx as nx



import heapq

class MaxHeap:
    def __init__(self,max_size = 64 ):
        self.heap = []
        self.max_size = max_size # Maximum size of the heap

    def push(self, val, obj):
        # Store a tuple of (inverted val, obj) to maintain max heap behavior
        if len(self.heap)+1 == self.max_size:
            self.heap.pop()  #notice heap.pop is totally different to pop
        heapq.heappush(self.heap, (-val, obj))
        # If the heap size exceeds max_size, remove the smallest element


    def pop(self):
        # Invert the value back and return both the value and the object
        val, obj = heapq.heappop(self.heap)
        return -val, obj

    def peek_top_n(self, n=10):
        # Return the top n values and their associated objects without removing them
        return [(-val, obj) for val, obj in heapq.nsmallest(min(n, len(self.heap)), self.heap)]

    def __len__(self):
        return len(self.heap)

class Trie_node():
    def __init__(self,vocab_size):
        assert isinstance(vocab_size,int)
        self._children = [None ]*vocab_size
        self.end_of_Sentence = False
    @property
    def children(self):
        return self._children


    def all_children(self):
        return np.arange(vocab_size)
    def add_child(self,index, node):
        ## TO DO: need to add element at its corresponding index
        self._children[index] = node




# class TrieNode:
#
#     # Trie node class
#     def __init__(self):
#         self.children = [None]*26
#
#         # isEndOfWord is True if node represent the end of the word
#         self.isEndOfWord = False

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


