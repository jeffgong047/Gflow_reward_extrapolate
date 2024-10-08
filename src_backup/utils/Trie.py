import networkx as nx

class Trie:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.root = "ROOT"
        self.graph.add_node(self.root)

    def insert(self, word, weight=1):
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
