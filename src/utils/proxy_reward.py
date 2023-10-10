import numpy as np
import pandas as pd

import pandas as pd
import networkx as nx
from scipy.special import gammaln
import networkx

def bdeu_score(data: pd.DataFrame, graph: nx.DiGraph, alpha: float) -> float:
    """
    Compute the BDeu score for a Bayesian network structure.

    Args:
    - data (pd.DataFrame): The dataset.
    - graph (nx.DiGraph): The Bayesian network structure.
    - alpha (float): Equivalent sample size (hyperparameter).

    Returns:
    - float: The BDeu score.
    """
    score = 0

    for node in graph.nodes():
        parents = list(graph.predecessors(node))
        states = data[node].unique()

        if parents:
            parent_states = [data[parent].unique() for parent in parents]
            for state in states:
                for parent_combination in pd.MultiIndex.from_product(parent_states):
                    mask = (data[node] == state) & all(data[parent] == parent_val for parent, parent_val in zip(parents, parent_combination))
                    N_ij = mask.sum()
                    N_ijk = len(parents) * [0]

                    score += gammaln(alpha / len(states)) - gammaln(N_ij + alpha / len(states))
                    for state_k, N_ijk_val in enumerate(N_ijk):
                        score += gammaln(N_ijk_val + alpha / (len(parents) * len(states))) - gammaln(alpha / (len(parents) * len(states)))
        else:
            for state in states:
                N_i = (data[node] == state).sum()
                score += gammaln(alpha) - gammaln(N_i + alpha)

    return score

# Example usage:
# data = pd.DataFrame({'A': ..., 'B': ..., 'C': ...})  # your data here
# G = nx.DiGraph()
# G.add_edges_from([('A', 'B'), ('B', 'C')])  # example structure
# print(compute_bdeu_score(data, G, alpha=1))
