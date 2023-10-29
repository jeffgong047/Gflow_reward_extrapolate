from transformer import transformer
class Gflow_extrapolate():
    def __init__(self,vocabulary,reward):
        '''

        :param vocabulary:
        :param reward: A dictionary containing all objects in string format and corresponding reward. e.g 1325:50
        Notes:
        Even though reward and state flows shares same set of keys i.e. the interpretation of the key should be different for subject of matter.
        For edge flows, the ordering of the vocabularies in the key indicates the ordering of actions to generate the object, and last vocabulary should be
        annoated with ~ imaginary to indicates flow ends with that action.

        If we dont care about the ordering in the key. We can just sort it such that duplication arises from combinations from odering is eliminated.
        We might want to index all states with a dictionary.

        Need to decide whether use backward_propagation for edge flows or state flows. We can store states in a trie
        '''
        self.edge_flows_dict = {combinations_choose2(vocabulary):None}
        self.states_flows_dict = {combinations(vocabulary):None}
        self._rewards = reward
    def predict_edge_flow(self,source_state, target_state):
        pass 
    
    def backward_reward_propagation(self,state):
        '''
        We use dynamic programming to get an estimate of edge flows between states
        Notice edge flows can be recursively decomposed into sub-edge flows.
        '''
        reacheable_states = next_states(state)
        if not self.states_flows_dict[state]:
            self.states_flows_dict[state] = 0
            for n_state in reacheable_states:
                if not self.edge_flows_dict[edge(state,n_state)]:
                    self.states_flows_dict +=



    def fit_edge_flow(self):
        pass

    def min_entropy(self):
        pass

    def max_entropy(self):
        pass

