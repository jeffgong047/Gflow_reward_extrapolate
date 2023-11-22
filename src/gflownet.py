# from transformer import transformer
from src.preprocessors import Word
from src.utils import Trie, Trie_node
import random
from abc import ABC


class Gflow_node(Trie_node):
    def __init__(self,vocab_size):
        super().__init__(vocab_size)
        self.flow = None

class Gflow_Trie(Trie):
    def __init__(self,vocab_size):
        # self.vocab = vocab
        self.vocab_size = vocab_size
        self.root = self.getNode()


    def getNode(self):
        return Gflow_node(self.vocab_size)

    # def _charToIndex(self,ch):
    #
    #     # private helper function
    #     # Converts key current character into index
    #     # use only 'a' through 'z' and lower case
    #     if ch in self.vocab:
    #         index = self.vocab[ch]
    #     else:
    #         raise Exception('The word is not in the vocabulary')
    #     return index

    def get_children(self,node):
        return node.children

    def get_root(self):
        return self.root

    def insert(self,sample):
        '''
        The inheritance is difficult in this case
        :param sample:
        :return:
        '''
        key = sample[0]
        reward = sample[1]
        # If not present, inserts key into trie
        # If the key is prefix of trie node,
        # just marks leaf node
        pCrawl = self.root
        length = len(key)
        for level in range(length):
            index = key[level]
            # if current character is not present
            if not pCrawl.children[index]:
                pCrawl.children[index] = self.getNode()
            pCrawl = pCrawl.children[index]
        # mark last node as leaf
        pCrawl.isEndOfWord = True
        pCrawl.flow = reward

    def get_edge_flow(self,source,target):
        state = [source,target]
        return self.get_state_flow(state)


    def get_state_flow(self, state):
        key = state
        pCrawl = self.root
        length = len(key)
        for level in range(length):
            index = key[level]
            if not pCrawl.children[index]:
                raise Exception('The edge does not exist in Gflownet sample structure')
            pCrawl = pCrawl.children[index]
        return pCrawl.flow

class Gflow_extrapolate(ABC):
    def __init__(self,args,word,annotated_samples):
        '''
        Mainly contains two orgain: 1. data structure to hold all the structured edge flows 2. Transformer that can utilize the edge flows to extrapolate
        Algorithms to calculate edge flows and train the transformer to fit the flows
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
        self.word = word
        self.samples = annotated_samples
        self.samples_structure = self.build_samples_structures()

    def predict_edge_flow(self,source_state, target_state):
        indexes = self.word.vocabulary([source_state,target_state])
        embedding = self.word.embed(indexes)
        return self.transformer(embedding)

    def build_samples_structures(self):
        gflow_trie = Gflow_Trie(len(self.word.vocabulary))
        for s in zip(self.samples['object'],self.samples['rewards']):
            gflow_trie.insert(s)
        return gflow_trie
    def backward_reward_propagation(self, cursor = None, propogate_rule='min_entropy'):
        '''
        We use dynamic programming to get an estimate of edge flows between states
        Notice edge flows can be recursively decomposed into sub-edge flows.
        Given Trie implementation, backward_reward_propagation is an algorithm to build a Trie (medium of Gflow Network)
        '''
        if cursor.end_of_Sentence:
            if current_node.flows != None:
                raise Exception('The sample is incomplete because it has no reward')
            return current_node.flows
        else:
            childrens = self.samples_structure.get_children(cursor)
            flow = 0
            for child in childrens:
                if child.flow ==None:
                    child.flow = self.backward_reward_propagation(child,propogate_rule='min_entropy')
                flow += child.flow

        #
        # if len(samples)==0:
        #     #??? need to handle broadcasting here
        #     return samples['reward']
        # reacheable_states = next_states(state)
        # if not self.states_flows_dict[state]:
        #     self.states_flows_dict[state] = 0
        #     for n_state in reacheable_states:
        #         if not self.edge_flows_dict[edge(state,n_state)]:
        #             if propogate_rule =='min_entropy':
        #                 p =  random.random()
        #                 if p>threshold:
        #                     self.states_flows_dict += self.backward_reward_propagation(samples[1:])
        #             elif propogate_rule == 'maximum_entropy':
        #                 self.states_flows_dict += self.backward_reward_propagation(samples[1:])


    def loss(self, params, target_params, samples):
        vmodel = vmap(self.model.apply, in_axes=(None, 0, 0))
        log_pi_t = vmodel(params, samples['adjacency'], samples['mask'])
        log_pi_tp1 = vmodel(
            target_params,
            samples['next_adjacency'],
            samples['next_mask']
        )

        return detailed_balance_loss(
            log_pi_t,
            log_pi_tp1,
            samples['actions'],
            samples['delta_scores'],
            samples['num_edges'],
            delta=self.delta
        )


    def fit_edge_flow(self,params, state, samples):
        grads, logs = grad(self.loss, has_aux=True)(
                params.online,
                params.target,
                samples
            )

        # Update the online params
        updates, opt_state = self.optimizer.update(
            grads,
            state.optimizer,
            params.online
        )
        state = DAGGFlowNetState(optimizer=opt_state, steps=state.steps + 1)
        online_params = optax.apply_updates(params.online, updates)

        # Update the target params periodically
        params = DAGGFlowNetParameters(
            online=online_params,
            target=optax.periodic_update(
                online_params,
                params.target,
                state.steps,
                self.update_target_every
            ),
        )

        return (params, state, logs)

    # def min_entropy(self,samples):
    #     self.env_structure[]
    #
    # def max_entropy(self,samples):
    #     pass

