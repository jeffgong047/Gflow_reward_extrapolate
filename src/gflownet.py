# from transformer import transformer
import torch
from src.preprocessors import Word
from src.utils import Trie, Trie_node
import random
from abc import ABC
from sklearn.metrics import mean_squared_error, log_loss
import optax
from torch import optim
import torch.nn as nn
import torch.optim as optim
from torch.nn import Flatten
import matplotlib.pyplot as plt
from gfn.gflownet.base import GFlowNet
from gfn.containers import Trajectories
from src.samplers import extrapolate_Policy
from gfn.states import States
import torch
from gfn.env import Env

class Gflow_node(Trie_node):
    def __init__(self,vocab_size,parent= None):
        super().__init__(vocab_size)
        self.flow = None
        self.attempts = 0
        self.parent = parent
        self.curiosity_budget = 0
class Gflow_Trie(Trie):
    def __init__(self,vocab_size):
        # self.vocab = vocab
        self.vocab_size = vocab_size
        self.root = self.getNode()
        self._num_sentences = 0

    @property
    def num_sentences(self):
        return self._num_sentences

    @num_sentences.setter
    def num_sentences(self, value):
        self._num_sentences = value

    def get_state(self,state_representation):
        print(state_representation)
        cursor = self.root
        for i in state_representation:
            if cursor is None:
                cursor.children[i] = self.getNode(cursor)
            cursor = cursor.children[i]
            assert cursor.end_of_Sentence == True
        return cursor

    def get_sentence(self, state):
        '''
        Go from a state to sentence
        :return:
        '''
        sentence = []
        assert state is not None
        parent = state.parent
        child = state
        while parent is not None:
            action = None
            for index, element in parent.children:
                if element == child:
                    action = index
            assert action is not None
            sentence.append(action)
            child = parent
            parent = parent.parent
        return list(reversed(sentence))


    def getNode(self,parent=None):
        return Gflow_node(self.vocab_size,parent)

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
    def get_num_trajectories_from(self,node):
        children = get_children(node)
        if children:
            num_trajectories = 0
            for c in children:
                num_trajectories += self.get_num_trajectories_from(c)
            return num_trajectories
        else:
            return 1

    def get_children(self,node):
        return node.children
    def get_non_visited_children(self,node):
        visited_children = self.get_children(node)
        all_possible_children = node.all_children()
        return list(filter(lambda c: c not in visited_children , all_possible_children))

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
        self.num_sentences += 1
        assert key[-1] == -1 # check if end token exist
        # If not present, inserts key into trie
        # If the key is prefix of trie node,
        # just marks leaf node
        pCrawl = self.root
        length = len(key)
        for level in range(length):
            index = key[level]
            # if current character is not present
            if not pCrawl.children[index]:
                pCrawl.add_child(index, self.getNode(parent=pCrawl))
            pCrawl = pCrawl.children[index]
            pCrawl.attempts +=1
            if index==-1: # handle padding in a lazy way
                break
        # mark last node as leaf
        pCrawl.end_of_Sentence = True  #pCrawl is after taking the end action
        try:
            assert index == -1
        except:
            breakpoint()
            a=1
            b=2
            c=a+b
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


    def get_All_states_flow(self):
        '''
        This algorithm returns topological sorted state flow values of nodes in the trie and corresponding ordered state features
        :return:
        '''
        flows = []
        # states = []
        queue = []
        queue.append(self.root)
        while(queue):
            node = queue.pop()
            flows.append(node.flow if node else None)
            # states.append(node)
            if node:
                children = node.children
                for i in range(self.vocab_size):
                    child = children[i]
                    queue.append(child)
            else:
                flows.append(None)
        # features = []
        # breakpoint()
        # for s in states:
        #     s_feature = []
        #     parent = s.parent
        #     while parent:
        #         assert s in self.get_children(parent) # why we cant use the node to find its children without the trie?
        #         s_feature.append(self.get_children(parent).index(s))
        #         parent  = parent.parent
        #     features.append(s_feature.reverse())
        return  flows

    def get_All_edge_flows(self):
        '''
        use topological sort to enumerate over all edge features, and compute edge flows accordingly
        :return:
        '''
        edge_flows = []
        edge_features = []
        queue = []
        queue.append(([],self.root))
        while(queue):
            parent = queue.pop()
            for index, child in enumerate(parent[1].children):
                if child:
                    #get all the edges
                    edges  = parent[0]+ [index]
                    edge_features.append(edges)
                    edge_flows.append(child.flow)
                    queue.append((edges,child))
                    # maps them into embeddings
        return edge_flows, edge_features

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(-1)

class Gflow_extrapolate(GFlowNet):
    def __init__(self,args,word=None,annotated_samples=None):
        '''
        Mainly contains two organ: 1. data structure to hold all the structured edge flows 2. Transformer that can utilize the edge flows to extrapolate
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
        super().__init__()
        self.word = word
        self.samples = annotated_samples
        self.samples_structure = self.preload_samples_structures()
        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=args.embedding_dim, nhead=8)
        transformer_encoder = nn.TransformerEncoder(transformer_encoder_layer, num_layers=2)
        linear = nn.Linear(args.embedding_dim, 1)
        flatten = nn.Flatten()
        additional_linear = nn.Linear(len(self.word.elements), 1)
        self.is_backward = False
        # Create a sequential model
        self.predictor = nn.Sequential(self.word.embedding,transformer_encoder, linear,flatten,additional_linear)
                                       # transformer_encoder, linear)

        self.optimizer = optim.Adam(self.predictor.parameters(), lr=0.001)
        self.loss_function = nn.MSELoss()
        # print(self.samples)
    #    self.transformer = transformer(word)
    #     self.optimizer = optim.Adam([var1, var2], lr=0.0001)

    def to_states(self,state):
        '''

        :param state: Convert from gflow++ states representation in Trie to torch.gfn States representation
        :return:
        '''
        sentence = self.samples_structure.get_sentence(state)

        return States(torch.tensor(sentence).unsqueeze(0))

    def __call__(self,*args,**kwargs):
        '''
        :param args:
        :param kwargs:
        :return: of shape (batch_size, action_cardinality)
        '''
        return sample(args,kwargs)

    def dynamic_insert(self,samples):
        '''
        This insertion will update the curiosity budget according to the time xxx
        :param sample:
        :return:
        '''
        breakpoint()
        for s in samples:
            self.samples_structure.insert(s)
            self.sample_wise_backward_reward_propagation(s)

    # Define some GFlowNet methods required by torchgfn
    def sample_trajectories(
            self, env: Env, n_samples: int
    ):
        '''
        This sampling can improve the knowledge in the sampler ;
        :param env:
        :param n_samples:
        :param sample_off_policy:
        :return:
        '''
        sampler = extrapolate_Policy(self.samples_structure)
        trajectories = sampler.sample_trajectories(env,[self.samples_structure.root]*n_samples,n_samples) # gflow++ provides only on policy
        return trajectories
        # first sample trajectories using current learnt gflownet(trie)

        #put these trajectories into Trajectories , padding

    def to_training_samples(self, trajectories):
        """Converts trajectories to training samples. The type depends on the GFlowNet."""
        return trajectories

    def loss(self, env: Env, training_objects):
        """Computes the loss given the training objects."""

        def loss(self,  samples):
            log_pi_predicted = self.predict_edge_flow(samples)
            log_pi_ground_truth = self.get_edge_flow(samples)

            return mean_squared_error(
            log_pi_predicted,
            log_pi_ground_truth
        )
        return loss(env, training_objects)



    def sample_wise_backward_reward_propagation(self, state,flows,path):
        '''
        Important assumption:
        We assume this function is called after and only after the insert function therefore we do not update node statistics like attempts(updated by the insert)

        This function could calculate curiosity for visited states and moreover,
        For unvisited states, the curiosity could be defined by (actual flow - expected flow) and we put it through exponential function
        For visited states, the curiosity should fade away as the states has been visited multiple times(sine we know a good deal out of it and want to move on. ),
         but sometimes we want to explore more.

        Another issue of this function is that, we do not give the active attempt full credits. (The credits should be shared exclusively to the states computed by the active explored path)

        we could allow it to activate forward sampling if it found something spicy (Later)
        :param state:
        :param flows:
        :param path: if path is provided, the reward will propagated to states over that path and curiosity budget will be updated over that path
        :return:
        '''
        curiosity_budget = lambda attempt,deviation:max(0, deviation**2/math.sqrt(attempt))
        # surprise_threshold = lambda x:somefunction(state_flows)  lets dont consider activation of this feature
        if path:
            parents = state.get_parent()
            while parents:
                propagated_flows = flows/len(parents) # we use the maximum entropy criterion here
                for p in parents:
                    if p.flow:
                        p.flow += propagated_flows
                    else:
                        p.flow += propagated_flows

                        p.curiosity_budget = curiosity_budget(attempt,propagated_flows - p.parent.flow/len(p.parent.children())) #how to scale the curiosity budget? curiosity(t,attempts,surprise)
                        # if p.curiosity_budget > surprise_threshold(p): # null model does not need this part...
                        #     self.sample(state)  # we need to handle multi-threading here
                    self.sample_wise_backward_reward_propagation(p, propagated_flows)  # might need to add dynamic programming here
        else:
            back_ward_path = reversed(path)
            for s in back_ward_path:
                s.flow += flows
                s.curiosity_budget = curiosity_budget(attempt,propagated_flows - p.parent.flow/len(p.parent.children()))
                # if p.curiosity_budget > suprise_threshold(p):
                #     self.sample(state)


    def sample(self, states):
        '''
        :param states:
        :return: (batch_size, action_cardinality)
        with trie implementation, we can not do broadcasting here.
        '''
    #    out = self.module(self.preprocessor(states))
        preprossed = self.preprocessor(states)
        child_flows =[]
        for t in preprossed:
            child_flow = self.get_child_flows(t)
            child_flows.append(child_flow)
        # if not self._output_dim_is_checked:
        #     self.check_output_dim(out)
        #     self._output_dim_is_checked = True
        return child_flows


    # def to_probability_distribution(
    #         self,
    #         states: DiscreteStates,
    #         module_output: TT["batch_shape", "output_dim", float],
    #         temperature: float = 1.0,
    #         sf_bias: float = 0.0,
    #         epsilon: float = 0.0,
    # ) -> Categorical:
    #     """Returns a probability distribution given a batch of states and module output.
    #
    #     Args:
    #         temperature: scalar to divide the logits by before softmax. Does nothing
    #             if set to 1.0 (default), in which case it's on policy.
    #         sf_bias: scalar to subtract from the exit action logit before dividing by
    #             temperature. Does nothing if set to 0.0 (default), in which case it's
    #             on policy.
    #         epsilon: with probability epsilon, a random action is chosen. Does nothing
    #             if set to 0.0 (default), in which case it's on policy."""
    #     masks = states.backward_masks if self.is_backward else states.forward_masks
    #     logits = module_output
    #     logits[~masks] = -float("inf")
    #
    #     # Forward policy supports exploration in many implementations.
    #     if temperature != 1.0 or sf_bias != 0.0 or epsilon != 0.0:
    #         logits[:, -1] -= sf_bias
    #         probs = torch.softmax(logits / temperature, dim=-1)
    #         uniform_dist_probs = masks.float() / masks.sum(dim=-1, keepdim=True)
    #         probs = (1 - epsilon) * probs + epsilon * uniform_dist_probs
    #
    #         return UnsqueezedCategorical(probs=probs)
    #
    #     # LogEdgeFlows are greedy, as are more P_B.
    #     else:
    #         return UnsqueezedCategorical(logits=logits)

    def get_root(self):
        return self.samples_structure.root

    def get_states_flows(self):
        return self.samples_structure.get_All_states_flow()

    def get_child_flows(self,node):
        child = self.samples_structure.get_children(node)
        return [c.flow for c in child]

    def get_edges_flows(self):
        return self.samples_structure.get_All_edge_flows()

    def preload_samples_structures(self):
        gflow_trie = Gflow_Trie(len(self.word.vocabulary))
        if self.samples:
            for s in zip(self.samples['object'],self.samples['rewards']):
                gflow_trie.insert(s)
        return gflow_trie

    def backward_reward_propagation(self, cursor = None, propogate_rule='max_entropy'):
        '''
        We use dynamic programming to get an estimate of edge flows between states
        Notice edge flows can be recursively decomposed into sub-edge flows.
        Given Trie implementation, backward_reward_propagation is an algorithm to build a Trie (medium of Gflow Network)
        #Make this dynamic programming and states based.
        '''
        if cursor.end_of_Sentence:
            if current_node.flows == None:
                raise Exception('The sample is incomplete because it has no reward')
            return current_node.flows
        else:
            assert cursor.flow ==None
            childrens = self.samples_structure.get_children(cursor)
            flow = 0
            for index, child in enumerate(childrens):
                end_index = len(self.word.vocabulary)-1 # assumes 'end' is the last token
                #update this node's flow by recursively backpropagate flows from children's node
                if child:
                    if child.flow ==None:
                        child.flow = self.backward_reward_propagation(child,propogate_rule='max_entropy')
                    flow += child.flow
            cursor.flow = flow
            return flow


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



    # def sample_actions(self, params, key, observations, epsilon):
    #     masks = observations['mask'].astype(jnp.float32)
    #     adjacencies = observations['adjacency'].astype(jnp.float32)
    #     batch_size = adjacencies.shape[0]
    #     key, subkey1, subkey2 = random.split(key, 3)
    #
    #     # Get the GFlowNet policy
    #     log_pi = vmap(self.model.apply, in_axes=(None, 0, 0))(
    #         params,
    #         adjacencies,
    #         masks
    #     )
    #
    #     # Get uniform policy
    #     log_uniform = uniform_log_policy(masks)
    #
    #     # Mixture of GFlowNet policy and uniform policy
    #     is_exploration = random.bernoulli(
    #         subkey1, p=1. - epsilon, shape=(batch_size, 1))
    #     log_pi = jnp.where(is_exploration, log_uniform, log_pi)
    #
    #     # Sample actions
    #     actions = batch_random_choice(subkey2, jnp.exp(log_pi), masks)
    #
    #     logs = {
    #         'is_exploration': is_exploration.astype(jnp.int32),
    #     }
    #     return (actions, key, logs)

    def train(self,  epochs, data_loader):
        for epoch in range(epochs):
            self.predictor.train()
            total_loss = 0
            for inputs, targets in data_loader:
                # Prepare your data here (e.g., tokenize, convert to tensor)
                targets  = targets.to(torch.float32)
                # Forward pass
                predictions = self.predictor(inputs).squeeze(-1)
                loss = self.loss_function(predictions, targets)
                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(data_loader.dataset)
            print(f"Epoch {epoch+1}, Loss: {avg_loss}")
            # self.validate()

    # def validate(self):
    #     self.predictor.eval()
    #     total_loss = 0
    #     with torch.no_grad():
    #         for data in self.val_data:
    #             inputs, targets = self.prepare_data(data)
    #             predictions = self.transformer(inputs)
    #             loss = self.loss_function(predictions, targets)
    #             total_loss += loss.item()
    #
    #     avg_loss = total_loss / len(self.val_data)
    #     print(f"Validation Loss: {avg_loss}"s)

    def test(self,data_loader):
        self.predictor.eval()
        predictions = []
        targets = []
        with torch.no_grad():
            for input, target in data_loader:
                prediction = self.predictor(input)
                predictions.append(prediction)
                targets.append(target)

        # Visualization or further analysis
        return [predictions, targets]





    def get_edge_flow(self,source,target):
        return self.samples_structure.get_edge_flow(source,target)



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


    def predict_edge_flow(self,source_state, target_state):
        indexes = self.word.vocabulary([source_state,target_state])
        embedding = self.word.embed(indexes)
        return self.predictor(embedding)


    # def min_entropy(self,samples):
    #     self.env_structure[]
    #
    # def max_entropy(self,samples):
    #     pass

