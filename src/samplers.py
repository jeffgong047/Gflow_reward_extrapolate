from gfn.samplers import Sampler
from gfn.states import States
from gfn.env import Env
import numpy as np
from src.utils import translator
import torch
import torch.nn.functional as F
from gfn.gym.discrete_ebm import DiscreteEBM
trans = translator()

class States_triv(States):
    def __init__(self,tensor):
        self.tensor = tensor

class extrapolate_Policy(Sampler):
    def __init__(
            self,
            memory,
            control,
            **probability_distribution_kwargs,
    ) -> None:
        '''
        Sampler encodes the forward wisdom of the model
        :param memory: Gflownet states and flows that are stored in a Trie
        :param probability_distribution_kwargs:
        '''
        self.memory = memory
        self.probability_distribution_kwargs = probability_distribution_kwargs
        self.T= 1
        self.control_protocol = control
    def sample_actions(
            self, env: Env, state
    ):
        """Samples actions from the given states.

        Args:
            env: The environment to sample actions from.
            states (States): A batch of states.

        Returns:
            A tuple of tensors containing:
             - An Actions object containing the sampled actions.
             - A tensor of shape (*batch_shape,) containing the log probabilities of
                the sampled actions under the probability distribution of the given
                states.
        """
        logits = self.exploration_strategy(state,env)
        # for the sake of getting results faster, we manually create a mask here if the environment is Ising
        dont_stop = type(env) == DiscreteEBM and len(self.memory.get_sentence(state))< env.ndim # this should be encapsulated within the environment instead of being specified by sampling function
        if dont_stop:
            logits[-1] = float('-inf')
        dist = F.softmax(logits,dim=0)
        print('distribution is: ', dist)
        action_index = torch.multinomial(dist, num_samples=1, replacement=True).item()
        print('selected action is: ', action_index)
        if dont_stop and action_index == len(dist):
            breakpoint()
        action_index = -1  if action_index +1 == len(dist) else action_index
        print('selected action is: ', action_index)
        return action_index


    def exploration_strategy(self,state ,env):
        '''
        This is a step wise strategy
        :state current state
        :env providing necessary constraints and information about the environment(such as reward)
        :return:
        '''
        # for child with no flows, the bias is that their flow = average flow given that state
        state_representation = self.memory.get_sentence(state)
        print('state representation is: ', state_representation)
        try:
            state_reward = env.log_reward(trans.translate(state_representation, self.memory,env))   # For each environment we need to define processor(temporary) that converts from Trie state to torch.gfn state
        except:
            state_reward = 0
        state_children = state.children
        assert len(state.children) == self.memory.vocab_size
        if state.flow:
            average_flow = state.flow/len(list(filter(lambda x:x is not None, state_children)))
        else:
            average_flow = state_reward
            # assert all(c is None for c in state.children) # exploration beyond state of zero flow is allowed, and without backward reward propagation, flow=0 can not indicates children existence
        # should i update leaf nodes during explorations?
        if state_children[-1] is None:
            state_children[-1] = self.memory.getNode()
            state_children[-1].add_parent(-1,state)
        state_children[-1].flow = state_reward
        state_children[-1].end_of_Sentence = True
        state_flows = []
        for child in state_children[:-1]:
            if child:
                if self.control_protocol == 'curiosity_guided':
                    state_flows.append(child.flow + child.curiosity_budget)
                else:
                    state_flows.append(child.flow)
            else:
                # the problem is if a state is not visited, how could it have curiosity budget?
                # are we also considering curiosity budget of visited states(how to effectively scale w.r.t to time and childrens).
                #Notice if we have curiosity budget, the average flow of visited states is larger than non-visited states
                state_flows.append(average_flow)
        state_flows.append(state_reward)
        z = max(sum(state_flows) + state_reward, 1)
        print('state flows are: ', torch.tensor(state_flows), 'normalizing constatnt is: ', z)
        logits = torch.tensor(state_flows).cuda()/abs(z)
        #we need to define curiosity for each state
        logits = torch.where(torch.isnan(logits), torch.tensor(0.0).cuda(), logits)
        return logits



    def sample_trajectories(
            self,
            env: Env,
            states,
            n_trajectories,
    ):
        '''
        The sampled trajectories should represent the intelligence of the gflownet
        1. If no such knowledge exists in the sampler, sampler trajectory function rely on its exploration strategy
        2. The sampled trajectory can dynamically update parameters of the sampler in return to improve the exploration policy
        :param env:
        :param off_policy:
        :param states:
        :param n_trajectories:
        :return
        '''
        trajectories = []
        for i in range(n_trajectories):
            state = states[i]
            while(True):
                action = self.sample_actions(env,state)
                print('original state is: ', state)
                print('action being taken is: ', action)
                actions = env.step_trie(self.memory.get_sentence(state), action) # need to specify how to communicate with environment
                assert actions is not None
                if actions[-1]!=-1:
                    state = self.memory.get_state(actions[:-1])
                    if state.children[action] is None:
                        state.children[action] = self.memory.getNode() #we could update curiosity budget here
                        state.children[action].add_parent(action,state)
                    state = state.children[action]
                    if action ==-1 :
                        state.end_of_Sentence = True
                    print('new state is: ', state)
                    print('current trajectory is: ', actions)
                else:
                    print('stop current trajectory ...', actions)
                    break
            trajectories.append(actions)
        return trajectories



