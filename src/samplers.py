from gfn.samplers import Sampler
from gfn.states import States
from gfn.env import Env
import numpy as np
from src.utils import translator

trans = translator()

class States_triv(States):
    def __init__(self,tensor):
        self.tensor = tensor

class extrapolate_Policy(Sampler):
    def __init__(
            self,
            memory,
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
        dist = self.exploration_strategy(state, env)
        action = np.random.multinomial(1, dist)
        return action


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
        state_reward = env.log_reward(trans.translate(state_representation, self.memory,env))   # For each environment we need to define processor(temporary) that converts from Trie state to torch.gfn state
        state_children = state.children
        if state.flow:
            average_flow = state.flow/len(list(filter(lambda x:x is not None, state_children)))
        else:
            average_flow = 1
            assert state.children == None
        state_flows = []
        for child in state_children:
            if child:
                state_flows.append(child.flow + child.curiosity_budget)
            else:
                flow = average_flow
                # the problem is if a state is not visited, how could it have curiosity budget?
                # are we also considering curiosity budget of visited states(how to effectively scale w.r.t to time and childrens).
                #Notice if we have curiosity budget, the average flow of visited states is larger than non-visited states
                state_flows.append(flow)
        z = sum(state_flows) + state_reward
        prob = state_flows/z

        #we need to define curiosity for each state
        return prob





    def sample_trajectories(
            self,
            env: Env,
            states,
            n_trajectories,
    ):
        '''
        :param env:
        :param off_policy:
        :param states:
        :param n_trajectories:
        :return
        '''
        trajectories = []
        for i in range(n_trajectories):
            trajectory = []
            state = states[i]
            while(True):
                action = self.sample_actions(env,state)
                state = env.step(action) # need to specify how to communicate with environment
                if state:
                    trajectory.append(state)
                else:
                    break
            trajectories.append(trajectory)
        return trajectories



