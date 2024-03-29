import sys
import random
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from pathlib import Path
import json
from src.gflownet import Gflow_extrapolate
from src.preprocessors import Word
import pandas as pd
from src.utils.data import get_data
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, random_split
from gfn.gym import DiscreteEBM
from gfn.gym import HyperGrid
import torch
import numpy as np
from src.utils import translator
import itertools

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def enumerate_Ising_states(ndim):  # equivalent to build_grid function for hypergrid experiment setting
    items = [0,1]

    all_states = list(itertools.product(items,repeat = ndim))
    return [s + (-1,) for s in all_states]


def visualize_prediction_accuracy(predictions, targets):
    plt.figure(figsize=(10, 6))

    # Scatter plot of predictions vs targets
    plt.scatter(targets, predictions, alpha=0.5)

    # Line for perfect predictions
    plt.plot(targets, targets, color='red', label='Perfect Predictions')

    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('Predictions vs True Values')
    plt.legend()
    plt.savefig('./prediction_accuracy.png')
    plt.show()

def update_dict_frequency(dict, keys):
    for k in keys:
        k = tuple(k)
        if k in dict.keys():
            dict[k] +=1
        else:
            dict[k] = 1


def main(args):
    #basic test on signaling pathway dataset
    # scorer,evidences = get_data(args)
    # args.scorer = scorer
    # word = Word(args,evidences)
    # #1)sample some flows and use backward decomposition to reconstruct the flows and check all flows reproduced exactly and and all other flows are zero
    # annotated_samples = word.rawData_to_samples(args.scorer)
    # # print(list(zip(annotated_samples['object'],annotated_samples['rewards'])))
    # random_samples_with_replacement = random.choices(list(zip(annotated_samples['object'],annotated_samples['rewards'])), k=3)
    # selected_samples = {'object':[],'rewards':[]}
    # for s in random_samples_with_replacement:
    #     selected_samples['object'].append(s[0])
    #     selected_samples['rewards'].append(s[1])
    # print('selected_samples are: ', selected_samples)
    # Gflownet = Gflow_extrapolate(args,word,selected_samples)
    # root = Gflownet.get_root()
    # Gflownet.backward_reward_propagation(root)
    # states_flows = Gflownet.get_states_flows()
    # #sanity test of the fitted model
    # print('based on the state_flows, total reward is: ', states_flows[0],'compared to ground truth' ,sum(selected_samples['rewards']))
    # print('all state flows are: ',states_flows)
    # edge_flows,edge_features = Gflownet.get_edges_flows()
    # # print(edge_flows)
    # scale = -edge_flows[0]
    # edge_flows = [f+scale for f in edge_flows]


    # we next check the statistics to ensure that dynamic insert works
    # In hyper-grid show that backward_propagation works
    # env = HyperGrid(
    #     2, 8, 0.1, 0.5, 2, reward_cos = True,device_str=0
    # )
    # evidences = {'full_samples':None,'elements':list(range(env.n_actions-1))}
    # word = Word(args, evidences)
    # selected_samples = [[0,1,-1,-1], [0,-1,-1,-1],[1,-1,-1,-1],[1,1,-1,-1],[1,0,1,-1]]
    # trans = translator()
    # selected_samples_states = trans.translate(selected_samples)
    # selected_samples_rewards = env.log_reward(final_states = selected_samples_states)
    # annotated_selected_samples = {'object':selected_samples, 'rewards': selected_samples_rewards}
    # Gflownet_active_learning_hypergrid = Gflow_extrapolate(args, word, annotated_selected_samples, control = args.control) # the selected samples are pre-learnt
    # Gflownet_active_learning_hypergrid.backward_reward_propagation(Gflownet_active_learning_hypergrid.get_root()) # if the initialization is with samples, backward propagation is required
    # trajectories = Gflownet_active_learning_hypergrid.sample_trajectories(n_samples = 10, env= env)
    # print('sampled trajectories are: ', trajectories)
    # trajectories_rewards = env.log_reward(final_states = trans.translate(trajectories))
    # print('corresponding rewards of these trajectories are: ', trajectories_rewards)
    # annotated_trajectories = {'object': trajectories, 'rewards': trajectories_rewards}
    # Gflownet_active_learning_hypergrid.dynamic_insert(annotated_trajectories)
    # print('lets take a look at top 3 values')
    # Gflownet_active_learning_hypergrid.samples_structure.top_flows.peek_top_n(3)
    # print(Gflownet_active_learning_hypergrid.get_states_flows())
    # breakpoint()
    #2)try Ising models
    all_sampled_trajectories = {}
    explored_states_n_rewards = {}
    env_Ising = DiscreteEBM(ndim=10, alpha=1.0, device_str=0)
    all_states = enumerate_Ising_states(env_Ising.ndim)
    trans = translator()
    all_states_rewards = env_Ising.log_reward(trans.translate(all_states, env= env_Ising))
    print( all_states_rewards )
    evidences = {'full_samples':None,'elements':list(range(2))} # Notice, if we do not care permutation of Ising model's states, we only need 2 actions
    word = Word(args, evidences) #initialization of the word object requires elements and end element have to be specified as 'end'
    selected_samples =  [[0,1,0,1,1,1,1,1,1,1,-1], [0,1,1,0,0,0,0,1,1,1,-1], [0,0,1,1,0,1,1,0,1,1,-1], [1,1,0,0,0,0,1,0,1,1,-1], [0,0,1,1,0,1,0,0,1,1,-1]]
    selected_samples_states = trans.translate(selected_samples,env= env_Ising)
    selected_samples_rewards = env_Ising.log_reward(final_states = selected_samples_states)
    breakpoint()
    # print('The reward of the root state is: ', env_Ising.log_reward(trans.translate([],env=env_Ising)))
    print('the rewards of known trajectories are: ', selected_samples_rewards)
    explored_states_n_rewards.update({tuple(s): r.item() for s , r in zip(selected_samples,selected_samples_rewards)})
    update_dict_frequency(all_sampled_trajectories,selected_samples)
    annotated_selected_samples = {'object':selected_samples, 'rewards': selected_samples_rewards}
    Gflownet_active_learning_Ising = Gflow_extrapolate(args, word, annotated_selected_samples) # the selected samples are pre-learnt
    Gflownet_active_learning_Ising.backward_reward_propagation(Gflownet_active_learning_Ising.get_root()) # if the initialization is with samples, backward propagation is required
    print('total unique trajectories is: ', len(explored_states_n_rewards), 'and it equals to???', Gflownet_active_learning_Ising.samples_structure.num_sentences, ' ???')
    print(Gflownet_active_learning_Ising.samples_structure.top_flows.peek_top_n(3))
    print(Gflownet_active_learning_Ising.get_states_flows())
    reward_intensity = []
    for i in range(18):
        print('check all the states flows: ', Gflownet_active_learning_Ising.samples_structure.get_All_states_flow(leaf_only=True))
        if len(explored_states_n_rewards) != Gflownet_active_learning_Ising.samples_structure.num_sentences:
            print('number of uniquely explored states ', len(explored_states_n_rewards), 'number of sentences recorded by Gflow++: ',Gflownet_active_learning_Ising.samples_structure.num_sentences)
            print('all of the sampled trajectories ', all_sampled_trajectories)
            breakpoint()
        trajectories = Gflownet_active_learning_Ising.sample_trajectories(n_samples = 5, env= env_Ising)
        print('sampled trajectories are: ', trajectories)
        translated_trajectories = trans.translate(trajectories)
        trajectories_rewards = env_Ising.log_reward(final_states = translated_trajectories)
        reward_intensity.append(sum(trajectories_rewards)/len(trajectories_rewards))
        annotated_trajectories = {'object': trajectories, 'rewards': trajectories_rewards}
        Gflownet_active_learning_Ising.dynamic_insert(annotated_trajectories)
        print('corresponding rewards of these trajectories are: ', trajectories_rewards)
        print('lets take a look at top 3 values')
        print('Total explored states are: ', Gflownet_active_learning_Ising.samples_structure.num_sentences)
        print(Gflownet_active_learning_Ising.samples_structure.top_flows.peek_top_n(10))
        print(Gflownet_active_learning_Ising.get_states_flows())
        explored_states_n_rewards.update({tuple(s): r.item() for s , r in zip(trajectories, trajectories_rewards)})
        update_dict_frequency(all_sampled_trajectories,trajectories)
        print('total unique trajectories is: ', len(explored_states_n_rewards), 'and it equals to???', Gflownet_active_learning_Ising.samples_structure.num_sentences, ' ???')
    total_reward = 0
    for v in explored_states_n_rewards.values():
        total_reward += v
    print('total reward is : ',total_reward, 'and it equals to???: ',Gflownet_active_learning_Ising.samples_structure.root.flow, ' ???')
    print('total unique trajectories is: ', len(explored_states_n_rewards), 'and it equals to???', Gflownet_active_learning_Ising.samples_structure.num_sentences, ' ???')
    print('top 3 flows :', Gflownet_active_learning_Ising.samples_structure.top_flows.peek_top_n(3))
    print('average reward is : ', total_reward/len(explored_states_n_rewards))
    print('check all the leaf flows: ', Gflownet_active_learning_Ising.samples_structure.get_All_states_flow(leaf_only=True))
    print('reward intensity', reward_intensity)
   # Supervised learning setting
    breakpoint()
    breakpoint()
    #we can improve upon the sampling of ground truths
    train_data_flows, test_data_flows = train_test_split(list(zip(edge_features,edge_flows)),test_size = 0.2)
    flows, rewards = [[d[0] for d in train_data_flows], [d[1] for d in train_data_flows]]
    # Example data



# Create and load custom dataset
# Create an instance of CustomDataset
    padded_train_data_flows = word.pad(flows)
    dataset = CustomDataset(padded_train_data_flows, rewards)

    # Specify the lengths for train and test sets
    train_size = int(0.8 * len(dataset))  # e.g., 80% of the dataset for training
    test_size = len(dataset) - train_size  # The rest for testing

    # Split the dataset
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create data loaders for train and test sets
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    Gflownet.train(100,train_loader)
    predictions, targets = Gflownet.test(test_loader)
    breakpoint()
    visualize_prediction_accuracy(predictions,targets)





if __name__ == '__main__':


    parser = ArgumentParser(description='DAG-GFlowNet for Strucure Learning.')

    # Environment
    environment = parser.add_argument_group('Environment')
    environment.add_argument('--num_envs', type=int, default=8,
                             help='Number of parallel environments (default: %(default)s)')
    environment.add_argument('--scorer', type=str, default='',
                             help='name of the scorer.')
    environment.add_argument('--prior', type=str, default='uniform',
                             choices=['uniform', 'erdos_renyi', 'edge', 'fair'],
                             help='Prior over graphs (default: %(default)s)')
    environment.add_argument('--prior_kwargs', type=json.loads, default='{}',
                             help='Arguments of the prior over graphs.')

    # Optimization
    optimization = parser.add_argument_group('Optimization')
    optimization.add_argument('--lr', type=float, default=1e-5,
                              help='Learning rate (default: %(default)s)')
    optimization.add_argument('--delta', type=float, default=1.,
                              help='Value of delta for Huber loss (default: %(default)s)')
    optimization.add_argument('--batch_size', type=int, default=32,
                              help='Batch size (default: %(default)s)')
    optimization.add_argument('--num_iterations', type=int, default=100_000,
                              help='Number of iterations (default: %(default)s)')

    optimization.add_argument('--control', type = str, default = 'likelihood', help = 'control strategy')

    # Replay buffer
    replay = parser.add_argument_group('Replay Buffer')
    replay.add_argument('--replay_capacity', type=int, default=100_000,
                        help='Capacity of the replay buffer (default: %(default)s)')
    replay.add_argument('--prefill', type=int, default=1000,
                        help='Number of iterations with a random policy to prefill '
                             'the replay buffer (default: %(default)s)')

    #Neural Network
    neural_network = parser.add_argument_group('Neural Network')
    neural_network.add_argument('--embedding_dim', type= int, default=256, help='word embedding dimension')

    # Exploration
    exploration = parser.add_argument_group('Exploration')
    exploration.add_argument('--min_exploration', type=float, default=0.1,
                             help='Minimum value of epsilon-exploration (default: %(default)s)')
    exploration.add_argument('--update_epsilon_every', type=int, default=10,
                             help='Frequency of update for epsilon (default: %(default)s)')

    # Miscellaneous
    misc = parser.add_argument_group('Miscellaneous')
    misc.add_argument('--objects',type = str,default='demo2',help = 'name of the object to perform generative modeling')
    misc.add_argument('--num_samples_posterior', type=int, default=1000,
                      help='Number of samples for the posterior estimate (default: %(default)s)')
    misc.add_argument('--update_target_every', type=int, default=1000,
                      help='Frequency of update for the target network (default: %(default)s)')
    misc.add_argument('--seed', type=int, default=0,
                      help='Random seed (default: %(default)s)')
    misc.add_argument('--num_workers', type=int, default=4,
                      help='Number of workers (default: %(default)s)')
    misc.add_argument('--mp_context', type=str, default='spawn',
                      help='Multiprocessing context (default: %(default)s)')
    misc.add_argument('--output_folder', type=Path, default='output',
                      help='Output folder (default: %(default)s)')

    subparsers = parser.add_subparsers(help='Type of graph', dest='graph')
    args = parser.parse_args()
    main(args)