import sys
import random
import matplotlib.pyplot as plt

print(sys.path)
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

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


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

def main(args):
    #basic test on signaling pathway dataset
    scorer,evidences = get_data(args)
    args.scorer = scorer
    word = Word(args,evidences)
    #1)sample some flows and use backward decomposition to reconstruct the flows and check all flows reproduced exactly and and all other flows are zero
    annotated_samples = word.rawData_to_samples(args.scorer)
    # print(list(zip(annotated_samples['object'],annotated_samples['rewards'])))
    random_samples_with_replacement = random.choices(list(zip(annotated_samples['object'],annotated_samples['rewards'])), k=3)
    selected_samples = {'object':[],'rewards':[]}
    for s in random_samples_with_replacement:
        selected_samples['object'].append(s[0])
        selected_samples['rewards'].append(s[1])
    print('selected_samples are: ', selected_samples)
    Gflownet = Gflow_extrapolate(args,word,selected_samples)
    root = Gflownet.get_root()
    Gflownet.backward_reward_propagation(root)
    states_flows = Gflownet.get_states_flows()
    #sanity test of the fitted model
    print('based on the state_flows, total reward is: ', states_flows[0],'compared to ground truth' ,sum(selected_samples['rewards']))
    print('all state flows are: ',states_flows)
    edge_flows,edge_features = Gflownet.get_edges_flows()
    # print(edge_flows)
    scale = -edge_flows[0]
    edge_flows = [f+scale for f in edge_flows]


    # we next check the statistics to ensure that dynamic insert works
    # In hyper-grid show that backward_propagation works
    env = HyperGrid(
        2, 8, 0.1, 0.5, 2, reward_cos = True,device_str=0
    )
    evidences = {'full_samples':None,'elements':list(range(env.n_actions-1))}
    word = Word(args, evidences)
    selected_samples = [[0,1,-1,-1], [0,-1,-1,-1],[1,-1,-1,-1],[1,1,-1,-1],[1,0,1,-1]]
    trans = translator()
    selected_samples_states = trans.translate(selected_samples)
    selected_samples_rewards = env.log_reward(final_states = selected_samples_states)
    annotated_selected_samples = {'object':selected_samples, 'rewards': selected_samples_rewards}
    Gflownet_active_learning_hypergrid = Gflow_extrapolate(args, word, annotated_selected_samples) # the selected samples are pre-learnt
    Gflownet_active_learning_hypergrid.backward_reward_propagation(Gflownet_active_learning_hypergrid.get_root()) # if the initialization is with samples, backward propagation is required
    trajectories = Gflownet_active_learning_hypergrid.sample_trajectories(n_samples = 10, env= env)
    print('sampled trajectories are: ', trajectories)
    trajectories_rewards = env.log_reward(final_states = trans.translate(trajectories))
    print('corresponding rewards of these trajectories are: ', trajectories_rewards)
    annotated_trajectories = {'object': trajectories, 'rewards': trajectories_rewards}
    Gflownet_active_learning_hypergrid.dynamic_insert(annotated_trajectories)
    print('lets take a look at top 3 values')
    Gflownet_active_learning_hypergrid.samples_structure.top_flows.peek_top_n(3)
    print(Gflownet_active_learning_hypergrid.get_states_flows())
    breakpoint()
    #2)try branching
    env = DiscreteEBM(ndim=args.ndim, alpha=args.alpha, device_str=device_str)
    Gflownet_active_learning = Gflow_extrapolate()
    Gflownet_active_learning.insert(trajectories)
    Gflownet_active_learning.backward_reward_propagation(Gflownet_active_learning.get_root())
    trajectories = Gflownet_active_learning.sample_trajectories(n_samples = 10 , env= env)
    Gflownet_active_learning.dynamic_insert(trajectories)
   # Supervised learning setting
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