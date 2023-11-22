import sys
print(sys.path)
from argparse import ArgumentParser
from pathlib import Path
import json
from src.gflownet import Gflow_extrapolate
from src.preprocessors import Word
import pandas as pd
from src.utils.data import get_data

def main(args):
    scorer,evidence_data ,elements,samples = get_data(args)
    args.scorer = scorer
    word = Word(args,evidence_data,elements,samples)
    a = Gflow_extrapolate(args,word, evidence_data)





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