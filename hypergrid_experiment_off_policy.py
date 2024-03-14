
from argparse import ArgumentParser
import pickle
import torch
import wandb
from tqdm import tqdm, trange

from gfn.containers import ReplayBuffer
from gfn.gflownet import (
    DBGFlowNet,
    FMGFlowNet,
    LogPartitionVarianceGFlowNet,
    ModifiedDBGFlowNet,
    SubTBGFlowNet,
    TBGFlowNet,
)
from gfn.gym import HyperGrid
from gfn.modules import DiscretePolicyEstimator, ScalarEstimator
from gfn.utils.common import validate, visit_distribution
from gfn.utils.modules import DiscreteUniform, NeuralNet, Tabular
from gfn.samplers import Sampler
from gfn.states import States
from gfn.utils.common import get_terminating_state_dist_pmf
torch.cuda.init()

DEFAULT_SEED = 4444


def off_policy_data_generation(protocol,env, pf , pb):
    if protocol =='grid_uni':
        return grid_wise_uniform_initialization(env, pb)
    elif protocol == 'policy_uni':
        return policy_wise_uniform_initialization(env, pf)

def grid_wise_uniform_initialization(env , pb):
    sampler = Sampler(pb)
    #create a tensor that each entry has equal chance of being one or zero. We use a tensor to represent the bernouli probability then sample from it.
    uniform_states = env.uniform_states()
    # print(uniform_states)
    trajectories = sampler.sample_trajectories(env,uniform_states)
    #since the trajectories are backward generated, we need to reverse the trajectory
    # actions_before = trajectories.actions.tensor[:,:3].detach().clone()
    # states_before = trajectories.states.tensor[:,:3].detach().clone()
    trajectories.reverse()
    # actions_after = trajectories.actions.tensor[:,:3].detach().clone()
    # states_after = trajectories.states.tensor[:,:3].detach().clone()
    # for i in range(3):
    #     print('actions_before, index:',i,'\n',actions_before[:,i])
    #     print('states_before, index:',i, '\n' ,states_before[:,i])
    #     print('actions_after, index:',i, '\n'  ,actions_after[:,i])
    #     print('states_after, index:',i, '\n', states_after[:,i])
    # print(trajectories.actions.tensor.shape)
    # print(trajectories.states.tensor.shape)
    return trajectories  # remove the corner case

def policy_wise_uniform_initialization(env ):
    pf_uniform_estimator = DiscretePolicyEstimator(
        module=DiscreteUniform(env.n_actions),
        n_actions=env.n_actions,
        preprocessor=env.preprocessor,
    )
    prob_args = {'sf_bias':3}
    sampler = Sampler(pf_uniform_estimator,**prob_args)
    trajectories = sampler.sample_trajectories(env, n_trajectories=args.training_sample_size,)
    training_samples = gflownet.to_training_samples(trajectories)
    return training_samples


def final_states_trajectory():
    print(training_samples.__dict__['states'].tensor,training_samples.__dict__['states'].tensor.shape)
    trajectories_states = training_samples.__dict__['states'].tensor.detach().clone()
    ending_states_indices = training_samples.__dict__['when_is_done'].detach().clone()-1
    print('ending_states_indices: ', ending_states_indices)
    terminating_states = trajectories_states[ending_states_indices, torch.arange(args.training_sample_size)].reshape(-1,args.ndim)
    return terminating_states


def main(args):
    seed = args.seed if args.seed != 0 else DEFAULT_SEED
    torch.manual_seed(seed)

    device_str = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"

    use_wandb = len(args.wandb_project) > 0
    if use_wandb:
        wandb.init(project=args.wandb_project)
        wandb.config.update(args)

    # 1. Create the environment
    env = HyperGrid(
        args.ndim, args.height, args.R0, args.R1, args.R2, reward_cos = True,device_str=device_str
    )

    # 2. Create the gflownets.
    #    For this we need modules and estimators.
    #    Depending on the loss, we may need several estimators:
    #       one (forward only) for FM loss,
    #       two (forward and backward) or other losses
    #       three (same, + logZ) estimators for TB.
    gflownet = None
    if args.loss == "FM":
        # We need a LogEdgeFlowEstimator
        if args.tabular:
            module = Tabular(n_states=env.n_states, output_dim=env.n_actions)
        else:
            module = NeuralNet(
                input_dim=env.preprocessor.output_dim,
                output_dim=env.n_actions,
                hidden_dim=args.hidden_dim,
                n_hidden_layers=args.n_hidden,
            )
        estimator = DiscretePolicyEstimator(
            module=module,
            n_actions=env.n_actions,
            preprocessor=env.preprocessor,
        )
        gflownet = FMGFlowNet(estimator)
    else:
        pb_module = None
        # We need a DiscretePFEstimator and a DiscretePBEstimator
        if args.tabular:
            pf_module = Tabular(n_states=env.n_states, output_dim=env.n_actions)
            if not args.uniform_pb:
                pb_module = Tabular(n_states=env.n_states, output_dim=env.n_actions - 1)
        else:
            pf_module = NeuralNet(
                input_dim=env.preprocessor.output_dim,
                output_dim=env.n_actions,
                hidden_dim=args.hidden_dim,
                n_hidden_layers=args.n_hidden,
            )
            if not args.uniform_pb:
                pb_module = NeuralNet(
                    input_dim=env.preprocessor.output_dim,
                    output_dim=env.n_actions - 1,
                    hidden_dim=args.hidden_dim,
                    n_hidden_layers=args.n_hidden,
                    torso=pf_module.torso if args.tied else None,
                )
        if args.uniform_pb:
            pb_module = DiscreteUniform(env.n_actions-1)

        assert (
                pf_module is not None
        ), f"pf_module is None. Command-line arguments: {args}"
        assert (
                pb_module is not None
        ), f"pb_module is None. Command-line arguments: {args}"


        pf_estimator = DiscretePolicyEstimator(
            module=pf_module,
            n_actions=env.n_actions,
            preprocessor=env.preprocessor,
        )
        pb_estimator = DiscretePolicyEstimator(
            module=pb_module,
            n_actions=env.n_actions,
            is_backward=True,
            preprocessor=env.preprocessor,
        )

        if args.loss == "ModifiedDB":
            gflownet = ModifiedDBGFlowNet(
                pf_estimator,
                pb_estimator,
                True if args.replay_buffer_size == 0 else False,
            )

        if args.loss in ("DB", "SubTB"):
            # We need a LogStateFlowEstimator
            assert (
                    pf_estimator is not None
            ), f"pf_estimator is None. Command-line arguments: {args}"
            assert (
                    pb_estimator is not None
            ), f"pb_estimator is None. Command-line arguments: {args}"

            if args.tabular:
                module = Tabular(n_states=env.n_states, output_dim=1)
            else:
                module = NeuralNet(
                    input_dim=env.preprocessor.output_dim,
                    output_dim=1,
                    hidden_dim=args.hidden_dim,
                    n_hidden_layers=args.n_hidden,
                    torso=pf_module.torso if args.tied else None,
                )

            logF_estimator = ScalarEstimator(
                module=module, preprocessor=env.preprocessor
            )
            if args.loss == "DB":
                gflownet = DBGFlowNet(
                    pf=pf_estimator,
                    pb=pb_estimator,
                    logF=logF_estimator,
                    on_policy=True if args.replay_buffer_size == 0 else False,
                )
            else:
                gflownet = SubTBGFlowNet(
                    pf=pf_estimator,
                    pb=pb_estimator,
                    logF=logF_estimator,
                    on_policy=True if args.replay_buffer_size == 0 else False,
                    weighting=args.subTB_weighting,
                    lamda=args.subTB_lambda,
                )
        elif args.loss == "TB":
            gflownet = TBGFlowNet(
                pf=pf_estimator,
                pb=pb_estimator,
                on_policy=True if args.replay_buffer_size == 0 else False,
            )
        elif args.loss == "ZVar":
            gflownet = LogPartitionVarianceGFlowNet(
                pf=pf_estimator,
                pb=pb_estimator,
                on_policy=True if args.replay_buffer_size == 0 else False,
            )

    assert gflownet is not None, f"No gflownet for loss {args.loss}"

    # Initialize the replay buffer ?

    replay_buffer = None
    if args.replay_buffer_size > 0:
        if args.loss in ("TB", "SubTB", "ZVar"):
            objects_type = "trajectories"
        elif args.loss in ("DB", "ModifiedDB"):
            objects_type = "transitions"
        elif args.loss == "FM":
            objects_type = "states"
        else:
            raise NotImplementedError(f"Unknown loss: {args.loss}")
        replay_buffer = ReplayBuffer(
            env, objects_type=objects_type, capacity=args.replay_buffer_size
        )

    # 3. Create the optimizer

    # Policy parameters have their own LR.
    params = [
        {
            "params": [
                v for k, v in dict(gflownet.named_parameters()).items() if k != "logZ"
            ],
            "lr": args.lr,
        }
    ]

    # Log Z gets dedicated learning rate (typically higher).
    if "logZ" in dict(gflownet.named_parameters()):
        params.append(
            {
                "params": [dict(gflownet.named_parameters())["logZ"]],
                "lr": args.lr_Z,
            }
        )

    optimizer = torch.optim.Adam(params)

    visited_terminating_states = env.States.from_batch_shape((0,))
    logs = {}
    states_visited = 0
    n_iterations = args.n_trajectories // args.batch_size
    validation_info = {"l1_dist": float("inf")}
    policies = {'ground_truth_policy':None,'Gflownet_policy':None}
    #sample trajectories for offline training
    trajectories = off_policy_data_generation(args.off_policy_data_generation_protocol,env,pf_estimator,pb_estimator)
    training_samples = gflownet.to_training_samples(trajectories)

    # print(training_samples.__dict__['states'].tensor,training_samples.__dict__['states'].tensor.shape)
    # raw_states = training_samples.__dict__['states'].tensor.reshape(-1,2)
    # mask =  torch.any(raw_states != torch.tensor([0, 0], device=raw_states.device), dim=1)
    # breakpoint()
    # selected_states = raw_states[mask,:]
    # result = get_terminating_state_dist_pmf(env,selected_states)
    # print(result)
    # uniform_ini_grid_path = 'uniform_ini_gridwise.pkl'
    # with open(uniform_ini_grid_path,'wb') as file:
    #     pickle.dump(result,file)

    # breakpoint()
    # prob_args = {'sf_bias':3}
    # sampler = Sampler(pf_uniform_estimator,**prob_args)
    # trajectories = sampler.sample_trajectories(env, n_trajectories=args.training_sample_size,)
    # training_samples = gflownet.to_training_samples(trajectories)
    # print(training_samples.__dict__['states'].tensor,training_samples.__dict__['states'].tensor.shape)
    # trajectories_states = training_samples.__dict__['states'].tensor.detach().clone()
    # ending_states_indices = training_samples.__dict__['when_is_done'].detach().clone()-1
    # print('ending_states_indices: ', ending_states_indices)
    # terminating_states = trajectories_states[ending_states_indices, torch.arange(args.training_sample_size)].reshape(-1,args.ndim)
    # # mask =  torch.all(terminating_states != torch.tensor([0, 0], device=terminating_states.device), dim=1)
    # # selected_states = terminating_states[mask,:]
    # result = get_terminating_state_dist_pmf(env, terminating_states)
    # print(result)
    # uniform_ini_policy_path = 'uniform_ini_policy_ternimating_states.pkl'
    # with open(uniform_ini_policy_path,'wb') as file:
    #     pickle.dump(result,file)
    # training_samples = gflownet.to_training_samples(trajectories)
    ''''
    One need to ensure the trajectories are forward before putting into the replay buffer
    code here:
    '''
    assert training_samples is not None
    replay_buffer.add(training_samples)
    for iteration in trange(n_iterations):
        if replay_buffer is not None:
            with torch.no_grad():
                training_objects = replay_buffer.sample(n_trajectories=args.batch_size)
        else:
            training_objects = training_samples

        optimizer.zero_grad()
        loss = gflownet.loss(env, training_objects)
        loss.backward()
        optimizer.step()

        visited_terminating_states.extend(trajectories.last_states)

        states_visited += len(trajectories)

        to_log = {"loss": loss.item(), "states_visited": states_visited}
        if use_wandb:
            wandb.log(to_log, step=iteration)
        if iteration % args.validation_interval == 0:
            if n_iterations-iteration<= args.validation_interval:
                validation_info, gflow_policy, ground_truth_policy = validate(
                    env,
                    gflownet,
                    args.validation_samples,
                    None,
                )
            else:
                validation_info, gflow_policy, ground_truth_policy = validate(
                    env,
                    gflownet,
                    args.validation_samples,
                    visited_terminating_states,
                )
            if use_wandb:
                wandb.log(validation_info, step=iteration)
            to_log.update(validation_info)
            tqdm.write(f"{iteration}: {to_log}")
            logs.update({iteration:to_log})
            policies['ground_truth_policy'] = ground_truth_policy
            policies['Gflownet_policy'] = gflow_policy
            print(logs)

    gflow_net_policy_state_visit_dist = visit_distribution((6,6),env,gflownet,20000)
    file_path = f'hypergrid_experiment_result_{args.ndim}_{args.height}_{args.R0}_{args.R1}_{args.R2}_Cosine_policywise_uniform_init.pkl'
    policy_path = f'hypergrid_experiment_result_{args.ndim}_{args.height}_{args.R0}_{args.R1}_{args.R2}_Cosine_policywise_uniform_init_policies.pkl'
    state_visit_distribution_path = f'hypergrid_experiment_result_{args.ndim}_{args.height}_{args.R0}_{args.R1}_{args.R2}_state_visit_dist.pkl'
    # Open the file in binary write mode and store the data using pickle.dump()
    with open(file_path, 'wb') as file:
        pickle.dump(logs, file)
    with open(policy_path,'wb') as file:
        pickle.dump(policies,file)
    with open(state_visit_distribution_path,'wb') as file:
        pickle.dump(gflow_net_policy_state_visit_dist,file)


    return validation_info["l1_dist"]






if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--no_cuda", action="store_true", help="Prevent CUDA usage")

    parser.add_argument(
        "--ndim", type=int, default=2, help="Number of dimensions in the environment"
    )
    parser.add_argument(
        "--height", type=int, default=8, help="Height of the environment"
    )
    parser.add_argument("--R0", type=float, default=0.1, help="Environment's R0")
    parser.add_argument("--R1", type=float, default=0.5, help="Environment's R1")
    parser.add_argument("--R2", type=float, default=2.0, help="Environment's R2")

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed, if 0 then a random seed is used",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size, i.e. number of trajectories to sample per training iteration",
    )
    parser.add_argument(
        "--replay_buffer_size",
        type=int,
        default=20000,
        help="If zero, no replay buffer is used. Otherwise, the replay buffer is used.",
    )

    parser.add_argument(
        "--off_policy_data_generation_protocol",
        type = str,
        choices = ["grid_uni", "policy_uni"],
        default= "grid_uni",
        help="how the data used for off-policy training generated"
    )

    parser.add_argument(
        "--training_sample_size",
        type = int,
        default=15000,
        help = "The number of data points used for off-policy training."
    )

    parser.add_argument(
        "--loss",
        type=str,
        choices=["FM", "TB", "DB", "SubTB", "ZVar", "ModifiedDB"],
        default="TB",
        help="Loss function to use",
    )
    parser.add_argument(
        "--subTB_weighting",
        type=str,
        default="geometric_within",
        help="weighting scheme for SubTB",
    )
    parser.add_argument(
        "--subTB_lambda", type=float, default=0.9, help="Lambda parameter for SubTB"
    )

    parser.add_argument(
        "--tabular",
        action="store_true",
        help="Use a lookup table for F, PF, PB instead of an estimator",
    )
    parser.add_argument("--uniform_pb", action="store_true", help="Use a uniform PB")
    parser.add_argument(
        "--tied", action="store_true", help="Tie the parameters of PF, PB, and F"
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=256,
        help="Hidden dimension of the estimators' neural network modules.",
    )
    parser.add_argument(
        "--n_hidden",
        type=int,
        default=2,
        help="Number of hidden layers (of size `hidden_dim`) in the estimators'"
             + " neural network modules",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for the estimators' modules",
    )
    parser.add_argument(
        "--lr_Z",
        type=float,
        default=0.1,
        help="Specific learning rate for Z (only used for TB loss)",
    )

    parser.add_argument(
        "--n_trajectories",
        type=int,
        default=int(1e6),
        help="Total budget of trajectories to train on. "
             + "Training iterations = n_trajectories // batch_size",
    )

    parser.add_argument(
        "--validation_interval",
        type=int,
        default=100,
        help="How often (in training steps) to validate the gflownet",
    )
    parser.add_argument(
        "--validation_samples",
        type=int,
        default=20000,
        help="Number of validation samples to use to evaluate the probability mass function.",
    )

    parser.add_argument(
        "--wandb_project",
        type=str,
        default="",
        help="Name of the wandb project. If empty, don't use wandb",
    )

    args = parser.parse_args()

    print(main(args))