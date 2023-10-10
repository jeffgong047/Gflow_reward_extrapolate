import pylab as plt
import numpy as np
from pgmpy.models import BayesianNetwork
import random
import networkx as nx
import pandas as pd
from pgmpy.factors.discrete import TabularCPD
from pgmpy.sampling import BayesianModelSampling
from pyvis.network import Network
import sys
import os
# import matplotlib.pyplot as plot
# from collections import deque

# Define the structure of the Bayesian network
# from dowhy import CausalModel

def generate_demo():
    edges = [('A', 'B'), ('C', 'B'), ('C', 'D')]
    model = BayesianNetwork(edges)
    user_input = input("Do you want to visualize the model, input 1 for yes： ")
    if user_input ==True:
        pos = nx.circular_layout(model)
        nx.draw(model,pos=pos ,with_labels=True)
        plt.show()



    # Define the conditional probability distributions
    cpd_a = TabularCPD(variable='A', variable_card=2, values=[[0.5],[0.5]])
    cpd_c = TabularCPD(variable='C', variable_card=2, values=[[0.5], [0.5]])
    cpd_b = TabularCPD(variable='B', variable_card=2,
                       values=[[0.8, 0.6, 0.4, 0.2], [0.2, 0.4, 0.6, 0.8]],
                       evidence=['A', 'C'], evidence_card=[2, 2])
    cpd_d = TabularCPD(variable='D', variable_card=2, values=[[0.7,0.2], [0.3,0.8]], evidence=['C'], evidence_card=[2])

    model.add_cpds(cpd_a, cpd_c, cpd_b, cpd_d)
    # Generate data from the Bayesian network
    inference = BayesianModelSampling(model)
    sample = inference.forward_sample(size=1000)
    print(sample)

    sample.to_csv('./data/demo.csv', index=False)

def generate_demo1():
    edges = [('A','B'),('B','C')]
    model = BayesianNetwork(edges)
    cpd_a = TabularCPD(variable='A',variable_card=2, values=[[0.5],[0.5]])
    cpd_b = TabularCPD(variable='B', variable_card=2, values=[[0.7,0.2], [0.3,0.8]], evidence=['A'], evidence_card=[2])
    cpd_c = TabularCPD(variable='C', variable_card=2, values=[[0.7,0.2], [0.3,0.8]], evidence=['B'], evidence_card=[2])
    model.add_cpds(cpd_a,cpd_b,cpd_c)
    inference = BayesianModelSampling(model)
    sample = inference.forward_sample(size=1000)
    print(sample)
    sample.to_csv('./data_results/data/synthetic_data/demo1.csv', index=False)


def generate_demo2():
    edges = [('A','B'),('C','B')]
    model = BayesianNetwork(edges)
    cpd_a = TabularCPD(variable='A',variable_card=2, values=[[0.5],[0.5]])
    cpd_b = TabularCPD(variable='B', variable_card=2,
                       values=[[0.8, 0.6, 0.4, 0.2], [0.2, 0.4, 0.6, 0.8]],
                       evidence=['A', 'C'], evidence_card=[2, 2])
    cpd_c = TabularCPD(variable='C',variable_card=2, values=[[0.5],[0.5]])
    model.add_cpds(cpd_a,cpd_b,cpd_c)
    inference = BayesianModelSampling(model)
    sample = inference.forward_sample(size=1000)
    print(sample)
    sample.to_csv('./data_results/data/synthetic_data/demo2.csv', index=False)


def generate_demo3():
    edges = [('B','A'),('B','C')]
    model = BayesianNetwork(edges)
    cpd_a = TabularCPD(variable='A', variable_card=2, values=[[0.7,0.2], [0.3,0.8]], evidence=['B'], evidence_card=[2])
    cpd_b = TabularCPD(variable='B',variable_card=2, values=[[0.5],[0.5]])
    cpd_c = TabularCPD(variable='C', variable_card=2, values=[[0.7,0.2], [0.3,0.8]], evidence=['B'], evidence_card=[2])

    model.add_cpds(cpd_a,cpd_b,cpd_c)
    inference = BayesianModelSampling(model)
    sample = inference.forward_sample(size=1000)
    print(sample)
    sample.to_csv('./data_results/data/synthetic_data/demo3.csv', index=False)

def generate_demo_interven():
    edges = [('A', 'B'), ('C', 'B'), ('C', 'D')]
    model = BayesianNetwork(edges)
    user_input = input("Do you want to visualize the model, input 1 for yes： ")
    if user_input ==True:
        pos = nx.circular_layout(model)
        nx.draw(model,pos=pos ,with_labels=True)
        plt.show()



    # Define the conditional probability distributions
    cpd_a = TabularCPD(variable='A', variable_card=2, values=[[0.5],[0.5]])
    cpd_c = TabularCPD(variable='C', variable_card=2, values=[[0.5], [0.5]])
    cpd_b = TabularCPD(variable='B', variable_card=2,
                       values=[[0.8, 0.6, 0.4, 0.2], [0.2, 0.4, 0.6, 0.8]],
                       evidence=['A', 'C'], evidence_card=[2, 2])
    cpd_d = TabularCPD(variable='D', variable_card=2, values=[[0.7,0.2], [0.3,0.8]], evidence=['C'], evidence_card=[2])

    model.add_cpds(cpd_a, cpd_c, cpd_b, cpd_d)
    # Generate data from the Bayesian network
    inference = BayesianModelSampling(model)
    sample = inference.forward_sample(size=1000)
    #mask variables B and D by making them all 1's
    sample.loc[:,'B']=1
    sample.loc[:,'D']=1
    sample.to_csv('./data/sample/demo_intervene.csv', index=False)




def do_structure():
    '''
    args: None
    return: Generate both observational and interventional data
    '''
    edges = [('Z', 'X'), ('X', 'Y'), ('Z', 'Y')]
    model = BayesianNetwork(edges)
    cpd_Z = TabularCPD(variable='Z', variable_card=2, values=[[0.5],[0.5]])
    cpd_X = TabularCPD(variable='X', variable_card=2, values=[[0.8,0.3], [0.2,0.7]], evidence=['Z'],evidence_card=[2])
    cpd_Y = TabularCPD(variable='Y', variable_card=2,
                       values=[[0.8, 0.6, 0.4, 0.2], [0.2, 0.4, 0.6, 0.8]],
                       evidence=['Z', 'X'], evidence_card=[2, 2])


    model.add_cpds(cpd_Z, cpd_X, cpd_Y)
    # Generate data from the Bayesian network
    inference = BayesianModelSampling(model)
    sample = inference.forward_sample(size=1000)
    print(sample)

    sample.to_csv('./data/do_structure.csv', index=False)

def do_structure_interven():
    edges = [('Z', 'X'), ('X', 'Y'), ('Z', 'Y')]
    model = BayesianNetwork(edges)
    cpd_Z = TabularCPD(variable='Z', variable_card=2, values=[[0.5],[0.5]])
    cpd_X = TabularCPD(variable='X', variable_card=2, values=[[0.8,0.3], [0.2,0.7]])
    cpd_Y = TabularCPD(variable='Y', variable_card=2,
                       values=[[0.8, 0.6, 0.4, 0.2], [0.2, 0.4, 0.6, 0.8]],
                       evidence=['Z', 'X'], evidence_card=[2, 2])
    data_Z= np.random.choice([0,1],size=1000,p=cpd_Z.values.flatten())



# def sampleSignalingPathway(queue, random_dag):
#      if len(queue)>0:
#         node_from = queue.pop()
#         mean_value = 0.5
#         lambd = 1/mean_value
#         upperbound = round(random.expovariate(lambd))
#         nod_to  = random.choice(list(queue),upperbound)
#      else:
#
#
#
# def sampleSignalingPathways(random_dag):
#     '''
#     recursively sample signaling pathways
#     :return: random_dag with added signaling pathways
#     '''
#     root = random_dag.predecessors()[0]
#     while len(list(random_dag.predecessors(root)))>0:
#         root = list(random_dag.predecessors(root))[0]
#     queue = deque()
#     queue.append(root)
#     sampleSignalingPathway(queue,random_dag)


def sampleSignalingPathways(root , G, num):
    nodes = {node:0 for node in G.nodes()}
    #   print(f'we are making {num} jumps.')
    for i in range(num):
        #watch variables node_status, nodeRemains, currentNode, nextNode
        # breakpoint()
        #     print(f'jump number {i}')
        nodes_status = nodes
        nodes_status[root]=1
        nodeRemains = len(nodes)
        currentNode = root
        while nodeRemains >0 and len(list(G.successors(currentNode)))>0:
            assert nodes_status[currentNode] ==1
            # breakpoint()
            nextNode = jump(currentNode,nodes_status,G)
            if nextNode is not None:
                nodes_status[nextNode] = 1
                nodeRemains -=1
                #remove nodes that are already in the pathway
                predecessors = G.predecessors(nextNode)
                for node in predecessors:
                    if nodes_status[node]==0:
                        nodeRemains -=1
                        nodes_status[node] =1
                #           print(f'nodes status {nodes_status}')
                #          print(f'current node {currentNode}, next node {nextNode}')
                currentNode = nextNode
            else:
                nodeRemains = 0
    #       print(f'node Remains {nodeRemains}, successors of current node {list(G.successors(currentNode))}')

def jump(current_Node,nodes_status,G):
    # Randomly choose a key among keys with value 0
    try_jump = random.randint(0,1)
    chose_key= random.choice(list(nodes_status.keys()))
    if try_jump and nodes_status[chose_key]==0:
        #    print(f'jump a large step to state {chose_key}')
        next_Node = chose_key
        G.add_edge(current_Node,next_Node)
    else:
        if list(G.successors(current_Node)):
            next_Node = random.choice(list(G.successors(current_Node)))
        else:
            next_Node = None
        if next_Node is not None:
            assert len(list(G.successors(current_Node)))>0
    #  print(f'walk slightly to state {next_Node}')
    return next_Node

def visualize_bn_with_pyvis_from_nx(nx_graph, directory_path,name):
    net = Network(notebook=True, cdn_resources='in_line',directed=True)
    net.from_nx(nx_graph)
    name = str(name)
    net.show(os.path.join(directory_path,name+'.html'))




def synthetic_signalinglike():
    # Set the number of nodes and the desired node labels
    choice  = 2
    exist_loop = False
    iters = 5
    # Set the number of nodes and the desired node labels
    nodes_range = [5,10,15]
    num_observations_range = [500,875,1000]
    num_nodes = nodes_range[choice]
    print(f'number of nodes {num_nodes}')
    node_labels = [i for i in range(1, num_nodes + 1)]
    for iter in range(iters):
        # Generate a random DAG with networkx
        random_network = nx.gn_graph(n=num_nodes)
        #Convet the random DAG to desired signaling_pathway alike format
        inverted_edges = [(v, u) for u, v in random_network.edges()]
        random_dag = nx.DiGraph(inverted_edges)
        #get root of the graph
        root = 0
        num_pathways = len(random_dag.edges())
        sampleSignalingPathways(root, random_dag, num_pathways) # sample more signaling pathways to the graph

        # Convert the networkx DAG to a pgmpy BayesianModel
        random_network = BayesianNetwork()
        for node in node_labels:
            random_network.add_node(node)

        for from_node, to_node in random_dag.edges():
            try:
                random_network.add_edge(node_labels[from_node], node_labels[to_node])
            except ValueError:
                exist_loop = True
        if exist_loop:
            iter -=1
            exist_loop = False
        else:
            # Set the number of states for each node (assuming all nodes have the same number of states)
            num_states = 2

            # Set the conditional probability distributions for the nodes using random values
            for node in random_network.nodes():
                node_parents = list(random_network.predecessors(node))
                if not node_parents:
                    cpd_value_0 = np.random.rand()
                    cpd_value_1 = 1-cpd_value_0
                    cpd = TabularCPD(variable=node, variable_card=num_states, values=[[cpd_value_0],[cpd_value_1]])
                else:
                    parent_combinations = num_states ** len(node_parents)
                    cpd_values_0 = np.random.rand(1, parent_combinations)
                    cpd_values_1 = 1 - cpd_values_0
                    cpd = TabularCPD(variable=node, variable_card=num_states,
                                     evidence=node_parents, evidence_card=[num_states] * len(node_parents),
                                     values=[cpd_values_0.tolist()[0],cpd_values_1.tolist()[0]])
                random_network.add_cpds(cpd)

            # Check if the CPDs are correctly defined and the network is valid
            assert random_network.check_model()
            #visualize the causal graph
            # pos = nx.spring_layout(random_dag)
            # nx.draw(random_dag, pos, with_labels=True, node_size=200, node_color='skyblue', font_size=10, font_weight='bold', arrowsize=20)
            #plt.show()
            # Set the number of observations you want to generate
            num_observations = num_observations_range[choice]

            # Generate the observations
            inference = BayesianModelSampling(random_network)
            observations = inference.forward_sample(size=num_observations)

            # Convert the observations to a pandas DataFrame and print the first few observations
            observations_df = pd.DataFrame(observations)
            print(observations_df.head())
            #write the observations and html graphs to local directory
            directory_path = f'./../data_results/data/synthetic_data/synthetic_{num_nodes}_{iter}'
            name = f'synthetic_{num_nodes}'
            if not os.path.exists(directory_path):
                os.mkdir(directory_path)
            observations_df.to_csv(os.path.join(directory_path, name+'.csv'), index = False)
            visualize_bn_with_pyvis_from_nx(random_dag, directory_path,name)



#cited from huawei-Noah Trustworthy AI/datasets
#https://github.com/huawei-noah/trustworthyAI/blob/master/datasets/synthetic_datasets.py
import numpy as np
from copy import deepcopy
from sklearn.preprocessing import PolynomialFeatures


def generate_W(d=6, prob=0.5, low=0.5, high=2.0):
    """
    generate a random weighted adjaceecy matrix
    :param d: number of nodes
    :param prob: prob of existing an edge
    :return:
    """
    g_random = np.float32(np.random.rand(d,d)<prob)
    g_random = np.tril(g_random, -1)
    U = np.round(np.random.uniform(low=low, high=high, size=[d, d]), 1)
    U[np.random.randn(d, d) < 0] *= -1
    W = (g_random != 0).astype(float) * U
    return W


def gen_data_given_model(b, s, c, n_samples=10, noise_type='lingam', permutate=False):
    """Generate artificial data based on the given model.
       Based on ICA-LiNGAM codes.
       https://github.com/cdt15/lingam
    Parameters
    ----------
    b : numpy.ndarray, shape=(n_features, n_features)
        Strictly lower triangular coefficient matrix.
        NOTE: Each row of `b` corresponds to each variable, i.e., X = BX.
    s : numpy.ndarray, shape=(n_features,)
        Scales of disturbance variables.
    c : numpy.ndarray, shape=(n_features,)
        Means of observed variables.

    Returns
    -------
    xs, b_, c_ : Tuple
        `xs` is observation matrix, where `xs.shape==(n_samples, n_features)`.
        `b_` is permuted coefficient matrix. Note that rows of `b_` correspond
        to columns of `xs`. `c_` if permuted mean vectors.
    """

    n_vars = b.shape[0]

    # Check args
    assert (b.shape == (n_vars, n_vars))
    assert (s.shape == (n_vars,))
    assert (np.sum(np.abs(np.diag(b))) == 0)
    np.allclose(b, np.tril(b))

    if noise_type == 'lingam':
        # Nonlinearity exponent, selected to lie in [0.5, 0.8] or [1.2, 2.0].
        # (<1 gives subgaussian, >1 gives supergaussian)
        q = np.random.rand(n_vars) * 1.1 + 0.5
        ixs = np.where(q > 0.8)
        q[ixs] = q[ixs] + 0.4

        # Generates disturbance variables
        ss = np.random.randn(n_samples, n_vars)
        ss = np.sign(ss) * (np.abs(ss) ** q)

        # Normalizes the disturbance variables to have the appropriate scales
        ss = ss / np.std(ss, axis=0) * s

    elif noise_type == 'gaussian':
        ss = np.random.randn(n_samples, n_vars) * s

    # Generate the data one component at a time
    xs = np.zeros((n_samples, n_vars))
    for i in range(n_vars):
        # NOTE: columns of xs and ss correspond to rows of b
        xs[:, i] = ss[:, i] + xs.dot(b[i, :]) + c[i]

        # Permute variables
    b_ = deepcopy(b)
    c_ = deepcopy(c)
    if permutate:
        p = np.random.permutation(n_vars)
        xs[:, :] = xs[:, p]
        b_[:, :] = b_[p, :]
        b_[:, :] = b_[:, p]
        c_[:] = c[p]

    return xs, b_, c_


def gen_data_given_model_2nd_order(b, s, c, n_samples=10, noise_type='lingam', permutate=False):
    """Generate artificial data based on the given model.
       Quadratic functions

    Parameters
    ----------
    b : numpy.ndarray, shape=(n_features, n_features)
        Strictly lower triangular coefficient matrix.
        NOTE: Each row of `b` corresponds to each variable, i.e., X = BX.
    s : numpy.ndarray, shape=(n_features,)
        Scales of disturbance variables.
    c : numpy.ndarray, shape=(n_features,)
        Means of observed variables.

    Returns
    -------
    xs, b_, c_ : Tuple
        `xs` is observation matrix, where `xs.shape==(n_samples, n_features)`.
        `b_` is permuted coefficient matrix. Note that rows of `b_` correspond
        to columns of `xs`. `c_` if permuted mean vectors.

    """
    # rng = np.random.RandomState(random_state)
    n_vars = b.shape[0]

    # Check args
    assert (b.shape == (n_vars, n_vars))
    assert (s.shape == (n_vars,))
    assert (np.sum(np.abs(np.diag(b))) == 0)
    np.allclose(b, np.tril(b))

    if noise_type == 'lingam':
        # Nonlinearity exponent, selected to lie in [0.5, 0.8] or [1.2, 2.0].
        # (<1 gives subgaussian, >1 gives supergaussian)
        q = np.random.rand(n_vars) * 1.1 + 0.5
        ixs = np.where(q > 0.8)
        q[ixs] = q[ixs] + 0.4

        # Generates disturbance variables
        ss = np.random.randn(n_samples, n_vars)
        ss = np.sign(ss) * (np.abs(ss) ** q)

        # Normalizes the disturbance variables to have the appropriate scales
        ss = ss / np.std(ss, axis=0) * s

    elif noise_type == 'gaussian':

        ss = np.random.randn(n_samples, n_vars) * s
    # Generate the data one component at a time

    xs = np.zeros((n_samples, n_vars))
    poly = PolynomialFeatures()
    newb = []

    for i in range(n_vars):
        # NOTE: columns of xs and ss correspond to rows of b
        xs[:, i] = ss[:, i] + c[i]
        col = b[i]
        col_false_true = np.abs(col) > 0.3
        len_parents = int(np.sum(col_false_true))
        if len_parents == 0:
            newb.append(np.zeros(n_vars, ))
            continue
        else:
            X_parents = xs[:, col_false_true]
            X_2nd = poly.fit_transform(X_parents)
            X_2nd = X_2nd[:, 1:]
            dd = X_2nd.shape[1]
            U = np.round(np.random.uniform(low=0.5, high=1.5, size=[dd, ]), 1)
            U[np.random.randn(dd, ) < 0] *= -1
            U[np.random.randn(dd, ) < 0] *= 0
            X_sum = np.sum(U * X_2nd, axis=1)
            xs[:, i] = xs[:, i] + X_sum

            # remove zero-weight variables
            X_train_expand_names = poly.get_feature_names()[1:]
            cj = 0
            new_reg_coeff = np.zeros(n_vars, )

            # hard coding; to be optimized for reading
            for ci in range(n_vars):
                if col_false_true[ci]:
                    xxi = 'x{}'.format(cj)
                    for iii, xxx in enumerate(X_train_expand_names):
                        if xxi in xxx:
                            if np.abs(U[iii]) > 0.3:
                                new_reg_coeff[ci] = 1.0
                                break
                    cj += 1
            newb.append(new_reg_coeff)

    # Permute variables
    b_ = deepcopy(np.array(newb))
    c_ = deepcopy(c)
    if permutate:
        p = np.random.permutation(n_vars)
        xs[:, :] = xs[:, p]
        b_[:, :] = b_[p, :]
        b_[:, :] = b_[:, p]
        c_[:] = c[p]

    return xs, b_, c_

if __name__ == '__main__':
    # generate_demo1()
    # generate_demo2()
    # generate_demo3()
    seeds = [8]

    #Linear
    for seed in seeds:
        np.random.seed(seed)
        d = 12
        W = generate_W(d=d, prob=0.5) # 0.2
        c = np.zeros(d)
        s = np.ones([d]) # s = np.round(np.random.uniform(low=0.5, high=2, size=[d]), 1) different varicne
        xs, b_, c_ = gen_data_given_model(W, s, c, n_samples=5000, noise_type='lingam', permutate=True)

    dir_name = os.path.join(os.getcwd(), 'lingam_same_noise_seed{}'.format(seed))
    os.mkdir(dir_name)
    np.save(os.path.join(dir_name, 'data.npy'), xs)
    np.save(os.path.join(dir_name, 'DAG.npy'), b_)

    #Quadratic
    seeds = [8]
    for seed in seeds:
        np.random.seed(seed)
        d = 10
        W = generate_W(d=d, prob=0.5)
        c = np.zeros(d)
        #s = np.round(np.random.uniform(low=0.5, high=2, size=[d]), 1)
        s = np.ones([d])
        xs, b_, c_ = gen_data_given_model_2nd_order(W, s, c, n_samples=5000, noise_type='lingam', permutate=True)

        # get the first 3000 samples
        xs_norm = np.linalg.norm(xs, axis=1)
        xs_th = sorted(xs_norm)[3000]
        xs = xs[xs_norm < xs_th]
    dir_name = os.path.join(os.getcwd(), 'lingam_quad_noise_seed{}'.format(seed))
    os.mkdir(dir_name)
    np.save(os.path.join(dir_name, 'data.npy'), xs)
    np.save(os.path.join(dir_name, 'DAG.npy'), b_)