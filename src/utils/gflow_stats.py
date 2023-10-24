class Gflow_parameters():
    def __init__(self,args):
        graph, data, score = get_data(args.graph, args, rng=default_rng())
        self.prior = get_prior(args.prior, **args.prior_kwargs)
        self._threepointInfoScorer =  threepoint_info_score(data,self.prior,**args.soft_info_constraint_kwargs)
        # self._MLE = Entropy_Score(data, prior, **args.scorer_kwargs)
        self._num_variables = len(data.columns)
        self._data = data

    def construct(self,env,gflownet, params):
        '''
        :param nodes: The collection of nodes
        :return: all combina
        '''
        columns = ['v_key' , 'weight','trans_prob','bayesian_scores','info_constraint']
        states = pd.DataFrame(columns=columns)
        env_state = deepcopy(env.reset())
        env_Closure = deepcopy(env._closure_T)
        global env_states
        env_states.update({"":[env_state,env_Closure]})
        states.loc[""] ={"weight":1, "trans_prob":[],"bayesian_scores":[],"info_constraint":[]}
        gflow_params = nx.DiGraph()
        gflow_params.add_node("")
        self.explore([],gflow_params,env,gflownet,params,states)
        return gflow_params, states, env_states



    def explore(self,trajectory,gflow_params,env,gflownet,params,states):
        '''

        :param trajectory: a list containing ordered actions that generates the trajectory
        :param gflow_params: a di_graph (networkx digraph)that records the trained Gflownet parameters (jump probabilities for all possible states in [bayesian score, soft-info-constraint,jump probability from previous state to current state])
        :param env : the environment is synchronized with the trajectory
        :return: the tree that contains the explored states and edges
        #the explore can be done via backward decomposition with dynamic programming
        '''
        global env_states
        mask  = np.squeeze(env._state['mask'], axis = 0)
        observation = env._state
        possible_actions = [ row_idx*len(row)+column_idx for row_idx, row in enumerate(mask) for column_idx, element in enumerate(row) if element==1]
        trans_probs = get_transition_prob(gflownet,params,observation,None)
        if len(possible_actions)<=0:
            return
        key_0 = ','.join(trajectory)
        if key_0 not in env_states.keys():
            e_s,e_c = env.get_state()
            env_states.update({key_0:[e_s,e_c]})
        for action in possible_actions:
            new_state = trajectory+[str(action)]
            trans_prob =  trans_probs[int(action)]
            try:
                observation = env._state
                adjacency_matrix = observation['adjacency'].astype(jnp.float32)
                source , target = divmod(action, self._num_variables)
                info_constraint = self._threepointInfoScorer.soft_info_constraint([source,target],adjacency_matrix)
                # MLE = self._MLE([],adjacency_matrix)
                next_observation,delta_score,dones,_ = env.step(np.asarray(np.array([action],dtype='int32')))
            except ValueError:
                print('Dude some actions are invalid: either the edge to be '
                      'added is already in the DAG, or adding this edge '
                      'would lead to a cycle.')
                breakpoint()
                continue
            key = ','.join(new_state)
            num_variables = env.num_variables
            v_key = []
            for a in new_state:
                source, target = divmod(int(a), num_variables)
                v_key.append(f'{self._data.columns[source]}->{self._data.columns[target]}')
            _,weight_0, trans_prob_0, delta_scores_0,info_constraint_0 = states.loc[key_0]
            assert all(v is not None for v in [weight_0, trans_prob_0, delta_scores_0]), "Some variables are None"
            states.loc[key] = {"v_key":v_key, "weight":weight_0*trans_prob, "trans_prob":trans_prob_0+[trans_prob], "bayesian_scores":delta_scores_0 + [delta_score],"info_constraint":info_constraint_0+[info_constraint]}
            gflow_params.add_node(key)
            gflow_params.add_edge(key_0,key)
            self.explore(new_state,gflow_params,env,gflownet,params,states)
            env_copy = env_states[key_0]
            env.set(deepcopy(env_copy[0]),deepcopy(env_copy[1])) # the critical bug is here.
            del env_copy
        return





def query_to_actions(query,num_nodes):
    assert len(query)%2==0
    transitions = list(query)
    actions = []
    for i in range(0,len(transitions)-1,2):
        actions.append(int(transitions[i])*num_nodes+int(transitions[i+1]))
    return actions

def step_statistics(action, env, gflownet, params):
    '''

    :param action: we assume the action is already mapped to integers
    :param previous_trajectory:
    :param env: The environment is a result of going through series of actions in previous_trajectory
    :param gflownet:
    :param params:
    :return: new_env after the action, trans prob i.e. jump prob for the action , delta scores
    '''
    #how to get observation from env
    observation,_ = deepcopy(env.get_observation)
    trans_prob =  get_transition_prob(gflownet,params,observation,np.array([action]))
    try:
        next_observation,delta_score,dones,_ = env.step(np.asarray(np.array([action],dtype='int32')))
    except ValueError:
        return [None,None,None]
    return new_env, trans_prob ,delta_score
def query_sampleStatistics(query,env,gflownet,params):
    '''

    :param query: string
    :return: sample weight of the query
    '''

    num_nodes = env.scorer.num_variables

    actions = query_to_actions(query,num_nodes)
    observation = env.reset()
    transition_probs = {}
    delta_scores=  {}
    for action in actions:
        trans_prob  = get_transition_prob(gflownet, params,observation,np.array([action],dtype='int32'))
        try:
            next_observation, delta_score, dones, _ = env.step(np.asarray(np.array([action],dtype='int32')))
        except ValueError:
            return [None, None, None,None]

        print('transition probabilities: ', transition_probs)
        transition_probs.update({str(action):trans_prob})
        delta_scores.update({str(action):delta_score})
        observation = next_observation
        weight =1
    for prob in transition_probs.values():
        weight = prob*weight
    return env, weight, transition_probs ,delta_scores

import numpy as np

def softmax(arr):
    exp_arr = np.exp(arr)
    sum_exp_arr = np.sum(exp_arr)
    softmax_arr = exp_arr / sum_exp_arr
    return softmax_arr



def get_transition_prob(gflownet,params,observations,action):
    masks = observations['mask'].astype(jnp.float32)
 #   print('action being executed is ', action, 'observation is ', observations)
    adjacencies = observations['adjacency'].astype(jnp.float32)
    vmodel = vmap(gflownet.model.apply, in_axes=(None, 0, 0))
    log_pi = vmodel(params,adjacencies,masks)
    log_pi = softmax(log_pi[0])
  #  print('action being executed is ', action, 'observation is ', observations)
    if action is None:
        return log_pi
    else:
        return log_pi[int(action)]

