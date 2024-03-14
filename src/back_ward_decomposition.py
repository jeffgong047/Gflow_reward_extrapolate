from gfn.states import States

class States_triv(States):
    def __init__(self,tensor):
        self.tensor = tensor

def reward_distribution(state):
    return env.log_reward(state)

def back_ward_decomposition(state,container, memory):
    '''
    The goal is to update the edge flows in the container
    :param state:
    :memory 2-d dictionary
    :return:
    '''
    global counts
    counts +=1
    # print(counts)
    current_position = state
    next_positions = []
    if current_position[0] <env_parameters[1]-1:
        next_positions.append((state[0]+1,state[1]))
    if current_position[1] < env_parameters[1]-1:
        next_positions.append((state[0],state[1]+1))
    if not next_positions:
        # print('return from states: ', state)
        return reward_distribution(States_triv(torch.tensor(state)))
    edges = [ (current_position,next_pos) for next_pos in next_positions]
    # print('current position: ', current_position)
    # print('edges from current position: ', edges)
    edge_flows=[]
    for edge in edges:
        linked_state = edge[1]
        if edge not in memory.keys():
            # print('states to being visited: ', linked_state)
            memory.update({edge:None})
            edge_flow = back_ward_decomposition(linked_state,container,memory)
            # print('flows returned from: ',linked_state, ' is: ',edge_flow)
            memory.update({edge:edge_flow})
            edge_flows.append(edge_flow)
            container[edge[0]][edge[1]].update({'flow':edge_flow})
        # print('edge flow of the edge: ',edge, 'is: ',edge_flow)
        else:
            edge_flows.append(container[edge[0]][edge[1]]['flow'])
        # print('edge flow of the edge: ',edge, 'is: ',container[edge[0]][edge[1]]['flow'])
    # print('return from states: ', state)
    return reward_distribution(States_triv(torch.tensor(state)))+ sum(edge_flows)

