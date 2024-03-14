import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import argparse
import seaborn as sns
import torch
import networkx as nx
from gfn.gym import HyperGrid
from gfn.states import States
import matplotlib.colors as mcolors
import time
# Create an ArgumentParser object
parser = argparse.ArgumentParser(description='Process input data files.')

# Add an argument to accept a list of input data files
parser.add_argument('files', nargs='+', help='List of input data files')

# Parse the command line arguments
args = parser.parse_args()

# Access the list of input data files using args.files
input_files = args.files

# Initialize an empty list to store DataFrames
data_frames = []
configurations = [0.1,0.01,0.001]
# Loop through input files to load dataframes of statistics given different configuration of hypergrid_experiment results

# for data_file in input_files:
# 	with open(data_file, 'rb') as file:
# 		data = pickle.load(file)
# 		df = pd.DataFrame.from_dict(data).T
#
# 		# Check for non-positive values in 'states_visited' before applying np.log
# 		if 'states_visited' in df.columns and (df['states_visited'] <= 0).any():
# 			print(f"Warning: Non-positive values in 'states_visited' for file {data_file}")
# 		# Apply np.log only if there are no non-positive values
# 		df['log_states_visited'] = np.log(df['states_visited'].astype(float))
# 		df['log_l1_dist'] = np.log(df['l1_dist'].astype(float))
#
# 		data_frames.append(df.copy())


# load policies
data_file = input_files[0]
with open(data_file,'rb') as file:
	data = pickle.load(file)
print(data)







def scatter_plot(data_frames,x,y):
	# Define custom colors for the DataFrames
	colors = ['blue', 'orange', 'green']

	plt.figure(figsize=(8, 6))

	for i, df in enumerate(data_frames):
		plt.scatter(
			x=df[x],
			y=df[y],
			label=f'R0={configurations[i]}',
			color=colors[i]  # Set the color based on the index
		)

	plt.xlabel(x)
	plt.ylabel(y)
	plt.title(f'{y} vs. {x}')
	plt.legend()

	plt.savefig(f'combined_scatter_plot_{y}_vs_{x}_4_8.png')






env_parameters = [2,200,0.1,0.5,2]
args.ndim, args.height, args.R0, args.R1, args.R2= env_parameters
#temporary implementation for flowmap with backward decomposition algorithm
env = HyperGrid(
	args.ndim, args.height, args.R0, args.R1, args.R2, reward_cos = True,device_str='cpu'
)

counts= 0

class States_triv(States):
	def __init__(self,tensor):
		self.tensor = tensor

def reward_distribution(state):
	return env.log_reward(state)


def step(current_position, step_size, boundaries):
	next_positions = []
	next_dim_0 = current_position[0] + step_size
	next_dim_1 = current_position[1] + step_size
	if next_dim_0 <boundaries and next_dim_0>=0:
		next_positions.append((next_dim_0, current_position[1]))
	if next_dim_1 < boundaries and next_dim_1>=0:
		next_positions.append((current_position[0],next_dim_1))
	return next_positions



def back_ward_decomposition(state,container, memory, boundaries):
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
	next_positions = step(state,1,boundaries)
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
			edge_flow = back_ward_decomposition(linked_state,container,memory,boundaries)/max(len(step(linked_state, -1,boundaries)),1)
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


def checker(G):
	'''
	The goal of the checker is to ensure that
	:param G: networkx graph
	:return:
	The goal is to traverse through all states to ensure that total flows in = total flows out from each state
	'''
	for s in G.nodes():
		# print('current node is: ',s)
		# print('neighbor of s are: ', list(G.neighbors(s)))
		neighbors = np.array(list(G.neighbors(s)))
		neighbors_relationship = neighbors.sum(axis=-1)-sum(s)
		neighbors_values = []
		for n, r in zip(G.neighbors(s), neighbors_relationship):
			if r==1:
				neighbors_values.append(G[s][n]['flow'])
			elif r==-1:
				neighbors_values.append(G[n][s]['flow'])
			else:
				raise ValueError
		neighbors_values = np.array(neighbors_values)
		# print(np.sum(neighbors_values*neighbors_relationship))
		if len(list(G.neighbors(s)))>2: #this is to avoid root
			try:
				assert abs(np.sum(neighbors_values*neighbors_relationship)+ reward_distribution(States_triv(torch.tensor(s)))) <1e-02
				print('in_flow and out_flow balanced, sum in flow = sum out flow + reward for state ', s)
			except Exception as e:
				print(s, list(G.neighbors(s)), neighbors_values)
				breakpoint()
def test_backward_decomposition_runtime():
	scales = np.linspace(5,205,10)
	t = []
	num_states = []
	for i in scales:
		global counts
		counts = 0
		global env_parameters
		env_parameters[1] = i
		G = nx.grid_2d_graph(int(i) , int(i))
		#use backward decomposition algorithm on reward_distribution
		root = list(G.nodes)[0]
		assert root ==(0,0)
		memory_hypergrid_dict = {}
		start_time = time.time()
		back_ward_decomposition(root,G, memory_hypergrid_dict,int(i) )
		end_time = time.time()
		execution_time = end_time - start_time
		t.append(execution_time)
		num_states.append(counts)
		print("Execution time:", execution_time, "seconds for ", 2*env_parameters[1]**2,'==' ,counts)

	plt.figure()
	plt.scatter(t, num_states)
	plt.xlabel('number of states')
	plt.ylabel('time')
	plt.title('Scatter Plot: num_states visited vs. time')
	plt.grid(True)
	plt.savefig('scatter_plot_num_states_visited_vs_time.png')




def flow_map_backward_decomposition():
	G = nx.grid_2d_graph(env_parameters[1], env_parameters[1])
	#use backward decomposition algorithm on reward_distribution
	root = list(G.nodes)[0]
	assert root ==(0,0)
	memory_hypergrid_dict = {}
	back_ward_decomposition(root,G, memory_hypergrid_dict,env_parameters[1])
	checker(G)
	breakpoint()
	print('total states visited: ', counts)
	pos = {(x, y): (y, -x) for x, y in G.nodes()}
	colors = plt.cm.viridis(np.linspace(0, 1, 1))
	for (u, v, wt) in G.edges.data('flow'):
		print(f"flow of edge ({u},{v}): {wt}")
		if wt is None:
			G[u][v]['flow'] = 0
	nx.draw(G, pos, with_labels=True, node_color='white', node_size=700, edge_color=colors, width=2)
	# Get weights and normalize them
	flows = np.array([G[u][v]['flow'] for u, v in G.edges()])
	norm = mcolors.Normalize(vmin=flows.min(), vmax=flows.max())
	# Create a colormap
	cmap = plt.cm.viridis
	# Draw the graph
	for (u, v) in G.edges():
		nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=2,
							   edge_color=[cmap(norm(G[u][v]['flow']))])

	nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=50)
	nx.draw_networkx_labels(G, pos)
	# # Add a colorbar
	# sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
	# sm.set_array([])
	# plt.colorbar(sm, label='Edge Weight')
	plt.savefig('flow_diagram_backward_decomposition_2_8.png')
		#plot flows




def flow_diagram(data):
	G = nx.grid_2d_graph(8, 8)
	keys = ['trajectory_actions','trajectory_probs']
	stop_probability = []
	actual_reward = []
	trajectory_actions = data[keys[0]].squeeze(-1).permute(1,0).cpu()
	trajectory_probs = data[keys[1]].permute(1,0).detach().cpu()
	num_samples = trajectory_actions.shape[1]
	print(trajectory_actions.shape)
	print(trajectory_probs.shape)
	for u, v in G.edges():
		G.edges[u,v]['weight'] = None
	for i in range(num_samples):
		actions = trajectory_actions[i,:]
		probs = trajectory_probs[i,:]
		right = 0 #for simplicity assume right -> dimension 1 down -> dimension 2
		down  = 0
		current_position = (right,down)
		for index, a in enumerate(actions):
			prob = probs[index]
			print(actions)
			if a==0:
				right+=1
			elif a==1:
				down +=1
			elif a==2:
				G.nodes[current_position]['weight'] = prob
				print(G.nodes[current_position]['weight'])
				true_reward = reward_distribution(States_triv(torch.tensor(current_position)))
				stop_probability.append(prob)
				actual_reward.append(true_reward)
			elif a==-1:
				assert actions[index-1]==2
				break
			pos_position = (right,down)
			try:
				if G.has_edge(current_position,pos_position):
					print(G[current_position][pos_position])
					if G[current_position][pos_position]['weight']:
						assert G[current_position][pos_position]['weight']==prob
					else:
						G.add_edge(current_position,pos_position,weight=prob)
				else:
					G.add_edge(current_position, pos_position, weight=prob)
			except KeyError:
				print(f"weight assignment might has problem between: {current_position} and {pos_position}, before {G[current_position][pos_position]['weight']}, now {prob} ")
			current_position = pos_position
	pos = {(x, y): (y, -x) for x, y in G.nodes()}
	# colors = plt.cm.viridis(np.linspace(0, 1, 1))
	for (u, v, wt) in G.edges.data('weight'):
		print(f"Weight of edge ({u},{v}): {wt}")
		if wt is None:
			G[u][v]['weight'] = 0
	# nx.draw(G, pos, with_labels=True, node_color=colors, node_size=700, edge_color=colors, width=2)
	# Get weights and normalize them
	weights_edges = np.array([G[u][v]['weight'] for u, v in G.edges()])
	weights_nodes = np.array([])
	for u in G.nodes():
		if 'weight' not in G.nodes[u].keys():
			G.nodes[u]['weight'] = 0
		weights_nodes = np.append(weights_nodes, G.nodes[u]['weight'])
		# print(G.nodes[u]['weight'])
		# print(weights_nodes)
		# breakpoint()
	weights = np.append(weights_nodes , weights_edges)
	# norm = mcolors.Normalize(vmin=weights.min(), vmax=weights.max())
	# # Create a colormap
	# cmap = plt.cm.viridis

	# fig, ax = plt.subplots(figsize=(8, 8))
# # Add a colorbar
# 	sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
# 	sm.set_array([])
	# plt.colorbar(sm, ax=ax, label='Edge Weight')
	# Draw the graph
	# for (u, v) in G.edges():
	# 	nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=2,
	# 			   edge_color=[cmap(norm(G[u][v]['weight']))])
	actual_reward = np.array(actual_reward)
	normalized_actual_reward = actual_reward/np.sum(actual_reward)
	stop_probability = np.array(stop_probability)
	normalized_stop_probability = stop_probability/np.sum(stop_probability)
	# for u in G.nodes():
	# 	nx.draw_networkx_nodes(G, pos, node_color=[cmap(norm(G.nodes[u]['weight']))], node_size=30)
	# nx.draw_networkx_labels(G, pos)

	# nx.draw_networkx_nodes(G, pos, node_size=50,  # Adjust size as needed
	# 				   node_color=[cmap(norm(G.nodes[u]['weight'])) for u in G.nodes()])

	plt.figure()
	plt.scatter(normalized_stop_probability, normalized_actual_reward)
	plt.xlabel('Normalized Stop Probability')
	plt.ylabel('Normalized Actual Reward')
	plt.title('Scatter Plot: Normalized Stop Probability vs. Normalized Actual Reward')
	plt.grid(True)
	plt.savefig('scatter_plot_stop_probability_vs_actual_reward.png')
	breakpoint()
	return flow_diagram





def heat_map(key, matrix):
	sns.set()  # Set Seaborn style (optional)
	plt.figure(figsize=(8, 6))  # Set the figure size (optional)

	breakpoint()
	# Create the heatmap
	sns.heatmap(matrix, annot=False, cmap="YlGnBu", fmt=".2f", linewidths=.5)

	# Customize labels, title, and ot
	# her properties (optional)
	plt.xlabel("X-axis")
	plt.ylabel("Y-axis")
	plt.title(key)

	# Display the heatmap
	plt.savefig(f'heat_map_{key}_cosine_policywise_uniform_init_2_30_bias_3.png')

# scatter_plot(data_frames,x='log_states_visited',y='log_l1_dist')
#

# state visit distribution
# matrix = data.reshape(30,30)
# heat_map('uniform_initialization_policywise_visit_distribution',matrix)

# flow_diagram(data) #for gflownet policies
#flow diagram and policies
# for key in policies.keys():
# 	raw_matrix = policies[key]
# 	matrix = raw_matrix.reshape(30,30)
# 	print(matrix)
# 	heat_map(key,matrix)

#
flow_map_backward_decomposition()

# test_backward_decomposition_runtime()