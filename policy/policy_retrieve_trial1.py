from MDP_TG.dra import Dra
import numpy as np
import itertools
from itertools import product
import time
import networkx
from networkx import single_source_shortest_path
from networkx.classes.digraph import DiGraph
import random

import os
import pandas as pd
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.stats as stats
# import pystan
import seaborn as sns
from pprint import pprint
import _pickle as pkl
from scipy.stats.mstats import normaltest, kruskalwallis
from math import *
from scipy.stats import bernoulli
from numpy.random import normal
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from scipy.stats import gaussian_kde, sem
from scipy.stats import norm
import scipy.stats as stats
from random import seed
import pickle
from collections import deque

t0 = time.time()
## Getting back the objects:
path='/home/cpsgroup/trust_aware_hrc/transition_probability/'

with open(path+'trustMM.pkl','rb') as f:  # Python 3: open(..., 'rb')
        trustMM = pickle.load(f)

with open(path+'actMM_TB.pkl','rb') as f:  # Python 3: open(..., 'rb')
        actMM_TB = pickle.load(f)

# -------- real example -------
# -------- Robot -------
Action_robot = [tuple('a1'), tuple('a2'), tuple('a3')] 
# vehicle actions: at each location, the vehicle has 2 route choices, which correspond to 'a1' and 'a2'. 
# 'a3' is a self-loop transition, which is defined specifically for the destintion L. 

# transitions enabled by action 'a1'
WS_position_edge_dict1={
    # incidents
    ('A1', 'B1'): 'pedestrian',
    ('A2', 'E1'): 'construction',
    ('A3', 'B1'): 'pedestrian',
    ('B1', 'C1'): 'empty',
    ('B2', 'A2'): 'pedestrian',
    ('B3', 'G1'): 'pedestrian',
    ('C1', 'D1'): 'truck',
    ('C2', 'B3'): 'empty',
    ('C3', 'B3'): 'empty',
    ('D1', 'K2'): 'pedestrian',
    ('D2', 'C3'): 'truck',
    ('D3', 'C3'): 'truck',
    ('E1', 'L1'): 'empty',
    ('E2', 'A3'): 'construction',
    ('E3', 'A3'): 'construction',
    ('F1', 'G2'): 'truck',
    ('F2', 'H1'): 'empty',
    ('F3', 'G2'): 'truck',
    ('G1', 'I2'): 'pedestrian',
    ('G2', 'B2'): 'pedestrian',
    ('G3', 'B2'): 'pedestrian',
    ('H1', 'I1'): 'empty',
    ('H2', 'L3'): 'truck',
    ('H3', 'I1'): 'empty',
    ('I1', 'G3'): 'pedestrian',
    ('I2', 'H2'): 'empty',
    ('I3', 'H2'): 'empty',
    ('J1', 'C2'): 'construction',
    ('J2', 'I3'): 'empty',
    ('J3', 'C2'): 'construction',
    ('K1', 'D2'): 'pedestrian',
    ('K2', 'J3'): 'truck',
    ('K3', 'D2'): 'pedestrian',
    ('L1', 'K3'): 'truck',
    ('L2', 'E3'): 'empty',
    ('L3', 'E3'): 'empty',
}

# transitions enabled by action 'a2'
WS_position_edge_dict2={
   # incidents
    ('A1', 'E1'): 'construction',
    ('A2', 'F1'): 'empty',
    ('A3', 'F1'): 'empty',
    ('B1', 'G1'): 'pedestrian',
    ('B2', 'C1'): 'empty',
    ('B3', 'A2'): 'pedestrian',
    ('C1', 'J2'): 'construction',
    ('C2', 'D1'): 'truck',
    ('C3', 'J2'): 'construction',
    ('D1', 'E2'): 'construction',
    ('D2', 'E2'): 'construction',
    ('D3', 'K2'): 'pedestrian',
    ('E1', 'D3'): 'construction',
    ('E2', 'L1'): 'empty',
    ('E3', 'D3'): 'construction',
    ('F1', 'H1'): 'empty',
    ('F2', 'A1'): 'empty',
    ('F3', 'A1'): 'empty',
    ('G1', 'F2'): 'truck',
    ('G2', 'I2'): 'pedestrian',
    ('G3', 'F2'): 'truck',
    ('H1', 'L3'): 'truck',
    ('H2', 'F3'): 'empty',
    ('H3', 'F3'): 'empty',
    ('I1', 'J1'): 'empty',
    ('I2', 'J1'): 'empty',
    ('I3', 'G3'): 'pedestrian',
    ('J1', 'K1'): 'truck',
    ('J2', 'K1'): 'truck',
    ('J3', 'I3'): 'empty',
    ('K1', 'L2'): 'truck',
    ('K2', 'L2'): 'truck',
    ('K3', 'J3'): 'truck',
    ('L1', 'H3'): 'truck',
    ('L2', 'H3'): 'truck',
    ('L3', 'K3'): 'truck',
}

# transitions enabled by action 'a3'
WS_position_edge_dict3={
    ('L1', 'L1'): 'empty',
    ('L2', 'L2'): 'empty',
    ('L3', 'L3'): 'empty',
}

# -------- Human -------
# human actions: 't' stands for takeover, 's' stands for standstill
Action_human = [tuple('t'), tuple('s')]

#trust levels
trustlevel=[1, 2, 3, 4, 5, 6, 7]
trust_number = len(trustlevel)

# Define trust transition matrix for different incidents: standstill
A1 = trustMM[(0,'Obstacle')]
A2 = trustMM[(0, 'Pedestrian')]
A3 = trustMM[(0, 'Truck')]
A4 = [[0] * trust_number for _ in range(trust_number)]
for i in range(trust_number):
    A4[i][i] = 1

WS_trust_transition_matrix_standstill_dict={
    'construction': A1,
    'pedestrian': A2,
    'truck': A3,
    'empty': A4,
}

# Define trust transition matrix for different incidents: takeover
B1 = trustMM[(1,'Obstacle')]
B2 = trustMM[(1, 'Pedestrian')]
B3 = trustMM[(1, 'Truck')]
B4 = A4.copy()

WS_trust_transition_matrix_takeover_dict={
    'construction': B1,
    'pedestrian': B2,
    'truck': B3,
    'empty': B4,
}

#----------Task (LDTL formula)----------------
ordered_reach_trust = '& F base1 & F base2 & F G base3_hightrust & G ! base3_nothightrust G ! lowtrust'
# translate the LDTL task into a Deterministic Rabin Automaton (DRA)
dra = Dra(ordered_reach_trust)

#-----------------------------
# Load the dictionary from the pickle file
trial_id = 1
with open('/home/cpsgroup/trust_aware_hrc/policy/pickle_data_trial{trial_id}_v2/robot_nodes.pkl'.format(trial_id=trial_id), 'rb') as pickle_file1:
    robot_nodes_dict = pickle.load(pickle_file1)

with open('/home/cpsgroup/trust_aware_hrc/policy/pickle_data_trial{trial_id}_v2/best_plan_prefix.pkl'.format(trial_id=trial_id), 'rb') as pickle_file2:
    best_plan_dict = pickle.load(pickle_file2)

with open('/home/cpsgroup/trust_aware_hrc/policy/pickle_data_trial{trial_id}_v2/prod_nodes.pkl'.format(trial_id=trial_id), 'rb') as pickle_file3:
    prod_nodes = pickle.load(pickle_file3)

with open('/home/cpsgroup/trust_aware_hrc/policy/pickle_data_trial{trial_id}_v2/prod_edges.pkl'.format(trial_id=trial_id), 'rb') as pickle_file4:
    prod_edge_dict = pickle.load(pickle_file4)

sampled_trust_point = []
for state in robot_nodes_dict.keys():
    belief = state[1]
    if belief not in sampled_trust_point:
        sampled_trust_point.append(tuple(belief))

def distance(point1, point2):
    distance_squared = sum((x2 - x1)**2 for x1, x2 in zip(point1, point2))
    return math.sqrt(distance_squared)

def closest_point(A, S):
    if not S:
        return None, -1
   
    closest = S[0]
    min_distance = distance(A, S[0])
    closest_index = 0

    for i, point in enumerate(S):
        dist = distance(A, point)
        if dist < min_distance:
            min_distance = dist
            closest = point
            closest_index = i

    return (closest_index, closest)
    
def successor(state):
    state_successor = []
    for state1, state2 in prod_edge_dict.items():
        if state1[0][0] == state[0][0] and distance(state1[0][1], state[0][1])<0.01 and state1[0][2] == state[0][2] and state1[2] == state[2]:
            state_successor.append(tuple(state2))
    return state_successor


def indexes_of_max_numbers(numbers):
    # Step 1: Find the maximum value in the set of numbers
    max_value = max(numbers, key=abs)

    # Step 2: Find the indexes of all occurrences of the maximum value
    indexes = [index for index, num in enumerate(numbers) if num == max_value]

    return indexes

def compute_distance_to_goal(graph, start_node, goal_node):
    # Initialize a queue for BFS
    queue = deque([(start_node, 0)])  # (node, distance from start_node)
    visited = set()

    while queue:
        node, distance = queue.popleft()

        # Check if the goal state is reached
        if node in goal_node:
            return distance

        if node not in visited:
            visited.add(node)

            # Enqueue all neighbors of the current node
            for neighbor in graph[node]:
                queue.append((neighbor, distance + 1))

    # If the goal state is not reachable from the start node
    return 1000

def find_min_value_and_index(set):
    # Convert the set to a list
    set_to_list = list(set)

    if not set_to_list:
        return None, None

    # Find the minimum value and its index in the list
    min_value = min(set_to_list)
    min_index = set_to_list.index(min_value)

    return min_value, min_index

def is_hightrust_tuple(t):
    if t[5]+t[6] < 0.5:
            return False
    return True

WS_graph={
    # incidents
    'A1': ['B1', 'E1'],
    'A2': ['E1', 'F1'],
    'A3': ['B1', 'F1'],
    'B1': ['C1', 'G1'],
    'B2': ['A2', 'C1'],
    'B3': ['G1', 'A2'],
    'C1': ['D1', 'J2'],
    'C2': ['B3', 'D1'],
    'C3': ['B3', 'J2'],
    'D1': ['K2', 'E2'],
    'D2': ['C3', 'E2'],
    'D3': ['C3', 'K2'], 
    'E1': ['L1', 'D3'],
    'E2': ['A3', 'L1'],
    'E3': ['A3', 'D3'], 
    'F1': ['G2', 'H1'],
    'F2': ['H1', 'A1'],
    'F3': ['G2', 'A1'], 
    'G1': ['I2', 'F2'],
    'G2': ['B2', 'I2'],
    'G3': ['B2', 'F2'],
    'H1': ['I1', 'L3'],
    'H2': ['L3', 'F3'],
    'H3': ['I1', 'F3'],
    'I1': ['G3', 'J1'],
    'I2': ['H2', 'J1'],
    'I3': ['H2', 'G3'],
    'J1': ['C2', 'K1'],
    'J2': ['I3', 'K1'],
    'J3': ['C2', 'I3'],
    'K1': ['D2', 'L2'],
    'K2': ['J3', 'L2'],
    'K3':  ['D2', 'J3'],
    'L1': ['K3', 'H3', 'L1'],
    'L2': ['E3', 'H3', 'L2'],
    'L3': ['E3', 'K3', 'L3'],
}


goal_set = ['L1', 'L2', 'L3']
distance_to_goal = dict()
for node in WS_graph.keys():
    distance_to_goal[node] = compute_distance_to_goal(WS_graph, node, goal_set)


init_trust_belief = (1/7, 1/7, 1/7, 1/7, 1/7, 1/7, 1/7)
incidents={'construction', 'pedestrian', 'truck'}

# the state of the trust-pomdp is defined as follows:
# ((vehicle location, trust belief, observation), label, DRA state)
# label maps each robot state to an atomic proposition of the LDTL formula that is satisfied at this state
# DRA state monitors the task evolution

def policy(current_state, a_h):
    fx = current_state[0][0] #vehicle location
    fy = current_state[0][1] #trust belief
    fo = current_state[0][2] #observation (i.e., human takeover decision)
    f_mdp_label = current_state[1] #label
    fz = current_state[2] #dra state

    possible_next_state = successor(current_state)
    next_dra_state = []
    for state in possible_next_state:
        if state[2] not in next_dra_state:
            next_dra_state.append(state[2])
    
    if fx in goal_set and is_hightrust_tuple(fy):
        print('Current state is the destination.')
        return (-1, ((-1, (), ()), -1, -1), -1)
    else:
        for state, policy in best_plan_dict.items():
            #given the current state, find the best action (there may be mutiple best actions, in this case we randomly choose one)
            if fx == state[0][0] and distance(fy, state[0][1])<0.01 and fz == state[2]:
                actions_tep = policy[0]
                actions = tuple(''.join(char for char in item) for item in actions_tep)
                action_prob = policy[1]
                indexes = indexes_of_max_numbers(action_prob)
                best_action_set = [actions[idx] for idx in indexes]
                #best_action = random.choice(best_action_set)
             
                #given the best action and the human action (i.e., a_h), compute the trust belief at the next step (i.e., ty)
                if len(best_action_set) == 0:
                    print('Best action not found!')
                elif len(best_action_set) == 1:
                    best_action = best_action_set[0]
                    if tuple(best_action) == Action_robot[0]:
                        for (x, y) in WS_position_edge_dict1.keys():
                            if fx == x:
                                tx = y
                                id = WS_position_edge_dict1[(fx, tx)]
                    if tuple(best_action) == Action_robot[1]:
                        for (x, y) in WS_position_edge_dict2.keys():
                            if fx == x:
                                tx = y
                                id = WS_position_edge_dict2[(fx, tx)]
                    if tuple(best_action) == Action_robot[2]:
                        for (x, y) in WS_position_edge_dict3.keys():
                            if fx == x:
                                tx = y
                                id = WS_position_edge_dict3[(fx, tx)]
                else:
                    next_state_set = []
                    value_act = []
                    id_set = []
                    for idx, act in enumerate(best_action_set):
                        if tuple(act) == Action_robot[0]:
                            for (x, y) in WS_position_edge_dict1.keys():
                                if fx == x:
                                    tx = y 
                                    next_state_set.append(tx)
                                    value_act.append(distance_to_goal[tx])
                                    incident = WS_position_edge_dict1[(fx, tx)]
                                    id_set.append(incident)
                        if tuple(act) == Action_robot[1]:
                            for (x, y) in WS_position_edge_dict2.keys():
                                if fx == x:
                                    tx = y 
                                    next_state_set.append(tx)
                                    value_act.append(distance_to_goal[tx])
                                    incident = WS_position_edge_dict2[(fx, tx)]
                                    id_set.append(incident)
                        if tuple(act) == Action_robot[2]:
                            for (x, y) in WS_position_edge_dict3.keys():
                                if fx == x:
                                    tx = y 
                                    next_state_set.append(tx)
                                    value_act.append(distance_to_goal[tx])
                                    incident = WS_position_edge_dict3[(fx, tx)]
                                    id_set.append(incident)

                    min_value, min_index = find_min_value_and_index(value_act)
                    best_action = best_action_set[min_index]
                    tx = next_state_set[min_index]
                    id = id_set[min_index]
                                    
                if a_h == Action_human[0]:
                    ty_temp = tuple(np.dot(np.transpose(WS_trust_transition_matrix_takeover_dict[id]), fy))
                    (ty_idx, ty)=closest_point(ty_temp, sampled_trust_point)
                if a_h == Action_human[1]:
                    ty_temp = tuple(np.dot(np.transpose(WS_trust_transition_matrix_standstill_dict[id]), fy))
                    (ty_idx, ty)=closest_point(ty_temp, sampled_trust_point)
                
                if tuple(best_action) == Action_robot[1]:
                    for (x, y) in WS_position_edge_dict2.keys():
                        if fx == x:
                            tx = y
                    id = WS_position_edge_dict2[(fx, tx)]
                    if a_h == Action_human[0]:
                        ty_temp = tuple(np.dot(np.transpose(WS_trust_transition_matrix_takeover_dict[id]), fy))
                        (ty_idx, ty)=closest_point(ty_temp, sampled_trust_point)
                    if a_h == Action_human[1]:
                        ty_temp = tuple(np.dot(np.transpose(WS_trust_transition_matrix_standstill_dict[id]), fy))
                        (ty_idx, ty)=closest_point(ty_temp, sampled_trust_point)
                
                if tuple(best_action) == Action_robot[2]:
                    for (x, y) in WS_position_edge_dict3.keys():
                        if fx == x:
                            tx = y 
                    id = WS_position_edge_dict3[(fx, tx)]
                    if a_h == Action_human[0]:
                        ty_temp = tuple(np.dot(np.transpose(WS_trust_transition_matrix_takeover_dict[id]), fy))
                        (ty_idx, ty)=closest_point(ty_temp, sampled_trust_point)
                    if a_h == Action_human[1]:
                        ty_temp = tuple(np.dot(np.transpose(WS_trust_transition_matrix_standstill_dict[id]), fy))
                        (ty_idx, ty)=closest_point(ty_temp, sampled_trust_point)

                t_mdp_node = (tx, ty, a_h) # robot state at the next step
                t_mdp_label = robot_nodes_dict[(tx, ty, a_h)]
                
                for t_dra_node in dra.nodes():
                    t_prod_node = (t_mdp_node, t_mdp_label, t_dra_node)
                    if t_dra_node in next_dra_state:
                        return (best_action, t_prod_node, id)

        #     print('NO return.')
        #     #print(fx == state[0][0] and distance(fy, state[0][1])<0.01 and fz == state[2])
        #     print(fx, distance(fy, state[0][1]), fz)
        #     print(fx == state[0][0], distance(fy, state[0][1])<0.01, fz == state[2])


# #------------------main------------------------
# current_state = (('A3', (0.14285714285714285,0.14285714285714285,0.14285714285714285,0.14285714285714285,0.14285714285714285,0.14285714285714285,0.14285714285714285), ('t',)), frozenset(), 1)
# a_h = tuple('s')
# while current_state[0][0] not in ['L1', 'L2', 'L3']:
#     (action, next_state, id) = policy(current_state, a_h)
#     print('Action: %s' % action)
#     print('Next_state: %s, %s, %s' % (next_state[0], next_state[1], next_state[2]))
#     current_state = next_state
#     if id in incidents:
#         a_h = random.choice(Action_human)
#     else:
#         a_h = Action_human[1]
#
#
# print('Destination is arrived.')





