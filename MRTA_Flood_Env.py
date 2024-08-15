"""
Author: Steve Paul 
Date: 4/14/22 """
""

# import numpy as np
# import gym
import time
from tkinter import CURRENT
from gym import Env
from collections import defaultdict
from gym.spaces import Discrete, Box, Dict
import matplotlib.pyplot as plt
import torch
from topology import *
import scipy.sparse as sp
from persim import wasserstein
from scipy.io import loadmat
import csv
from sklearn.linear_model import LinearRegression
import math
import os 
import statsmodels.api as sm
import copy
Paret = loadmat('paretos.mat')
Paret = np.array(Paret['Paret'])

Capas = Paret[:, [6, 7, 8]]
payloads = -Capas[:, 0]
speeds = -Capas[:, 2]
ranges = -Capas[:, 1]

# Preparing the data
X = ranges.reshape(-1, 1)
X = sm.add_constant(X)  # Adding a constant term for intercept
y = speeds

# Define and fit the model for the 5th percentile (lower boundary)
quantile_05_model = sm.regression.quantile_regression.QuantReg(y, X).fit(q=0.05)

# Define and fit the model for the 95th percentile (upper boundary)
quantile_95_model = sm.regression.quantile_regression.QuantReg(y, X).fit(q=0.9)

# Constructing the polynomial features matrix
X_poly = np.column_stack([
    speeds**2,
    speeds,
    ranges
])

# Fitting a linear regression model
poly_model = LinearRegression().fit(X_poly, payloads)


def predict_payload(speed, range_val):
    """
    Predict the payload based on given speed and range using the polynomial model.
    
    Args:
    - speed (float): The speed value.
    - range_val (float): The range value.
    
    Returns:
    - float: Predicted payload value.
    """
    # Constructing the polynomial features for the given values
    X_pred = np.array([[speed**2, speed, range_val]])
    
    # Predicting using the polynomial model
    payload_pred = poly_model.predict(X_pred)[0]
    
    return payload_pred

def predict_quantile_boundaries_fixed(range_val):
    """
    Predict the 5th and 95th percentile speed values for a given range using quantile regression models.
    
    Args:
    - range_val (float): The range value.
    
    Returns:
    - (float, float): Tuple containing the predicted 5th and 95th percentile speed values.
    """
    # Preparing the range value for prediction
    X_pred = np.array([[1.0, range_val]])  # Added constant term for intercept
    
    # Predicting using the quantile regression models
    speed_05_pred = quantile_05_model.predict(X_pred)[0]
    speed_95_pred = quantile_95_model.predict(X_pred)[0]
    
    return speed_05_pred, speed_95_pred


def normalize(x):
    return (x - x.min()) / (x.max() - x.min())

import numpy as np

def scale_action_values(action_values):
    assert len(action_values) == 3, "The input array should have 3 values."
    scaled_values = np.zeros_like(action_values, dtype=float)

    scaled_values[0] = (action_values[0] * (15.17 - 7.17)) + 7.17
    
    quantile_boundary = predict_quantile_boundaries_fixed(scaled_values[0])
    min_value = quantile_boundary[0]
    max_value = quantile_boundary[1]

    speed_value = (action_values[1] * (max_value - min_value)) + min_value
    if speed_value > max_value:
        penalty = speed_value - max_value
        return False, penalty
    elif speed_value < min_value:
        penalty = min_value - speed_value
        return False, penalty
    else:
        under_con = True
        scaled_values[1] = speed_value
        scaled_values[2] = round(predict_payload(scaled_values[1], scaled_values[0]))
        scaled_values[2] = np.clip(scaled_values[2], 2, 7)
        #0 - range, 1 - speed, 2 - payload
        #print(scaled_values)
        return True, scaled_values
     
class MRTA_Flood_Env(Env):
    def __init__(self,
                 n_locations=100,
                 visited=[],
                 n_agents=2,
                 total_distance_travelled=0.0,
                 max_capacity=6,
                 max_range=4,
                 enable_dynamic_tasks = False,
                 n_initial_tasks = 30,
                 display = False,
                 enable_topological_features = False,
                 training = True,
                 with_morphology=False
                 ):
        # Action will be choosing the next task. (Can be a task that is alraedy done)
        # It would be great if we can force the agent to choose not-done task
        super(MRTA_Flood_Env, self).__init__()
        self.n_locations = n_locations
        self.with_morphology = with_morphology

        self.action_space =  [Box(0,1,(2,), dtype=np.float64), Discrete(1)]
        self.locations = np.random.random((n_locations, 2))*4
        self.depot = self.locations[0, :]
        self.visited = visited
        self.n_agents = n_agents
        self.agents_prev_location = np.zeros((n_agents, 1), dtype=int)
        self.agents_next_location = np.zeros((n_agents, 1), dtype=int)
        self.agents_distance_travelled = np.zeros((n_agents, 1))
        self.total_distance_travelled = total_distance_travelled
        self.agent_taking_decision = 0
        self.current_location_id = 0
        self.nodes_visited = np.zeros((n_locations, 1))
        self.n_locations = n_locations
        self.enable_dynamic_tasks = enable_dynamic_tasks
        self.agents_distance_to_destination = np.zeros(
            (n_agents, 1))  # tracks the distance to destination from its current point for all robots
        self.actions_vals = []
        self.distance_matrix = np.linalg.norm(self.locations[:, None, :] - self.locations[None, :, :], axis=-1)
        self.time = 0.0
        self.agents_next_decision_time = np.zeros((n_agents, 1))
        self.agents_prev_decision_time = np.zeros((n_agents, 1))
        self.agents_destination_coordinates = np.ones((n_agents, 1)) * self.depot

        self.total_reward = 0.0
        self.total_length = 0
        self.max_capacity = 5
        self.max_range = 5.68

        self.time_deadlines = (torch.tensor(np.random.random((1, n_locations)))*.5 + .5)*2
        self.time_deadlines[0, 0] = 1000000
        self.location_demand = torch.ones((1, n_locations), dtype=torch.float32)
        self.task_done = torch.zeros((1, n_locations), dtype=torch.float32)
        self.deadline_passed = torch.zeros((1, n_locations), dtype=torch.float32)
        self.depot_id = 0
        self.active_tasks = ((self.nodes_visited == 0).nonzero())[0]
        self.available_tasks = torch.zeros((n_locations, 1), dtype=torch.float32)
        self.actions_vals = []
        #new
        if not self.enable_dynamic_tasks:
            n_initial_tasks = n_locations
        self.n_initial_tasks = n_initial_tasks
        self.available_tasks[0: n_initial_tasks, 0] = 1 # set the initial tasks available
        self.time_start = self.time_deadlines*(torch.rand((n_locations,1))*0).T
        self.time_start[0,0:self.n_initial_tasks] = 0
        self.display = display
        self.enable_topological_features = enable_topological_features

        self.task_graph_node_dim = self.generate_task_graph()[0].shape[1]
        self.agent_node_dim = self.generate_agents_graph()[0].shape[1]
 

        self.step_count = 0
        self.action_0_bounds = {
                2: (8, 15.1),
                3: (7.76, 14.97),
                4: (6.73, 11.59),
                5: (5.81, 9.4),
                6: (4.82, 7.99),
                7: (4.16, 6.80)
            }
        self.talent_beginned = [0.5,0.5]
        if self.enable_topological_features:
            self.observation_space = Dict(
                dict(
                    depot=Box(low=0, high=1, shape=(1, 2)),
                    mask=Box(low=0, high=1, shape=self.nodes_visited.shape),
                    topo_laplacian=Box(low=0, high=1, shape=(n_locations-1,n_locations-1)),
                    task_graph_nodes=Box(low=0, high=1, shape=(n_locations - 1, self.task_graph_node_dim)),
                    agents_graph_nodes=Box(low=0, high=1, shape=(n_agents, self.agent_node_dim)),
                    agent_taking_decision=Box(low=0, high=n_agents, shape=(1,1), dtype=int),
                ))
            self.topo_laplacian = None
            state = self.get_encoded_state()
            topo_laplacian = self.get_topo_laplacian(state)
            state["topo_laplacian"] = topo_laplacian
            self.topo_laplacian = topo_laplacian
        else:
            self.observation_space = Dict(
                dict(
                    depot=Box(low=0, high=1, shape=(1, 2)),
                    mask=Box(low=0, high=1, shape=self.nodes_visited.shape),
                    task_graph_nodes=Box(low=0, high=1, shape=(n_locations-1,self.task_graph_node_dim)),
                    task_graph_adjacency=Box(low=0, high=1, shape=(n_locations-1, n_locations-1)),
                    agents_graph_nodes=Box(low=0, high=1, shape=(n_agents, self.agent_node_dim)),
                    agent_taking_decision=Box(low=0, high=n_agents, shape=(1,1), dtype=int),
                    agent_talents=Box(low=4, high=14.97, shape=(1,2), dtype=np.float32),
                    step= Box(low=0, high=55, shape=(1,), dtype=int),
                    talent_beginned= Box(low=0, high=1, shape=(2,), dtype=np.float32)
                ))

        self.training = training
        self.distance = 0.0
        self.done = False
        self.mask = np.zeros(shape=(self.n_locations,1))
        self.mask[0,0] = 1

        
    def initialize(self, talents):
    #0 - range, 1 - speed, 2 - payload
        self.max_capacity = talents[2]
        self.max_range = talents[0]
        speed = talents[1]
        self.agent_speed = speed  # this param should be handles=d carefully. Makesure this is the same for the baselines
        #print(self.max_capacity, self.max_range, self.agent_speed)
        self.agents_current_range = torch.ones((1,self.n_agents), dtype=torch.float32)*self.max_range
        self.agents_current_payload = torch.ones((1,self.n_agents), dtype=torch.float32)*self.max_capacity
        saving = [self.max_capacity, self.max_range, self.agent_speed]
        #print(saving, "saving")
        self.agent_distance_travelled = torch.zeros((1, self.n_agents), dtype=torch.float32)
        self.agent_distance_travelled_per_trip = list()
        self.agent_task_completed = torch.zeros((1, self.n_agents), dtype=torch.float32)
        self.completed_tasks_distance = list()
        self.deadline_passed_distances = list()
        self.task_completed_info = list()
        self.time_deadlines_copy = copy.deepcopy(self.time_deadlines)
        saving = [self.max_capacity, self.max_range, self.agent_speed]
        self.packages_per_trip = {}
        for i in range(self.n_agents):
            key = i
            value = [0]
            self.packages_per_trip[key] = value
        self.distances_per_trip = {}
        for i in range(self.n_agents):
            key = i
            value = [0]
            self.distances_per_trip[key] = value
        with open('data1.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(saving)


    def get_state(self):
        # include locations visited into the state
        return np.concatenate((np.concatenate((self.locations, self.agents_destination_coordinates,
                                               self.agents_destination_coordinates[self.agent_taking_decision,
                                               :].reshape(1, 2)), axis=0).reshape(-1, 1),
                               self.nodes_visited.reshape(-1, 1)))



    def get_encoded_state(self):
        self.mask = self.get_mask()
        task_graph_nodes, task_graph_adjacency = self.generate_task_graph()
        agents_graph_nodes, agents_graph_adjacency = self.generate_agents_graph()
        agent_talents = np.array([ self.max_capacity, self.max_range]).reshape(2,1)
    
        # Normalize variables
        depot_normalized = normalize(self.depot.reshape(1, 2))
        task_graph_nodes_normalized = normalize(task_graph_nodes)
        task_graph_adjacency_normalized = normalize(task_graph_adjacency)
        agents_graph_nodes_normalized = normalize(agents_graph_nodes)
        step = np.array([self.step_count])
        #agent_taking_decision_normalized = normalize(self.agent_taking_decision)

        if self.enable_topological_features:
            state = {
                'depot': depot_normalized,
                'mask': self.mask,
                'task_graph_nodes': task_graph_nodes_normalized,
                'topo_laplacian': self.topo_laplacian, 
                'agents_graph_nodes': agents_graph_nodes_normalized,
                'agent_taking_decision': self.agent_taking_decision,
            }
        else:
            state = {
                'depot': self.depot.reshape(1, 2),
                'mask': self.mask,
                'task_graph_nodes': task_graph_nodes_normalized,
                'task_graph_adjacency': task_graph_adjacency_normalized,
                'agents_graph_nodes': agents_graph_nodes_normalized,
                'agents_graph_adjacency':agents_graph_adjacency,
                'agent_taking_decision': torch.tensor([[self.agent_taking_decision]]),
                'agent_talents': torch.tensor(agent_talents).reshape(1,2),
                'step': torch.tensor(step).reshape(1,),
                'talent_beginned': torch.tensor(self.talent_beginned)
            }
        return state


    def var_preprocess(self, adj, r):
        adj_ = adj + sp.eye(adj.shape[0])
        adj_ = adj_ ** r
        adj_[adj_ > 1] = 1
        rowsum = adj_.sum(1).A1
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5))
        adj_normalized = adj_.dot(degree_mat_inv_sqrt).T.dot(degree_mat_inv_sqrt).tocsr()
        return adj_normalized

    def get_topo_laplacian(self, data):
        # active_tasks = ((data['nodes_visited'] == 0).nonzero())[0]
        X_loc = (data['task_graph_nodes'].numpy())[None,:]
        # X_loc = X_loc[:, active_tasks[1:] - 1, :]
        # distance_matrix = ((((X_loc[:, :, None] - X_loc[:, None]) ** 2).sum(-1)) ** .5)[0]
        distance_matrix = torch.cdist(torch.tensor(X_loc), torch.tensor(X_loc),p=2)[0]

        adj_ = np.float32(distance_matrix < 0.3)

        adj_ = adj_ * (self.available_tasks[1:, :].T).numpy()
        adj_ = adj_ * (self.available_tasks[1:, :]).numpy()

        dt = defaultdict(list)
        for i in range(adj_.shape[0]):
            n_i = adj_[i, :].nonzero()[0].tolist()

            dt[i] = n_i

        adj = nx.adjacency_matrix(nx.from_dict_of_lists(dt))
        adj_array = adj.toarray().astype(np.float32)
        var_laplacian = self.var_preprocess(adj=adj, r=2).toarray()

        secondorder_subgraph = k_th_order_weighted_subgraph(adj_mat=adj_array, w_adj_mat=distance_matrix, k=2)

        reg_dgms = list()
        for i in range(len(secondorder_subgraph)):
            # print(i)
            tmp_reg_dgms = simplicial_complex_dgm(secondorder_subgraph[i])
            if tmp_reg_dgms.size == 0:
                reg_dgms.append(np.array([]))
            else:
                reg_dgms.append(np.unique(tmp_reg_dgms, axis=0))

        reg_dgms = np.array(reg_dgms)

        row_labels = np.where(var_laplacian > 0.)[0]
        col_labels = np.where(var_laplacian > 0.)[1]

        topo_laplacian_k_2 = np.zeros(var_laplacian.shape, dtype=np.float32)

        for i in range(row_labels.shape[0]):
            tmp_row_label = row_labels[i]
            tmp_col_label = col_labels[i]
            tmp_wasserstin_dis = wasserstein(reg_dgms[tmp_row_label], reg_dgms[tmp_col_label])
            # if tmp_wasserstin_dis == 0.:
            #     topo_laplacian_k_2[tmp_row_label, tmp_col_label] = 1. / 1e-1
            #     topo_laplacian_k_2[tmp_col_label, tmp_row_label] = 1. / 1e-1
            # else:
            topo_laplacian_k_2[tmp_row_label, tmp_col_label] = 1. / (tmp_wasserstin_dis+1)
            topo_laplacian_k_2[tmp_col_label, tmp_row_label] = 1. / (tmp_wasserstin_dis+1)

        return topo_laplacian_k_2

    def step(self, action):


        if self.step_count == 0:
            self.talent_beginned = [action[0],action[1]]
            condition, action_scaled = scale_action_values(action)
            if condition == True:
                self.initialize(action_scaled)
            elif condition == False:
                done = True
                obs = self.get_encoded_state()
                #print("failed," , action_scaled)
                return obs, -abs(action_scaled), done, {}  # Return the negative absolute reward value


        action = int(action[2])
        self.step_count += 1
        reward = 0.0

        agent_taking_decision = self.agent_taking_decision  # id of the agent taking action
        current_location_id = self.current_location_id  # current location id of the robot taking decision
        self.total_length = self.total_length + 1

        info = {}
        travel_distance = self.distance_matrix[current_location_id, action]
        self.agent_distance_travelled[0, agent_taking_decision] += travel_distance
        #agent_range = copy.deepcopy(self.agents_current_range[0, agent_taking_decision])
        self.agents_current_range[0, agent_taking_decision] -= travel_distance
        #if self.agents_current_range[0, agent_taking_decision] < 0:
            #print(["current location", current_location_id,"distance", travel_distance, "agent range", agent_range])
        self.agents_prev_decision_time[agent_taking_decision, 0] = self.time
        self.visited.append((action, self.agent_taking_decision))
        if action == self.depot_id: # if action is depot, then capacity is full, range is full
            rech_time = self.calculate_recharge_time(self.agents_current_range[0, agent_taking_decision])
            #print("remaining_range", self.agents_current_range[0, agent_taking_decision])
            #print(rech_time)
            self.agents_next_decision_time[agent_taking_decision] = self.time + travel_distance / self.agent_speed + rech_time
            self.agents_current_payload[0, agent_taking_decision] = torch.tensor(self.max_capacity)
            self.agents_current_range[0, agent_taking_decision] = torch.tensor(self.max_range)
            self.nodes_visited[action] = 0
            self.packages_per_trip[agent_taking_decision].append(0)
            self.distances_per_trip[agent_taking_decision].append(0)


        if self.nodes_visited[action] != 1 and action != self.depot_id:
            # range is reduced, capacity is reduced by 1
            self.packages_per_trip[agent_taking_decision][-1] +=1
            self.task_completed_info.append([self.locations[action], self.time + (travel_distance / self.agent_speed)])
            distance_covered = self.total_distance_travelled + travel_distance
            self.distances_per_trip[agent_taking_decision][-1] += travel_distance
            self.total_distance_travelled = distance_covered
            self.agents_distance_travelled[agent_taking_decision] += travel_distance
            self.agents_current_payload[0, agent_taking_decision] -= self.location_demand[0, action].item()
            #print(self.distance_matrix[self.depot_id, action])
            self.completed_tasks_distance.append([self.distance_matrix[self.depot_id, action], self.time_deadlines_copy[0, action]])
            #print("working here")
            # update the  status of the node_visited that was chosen
            self.nodes_visited[action] = 1
            self.agent_task_completed[0, agent_taking_decision] += 1
            #print(travel_distance / self.agent_speed)
            
            self.agents_next_decision_time[agent_taking_decision] = self.time + travel_distance / self.agent_speed
            if self.time_deadlines[0, action] < torch.tensor(self.time + (travel_distance / self.agent_speed)):
                self.deadline_passed[0, action] = 1
            else:
                self.task_done[0, action] = 1
                # reward = 1/(self.n_locations-1)
        self.total_reward += reward
        #print(self.time)
            # change destination of robot taking decision
        self.agents_next_location[agent_taking_decision] = action
        self.agents_prev_location[agent_taking_decision] = current_location_id
        self.agents_destination_coordinates[agent_taking_decision] = self.locations[action].copy()
        self.agents_distance_to_destination[agent_taking_decision] = travel_distance
        
        if self.display:
            self.render(action)

        # finding the agent which takes the next decision
        self.agent_taking_decision = np.argmin(self.agents_next_decision_time)
        self.current_location_id = self.agents_next_location[self.agent_taking_decision][0].copy()
        self.time = self.agents_next_decision_time[self.agent_taking_decision][0].copy()
        deadlines_passed_ids = (self.time_deadlines < torch.tensor(self.time)).nonzero()
        # if a task deadline is over, we set it as visited so its not selected again
        if deadlines_passed_ids.shape[0] != 0:

            self.deadline_passed[0, deadlines_passed_ids[:,1]] = 1
            self.nodes_visited[deadlines_passed_ids[:, 1], 0] = 1
        # print("Active tasks before update: ", self.active_tasks)
        self.active_tasks = ((self.nodes_visited == 0).nonzero())[0]


        self.available_tasks = (self.time_start <= self.time).to(torch.float32).T # making new tasks available

        if sum(self.nodes_visited) == self.n_locations - 1:
            final_distance_to_depot = torch.cdist(torch.tensor(self.agents_destination_coordinates), torch.tensor(self.depot[None,:])).sum().item()
            if self.task_done.sum() >= self.n_locations - 1:
                print("more success")
                reward = (self.n_locations - (self.n_locations - self.task_done.sum()))/self.n_locations
                reward = reward *10
            else:
                #modifying reward to be positive, max_tasks
                reward = (self.n_locations - (self.n_locations - self.task_done.sum()))/self.n_locations
                reward = reward *10
                #reward = -((self.n_locations - 1) - self.task_done.sum())/(self.n_locations - 1)
            #print(self.time)
            self.total_reward = reward
            self.done = True
            for index, value in np.ndenumerate(self.deadline_passed):
                if value == 1:
                    self.deadline_passed_distances.append([self.distance_matrix[self.depot_id, index], self.time_deadlines_copy[0,index]])
            result_scenario = dict()
            result_scenario["agent_distance_travelled"]= self.agent_distance_travelled
            result_scenario["agent_task_done"] = self.agent_task_completed
            result_scenario["total_distance_travelled"] = self.total_distance_travelled
            result_scenario["packs_per_trip"] = self.packages_per_trip
            result_scenario["missed_deadline_distances"] = self.deadline_passed_distances
            result_scenario["completed_mission_distances"] = self.completed_tasks_distance
            result_scenario["distances_per_trip"] = self.distances_per_trip
            result_scenario["task_completed_info"] = self.task_completed_info
            # Check if directory exists
            directory = "scenario_results_"+str(self.n_locations)+"_robs_" + str(self.n_agents)
            if not os.path.exists(directory):
                os.makedirs(directory)

            llst = len(os.listdir(directory))
            file_name = directory + "/result" + str(llst+1)+".npy"
            np.save(file_name, result_scenario)
            info = {"is_success": self.done,
                    "episode": {
                        "r": self.total_reward,
                        "l": self.total_length
                    }
                    }
            reward = reward.item()

        return self.get_encoded_state(), reward, self.done, info
    
    def calculate_recharge_time(self, current_range):
        if self.max_range > 1:  # Ensure division is valid
            s = current_range / (self.max_range - 1)
        else:
            s = 0  # Avoid division by zero if max_range is 1
        s = min(s, 1)
        recharge_time = 0.02 + (0.8 - 0.02) * (1 - s)
        #print(s, recharge_time, current_range, self.max_range)
        return recharge_time

    def get_mask(self):
        # masking:
        #   nodes visited - done
        #   capacity = 0 -> depot - done
        #   Range not sufficient to reach depot -> depot
        #   deadlines passed done
        #    if current location is depot, then mask the depot - done
        agent_taking_decision = self.agent_taking_decision
        mask = self.nodes_visited.copy()
        current_location_id = self.current_location_id
        if self.agents_current_payload[0, agent_taking_decision] == 0:
            mask[1:,0] = 1
            mask[0, 0] = 0
        else:
            unreachbles = (self.distance_matrix[0,:] + self.distance_matrix[current_location_id,:] > self.agents_current_range[0, agent_taking_decision].item()).nonzero()
            #print(self.distance_matrix[0,:], "distance matrix")
            #print(self.distance_matrix[current_location_id,:], "from current location")
            #print(self.agents_current_range[0, agent_taking_decision], "current_range")
            #print(unreachbles, "no reaching")
            if unreachbles[0].shape[0] != 0:
                mask[unreachbles[0], 0] = 1
            mask = np.logical_or(mask, (self.deadline_passed.T).numpy()).astype(mask.dtype)
            if mask[1:,0].prod() == 1: # if no other feasible locations, then go to depot
                mask[0,0] = 0
        if current_location_id == self.depot_id:
            mask[0, 0] = 1
        if mask.prod() != 0.0:
            mask[0,0] = 0
        #print("mask before numpy", mask)
        mask = mask*(self.available_tasks).numpy() # making unavailable tasks
        #print("mask after numpy", mask)
        return mask

    def generate_task_graph(self):

        locations = torch.tensor(self.locations)
        time_deadlines = self.time_deadlines.T
        location_demand = self.location_demand.T
        deadlines_passed = self.deadline_passed.T
        nodes_visited = torch.tensor(self.nodes_visited)

        #print(deadlines_passed.shape)
        node_properties = torch.cat((locations, time_deadlines, location_demand, deadlines_passed, nodes_visited), dim=1)
        node_properties = node_properties[1:, :] # excluding the depot
        node_properties[:, 0:4] = node_properties[:, 0:4]/node_properties[:, 0:4].max(dim=0).values # normalizing all except deadline_passed
        distance_matrix = torch.cdist(node_properties, node_properties)
        adjacency_matrix = 1/(1+torch.cdist(node_properties, node_properties))
        adjacency_matrix = adjacency_matrix*(distance_matrix>0).to(torch.float32) # setting diagonal elements as 0
        node_properties = node_properties[:,:]*self.available_tasks[1:,:] # masking the unavailable tasks
        adjacency_matrix = adjacency_matrix*(self.available_tasks[1:,:].T)
        adjacency_matrix = adjacency_matrix*self.available_tasks[1:,:]
        return node_properties, adjacency_matrix

    def generate_agents_graph(self):
        try:
            node_properties = torch.cat((torch.tensor(self.agents_destination_coordinates), self.agents_current_range.T, self.agents_current_payload.T, torch.tensor(self.agents_next_decision_time)), dim=1)
        except:
            self.agents_current_range = torch.ones((1,self.n_agents), dtype=torch.float32)*5
            self.agents_current_payload = torch.ones((1,self.n_agents), dtype=torch.float32)*7
            node_properties = torch.cat((torch.tensor(self.agents_destination_coordinates), self.agents_current_range.T, self.agents_current_payload.T, torch.tensor(self.agents_next_decision_time)), dim=1)

        distance_matrix = torch.cdist(node_properties, node_properties)
        adjacency_matrix = 1 / (1 + torch.cdist(node_properties, node_properties))
        adjacency_matrix = adjacency_matrix * (distance_matrix > 0).to(torch.float32) # setting diagonal elements as 0
        return node_properties, adjacency_matrix

    def render(self, action):

        # Show the locations

        plt.plot(self.locations[0, 0], self.locations[0, 1], 'bo')
        for i in range(1, self.n_locations):
            if self.available_tasks[i, 0] == 1:
                if self.task_done[0, i] == 1:
                    plt.plot(self.locations[i, 0], self.locations[i, 1], 'go')
                elif self.nodes_visited[i, 0] == 0 and self.deadline_passed[0, i] == 0:
                    plt.plot(self.locations[i, 0], self.locations[i, 1], 'ro')
                elif self.deadline_passed[0, i] == 1:
                    plt.plot(self.locations[i, 0], self.locations[i, 1], 'ko')
        plt.plot(self.locations[action, 0], self.locations[action, 1], 'mo')
        prev_loc = self.locations[self.agents_prev_location][:, 0, :]
        next_loc = self.locations[self.agents_next_location][:, 0, :]
        diff = next_loc - prev_loc
        velocity = np.zeros((self.n_agents, 2))
        for i in range(self.n_agents):
            if diff[i, 0] == 0 and diff[i, 1] == 0:
                velocity[i, 0] = 0
                velocity[i, 1] = 0
            else:
                direction = diff[i, :] / (np.linalg.norm(diff[i, :]))
                velocity[i, :] = direction * self.agent_speed

        prev_time = self.time
        current_agent_locations = prev_loc + (prev_time - self.agents_prev_decision_time) * velocity

        agent_taking_decision = np.argmin(self.agents_next_decision_time)
        # current_location_id = self.agents_next_location[agent_taking_decision][0].copy()
        next_time = self.agents_next_decision_time[agent_taking_decision][0].copy()
        delta_t = (next_time - prev_time) / 10
        curr_time = prev_time
        # for i in range(10):

        current_agent_locations = current_agent_locations + velocity * delta_t
        plt.plot(current_agent_locations[:, 0], current_agent_locations[:, 1], 'mv')
        curr_time = curr_time + delta_t
        deadlines_passed_ids = (self.time_deadlines < torch.tensor(curr_time)).nonzero()
        time.sleep(0.01)

        # print(prev_loc)
        # print(next_loc)
        # print("***********")
        for i in range(self.n_agents):
            plt.arrow(prev_loc[i, 0], prev_loc[i, 1], diff[i, 0]*0.95, diff[i, 1]*0.95, width=0.005)
        plt.draw()
        time.sleep(1)
        plt.show()
        plt.clf()
        #   Grey as unavailable
        #   Red as active
        #   Green as done
        #   Black as deadline passed and not completed
        # Current location of the robots
        # Show arrow for destination
        # Encircle robot taking decision
        # encircle decision taken
        # Show movement inbetween decision-making


    def reset(self):
        self.actions_vals = []
        if self.training:
            self.step_count = 0
            self.locations = np.random.random((self.n_locations, 2)) * 5
            self.depot = self.locations[0, :]
            self.visited = []
            self.agent_taking_decision = 1
            self.agents_prev_location = np.zeros((self.n_agents, 1), dtype=int)
            self.agents_next_location = np.zeros((self.n_agents, 1), dtype=int)
            self.agents_distance_travelled = np.zeros((self.n_agents, 1))
            self.total_distance_travelled = 0.0
            self.agent_taking_decision = 0
            self.current_location_id = 0
            self.nodes_visited = np.zeros((self.n_locations, 1))
            self.agents_distance_to_destination = np.zeros(
                (self.n_agents, 1))  # tracks the distance to destination from its current point for all robots
            self.distance_matrix = np.linalg.norm(self.locations[:, None, :] - self.locations[None, :, :], axis=-1)
            self.time = 0.0
            #self.agent_speed = 0.4
            self.agents_next_decision_time = np.zeros((self.n_agents, 1))
            self.agents_prev_decision_time = np.zeros((self.n_agents, 1))
            self.agents_destination_coordinates = np.ones((self.n_agents, 1)) * self.depot
            self.total_reward = 0.0
            self.total_length = 0
            #self.agents_current_range = torch.ones((1, self.n_agents), dtype=torch.float32) * self.max_range
            #self.agents_current_payload = torch.ones((1, self.n_agents), dtype=torch.float32) * self.max_capacity
            self.time_deadlines = (torch.tensor(np.random.random((1, self.n_locations))) * .3 + .7) * 2
            self.time_deadlines[0, 0] = 1000000 # large number for depot,
            self.location_demand = torch.ones((1, self.n_locations), dtype=torch.float32)
            self.task_done = torch.zeros((1, self.n_locations), dtype=torch.float32)
            self.deadline_passed = torch.zeros((1, self.n_locations), dtype=torch.float32)
            self.active_tasks = ((self.nodes_visited == 0).nonzero())[0]
            # Reset the number of not-done tasks
            self.done = False

            if not self.enable_dynamic_tasks: # this conditional moight be unnecessary
                n_initial_tasks = self.n_locations
            else:
                n_initial_tasks = self.n_initial_tasks
            self.n_initial_tasks = n_initial_tasks
            self.available_tasks[0: n_initial_tasks, 0] = 1 # set the initial tasks available
            self.time_start = self.time_deadlines*(torch.rand((self.n_locations,1))*0).T
            self.time_start[0,0:self.n_initial_tasks] = 0
            self.actions_vals = []
        # self.mask = np.zeros(shape=(self.n_locations, 1))
        # self.mask[0, 0] = 1
        state = self.get_encoded_state()
        if self.enable_topological_features:
            self.topo_laplacian = None

            topo_laplacian = self.get_topo_laplacian(state)
            state["topo_laplacian"] = topo_laplacian
            self.topo_laplacian = topo_laplacian
        return state