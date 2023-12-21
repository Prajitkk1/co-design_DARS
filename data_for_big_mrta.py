"""
Author: Steve Paul 
Date: 7/4/22 """

import pickle
import numpy as np
import os

problem = "MRTA_Flood"
task_type = "ND"

trained_model_n_loc = 51
trained_model_n_robots = 6
loc_test_multipliers = [0.5,1,2,5,10]
robot_test_multipliers = [0.5,1,2]

for loc_mult in loc_test_multipliers:
    for rob_mult in robot_test_multipliers:
        n_robots_test = int(rob_mult*loc_mult*trained_model_n_robots) + 1
        n_loc_test = int(trained_model_n_loc*loc_mult)

        path = "Test_data/" + problem + "/"
        nAllTasks = n_loc_test
        nRobot = n_robots_test
        file_name = path + problem \
                    + "_nloc_" + str(n_loc_test) \
                    + "_nrob_" + str(n_robots_test) + "_" + task_type + ".pkl"
        with open(file_name, 'rb') as fl:
            test_envs = pickle.load(fl)
        fl.close()
        data_list = []
        for env in test_envs:
            locations = env.envs[0].locations[1:,:]
            deadlines = env.envs[0].time_deadlines[0, 1:].unsqueeze(dim=1).numpy()
            depot = locations[0,:]
            rnd = np.random.random((n_loc_test-1, 1))
            dat = np.concatenate([locations, rnd, deadlines], axis=1)
            data = {
                'taskData': dat,
                'depotLocation': depot,
                'n_locations': n_loc_test,
                'n_robots': n_robots_test
            }
            data_list.append(data)
        mode = 0o755
        path = "baseline_methods/BIGMRTA/Data/" + problem + "/"
        if not os.path.exists(path):
            os.makedirs(path, mode)
        file_name = path + problem \
                    + "_nloc_" + str(n_loc_test) \
                    + "_nrob_" + str(n_robots_test) + "_" + task_type + ".pkl"
        with open(file_name, 'wb') as fl:
            pickle.dump(data_list, fl)
        fl.close()