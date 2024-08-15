"""
Author: Steve Paul 
Date: 6/18/22 """
import pickle
from stable_baselines3.common.vec_env import DummyVecEnv#, SubprocVecEnv
from MRTA_Flood_Env import MRTA_Flood_Env
from MRTA_Flood_PO_Env import MRTA_Flood_PO_Env
import os

problem_types = ["PO", None] # set this as none or false if partial observation envs are not required
for problem_type in problem_types:
    if problem_type:
        from training_config_PO import get_config
    else:
        from training_config import get_config
    config = get_config()
    n_test_Scenarios = 250 # number of test scenarios
    trained_model_n_loc = config.n_locations
    trained_model_n_robots = config.n_robots
    #loc_test_multipliers = [1,1.25,1.5, 2, 3]
    #robot_test_multipliers = [1, 1.25,1.5, 2 ,3]
    loc_test_multipliers = [51,101,151]
    robot_test_multipliers = [5,10,15]
    for loc_mult in loc_test_multipliers:
        for rob_mult in robot_test_multipliers:
            # n_robots_test = int(rob_mult*loc_mult*trained_model_n_robots)
            # n_loc_test = int(trained_model_n_loc*loc_mult)
            n_robots_test = int(rob_mult)
            n_loc_test = int(loc_mult)

            env_list = []
            for i in range(n_test_Scenarios):
                if config.problem == "MRTA_Flood":
                    env = DummyVecEnv([lambda: MRTA_Flood_Env(
                            n_locations = n_loc_test,
                            n_agents = n_robots_test,
                            max_capacity = config.max_capacity,
                            max_range = config.max_range,
                            enable_dynamic_tasks=config.enable_dynamic_tasks,
                            display = False,
                            enable_topological_features = config.enable_topological_features
                    )])
                else:
                    env = DummyVecEnv([lambda: MRTA_Flood_PO_Env(
                        n_locations=n_loc_test,
                        n_agents=n_robots_test,
                        max_capacity=config.max_capacity,
                        max_range=config.max_range,
                        enable_dynamic_tasks=config.enable_dynamic_tasks,
                        display=False,
                        enable_topological_features=config.enable_topological_features,
                        agents_info_exchange_distance_threshold=config.agents_info_exchange_distance_threshold
                    )])
                env_list.append(env)

            if config.enable_dynamic_tasks:
                task_type = "D"
            else:
                task_type = "ND"
            mode =  0o755
            path =  "Test_data/" + config.problem + "/"
            if not os.path.exists(path):
                os.makedirs(path, mode)
            file_name = path + config.problem\
                                    + "_nloc_" + str(n_loc_test)\
                                     + "_nrob_" + str(n_robots_test) + "_" + task_type + ".pkl"
            with open(file_name, 'wb') as fl:
                pickle.dump(env_list, fl)
            fl.close()