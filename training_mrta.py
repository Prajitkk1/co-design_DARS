"""
Author: Steve Paul 
Date: 4/15/22 """
""
import warnings
from stable_baselines3 import PPO
from MRTA_Flood_Env import MRTA_Flood_Env
import torch
from topology import *
import pickle
import os
from CustomPolicies import ActorCriticGCAPSPolicy
from training_config import get_config
from stable_baselines3.common.vec_env import DummyVecEnv,VecCheckNan #, SubprocVecEnv
import gc
from typing import Callable, Dict, List, Optional, Tuple, Type, Union
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback

gc.collect()
warnings.filterwarnings('ignore')
torch.cuda.empty_cache()
torch.autograd.set_detect_anomaly(True)
from stable_baselines3.common.vec_env import VecMonitor
def as_tensor(observation):
    for key, obs in observation.items():
        observation[key] = torch.tensor(obs)
    return observation
n_envs = 15 # Number of environments you want to run in parallel, 16 for training, 1 for test
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
#log_dir = "/results"
#os.makedirs(log_dir, exist_ok=True)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results


def make_env(config, seed, log_dir):
    def _init():
        env = MRTA_Flood_Env(
            n_locations=config.n_locations,
            n_agents=config.n_robots,
            max_capacity=config.max_capacity,
            max_range=config.max_range,
            enable_dynamic_tasks=config.enable_dynamic_tasks,
            display=False,
            enable_topological_features=config.enable_topological_features
        )
        env.seed(seed)
        env = Monitor(env,os.path.join(log_dir, str(seed)) )
        return env
    return _init
config = get_config()
test = True 	  # if this is set as true, then make sure the test data is generated.
# Otherwise, run the test_env_generator script
config.device = torch.device("cuda:0" if config.use_cuda else "cpu")
#config.device = torch.device( "cpu")
from stable_baselines3.common.monitor import load_results

from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import os
from stable_baselines3.common.callbacks import BaseCallback

from stable_baselines3.common.callbacks import BaseCallback
import os
import numpy as np

log_dir = "tmp2"
os.makedirs(log_dir, exist_ok=True)



class SaveOnBestTrainingRewardCallback(BaseCallback):
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_dir = os.path.join(log_dir, 'models')
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        os.makedirs(self.save_dir, exist_ok=True)
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Save model at every check_freq
            interval_save_path = os.path.join(self.save_dir, f"model_at_step_{self.num_timesteps}.zip")
            self.model.save(interval_save_path)

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print(f"Saving new best model to {self.save_path}.zip")
                    self.model.save(self.save_path)

        return True





def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func

def learning_rate_schedule(initial_value: float) -> Callable[[float], float]:

    def func(progress_remaining: float) -> float:
        decay_rate = 1.2
        return initial_value * math.exp((-progress_remaining**2*decay_rate))

        # return  initial_value
    return func

if __name__ == '__main__':
    if config.enable_dynamic_tasks:
        task_type = "D"
    else:
        task_type = "ND"

    if config.node_encoder == "CAPAM" or config.node_encoder == "MLP":
        tb_logger_location = config.logger+config.problem\
                         + "/" + config.node_encoder + "/" \
                        + config.problem\
                              + "_nloc_" + str(config.n_locations)\
                             + "_nrob_" + str(config.n_robots) + "_" + task_type + "_"\
                             + config.node_encoder\
                             + "_K_" + str(config.K) \
                             + "_P_" + str(config.P) + "_Le_" + str(config.Le) \
                             + "_h_" + str(config.features_dim)
        save_model_loc = config.model_save+config.problem\
                         + "/" + config.node_encoder + "/" \
                        + config.problem\
                              + "_nloc_" + str(config.n_locations)\
                             + "_nrob_" + str(config.n_robots) + "_" + task_type + "_"\
                             + config.node_encoder\
                             + "_K_" + str(config.K) \
                             + "_P_" + str(config.P) + "_Le_" + str(config.Le) \
                             + "_h_" + str(config.features_dim)
    elif config.node_encoder == "AM":
        tb_logger_location = config.logger + config.problem \
                             + "/" + config.node_encoder + "/" \
                             + config.problem \
                             + "_nloc_" + str(config.n_locations) \
                             + "_nrob_" + str(config.n_robots) + "_" + task_type + "_" \
                             + config.node_encoder \
                             + "_n_heads_" + str(config.n_heads) \
                             + "_Le_" + str(config.Le) \
                             + "_h_" + str(config.features_dim)
        save_model_loc = config.model_save + config.problem \
                         + "/" + config.node_encoder + "/" \
                         + config.problem \
                         + "_nloc_" + str(config.n_locations) \
                         + "_nrob_" + str(config.n_robots) + "_" + task_type + "_" \
                         + config.node_encoder \
                         + "_n_heads_" + str(config.n_heads) \
                         + "_Le_" + str(config.Le) \
                         + "_h_" + str(config.features_dim)
    single_env_creator = make_env(config, seed=0,log_dir = log_dir)
    single_env = single_env_creator()
    task_graph_node_dim = single_env.task_graph_node_dim
    agent_node_dim = single_env.agent_node_dim

    policy_kwargs=dict(
    features_extractor_kwargs=dict(
        feature_extractor=config.node_encoder,
        features_dim=config.features_dim,
        K=config.K,
        Le=config.Le,
        P=config.P,
        node_dim=task_graph_node_dim,
        agent_node_dim=agent_node_dim,
        n_heads=config.n_heads,
        tda=config.tda,
        tanh_clipping=config.tanh_clipping,
        mask_logits=config.mask_logits,
        temp=config.temp
    ),
    device=config.device
)




    envs = [make_env(config, seed=i, log_dir = log_dir) for i in range(n_envs)]
    env = SubprocVecEnv(envs)
    env = VecMonitor(env)
    #env= VecCheckNan(env, raise_exception=True)
    model = PPO(
     
        ActorCriticGCAPSPolicy,
            env,
            gamma=config.gamma,
            verbose=1,
            n_epochs=config.n_epochs,
            batch_size=config.batch_size,
            tensorboard_log=tb_logger_location,
            # create_eval_env=True,
            n_steps=config.n_steps,
            normalize_advantage = True,
            learning_rate= 0.00005,
            policy_kwargs = policy_kwargs,
            ent_coef=config.ent_coef,
            vf_coef=config.val_coef,
            device=config.device
        )
    custom_objects = {'learning_rate': 0.00001}
    #model = PPO.load("best_model", env=env, custom_objects = custom_objects)
    reward_threshold = 10.005
    #save_path = save_model_loc
    callback = SaveOnBestTrainingRewardCallback(check_freq=2000, log_dir=log_dir)
    
    if not test:

        model.learn(total_timesteps=config.total_steps,reset_num_timesteps=True, callback=callback)

        obs = env.reset()
        model.save(save_model_loc)
    if test:
        model = PPO.load(save_model_loc, env=env)

        trained_model_n_loc = config.n_locations
        trained_model_n_robots = config.n_robots
        loc_test_multipliers = [51,101,151]
        robot_test_multipliers = [5,10,15]
        path =  "Test_data/" + config.problem + "/"
        for loc_mult in loc_test_multipliers:
            for rob_mult in robot_test_multipliers:
                #n_robots_test = int(rob_mult*loc_mult*trained_model_n_robots)
                #n_loc_test = int(trained_model_n_loc*loc_mult)
                n_robots_test = int(rob_mult)
                n_loc_test = int(loc_mult)

                env = DummyVecEnv([lambda: MRTA_Flood_Env(
                        n_locations = n_loc_test,
                        n_agents = n_robots_test,
                        max_capacity = config.max_capacity,
                        max_range = config.max_range,
                        enable_dynamic_tasks=config.enable_dynamic_tasks,
                        display = False,
                        enable_topological_features = config.enable_topological_features
                )])

                file_name = path + config.problem\
                                        + "_nloc_" + str(n_loc_test)\
                                         + "_nrob_" + str(n_robots_test) + "_" + task_type + ".pkl"
                with open(file_name, 'rb') as fl:
                    test_envs = pickle.load(fl)
                fl.close()
                total_rewards_list = []
                distance_list = []
                total_tasks_done_list = []
                for env in test_envs:
                    env.envs[0].training = False
                    model.env = env
                    obs = env.reset()
                    obs = as_tensor(obs)
                    for i in range(1000000):
                            with torch.no_grad():
                                model.policy.set_training_mode(False)
                                action = model.policy._predict(obs)
                                action = action.cpu().detach().numpy()
                                #print(action)
                                obs, reward, done, _ = env.step(action)
                                obs = as_tensor(obs)
                                if done:
                                        total_rewards_list.append(reward)
                                        distance_list.append(env.envs[0].total_distance_travelled)
                                        total_tasks_done_list.append(env.envs[0].task_done.sum())
                                        print(env.envs[0].task_done.sum())
                                        break

                total_rewards_array = np.array(total_rewards_list)
                distance_list_array = np.array(distance_list)
                total_tasks_done_array = np.array(total_tasks_done_list)
                if config.node_encoder == "CAPAM" or config.node_encoder == "MLP":
                    encoder = config.node_encoder\
                                             + "_K_" + str(config.K) \
                                             + "_P_" + str(config.P) + "_Le_" + str(config.Le) \
                                             + "_h_" + str(config.features_dim)
                else:
                    encoder = config.node_encoder \
                             + "_n_heads_" + str(config.n_heads) \
                             + "_Le_" + str(config.Le) \
                             + "_h_" + str(config.features_dim)
                data = {
                    "problem": config.problem,
                    "n_locations": n_loc_test,
                    "n_robots": n_robots_test,
                    "dynamic_task": config.enable_dynamic_tasks,
                    "policy":encoder,
                    "total_tasks_done": total_tasks_done_array,
                    "total_rewards": total_rewards_array,
                    "distance": distance_list_array
                }

                result_path = "Results/" + config.problem + "/"

                result_file = result_path + config.problem + "_nloc_" + str(n_loc_test) \
                              + "_nrob_" + str(n_robots_test) + "_" + task_type + "_" + encoder
                mode = 0o755
                if not os.path.exists(result_path):
                    os.makedirs(result_path, mode)
                with open(result_file, 'wb') as fl:
                    pickle.dump(data, fl)
                fl.close()
