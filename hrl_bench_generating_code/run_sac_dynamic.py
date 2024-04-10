import pickle

import gym
import torch.optim
from typing import Union, List

from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.atari_wrappers import AtariWrapper
import sys
import os
import time
import gym
import numpy as np
import matplotlib.pyplot as plt
import json
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecMonitor

ATARI_ENVS = ['Pong-v0', 'Alien-v0', 'BankHeist-v0',
              'BeamRider-v0', 'Breakout-v0', 'Enduro-v0',
              'Phoenix-v0', 'Seaquest-v0', 'SpaceInvaders-v0',
              'Riverraid-v0', 'Tennis-v0', 'Skiing-v0', 'Boxing-v0',
              'Bowling-v0', 'Asteroids-v0']
CONTROL_ENVS = ['CartPole-v1', 'MountainCar-v0', 'Acrobot-v1', 'Pendulum-v0']

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level.
    """
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1, seed: int = 12345, start_time: float = time.time()):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf
        self.seed = seed
        self.start_time = time.time()
        self.timestamps = []
        self.total = 0

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward

        return True
    def _on_rollout_end(self) -> None:
        self.timestamps.append(time.time() - self.start_time)

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        # Retrieve training reward
        x, y = ts2xy(load_results(self.log_dir), 'timesteps')
        if len(x) > 0:
              # Mean training reward over the last 100 episodes
             mean_reward = np.mean(y[-100:])
             # for idx, re in enumerate(y):
             #     print("Episode %s: %s" % (idx+1, re))
             self.total += self.num_timesteps
             if self.verbose > 0:
                print(f"Num timesteps: {self.num_timesteps} ({self.total})")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")
             print("Rewards Train: %s" % y[-1])
             print("Times Train: %s" % x[-1])
             print("Times Train: %s" % self.timestamps[-1])
             with open(os.path.join(self.log_dir, "rewards_train"), 'wb') as f:
                pickle.dump(y.tolist(), f)
             with open(os.path.join(self.log_dir, "timesteps_train"), 'wb') as g:
                pickle.dump(x.tolist(), g)
        with open(os.path.join(self.log_dir, "timestamps_train"), 'wb') as g:
            pickle.dump(self.timestamps, g)


        return True


def dynamic_change_hyperparameters(model, learning_rate, gamma):
    model.learning_rate = 10**learning_rate
    model._setup_lr_schedule()
    model.gamma = gamma
    print(model.learning_rate, model.gamma)


def run_ppo(learning_rate: Union[List[float], float], gamma: Union[List[float], float],
            environment: str = 'CartPole-v1', policy: str = 'MlpPolicy',
            total_timesteps: float = 1e6, switch_every: int = 0, seed: int = 12345):
    # seed = 0
    # Create log dir
    if isinstance(learning_rate, list):
        lr_str = f'{"".join(str(lr) for lr in learning_rate)}'
    else:
        lr_str = str(learning_rate)
    if isinstance(gamma, list):
        gamma_str = f'{"".join(str(g) for g in gamma)}'
    else:
        gamma_str = str(gamma)
    log_dir = "tmp%s_%s_%s_%s/" % (environment, lr_str, gamma_str, str(seed))
    os.makedirs(log_dir, exist_ok=True)
    # Create the callback: check every 1000 steps
    model = None
    if environment in ATARI_ENVS:
        env = make_atari_env(environment, n_envs=8, seed=seed)
        env = VecMonitor(env, log_dir)
        # env = AtariWrapper(env)
    else:
        env = gym.make(environment)
        env = Monitor(env, log_dir)
    start = time.time()
    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir, seed=seed, start_time=start)
    rewards = []
    std_rewards = []
    times_eval = []
    timesteps_eval = []
    used_config = []

    # Train the agent
    steps_per_learn_call = 1e4
    timesteps = int(total_timesteps/steps_per_learn_call)
    switch_index = 0
    total_steps = 0
    next_switch = switch_every
    for i in range(timesteps):
        # Create and wrap the environment
        eval_env = gym.make(environment)
        if model is None:
            model = SAC(policy, env, verbose=0,
                        learning_rate=10**learning_rate[switch_index], gamma=gamma[switch_index],
                        seed=seed)
            switch_index += 1
        else:
            model = SAC.load(path=os.path.join(log_dir, "ppo_model"), env=env)
        env.reset()
        print(total_steps, next_switch)
        if switch_every > 0 and total_steps >= next_switch and switch_index < len(learning_rate):
            print(switch_index, learning_rate, gamma)
            dynamic_change_hyperparameters(model, learning_rate[switch_index], gamma[switch_index])
            switch_index += 1
            next_switch += switch_every
            print('SWITCHED')
        model.learn(total_timesteps=int(steps_per_learn_call), callback=callback)  # calls train which calls update_learning_rate which sets the optimizers learning rate according to the schedule
        total_steps += steps_per_learn_call
        # Returns average and standard deviation of the return from the evaluation
        r, std_r = evaluate_policy(model=model, env=eval_env)
        times_eval.append(time.time() - start)
        used_config.append((learning_rate[switch_index-1], gamma[switch_index-1]))
        rewards.append(r)
        std_rewards.append(std_r)
        with open(os.path.join(log_dir, "timesteps_train"), 'rb') as f:
            timesteps_train = pickle.load(f)
        timesteps_eval.append(timesteps_train[-1])
        print("Rewards %s" % rewards[-1])
        print("Std rewards %s" % std_rewards[-1])
        print("Times: %s" % times_eval[-1])
        print("Timesteps Eval: %s" % timesteps_eval[-1])
        print(f"Used Config: lr {used_config[-1][0]} gamma {used_config[-1][1]}")
        model.save(os.path.join(log_dir, "ppo_model"))
    with open(os.path.join(log_dir, "rewards_train"), 'rb') as f:
        rewards_train = pickle.load(f)
    with open(os.path.join(log_dir, "timesteps_train"), 'rb') as f:
        timesteps_train = pickle.load(f)
    with open(os.path.join(log_dir, "timestamps_train"), 'rb') as f:
        timestamps_train = pickle.load(f)
    data = {"gamma": gamma, "learning_rate": learning_rate,
                         "returns_eval": rewards, "std_returns_eval": std_rewards, "timestamps_eval": times_eval,
            "timesteps_eval": timesteps_eval,
            "returns_train": rewards_train, "timesteps_train": timesteps_train, "timestamps_train": timestamps_train,
            "switch_every": switch_every, "seed": seed, "configs": used_config}
    with open("%s_SAC_random_lr_%s_gamma_%s_seed%s_eval.json" % (environment, lr_str, gamma_str, seed),
              'w+') as f:
        json.dump(data, f)
    return rewards, std_rewards


seed = int(sys.argv[1])
env_id = int(sys.argv[2])
with open('config_space_ppo', 'rb') as f:
    config_space_dqn = pickle.load(f)
new_config_space = []
for elem in config_space_dqn:
    if elem[1] < 0.95 or elem[1] > 0.99:
        pass
    elif elem[0] == -6 or elem[0] > -3:
        pass
    else:
        if elem[:-1] not in new_config_space:
            new_config_space.append(elem[:-1])
config_space_dqn = new_config_space
print(config_space_dqn)
print(len(config_space_dqn))

switch_every=int(sys.argv[3])
total_timesteps = 1e6
switches = int(total_timesteps/switch_every)
print(f'Will switch {switches - 1} times')
config_ids = list(map(int, sys.argv[4:]))
print(len(config_ids), switches)
assert len(config_ids) == switches

lr = []
gamma = []
for conf_id in config_ids:
    lr.append(config_space_dqn[conf_id][0])
    gamma.append(config_space_dqn[conf_id][1])

print(lr)
print(gamma)
if isinstance(lr, list):
    lr_str = f'{"".join(str(l) for l in lr)}'
else:
    lr_str = str(lr)
if isinstance(gamma, list):
    gamma_str = f'{"".join(str(g) for g in gamma)}'
else:
    gamma_str = str(gamma)
if not os.path.exists("%s_SAC_random_lr_%s_gamma_%s_seed%s_eval.json" % (CONTROL_ENVS[env_id], lr_str, gamma_str, seed)):
    r, std_r = run_ppo(learning_rate=lr, gamma=gamma, environment=CONTROL_ENVS[env_id], policy='MlpPolicy',
                   switch_every=switch_every, seed=seed, total_timesteps=total_timesteps)
