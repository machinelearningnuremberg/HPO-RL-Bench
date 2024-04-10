import pickle

import gym

from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecEnvWrapper, DummyVecEnv
from stable_baselines3.common.atari_wrappers import AtariWrapper
import sys
import os
import time
import gym
import numpy as np
import matplotlib.pyplot as plt
import json
from stable_baselines3 import TD3
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
MUJOCO_ENVS = ['Ant-v2', 'Hopper-v2', 'Humanoid-v2']
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
             if self.verbose > 0:
                print(f"Num timesteps: {self.num_timesteps}")
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

def make_environment(environment, seed):
    def _init():
        env = gym.make(environment)
        env.seed(seed)
        return env
    return _init

def run_ppo(learning_rate: float, gamma: float, clip: float, layers:int, units:int,
            environment: str = 'CartPole-v1', policy: str = 'MlpPolicy',
            total_timesteps: float = 1e6, seed: int = 1):
    net_arch = []
    for layer in range(layers):
        net_arch.append(units)
    policy_kwargs = dict(net_arch=[dict(pi=net_arch.copy(), vf=net_arch.copy())])
    # Create log dir
    log_dir = "/work/dlclarge2/shalag-HPORLBench/tmp_DQN_%s_%s_%s_%s_%s_%s_%s/" % (environment, learning_rate, gamma, clip, layers, units, seed)
    os.makedirs(log_dir, exist_ok=True)
    # Create the callback: check every 1000 steps
    model = None
    if environment in ATARI_ENVS:
        env = make_atari_env(environment, n_envs=8, seed=0)
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

    # Train the agent
    timesteps = int(total_timesteps/1e4)
    print("Running for %s iterations" % timesteps)
    for i in range(timesteps):
        print("Iteration %s" % i)
        if environment in ATARI_ENVS:
            eval_env = make_atari_env(environment, n_envs=5, seed=0)
        else:
            eval_env = gym.make(environment)
            eval_env = Monitor(eval_env)
        if model is None:
            # Because we use parameter noise, we should use a MlpPolicy with layer normalization
            model = DQN(policy, env, verbose=0, learning_rate=10**learning_rate, gamma=gamma, seed=seed, batch_size=64,
                        buffer_size=100000, learning_starts=1000, target_update_interval=10, train_freq=256,
                        gradient_steps=128,
                        exploration_initial_eps=clip, policy_kwargs=policy_kwargs)
        else:
            model = DQN.load(path=os.path.join(log_dir, "ppo_model"), env=env)
        env.reset()
        model.learn(total_timesteps=int(1e4), callback=callback)
        # Returns average and standard deviation of the return from the evaluation
        r, std_r = evaluate_policy(model=model, env=eval_env)
        times_eval.append(time.time() - start)
        rewards.append(r)
        std_rewards.append(std_r)
        with open(os.path.join(log_dir, "timesteps_train"), 'rb') as f:
            timesteps_train = pickle.load(f)
        timesteps_eval.append(timesteps_train[-1])
        print("Rewards %s" % rewards[-1])
        print("Std rewards %s" % std_rewards[-1])
        print("Times: %s" % times_eval[-1])
        print("Timesteps Eval: %s" % timesteps_eval[-1])
        model.save(os.path.join(log_dir, "ppo_model"))
        with open(os.path.join(log_dir, "rewards_train"), 'rb') as f:
            rewards_train = pickle.load(f)
        with open(os.path.join(log_dir, "timesteps_train"), 'rb') as f:
            timesteps_train = pickle.load(f)
        with open(os.path.join(log_dir, "timestamps_train"), 'rb') as f:
            timestamps_train = pickle.load(f)
        data = {"gamma": gamma, "learning_rate": learning_rate, "epsilon": clip, "layers": layers, "units": units,
                         "returns_eval": rewards, "std_returns_eval": std_rewards, "timestamps_eval": times_eval,
            "timesteps_eval": timesteps_eval,
            "returns_train": rewards_train, "timesteps_train": timesteps_train, "timestamps_train": timestamps_train}
        with open("/work/dlclarge2/shalag-HPORLBench/data_hpo_rl_bench/DQN/%s/%s_DQN_lr_%s_gamma_%s_epsilon_%s_layers_%s_units_%s_seed%s.json" % (environment, environment, learning_rate, gamma, clip,
                                                                                layers, units, seed),
              'w+') as f:
            json.dump(data, f)
    return rewards, std_rewards

seed = int(sys.argv[3])
config_id = int(sys.argv[1])
env_id = int(sys.argv[2])
with open('config_space_dqn_extended', 'rb') as f:
    config_space_ppo = pickle.load(f)
configuration = config_space_ppo[config_id]
lr = configuration[0]
gamma = configuration[1]
clip = configuration[2]
layers = configuration[3]
units = configuration[4]
if not os.path.exists("/work/dlclarge2/shalag-HPORLBench/data_hpo_rl_bench/DQN/%s/%s_DQN_lr_%s_gamma_%s_epsilon_%s_layers_%s_units_%s_seed%s.json" % (CONTROL_ENVS[env_id], CONTROL_ENVS[env_id], lr, gamma, clip,
    layers, units, seed)):
    r, std_r = run_ppo(learning_rate=lr, gamma=gamma, clip=clip, layers=layers, units=units,
                   environment=CONTROL_ENVS[env_id], policy='MlpPolicy', seed=seed)