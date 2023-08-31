import logging
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining
logging.basicConfig(level=logging.INFO)
from stable_baselines3.common.utils import get_schedule_fn
from ray.tune.trial import ExportFormat
from stable_baselines3.common.env_util import make_atari_env
import time
from stable_baselines3 import A2C
from ray.tune.schedulers.pb2 import PB2
from stable_baselines3.common.evaluation import evaluate_policy
import pickle
import os
import gym
import numpy as np
import json
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback
import sys
from stable_baselines3.common.vec_env import VecMonitor

ATARI_ENVS = ['Pong-v0', 'Alien-v0', 'BankHeist-v0',
              'BeamRider-v0', 'Breakout-v0', 'Enduro-v0',
              'Phoenix-v0', 'Seaquest-v0', 'SpaceInvaders-v0',
              'Riverraid-v0', 'Tennis-v0', 'Skiing-v0', 'Boxing-v0',
              'Bowling-v0', 'Asteroids-v0']
MUJOCO_ENVS = ['Ant-v2', 'Hopper-v2', 'Humanoid-v2']
ENVIRONMENTS = ['Pong-v0', 'Alien-v0', 'BankHeist-v0',
                'BeamRider-v0', 'Breakout-v0', 'Enduro-v0',
                'Phoenix-v0', 'Seaquest-v0', 'SpaceInvaders-v0',
                'Riverraid-v0', 'Tennis-v0', 'Skiing-v0', 'Boxing-v0',
                'Bowling-v0', 'Asteroids-v0',
                'Ant-v2', 'Hopper-v2', 'Humanoid-v2',
                'CartPole-v1', 'MountainCar-v0', 'Acrobot-v1', 'Pendulum-v0']
GAMMAS = [0.8, 0.9, 0.95, 0.98, 0.99, 1.0]
CLIPS = [0.2, 0.3, 0.4]
EPSILONS = [0.1, 0.2, 0.3]

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level.
    """
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1, seed: int = 12345,
                 checkpoint_dir: str = None,
                 start_time: float = time.time()):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf
        self.seed = seed
        self.start_time = start_time
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
             print("Rewards Train: ")
             with open(os.path.join("rewards_train"), 'wb') as f:
                pickle.dump(y.tolist(), f)
             with open(os.path.join("timesteps_train"), 'wb') as g:
                pickle.dump(x.tolist(), g)
        with open(os.path.join("timestamps_train"), 'wb') as g:
            pickle.dump(self.timestamps, g)


        return True


class RunRLAlgorithm:
    def __init__(self, search_space: str, environment: str, seed: int):
        self.search_space = search_space
        self.environment = environment
        self.seed = seed


    def dynamic_change_hyperparameters(self, model, learning_rate, gamma):
        model.learning_rate = learning_rate
        model._setup_lr_schedule()
        model.gamma = gamma

    # Target Algorithm
    def ppo_cartpole(self, cfg, checkpoint_dir=None):
        step = 0
        learning_rate = 10 ** cfg.get("learning_rate", 0.01)
        gamma = cfg.get("gamma", 0.99)
        # Create log dir
        log_dir = "tmp%s_%s_%s/" % (learning_rate, gamma, self.seed)
        os.makedirs(log_dir, exist_ok=True)

        # Create the callback: check every 1000 steps
        if checkpoint_dir is not None:
            with open(os.path.join(checkpoint_dir, 'state.json')) as f:
                data_state = json.load(f)
            step = data_state["step"]
            start = data_state["start_time"]
            config = data_state["config"]
            with open(os.path.join("%s_seed%s.json" % (self.search_space, self.seed))) as f:
                for obj in f:
                    data = json.loads(obj)
            rewards = data["returns_eval"]
            std_rewards = data["std_returns_eval"]
            times_eval = data["timestamps_eval"]
            rewards_train = data["returns_train"]
            timesteps_eval = data["timesteps_eval"]
            timesteps_train = data["timesteps_train"]
            timestamps_train = data["timestamps_train"]
        else:
            start = time.time()
            rewards = []
            std_rewards = []
            times_eval = []
            timesteps_eval = []
            timesteps_train = []
            timestamps_train = []
            rewards_train = []
            config =[learning_rate, gamma]

        callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=os.path.join(log_dir),
                                                    seed=self.seed, start_time=start)

        changed_hps = False
        while True:
            # Create and wrap the environment
            if self.environment in ATARI_ENVS:
                env = make_atari_env(self.environment, n_envs=8, seed=self.seed)
                env = VecMonitor(env, log_dir)
                policy = 'CnnPolicy'
                eval_env = make_atari_env(self.environment, n_envs=5, seed=self.seed)
            else:
                env = gym.make(self.environment)
                env = Monitor(env, log_dir)
                policy = 'MlpPolicy'
                eval_env = gym.make(self.environment)
                eval_env = Monitor(eval_env)
            print(self.environment)
            if checkpoint_dir is None:
                # Because we use parameter noise, we should use a MlpPolicy with layer normalization
                model = A2C(policy, env, verbose=0, learning_rate=learning_rate, gamma=gamma, seed=self.seed)
            else:
                model = A2C.load(path=os.path.join(checkpoint_dir, "checkpoint_a2c"), env=env)
                if not changed_hps:
                    self.dynamic_change_hyperparameters(model, learning_rate, gamma)
                    changed_hps = True
            model.learn(total_timesteps=int(1e4), callback=callback)
            start_eval = time.time()
            r, std_r = evaluate_policy(model=model, env=eval_env)
            end_eval = time.time() -start_eval
            rewards.append(r)
            std_rewards.append(std_r)
            with open(os.path.join("rewards_train"), 'rb') as f:
                rewards_train_new = pickle.load(f)
            with open(os.path.join("timesteps_train"), 'rb') as f:
                timesteps_train_new = pickle.load(f)
            with open(os.path.join("timestamps_train"), 'rb') as f:
                timestamps_train_new = pickle.load(f)
            if len(timesteps_train) > 0:
                last_timestep = timesteps_train[-1]
            if checkpoint_dir is not None:
                switched_config = config[0] != learning_rate or config[1] != gamma
                if switched_config:
                    print("SWITCHED")
                    for timestep in timesteps_train_new:
                        timesteps_train.append(timestep + last_timestep)
                    for timestamp in timestamps_train_new:
                        timestamps_train.append(timestamp)
                    for reward in rewards_train_new:
                        rewards_train.append(reward)
                else:
                    for timestep in timesteps_train_new:
                        timesteps_train.append(timestep + last_timestep)
                    timestamps_train = timestamps_train_new.copy()
                    for reward in rewards_train_new:
                        rewards_train.append(reward)
            else:
                for timestep in timesteps_train_new:
                    timesteps_train.append(timestep)
                for timestamp in timestamps_train_new:
                    timestamps_train.append(timestamp)
                for reward in rewards_train_new:
                    rewards_train.append(reward)
            timesteps_eval.append(timesteps_train[-1])
            times_eval.append(timestamps_train[-1] + end_eval)
            print("Rewards %s" % rewards[-1])
            print("Std rewards %s" % std_rewards[-1])
            print("Times Train: %s" % np.unique(timestamps_train).tolist())
            print("Timesteps Train: %s" % timesteps_train)
            print("Times: %s" % times_eval[-1])
            print("Timesteps Eval: %s" % timesteps_eval[-1])
            # model.save(os.path.join(log_dir, "ppo_model"))
            data = {"gamma": gamma, "learning_rate": learning_rate,
                    "returns_eval": rewards, "std_returns_eval": std_rewards, "timestamps_eval": times_eval,
                    "timesteps_eval": timesteps_eval,
                    "returns_train": rewards_train, "timesteps_train": timesteps_train,
                    "timestamps_train": np.unique(timestamps_train).tolist()}
            with open(os.path.join("%s_seed%s.json" % (self.search_space, self.seed)),
                      'w+') as f:
                json.dump(data, f)
                f.write("\n")
            score = r
            # Every 5 steps, checkpoint our current state.
            # First get the checkpoint directory from tune.
            with tune.checkpoint_dir(step=step) as checkpoint_dir:
                # Then create a checkpoint file in this directory.
                path = os.path.join(checkpoint_dir, "checkpoint_a2c")
                # Save state to checkpoint file.
                # No need to save optimizer for SGD.
                model.save(path=path)
                state = {"step": step, "start_time": time.time(), "config": [learning_rate, gamma]}
                with open(os.path.join(checkpoint_dir, 'state.json'), 'w+') as f:
                    json.dump(state, f)
            step += 1
            tune.report(mean_accuracy=score, training_iteration=step)