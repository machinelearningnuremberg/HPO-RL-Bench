import json
import os
from typing import List

import numpy as np
import pandas as pd
import pickle as pkl
from dyhpo.benchmarks.benchmark import BaseBenchmark
from autorlbench_utils import get_metrics

class HPORLBench_Extended(BaseBenchmark):

    nr_hyperparameters_dict = {
        "A2C": 324,
        "PPO": 972,
        "DDPG": 972,
        "SAC": 972,
        "TD3": 972
    }
    max_budget = 100
    hp_names_dict = {
        "A2C": ["learning_rate", "gamma", "n_layers", "n_units"],
        "PPO": ["learning_rate", "gamma", "clip", "n_layers", "n_units"],
        "DDPG": ["learning_rate", "gamma", "tau", "n_layers", "n_units"],
        "SAC": ["learning_rate", "gamma", "tau", "n_layers", "n_units"],
        "TD3": ["learning_rate", "gamma", "tau", "n_layers", "n_units"],
    }

    log_indicator = [False, False, False, False, False]

    def __init__(self, path_to_json_files: str, ss_name: str, env_name: str, seed: int=0):

        super().__init__(path_to_json_files)
        self.ss_name = ss_name
        self.env_name = env_name
        self.seed = seed
        self.hp_candidates = []
        self.reward_curves = []
        self.min_value = 0
        self.max_value = 0
        self.hp_names = self.hp_names_dict[self.ss_name]
        self.nr_hyperparameters = self.nr_hyperparameters_dict[self.ss_name]
        self._load_benchmark()
        self.reward_curves = np.array(self.reward_curves)
        self.hp_candidates = np.array(self.hp_candidates)
        self.min_value = self.get_worst_performance()
        self.max_value = self.get_best_performance()
        print(type(self.reward_curves[0]))
        self.reward_curves = (self.reward_curves - self.min_value)/(self.max_value - self.min_value)


    def get_worst_performance(self):

        # for taskset we have loss, so the worst value is infinity
        min_value = np.inf
        for hp_index in range(0, self.reward_curves.shape[0]):
            val_curve = self.reward_curves[hp_index]
            worst_performance_hp_curve = min(val_curve)
            if worst_performance_hp_curve < min_value:
                min_value = worst_performance_hp_curve

        return min_value

    def get_best_performance(self):

        # for taskset we have loss, so the worst value is infinity
        max_value = -np.inf
        for hp_index in range(0, self.reward_curves.shape[0]):
            val_curve = self.reward_curves[hp_index]
            best_performance_hp_curve = max(val_curve)
            if best_performance_hp_curve > max_value:
                max_value = best_performance_hp_curve

        return max_value

    def get_curve(self, hp_index: int, budget: int) -> List[float]:

        val_curve = self.reward_curves[hp_index]
        # val_curve = val_curve[1:]
        budget = int(budget)

        return val_curve[0:budget].tolist()

    def _load_benchmark(self):

        dataset_file = os.path.join(f'config_space_{self.ss_name}_extended')

        with open(dataset_file, 'rb') as fp:
            dataset_info = pkl.load(fp)

        for optimization_iteration in dataset_info:
            hp_configuration = list(optimization_iteration)
            if self.ss_name == "A2C":
                config = {
                    "lr": optimization_iteration[0],
                    "gamma": optimization_iteration[1],
                    "n_layers": optimization_iteration[2],
                    "n_units": optimization_iteration[3]
                }
            elif self.ss_name == "PPO":
                config = {
                    "lr": optimization_iteration[0],
                    "gamma": optimization_iteration[1],
                    "clip": optimization_iteration[2],
                    "n_layers": optimization_iteration[3],
                    "n_units": optimization_iteration[4]
                }
            else:
                config = {
                    "lr": optimization_iteration[0],
                    "gamma": optimization_iteration[1],
                    "tau": optimization_iteration[2],
                    "n_layers": optimization_iteration[3],
                    "n_units": optimization_iteration[4]
                }
            results = get_metrics(search_space=self.ss_name, environment=self.env_name, config=config, seed=self.seed,
                                  extended=True)
            reward_curve = results["eval_avg_returns"]

            # keep a fixed order for the hps and their values,
            # just in case
            new_hp_configuration = []
            for i, hp_name in enumerate(self.hp_names):
                new_hp_configuration.append(hp_configuration[i])
            self.hp_candidates.append(new_hp_configuration)
            self.reward_curves.append(np.array(reward_curve))

    def load_dataset_names(self) -> List[str]:

        dataset_file_names = [
            dataset_file_name for dataset_file_name in os.listdir(self.path_to_json_file)
            if os.path.isfile(os.path.join(self.path_to_json_file, dataset_file_name))
        ]

        return dataset_file_names

    def get_hyperparameter_candidates(self) -> np.ndarray:

        return np.array(self.hp_candidates)

    def get_performance(self, hp_index: int, budget: int) -> float:

        val_curve = self.reward_curves[hp_index]

        budget = int(budget)

        return val_curve[budget - 1]

    def get_incumbent_curve(self):

        best_value = -np.inf
        best_index = -1
        for index in range(0, self.reward_curves.shape[0]):
            val_curve = self.reward_curves[index]
            max_reward = max(val_curve)

            if max_reward > best_value:
                best_value = max_reward
                best_index = index

        return self.reward_curves[best_index]

    def get_incumbent_config_index(self):

        best_value = -np.inf
        best_index = -1
        for index in range(0, self.reward_curves.shape[0]):
            val_curve = self.reward_curves[index]
            min_loss = max(val_curve)

            if min_loss > best_value:
                best_value = min_loss
                best_index = index

        return best_index

    def log_transform_labels(self):

        validation_curves = np.array(self.validation_curves).flatten()
        max_value = np.amax(validation_curves)
        min_value = np.amin(validation_curves)
        self.max_value = max_value
        self.min_value = min_value

        f = lambda x: (np.log(x) - np.log(min_value)) / (np.log(max_value) - np.log(min_value))

        log_transformed_values = f(self.validation_curves)

        return log_transformed_values.tolist()

    def filter_curves(self):

        validation_curves = np.array(self.validation_curves)
        validation_curves = pd.DataFrame(validation_curves)
        # TODO do a query for both values instead of going through the df twice
        non_nan_idx = validation_curves.notnull().all(axis=1)
        non_diverging_idx = (validation_curves < validation_curves.quantile(0.95).min()).all(axis=1)

        idx = non_nan_idx & non_diverging_idx

        return idx


class HPORLBench(BaseBenchmark):

    nr_hyperparameters_dict = {
        "A2C": 36,
        "PPO": 108,
        "DDPG": 108,
        "SAC": 108,
        "TD3": 108
    }
    max_budget = 100
    hp_names_dict = {
        "A2C": ["learning_rate", "gamma"],
        "PPO": ["learning_rate", "gamma", "clip"],
        "DDPG": ["learning_rate", "gamma", "tau"],
        "SAC": ["learning_rate", "gamma", "tau"],
        "TD3": ["learning_rate", "gamma", "tau"],
    }

    log_indicator = [False, False, False]

    def __init__(self, path_to_json_files: str, ss_name: str, env_name: str, seed: int=0):

        super().__init__(path_to_json_files)
        self.ss_name = ss_name
        self.env_name = env_name
        self.seed = seed
        self.hp_candidates = []
        self.reward_curves = []
        self.min_value = 0
        self.max_value = 0
        self.hp_names = self.hp_names_dict[self.ss_name]
        self.nr_hyperparameters = self.nr_hyperparameters_dict[self.ss_name]
        self._load_benchmark()
        self.reward_curves = np.array(self.reward_curves)
        self.hp_candidates = np.array(self.hp_candidates)
        self.min_value = self.get_worst_performance()
        self.max_value = self.get_best_performance()
        self.reward_curves = (self.reward_curves - self.min_value)/(self.max_value - self.min_value)


    def get_worst_performance(self):

        # for taskset we have loss, so the worst value is infinity
        min_value = np.inf
        for hp_index in range(0, self.reward_curves.shape[0]):
            val_curve = self.reward_curves[hp_index]
            worst_performance_hp_curve = min(val_curve)
            if worst_performance_hp_curve < min_value:
                min_value = worst_performance_hp_curve

        return min_value

    def get_best_performance(self):

        # for taskset we have loss, so the worst value is infinity
        max_value = -np.inf
        for hp_index in range(0, self.reward_curves.shape[0]):
            val_curve = self.reward_curves[hp_index]
            best_performance_hp_curve = max(val_curve)
            if best_performance_hp_curve > max_value:
                max_value = best_performance_hp_curve

        return max_value

    def get_curve(self, hp_index: int, budget: int) -> List[float]:

        val_curve = self.reward_curves[hp_index]
        # val_curve = val_curve[1:]
        budget = int(budget)

        return val_curve[0:budget].tolist()

    def _load_benchmark(self):

        dataset_file = os.path.join(f'config_space_{self.ss_name}')

        with open(dataset_file, 'rb') as fp:
            dataset_info = pkl.load(fp)

        for optimization_iteration in dataset_info:
            hp_configuration = list(optimization_iteration)
            if self.ss_name == "A2C":
                config = {
                    "lr": optimization_iteration[0],
                    "gamma": optimization_iteration[1]
                }
            elif self.ss_name == "PPO":
                config = {
                    "lr": optimization_iteration[0],
                    "gamma": optimization_iteration[1],
                    "clip": optimization_iteration[2]
                }
            else:
                config = {
                    "lr": optimization_iteration[0],
                    "gamma": optimization_iteration[1],
                    "tau": optimization_iteration[2]
                }
            results = get_metrics(search_space=self.ss_name, environment=self.env_name, config=config, seed=self.seed)
            reward_curve = results["eval_avg_returns"]

            # keep a fixed order for the hps and their values,
            # just in case
            new_hp_configuration = []
            for i, hp_name in enumerate(self.hp_names):
                new_hp_configuration.append(hp_configuration[i])
            self.hp_candidates.append(new_hp_configuration)
            self.reward_curves.append(reward_curve)

    def load_dataset_names(self) -> List[str]:

        dataset_file_names = [
            dataset_file_name for dataset_file_name in os.listdir(self.path_to_json_file)
            if os.path.isfile(os.path.join(self.path_to_json_file, dataset_file_name))
        ]

        return dataset_file_names

    def get_hyperparameter_candidates(self) -> np.ndarray:

        return np.array(self.hp_candidates)

    def get_performance(self, hp_index: int, budget: int) -> float:

        val_curve = self.reward_curves[hp_index]

        budget = int(budget)

        return val_curve[budget - 1]

    def get_incumbent_curve(self):

        best_value = -np.inf
        best_index = -1
        for index in range(0, self.reward_curves.shape[0]):
            val_curve = self.reward_curves[index]
            max_reward = max(val_curve)

            if max_reward > best_value:
                best_value = max_reward
                best_index = index

        return self.reward_curves[best_index]

    def get_incumbent_config_index(self):

        best_value = -np.inf
        best_index = -1
        for index in range(0, self.reward_curves.shape[0]):
            val_curve = self.reward_curves[index]
            min_loss = max(val_curve)

            if min_loss > best_value:
                best_value = min_loss
                best_index = index

        return best_index

    def log_transform_labels(self):

        validation_curves = np.array(self.validation_curves).flatten()
        max_value = np.amax(validation_curves)
        min_value = np.amin(validation_curves)
        self.max_value = max_value
        self.min_value = min_value

        f = lambda x: (np.log(x) - np.log(min_value)) / (np.log(max_value) - np.log(min_value))

        log_transformed_values = f(self.validation_curves)

        return log_transformed_values.tolist()

    def filter_curves(self):

        validation_curves = np.array(self.validation_curves)
        validation_curves = pd.DataFrame(validation_curves)
        # TODO do a query for both values instead of going through the df twice
        non_nan_idx = validation_curves.notnull().all(axis=1)
        non_diverging_idx = (validation_curves < validation_curves.quantile(0.95).min()).all(axis=1)

        idx = non_nan_idx & non_diverging_idx

        return idx
