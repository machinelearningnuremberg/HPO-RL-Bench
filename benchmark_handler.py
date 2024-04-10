import json
import os
import numpy as np
import itertools
from typing import Union

try:
    from utils import run_rl_algorithm
except:
    print("utils not found, cannot run RL algorithm")

class BenchmarkHandler:

    def __init__(self, data_path: str = "", environment: str = None, search_space: str = None,
                 set: str = "static", return_metrics=None, seed=0, rl_algorithm=None):
        """
        A handler for interacting with and running HPO-RL-Bench.

        Parameters:
        - data_path (str): Path to the parent folder of the `data_hpo_rl_bench`. If not in the same folder
          as the `benchmark_handler.py` script, this path needs to be provided.
        - environment (str): Name of the environment to benchmark the RL algorithm on. This should be one
          of ["Pong-v0", "Ant-v2", "Alien-v0", "BankHeist-v0", "BeamRider-v0", "Breakout-v0", "Enduro-v0",
          "Phoenix-v0","Seaquest-v0", "SpaceInvaders-v0", "Riverraid-v0", "Tennis-v0", "Skiing-v0", "Boxing-v0",
          "Bowling-v0", "Asteroids-v0", "Hopper-v2", "Humanoid-v2", "CartPole-v1", "MountainCar-v0", "Acrobot-v1",
          "Pendulum-v0"]
        - search_space (str): Name of the RL algorithm.
          Must be one of ["PPO", "A2C", "DDPG", "SAC", "TD3", "DQN"].
        - set (str): Subset of the benchmark data to use. Can be one of ["static", "dynamic", "extended"].
          The default value is "static".
        - return_metrics (list): A list of metrics to return for the results. Can be a subset
          of ["eval_avg_returns", "eval_std_returns", "eval_timestamps", "eval_timesteps"].
        - seed (int): Seed of the benchmark runs. Can be one of [0, 1, 2, 3, 4,
          5, 6, 7, 8, 9]. The default value is 0.
        - rl_algorithm: Class of the RL Algorithm that is compatible with the interface provided by
          stable-baselines3 RL algorithms. This is used for running an actual RL algorithm
          rather than querying the benchmark dataset.

        This handler is designed to facilitate easy access to HPO-RLBench. It supports
        querying pre-computed datasets as well as running new RL algorithms.
        """

        self.data_path = data_path

        self.environment = environment
        if type(search_space) == str or search_space is None:
            self.search_space = search_space
        else:
            for ss_name, ss_dict in search_space.items():
                self.search_space = ss_name
                self.search_space_dict = ss_dict
            assert rl_algorithm is not None, "For new search spaces, an RL Algorithm object must be provided"
        self.rl_algorithm = rl_algorithm
        self.seed = seed
        self.return_metrics = return_metrics
        self.set = set
        self.search_space_structure = {"static": {"PPO": {"lr": [-6, -5, -4, -3, -2,-1],
                                                            "gamma": [0.8, 0.9, 0.95, 0.98, 0.99,1.0],
                                                            "clip": [0.2, 0.3, 0.4]},
                                                    "DQN": {"lr": [-6, -5, -4, -3, -2,-1],
                                                            "gamma": [0.8, 0.9, 0.95, 0.98, 0.99,1.0],
                                                            "epsilon": [0.1, 0.2, 0.3]},
                                                    "A2C": {"lr": [-6, -5, -4, -3, -2, -1],
                                                            "gamma": [0.8, 0.9, 0.95, 0.98, 0.99, 1.0]},
                                                    "DDPG": {"lr": [-6, -5, -4, -3, -2, -1],
                                                            "gamma": [0.8, 0.9, 0.95, 0.98, 0.99, 1.0],
                                                            "tau": [0.0001, 0.001, 0.005]},
                                                    "SAC": {"lr": [-6, -5, -4, -3, -2, -1],
                                                            "gamma": [0.8, 0.9, 0.95, 0.98, 0.99, 1.0],
                                                            "tau": [0.0001, 0.001, 0.005]},
                                                    "TD3": {"lr": [-6, -5, -4, -3, -2, -1],
                                                            "gamma": [0.8, 0.9, 0.95, 0.98, 0.99, 1.0],
                                                            "tau": [0.0001, 0.001, 0.005]}
                                                            },
                                        "dynamic": {"PPO": {"lr": [-5, -4, -3],
                                                            "gamma": [0.95, 0.98, 0.99]}},
                                        "extended": {"PPO": {"lr": [-6, -5, -4, -3, -2,-1],
                                                             "gamma": [0.8, 0.9, 0.95, 0.98, 0.99,1.0],
                                                             "clip": [0.1, 0.2, 0.3],
                                                             "n_layers":[1,2,3],
                                                             "n_units":[32,64,128, 256]},
                                                    "DQN":{"lr": [-6, -5, -4, -3, -2,-1],
                                                             "gamma": [0.8, 0.9, 0.95, 0.98, 0.99,1.0],
                                                             "epsilon": [0.1, 0.2, 0.3],
                                                             "n_layers":[1,2,3],
                                                             "n_units":[32,64,128, 256]},
                                                    "A2C": {"lr": [-6, -5, -4, -3, -2, -1],
                                                            "gamma": [0.8, 0.9, 0.95, 0.98, 0.99, 1.0],
                                                            "n_layers":[1,2,3],
                                                            "n_units":[32,64,128, 256]},
                                                    "DDPG": {"lr": [-6, -5, -4, -3, -2, -1],
                                                            "gamma": [0.8, 0.9, 0.95, 0.98, 0.99, 1.0],
                                                            "tau": [0.0001, 0.001, 0.005],
                                                            "n_layers":[1,2,3],
                                                            "n_units":[32,64,128]},
                                                    "SAC": {"lr": [-6, -5, -4, -3, -2, -1],
                                                            "gamma": [0.8, 0.9, 0.95, 0.98, 0.99, 1.0],
                                                            "tau": [0.0001, 0.001, 0.005],
                                                            "n_layers":[1,2,3],
                                                            "n_units":[32,64,128]},
                                                    "TD3": {"lr": [-6, -5, -4, -3, -2, -1],
                                                            "gamma": [0.8, 0.9, 0.95, 0.98, 0.99, 1.0],
                                                            "tau": [0.0001, 0.001, 0.005],
                                                            "n_layers":[1,2,3],
                                                            "n_units":[32,64,128]}
                                                    }
                                        }
        self.environment_dict = {"Atari": ["Pong-v0", "Alien-v0", "BankHeist-v0", "BeamRider-v0", "Breakout-v0", "Enduro-v0", "Phoenix-v0",
                                "Seaquest-v0", "SpaceInvaders-v0", "Riverraid-v0", "Tennis-v0", "Skiing-v0", "Boxing-v0",
                                "Bowling-v0", "Asteroids-v0"],
                                 "Mujoco" : ["Ant-v2", "Hopper-v2", "Humanoid-v2"],
                                 "Control" : ["CartPole-v1", "MountainCar-v0", "Acrobot-v1", "Pendulum-v0"]}

        self.environment_list = ["Pong-v0", "Ant-v2", "Alien-v0", "BankHeist-v0", "BeamRider-v0", "Breakout-v0", "Enduro-v0", "Phoenix-v0",
                                "Seaquest-v0", "SpaceInvaders-v0", "Riverraid-v0", "Tennis-v0", "Skiing-v0", "Boxing-v0",
                                "Bowling-v0", "Asteroids-v0", "Hopper-v2", "Humanoid-v2", "CartPole-v1", "MountainCar-v0", "Acrobot-v1", "Pendulum-v0"]
        self.seeds_list = [0,1,2,3,4]
        self.env_types = ["Atari", "Mujoco", "Control"]
        self.valid_dynamic_spaces = ["PPO"]
        self.valid_extended_spaces = ["PPO", "A2C", "DDPG", "SAC", "TD3", "DQN"]

        if self.set == "extended":
            self.environment_list = self.environment_dict["Control"]+self.environment_dict["Mujoco"]

        if self.return_metrics is None:
            self.return_metrics = ["eval_avg_returns", "eval_std_returns", "eval_timestamps", "eval_timesteps"]

        if self.search_space is not None:
            self._precompute_configurations()

    def set_env_space_seed(self, search_space: str, environment : str, seed: int):
        """
        Sets the search space, environment, and seed for the benchmark handler.

        This method configures the benchmark handler with a specific RL algorithm's search space,
        a given environment, and a seed for reproducibility.

        Parameters:
        - search_space (str): The name of the search space that defines the hyperparameter space
         of the RL algorithm. It should match one of the supported RL algorithms, such as "PPO",
         "A2C", "DDPG", "SAC", "TD3", "DQN".
        - environment (str): The name of the environment on which the RL algorithm is to be benchmarked.
         This should be one of the supported environments.
        - seed (int): The seed of the benchmark runs.

        Note: Before calling this method, ensure that the `BenchmarkHandler` instance is initialized with
        appropriate `data_path` if the benchmark data is not located in the default path.
        """
        self.search_space = search_space
        self.environment = environment
        self.seed = seed
        self._precompute_configurations()


    def _build_return_dict(self, data, budget):
        """
       Constructs a dictionary of return metrics based on the specified budget.

       This internal method filters and returns the benchmarking data up to a given budget limit.
       It supports a dynamic selection of metrics to include in the return dictionary based on the
       `return_metrics` attribute of the `BenchmarkHandler` instance. If `return_metrics` is None,
       all available metrics are included.

       Parameters:
       - data (dict): A dictionary containing full data dictionary including evaluation average returns,
         standard deviations of returns, timestamps, and timesteps.
       - budget (int): The budget limit for which data is to be returned.

       Returns:
       A dictionary containing the requested metrics up to the specified budget.

       Raises:
       AssertionError: If the specified budget exceeds the maximum budget allowed.
       """
        max_budget_allowed = len(data["eval_timesteps"])
        assert budget <= max_budget_allowed, f"Budget should be lower than {max_budget_allowed}"

        if self.return_metrics is None:
                    return data

        else:
            return_dict = {}
            for key in data.keys():
                if key in self.return_metrics:
                    return_dict[key] = data[key]
            return return_dict

    def get_environments_groups (self):
        """
        Retrieves a list of environment groups available in the benchmark dataset.

        This method provides the names of different environment groups that have been predefined
        and stored within the `BenchmarkHandler` instance.

        Returns:
        A list of strings, where each string is the name of an environment group within
        the benchmark dataset.
        """
        return list(self.environment_dict.keys())

    def get_environments_per_group (self, env_group: str = "Atari"):
        """
        Retrieves a list of environment names belonging to a specified group.

        Args:
            env_group (str): The name of the environment group. Can be one of ["Atari", "Mujoco", "Control"].
            Defaults to "Atari".

        Returns:
            list: A list of environment names associated with the specified group.
        """
        return self.environment_dict[env_group]

    def get_environments(self):
        """
        Returns:
            list: A list containing the names of all environments in the benchmark.
        """
        return self.environment_list

    def get_search_spaces_names(self, set: str = "static"):
        """
        Args:
            set (str): The name of the benchmark subset for which to retrieve search spaces.
             Can be one of ["static", "dynamic", "extended"]. Defaults to "static".

        Returns:
            list: A list of names of search spaces within the specified benchmark subset.
        """
        return list(self.search_space_structure[set].keys())

    def get_metrics(self, config: dict, search_space: str = '', environment: str = '', seed: int = 0,
                    budget: int = 100, set: str = "static", return_final_only=False):
        """
        Retrieves the performance metrics for a given configuration, search space, environment, and other parameters.
        The function can return either all metrics across the specified budget or only the final metric, based on
        the 'return_final_only' flag.

        Args:
            config (dict): The configuration for which metrics are being retrieved.
            Must have the same structure as tthe search space dict of the respective search space.
            search_space (str): The search space within which the configuration exists.
            environment (str): The environment for which metrics are to be retrieved.
            seed (int): The seed for the random number generator, ensuring reproducibility.Default is 0.
            budget (int): The total number of evaluations to consider for metrics retrieval. Defaults to 100.
            set (str): The benchmark subset to query. Defaults to "static".
            return_final_only (bool): Flag to indicate whether to return only the final metric. Defaults to False.

        Returns:
            dict or float: Depending on 'return_final_only', returns either a dictionary of metrics across the
                         specified budget or the final metric.
        """
        budget = int(budget)
        if self.rl_algorithm is None:
            if search_space == "":
                search_space = self.search_space
            if environment == "":
                environment  = self.environment
            if seed == -np.inf:
                seed = self.seed
            if set in ["static", "extended"]:
                lr = int(config.get("lr"))
                gamma = config.get("gamma")

                if search_space == "DQN":
                    epsilon = config.get("epsilon")
                    if set == "static":
                        if os.path.exists(os.path.join(self.data_path, 'data_hpo_rl_bench', search_space, environment,
                                                       '%s_%s_random_lr_%s_gamma_%s_epsilon_%s_seed%s_eval.json' % (
                                                       environment, search_space,
                                                       lr, gamma, epsilon, seed))):
                            with open(os.path.join(self.data_path, 'data_hpo_rl_bench', search_space, environment,
                                                   '%s_%s_random_lr_%s_gamma_%s_epsilon_%s_seed%s_eval.json' % (
                                                   environment, search_space,
                                                   lr, gamma, epsilon, seed))) as f:
                                data = json.load(f)
                            if not return_final_only:
                                return self._build_return_dict(data={
                                    "eval_avg_returns": data["returns_eval"][:budget],
                                    "eval_std_returns": data["std_returns_eval"][:budget],
                                    "eval_timestamps": data["timestamps_eval"][:budget],
                                    "eval_timesteps": data["timesteps_eval"][:budget],
                                }, budget=budget)
                            else:
                                return data["returns_eval"][budget-1]
                    else:
                        n_layers = config.get("n_layers")
                        n_units = config.get("n_units")
                        if os.path.exists(os.path.join(self.data_path, 'data_hpo_rl_bench', search_space, environment,
                                                       '%s_%s_lr_%s_gamma_%s_epsilon_%s_layers_%s_units_%s_seed%s.json' %
                                                       (environment, search_space, lr, gamma, epsilon, n_layers, n_units,
                                                        seed))):
                            with open(os.path.join(self.data_path, 'data_hpo_rl_bench', search_space, environment,
                                                   '%s_%s_lr_%s_gamma_%s_epsilon_%s_layers_%s_units_%s_seed%s.json' %
                                                   (environment, search_space, lr, gamma, epsilon, n_layers, n_units,
                                                    seed))) as f:
                                data = json.load(f)
                                for i in range(budget):
                                    if len(data["timesteps_eval"]) > i:
                                        timestep_eval = data["timesteps_eval"][i]
                                        if np.isnan(timestep_eval):
                                            data["timesteps_eval"][i] = (i + 1) * 10000
                                    else:
                                        data["timesteps_eval"].append(data["timesteps_eval"][-1])
                            if not return_final_only:
                                return self._build_return_dict(data={
                                    "eval_avg_returns": data["returns_eval"][:budget],
                                    "eval_std_returns": data["std_returns_eval"][:budget],
                                    "eval_timestamps": data["timestamps_eval"][:budget],
                                    "eval_timesteps": data["timesteps_eval"][:budget],
                                }, budget=budget)
                            else:
                                if len(data["returns_eval"])>= budget:
                                    return data["returns_eval"][budget-1]
                                else:
                                    return data["returns_eval"][-1]

                        else:
                            print(os.path.join(self.data_path, 'data_hpo_rl_bench', search_space, environment,
                                                       '%s_%s_lr_%s_gamma_%s_epsilon_%s_layers_%s_units_%s_seed%s.json' %
                                               (environment, search_space, lr, gamma, epsilon, n_layers, n_units,
                                                        seed)))
                elif search_space == "PPO":
                    clip = config.get("clip")
                    if set == "static":
                        with open(os.path.join(self.data_path, 'data_hpo_rl_bench', search_space, environment,
                                               '%s_%s_random_lr_%s_gamma_%s_clip_%s_seed%s_eval.json' % (
                                               environment, search_space,
                                               lr, gamma, clip, seed))) as f:
                            data = json.load(f)
                            for i in range(budget):
                                timestep_eval = data["timesteps_eval"][i]
                                if np.isnan(timestep_eval):
                                    data["timesteps_eval"][i] = (i + 1) * 10000
                    else:
                        n_layers = config.get("n_layers")
                        n_units = config.get("n_units")
                        with open(os.path.join(self.data_path, 'data_hpo_rl_bench', search_space, environment,
                                               '%s_%s_lr_%s_gamma_%s_clip_%s_layers_%s_units_%s_seed%s.json' %
                                               (environment, search_space, lr, gamma, clip, n_layers, n_units, seed))) as f:
                            data = json.load(f)
                            for i in range(budget):
                                if len(data["timesteps_eval"]) > i:
                                    timestep_eval = data["timesteps_eval"][i]
                                    if np.isnan(timestep_eval):
                                        data["timesteps_eval"][i] = (i + 1) * 10000
                                else:
                                    data["timesteps_eval"].append(data["timesteps_eval"][-1])
                    if not return_final_only:
                        return self._build_return_dict(data={
                            "eval_avg_returns": data["returns_eval"][:budget],
                            "eval_std_returns": data["std_returns_eval"][:budget],
                            "eval_timestamps": data["timestamps_eval"][:budget],
                            "eval_timesteps": data["timesteps_eval"][:budget],
                        }, budget=budget)
                    else:
                                if len(data["returns_eval"])>= budget:
                                    return data["returns_eval"][budget-1]
                                else:
                                    return data["returns_eval"][-1]
                elif search_space == "A2C":
                    if set == "static":
                        with open(os.path.join(self.data_path, 'data_hpo_rl_bench', search_space, environment,
                                               '%s_%s_random_lr_%s_gamma_%s_seed%s_eval.json' % (environment, search_space,
                                                                                                 lr, gamma, seed))) as f:
                            data = json.load(f)
                    else:
                        n_layers = config.get("n_layers")
                        n_units = config.get("n_units")
                        with open(os.path.join(self.data_path, 'data_hpo_rl_bench', search_space, environment,
                                               '%s_%s_lr_%s_gamma_%s_layers_%s_units_%s_seed%s.json' %
                                               (environment, search_space, lr, gamma, n_layers, n_units, seed))) as f:
                            data = json.load(f)
                    if not return_final_only:
                        return self._build_return_dict(data={
                            "eval_avg_returns": data["returns_eval"][:budget],
                            "eval_std_returns": data["std_returns_eval"][:budget],
                            "eval_timestamps": data["timestamps_eval"][:budget],
                            "eval_timesteps": data["timesteps_eval"][:budget],
                        }, budget=budget)
                    else:
                                if len(data["returns_eval"])>= budget:
                                    return data["returns_eval"][budget-1]
                                else:
                                    return data["returns_eval"][-1]
                elif search_space in ["DDPG", "TD3", "SAC"]:
                    tau = config.get("tau")
                    if set == "static":
                        with open(os.path.join(self.data_path, 'data_hpo_rl_bench', search_space, environment,
                                               '%s_%s_random_lr_%s_gamma_%s_tau_%s_seed%s_eval.json' % (
                                               environment, search_space,
                                               lr, gamma, tau, seed))) as f:
                            data = json.load(f)
                    else:
                        n_layers = config.get("n_layers")
                        n_units = config.get("n_units")
                        with open(os.path.join(self.data_path, 'data_hpo_rl_bench', search_space, environment,
                                               '%s_%s_lr_%s_gamma_%s_tau_%s_layers_%s_units_%s_seed%s.json' %
                                               (environment, search_space, lr, gamma, tau, n_layers, n_units, seed))) as f:
                            data = json.load(f)
                    if not return_final_only:
                        return self._build_return_dict(data={
                            "eval_avg_returns": data["returns_eval"][:budget],
                            "eval_std_returns": data["std_returns_eval"][:budget],
                            "eval_timestamps": data["timestamps_eval"][:budget],
                            "eval_timesteps": data["timesteps_eval"][:budget],
                        }, budget=budget)
                    else:
                                if len(data["returns_eval"])>= budget:
                                    return data["returns_eval"][budget-1]
                                else:
                                    return data["returns_eval"][-1]
            else:
                lrs = config.get("lr")
                if len(lrs) == 1:
                    lrs = lrs * 3
                elif len(lrs) == 2:
                    lrs.append(lrs[1])

                gammas = config.get("gamma")
                if len(gammas) == 1:
                    gammas = gammas * 3
                elif len(gammas) == 2:
                    gammas.append(gammas[1])
                if search_space in ["PPO", "TD3", "SAC"]:
                    with open(os.path.join(self.data_path, 'data_hpo_rl_bench', search_space, environment,
                                           '%s_%s_random_lr_%s%s%s_gamma_%s%s%s_seed%s_eval.json' % (environment, search_space,
                                                                                                     lrs[0], lrs[1], lrs[2],
                                                                                                     gammas[0], gammas[1],
                                                                                                     gammas[2], seed))) as f:
                        data = json.load(f)
                    if not return_final_only:
                        return self._build_return_dict(data={
                            "eval_avg_returns": data["returns_eval"][:budget],
                            "eval_std_returns": data["std_returns_eval"][:budget],
                            "eval_timestamps": data["timestamps_eval"][:budget],
                            "eval_timesteps": data["timesteps_eval"][:budget],
                        }, budget=budget)
                    else:
                        if len(data["returns_eval"]) >= budget:
                            return data["returns_eval"][budget-1]
                        else:
                            return data["returns_eval"][-1]
        else:
            for key in self.search_space_dict.keys():
                assert key in config.keys(), "The configuration must define a value for all the hyperparameters in the search space."
            data = run_rl_algorithm(rl_algorithm=self.rl_algorithm, rl_algorithm_name=self.search_space, config=config,
                            environment=self.environment, seed=self.seed, total_timesteps=budget * 1e4)
            return self._build_return_dict(data, budget)

    def get_search_space(self, search_space, set: str = "static"):
        """
        Retrieves the search space dictionary of information for a given search space name and benchmark subset.

        Args:
            search_space (str): The name of the search space to retrieve.
            set (str): The set category of the search space. Defaults to "static".

        Returns:
            dict: The search space details as a dictionary.
        """
        if self.rl_algorithm is None:
            return self.search_space_structure[set][search_space]
        else:
            return self.search_space_dict

    def _precompute_configurations(self):
        """
        Precomputes all possible configurations of hyperparameters for the current search space and set.
        This method populates the variable 'precomputed_configurations' with a list of dictionaries,
        each representing a unique combination of hyperparameters.
        """
        self.precomputed_configurations = []
        search_space_structure = self.get_search_space(self.search_space, self.set)
        hps_names = list(search_space_structure.keys())
        for hps in itertools.product(*tuple(list(search_space_structure.values()))):
            self.precomputed_configurations.append(dict(zip(hps_names, hps)))


    def sample_configuration(self):
        """
        Samples a random configuration from the current search space and set.
        This method iterates through the hyperparameters (hp) in the search space, randomly selecting a value for each
        from its respective range of values. It constructs and returns a dictionary representing a single configuration,
        where each key is a hyperparameter name, and the corresponding value is the randomly selected value for that
        hyperparameter.

        Returns:
            dict: A randomly sampled configuration as a dictionary, where keys are hyperparameter names and values are
            the selected values for those hyperparameters.
        """
        configuration = {}
        for hp, values in self.get_search_space(self.search_space, self.set):
            configuration[hp] = np.random.choice(values)

        return configuration

    def run_bo(self, optimizer, iterations):
        """
        Executes Bayesian Optimization (BO) over a predefined space of configurations for a specified number of epochs.
        It starts with a random configuration, evaluates it, and then iterates, allowing the optimizer to suggest the next
        configuration based on observed performances. This method updates and tracks the observed learning curves (LCs),
        budgets, and the configurations' performances.

        Args:
            optimizer: The Bayesian Optimization object responsible for suggesting configurations based on past observations.
            iterations (int): The number of iterations for which the optimization process should run.

        Returns:
            tuple: A tuple containing:
                   - observed_lc (dict): A dictionary mapping configuration indices to their observed learning curves.
                   - max_per_lc (list): A list of the maximum values found in each observed learning curve.
                   - configurations[np.argmax(max_per_lc)]: The configuration that achieved the highest value in its
                   learning curve.
                   - list(observed_lc.keys())[np.argmax(max_per_lc)]: The index of the configuration that achieved the
                   highest value.
        """

        observed_lc = {}
        configurations = self.precomputed_configurations.copy()

        next_conf_ix = np.random.randint(0, len(configurations))
        observed_lc[next_conf_ix]  = self.get_metrics(configurations[next_conf_ix], budget = 1)["eval_avg_returns"]

        for _ in range(iterations):
            next_conf_ix, budget = optimizer.observe_and_suggest(configurations, observed_lc)

            assert budget>0,"Negative budgets are not allowed"
            assert budget<100, "Upper bound of budget reached"

            observed_lc[next_conf_ix]  = self.get_metrics(configurations[next_conf_ix], budget = budget)["eval_avg_returns"]

        max_per_lc = [max(lc) for lc in observed_lc.values()]

        return observed_lc, max_per_lc, configurations[np.argmax(max_per_lc)],  list(observed_lc.keys())[np.argmax(max_per_lc)]
