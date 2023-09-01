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

    def __init__(self, data_path: str="/data_arl_bench", environment: str=None, search_space: str=None,
                       static: bool=True, return_names=None, seed=None, rl_algorithm=None, extended=False):

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
        self.seed =  seed
        self.return_names = return_names
        self.static = static
        self.extended = extended
        self.search_space_structure =  {"static": {"PPO": {"lr": [-6, -5, -4, -3, -2,-1],
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
        self.env_types = ["atari", "mujoco", "control"]
        self.valid_dynamic_spaces = ["PPO"]
        self.valid_extended_spaces = ["PPO", "A2C"]

        if self.extended:
            self.environment_list = self.environment_dict["Control"]+self.environment_dict["Mujoco"]
        if self.return_names == None:
            self.return_names = ["eval_avg_returns", "eval_std_returns", "eval_timestamps", "eval_timesteps"]

        if self.search_space is not None:
            self._precompute_configurations()

    def set_env_space_seed(self, search_space: str, environment : str, seed: int):

        self.search_space = search_space
        self.environment = environment
        self.seed = seed
        self._precompute_configurations()


    def _build_return_dict(self, data, budget):

        max_budget_allowed = len(data["timesteps_eval"])
        assert budget < max_budget_allowed, f"Budget should be lower than {max_budget_allowed}"

        if self.return_names == None:
                    return {
                        "eval_avg_returns": data["returns_eval"][:budget],
                        "eval_std_returns": data["std_returns_eval"][:budget],
                        "eval_timestamps": data["timestamps_eval"][:budget],
                        "eval_timesteps": data["timesteps_eval"][:budget],
                    }

        else:
            return_dict = {}
            for name in self.return_names:
                if "eval" in name:
                    return_dict[name] = data[name][:budget]
                elif "train" in name:
                    return_dict[name] = data[name][:train_timesteps_index]
            return return_dict

    def get_environments_groups (self):
        return list(self.environment_dict)

    def get_environments_per_group (self, env_group :str="atari"):

        return self.environment_dict[env_group]

    def get_environments(self):

        return self.environment_list

    def get_search_spaces_names(self, set: str = "static"):
        return list(self.search_space_structure[set].keys())

    def get_metrics(self, config: dict, search_space: str = '', environment: str = '', seed: int = -np.inf, budget: int = 100,
                    static: bool = True,
                    extended: bool = None,
                    return_final_only = False):
        budget = int(budget)
        if self.rl_algorithm is None:
            if search_space == "":
                search_space = self.search_space
            if environment == "":
                environment  = self.environment
            if seed == -np.inf:
                seed = self.seed
            DATA_PATH = self.data_path
            if extended is None:
                extended = self.extended
            if static:
                lr = int(config.get("lr"))
                gamma = config.get("gamma")

                if search_space == "DQN":
                    epsilon = config.get("epsilon")
                    if not extended:
                        if os.path.exists(os.path.join(DATA_PATH, 'data_arl_bench', search_space, environment,
                                                       '%s_%s_random_lr_%s_gamma_%s_epsilon_%s_seed%s_eval.json' % (
                                                       environment, search_space,
                                                       lr, gamma, epsilon, seed))):
                            with open(os.path.join(DATA_PATH, 'data_arl_bench', search_space, environment,
                                                   '%s_%s_random_lr_%s_gamma_%s_epsilon_%s_seed%s_eval.json' % (
                                                   environment, search_space,
                                                   lr, gamma, epsilon, seed))) as f:
                                data = json.load(f)
                            if not return_final_only:
                                return {
                                    "eval_avg_returns": data["returns_eval"][:budget],
                                    "eval_std_returns": data["std_returns_eval"][:budget],
                                    "eval_timestamps": data["timestamps_eval"][:budget],
                                    "eval_timesteps": data["timesteps_eval"][:budget],
                                }
                            else:
                                return data["returns_eval"][budget-1]
                    else:
                        n_layers = config.get("n_layers")
                        n_units = config.get("n_units")
                        if os.path.exists(os.path.join(DATA_PATH, 'data_arl_bench', search_space, environment,
                                                       '%s_%s_lr_%s_gamma_%s_epsilon_%s_layers_%s_units_%s_seed%s.json' %
                                                       (environment, search_space, lr, gamma, epsilon, n_layers, n_units,
                                                        seed))):
                            with open(os.path.join(DATA_PATH, 'data_arl_bench', search_space, environment,
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
                                return {
                                    "eval_avg_returns": data["returns_eval"][:budget],
                                    "eval_std_returns": data["std_returns_eval"][:budget],
                                    "eval_timestamps": data["timestamps_eval"][:budget],
                                    "eval_timesteps": data["timesteps_eval"][:budget],
                                }
                            else:
                                if len(data["returns_eval"])>= budget:
                                    return data["returns_eval"][budget-1]
                                else:
                                    return data["returns_eval"][-1]

                        else:
                            print(os.path.join(DATA_PATH, 'data_arl_bench', search_space, environment,
                                                       '%s_%s_lr_%s_gamma_%s_epsilon_%s_layers_%s_units_%s_seed%s.json' %
                                                       (environment, search_space, lr, gamma, epsilon, n_layers, n_units,
                                                        seed)))
                elif search_space == "PPO":
                    clip = config.get("clip")
                    if not extended:
                        with open(os.path.join(DATA_PATH, 'data_arl_bench', search_space, environment,
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
                        with open(os.path.join(DATA_PATH, 'data_arl_bench', search_space, environment,
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
                                return {
                                    "eval_avg_returns": data["returns_eval"][:budget],
                                    "eval_std_returns": data["std_returns_eval"][:budget],
                                    "eval_timestamps": data["timestamps_eval"][:budget],
                                    "eval_timesteps": data["timesteps_eval"][:budget],
                                }
                    else:
                                if len(data["returns_eval"])>= budget:
                                    return data["returns_eval"][budget-1]
                                else:
                                    return data["returns_eval"][-1]
                elif search_space == "A2C":
                    if not extended:
                        with open(os.path.join(DATA_PATH, 'data_arl_bench', search_space, environment,
                                               '%s_%s_random_lr_%s_gamma_%s_seed%s_eval.json' % (environment, search_space,
                                                                                                 lr, gamma, seed))) as f:
                            data = json.load(f)
                    else:
                        n_layers = config.get("n_layers")
                        n_units = config.get("n_units")
                        with open(os.path.join(DATA_PATH, 'data_arl_bench', search_space, environment,
                                               '%s_%s_lr_%s_gamma_%s_layers_%s_units_%s_seed%s.json' %
                                               (environment, search_space, lr, gamma, n_layers, n_units, seed))) as f:
                            data = json.load(f)
                    if not return_final_only:
                                return {
                                    "eval_avg_returns": data["returns_eval"][:budget],
                                    "eval_std_returns": data["std_returns_eval"][:budget],
                                    "eval_timestamps": data["timestamps_eval"][:budget],
                                    "eval_timesteps": data["timesteps_eval"][:budget],
                                }
                    else:
                                if len(data["returns_eval"])>= budget:
                                    return data["returns_eval"][budget-1]
                                else:
                                    return data["returns_eval"][-1]
                elif search_space in ["DDPG", "TD3", "SAC"]:
                    tau = config.get("tau")
                    if not extended:
                        with open(os.path.join(DATA_PATH, 'data_arl_bench', search_space, environment,
                                               '%s_%s_random_lr_%s_gamma_%s_tau_%s_seed%s_eval.json' % (
                                               environment, search_space,
                                               lr, gamma, tau, seed))) as f:
                            data = json.load(f)
                    else:
                        n_layers = config.get("n_layers")
                        n_units = config.get("n_units")
                        with open(os.path.join(DATA_PATH, 'data_arl_bench', search_space, environment,
                                               '%s_%s_lr_%s_gamma_%s_tau_%s_layers_%s_units_%s_seed%s.json' %
                                               (environment, search_space, lr, gamma, tau, n_layers, n_units, seed))) as f:
                            data = json.load(f)
                    if not return_final_only:
                                return {
                                    "eval_avg_returns": data["returns_eval"][:budget],
                                    "eval_std_returns": data["std_returns_eval"][:budget],
                                    "eval_timestamps": data["timestamps_eval"][:budget],
                                    "eval_timesteps": data["timesteps_eval"][:budget],
                                }
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
                    with open(os.path.join(DATA_PATH, 'data_arl_bench', search_space, environment,
                                           '%s_%s_random_lr_%s%s%s_gamma_%s%s%s_seed%s_eval.json' % (environment, search_space,
                                                                                                     lrs[0], lrs[1], lrs[2],
                                                                                                     gammas[0], gammas[1],
                                                                                                     gammas[2], seed))) as f:
                        data = json.load(f)
                    if not return_final_only:
                        return {
                                "eval_avg_returns": data["returns_eval"][:budget],
                                "eval_std_returns": data["std_returns_eval"][:budget],
                                "eval_timestamps": data["timestamps_eval"][:budget],
                                "eval_timesteps": data["timesteps_eval"][:budget],
                                }
                    else:
                        if len(data["returns_eval"])>= budget:
                            return data["returns_eval"][budget-1]
                        else:
                            return data["returns_eval"][-1]
        else:
            for key in self.search_space_dict.keys():
                assert key in config.keys(), "The configuration must define a value for all the hyperparameters in the search space."
            data = run_rl_algorithm(rl_algorithm=self.rl_algorithm, rl_algorithm_name=self.search_space, config=config,
                            environment=self.environment, seed=self.seed, total_timesteps=budget * 1e4)
            train_timesteps_index = budget
            return self._build_return_dict(data, budget, train_timesteps_index)

    def get_search_space(self, search_space, static: bool=True):

        if self.rl_algorithm is None:
            if self.extended:
                return self.search_space_structure["extended"][search_space]
            if static:
                return self.search_space_structure["static"][search_space]
            else:
                return self.search_space_structure["dynamic"][search_space]
        else:
            return self.search_space_dict


    def get_metrics_deprecated(self, config: dict, search_space: str="", environment: str="" , seed: int=None, budget: int=199, static: bool=True):

        if search_space == "":
            assert self.search_space != None, "Please set the search space"
            search_space = self.search_space
            static = self.static

        if environment == "":
            assert self.environment != None, "Please set the environment"
            environment = self.environment

        if seed == None:
            assert self.seed != None, "Please set a seed"
            seed = self.seed

        if self.rl_algorithm is None:
            if static:
                if os.path.exists(os.path.join(DATA_PATH, '%s_%s_%s.json' % (search_space, environment, seed))):
                    with open(os.path.join(DATA_PATH, '%s_%s_%s.json' % (search_space, environment, seed))) as mm:
                        min_max = json.load(mm)
                lr = int(config.get("lr"))
                gamma = config.get("gamma")
                if search_space in ["DDPG", "SAC", "TD3"]:
                    tau = config.get("tau")
                    with open(os.path.join(self.data_path, 'data_arl_bench', search_space, environment,
                                        '%s_%s_random_lr_%s_gamma_%s_tau_%s_seed%s_eval.json'%(environment, search_space,
                                                                                            lr, gamma, tau, seed))) as f:
                        data = json.load(f)
                        train_timesteps_index = budget
                    return self._build_return_dict(data, budget, train_timesteps_index)

                elif search_space == "PPO":
                    clip = config.get("clip")
                    with open(os.path.join(self.data_path, 'data_arl_bench', search_space, environment,
                                        '%s_%s_random_lr_%s_gamma_%s_clip_%s_seed%s_eval.json'%(environment, search_space,
                                                                                            lr, gamma, clip, seed))) as f:
                        data = json.load(f)
                        train_timesteps_index = budget

                    return self._build_return_dict(data, budget, train_timesteps_index)

                elif search_space == "A2C":
                    with open(os.path.join(self.data_path, 'data_arl_bench', search_space, environment,
                                        '%s_%s_random_lr_%s_gamma_%s_seed%s_eval.json'%(environment, search_space,
                                                                                            lr, gamma, seed))) as f:
                        data = json.load(f)
                        train_timesteps_index = budget
                    return self._build_return_dict(data, budget, train_timesteps_index)

            else:

                assert search_space in self.valid_dynamic_spaces, "This is not a valid dynamic space"
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

                with open(os.path.join(self.data_path, 'data_arl_bench', search_space, environment,
                                    '%s_%s_random_lr_%s%s%s_gamma_%.2f%.2f%.2f_seed%s_eval.json'%(environment, search_space,
                                                                                            lrs[0], lrs[1], lrs[2],
                                                                                            gammas[0], gammas[1],
                                                                                            gammas[2], seed))) as f:
                    data = json.load(f)
                    train_timesteps_index = data["timesteps_train"].index(data["timesteps_eval"][budget-1])
                return self._build_return_dict(data, budget, train_timesteps_index)
        else:
            for key in self.search_space_dict.keys():
                assert key in config.keys(), "The configuration must define a value for all the hyperparameters in the search space."
            data = run_rl_algorithm(rl_algorithm=self.rl_algorithm, rl_algorithm_name=search_space, config=config,
                                    environment=environment, seed=seed, total_timesteps=budget * 1e4)
            train_timesteps_index = budget
            return self._build_return_dict(data, budget, train_timesteps_index)

    def _precompute_configurations(self):

        self.precomputed_configurations = []
        search_space_structure = self.get_search_space(self.search_space)
        hps_names = list(search_space_structure.keys())
        for hps in itertools.product(*tuple(list(search_space_structure.values()))):
            self.precomputed_configurations.append(dict(zip(hps_names, hps)))


    def sample_configuration(self):

        configuration = {}
        for hp, values in self.get_search_space(self.search_space):
            configuration[hp] = np.random.choice(values)

        return configuration

    def run_bo(self, optimizer, epochs):

        observed_lc = {}
        configurations = self.precomputed_configurations.copy()

        next_conf_ix = np.random.randint(0, len(configurations))
        observed_lc[next_conf_ix]  = self.get_metrics(configurations[next_conf_ix], budget = 1)["eval_avg_returns"]

        for _ in range(epochs):
            next_conf_ix, budget = optimizer.observe_and_suggest(configurations, observed_lc)

            assert budget>0,"Negative budgets are not allowed"
            assert budget<100, "Upper bound of budget reached"

            observed_lc[next_conf_ix]  = self.get_metrics(configurations[next_conf_ix], budget = budget)["eval_avg_returns"]

        max_per_lc = [max(lc) for lc in observed_lc.values()]

        return observed_lc, max_per_lc, configurations[np.argmax(max_per_lc)],  list(observed_lc.keys())[np.argmax(max_per_lc)]
